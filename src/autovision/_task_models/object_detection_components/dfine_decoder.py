# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE decoder: the RT-DETRv2 skeleton with distributional refinement.

Shares the RT-DETRv2 pipeline verbatim (encoder proposals, contrastive
denoising, self/cross attention layers, fp32 islands around box heads) and
swaps the box-delta head for D-FINE's distributional head: each layer
predicts per-side distance distributions over ``2 × reg_max + 1`` bins and
refines the previous layer's box (the fine-grained distribution refinement
loop). See :mod:`dfine_utils` for the distance parameterization.

Training-only output extensions (consumed by ``DFINECriterionV2``):

- ``pred_corners_list`` / ``ref_boxes_list``: per-layer raw corner logits
  and the reference box each layer refined from (matched-query segment);
- ``dn_pred_corners_list`` / ``dn_ref_boxes_list``: same for the
  contrastive-denoising segment (denoising queries are prepended, so the
  matched segment starts at ``dn_num_split[0]``).

The export/inference contract is unchanged: ``pred_logits`` + normalized
cxcywh ``pred_boxes`` (final layer), feeding the shared decode path.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.init as init
from torch import nn

from autovision._task_models.object_detection_components.denoising import (
    get_contrastive_denoising_training_group,
)
from autovision._task_models.object_detection_components.dfine_utils import (
    distance2bbox,
    integrate_corners,
)
from autovision._task_models.object_detection_components.rtdetrv2_decoder import (
    MLP,
    RTDETRv2Decoder,
)

__all__ = ["DFINEv2Decoder"]


class DFINEv2Decoder(RTDETRv2Decoder):
    """RT-DETRv2 decoder + D-FINE distributional bbox heads (FDR loop)."""

    def __init__(self, *args, reg_max: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reg_max = reg_max
        num_bins = 2 * reg_max + 1
        self.register_buffer(
            "proj",
            (torch.arange(num_bins, dtype=torch.float32) - reg_max) / reg_max,
            persistent=False,
        )
        # Rebuild the per-layer box heads as distributional heads: the base
        # class initialized them for 4-dim box deltas. Zero-initialized last
        # layers emit uniform distributions → the initial box IS the
        # reference (refinement starts from the encoder proposals).
        self.dec_bbox_head = nn.ModuleList(
            MLP(self.hidden_dim, self.hidden_dim, 4 * num_bins, 3)
            for _ in range(self.num_layers)
        )
        for head in self.dec_bbox_head:
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)

    @property
    def num_bins(self) -> int:
        return 2 * self.reg_max + 1

    def forward(self, feats: List[torch.Tensor], targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None, None, None, None,
            )

        init_ref_contents, init_ref_points_unact = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        # D-FINE refinement loop in absolute (sigmoid) space: layer i refines
        # the box of layer i-1 through a residual side-distance distribution.
        ref_boxes = init_ref_points_unact.sigmoid().float()
        output = init_ref_contents
        out_bboxes: list[torch.Tensor] = []
        out_logits: list[torch.Tensor] = []
        corners_list: list[torch.Tensor] = []
        refs_list: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.decoder.layers):
            refs_list.append(ref_boxes)
            ref_detached = ref_boxes.detach()
            output = layer(
                output,
                ref_detached.unsqueeze(2),
                memory,
                spatial_shapes,
                attn_mask,
                None,
                self.query_pos_head(ref_detached),
            )

            # fp32 island: the distribution head + integration is the AMP
            # overflow hotspot (softmax over 2·reg_max+1 bins × 4 sides).
            with torch.autocast(device_type=output.device.type, enabled=False):
                corners = self.dec_bbox_head[layer_index](output.float())
                distances = integrate_corners(corners, self.proj.to(corners.dtype))
                bbox = distance2bbox(ref_boxes.float(), distances)

            out_bboxes.append(bbox)
            out_logits.append(self.dec_score_head[layer_index](output))
            corners_list.append(corners)

            if not self.training and layer_index == self.decoder.eval_idx:
                break
            # Next layer's reference: the detached decoded box (the refinement
            # loop is greedy — gradients reach layer i's head only through its
            # own corners).
            ref_boxes = bbox.detach()

        out_bboxes = torch.stack(out_bboxes)
        out_logits = torch.stack(out_logits)

        if self.training and dn_meta is not None:
            dn_out_bboxes, matched_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, matched_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_num = dn_meta["dn_num_split"][0]
            dn_corners_list = [c[:, :dn_num] for c in corners_list]
            dn_refs_list = [r[:, :dn_num] for r in refs_list]
            corners_list = [c[:, dn_num:] for c in corners_list]
            refs_list = [r[:, dn_num:] for r in refs_list]
        else:
            matched_bboxes, matched_logits = out_bboxes, out_logits
            dn_out_bboxes = dn_out_logits = None
            dn_corners_list = dn_refs_list = []

        out: dict = {
            "pred_logits": matched_logits[-1],
            "pred_boxes": matched_bboxes[-1],
        }

        if self.training:
            out["aux_outputs"] = self._set_aux_loss(
                matched_logits[:-1], matched_bboxes[:-1]
            )
            out["pred_corners_list"] = corners_list
            out["ref_boxes_list"] = refs_list
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits, dn_out_bboxes
                )
                out["dn_meta"] = dn_meta
                out["dn_pred_corners_list"] = dn_corners_list
                out["dn_ref_boxes_list"] = dn_refs_list

        return out
