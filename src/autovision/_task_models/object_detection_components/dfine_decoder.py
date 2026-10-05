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
  matched segment starts at ``dn_num_split[0]``);
- ``query_features``: the final layer's hidden states after the
  denoising split — the exact input of the last ``dec_score_head``
  (the classification head). Emitted only in training; the per-object
  SSL contrastive head (design doc 14) consumes it.
- ``ssl_candidate_features`` (Route B, ``ssl_candidate_features=True``):
  the top-k candidate features gathered from the encoder memory before
  detach — the label-free SSL-only pool. With the flag set, both this
  and ``query_features`` are emitted in EVAL mode as well, so the
  validation loss is the held-out contrastive loss (design doc 14
  §4.10). Inert for every baseline contract.

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

    def __init__(self, *args, reg_max: int = 32, ssl_candidate_features: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reg_max = reg_max
        # Route B emission switch (design doc 14 §4.10): when set, the
        # decoder exposes the top-k candidate features (the SSL-only pool)
        # and the final-layer hidden states in EVAL mode too — the
        # contrastive loss is the whole objective, so the validation loss
        # is the held-out SSL loss. Inert for every baseline contract.
        self._emit_query_features = ssl_candidate_features
        self._emit_candidate_features = ssl_candidate_features
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

        (
            matched_bboxes,
            matched_logits,
            corners_list,
            refs_list,
            dn_out_bboxes,
            dn_out_logits,
            dn_corners_list,
            dn_refs_list,
            matched_hidden,
        ) = self._refine_layers(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
            attn_mask, dn_meta,
        )

        out: dict = {
            "pred_logits": matched_logits[-1],
            "pred_boxes": matched_bboxes[-1],
        }

        if self.training or self._emit_query_features:
            out["query_features"] = matched_hidden
            if self._emit_candidate_features:
                # Set during _get_decoder_input above (same forward call).
                out["ssl_candidate_features"] = self._candidate_features

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

    def _refine_layers(
        self,
        init_ref_contents: torch.Tensor,
        init_ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes,
        attn_mask: torch.Tensor | None,
        dn_meta: dict | None,
    ):
        """FDR refinement loop + matched/denoising split (shared contract).

        Consumes the prepared query contents/boxes (baseline top-k or the
        density-guided dynamic selection in the Dome subclass) and runs the
        per-layer refinement; returns the stacked per-layer outputs split
        into the matched and denoising segments, plus the final layer's
        hidden states (matched segment) for the SSL contrastive head:

        ``(matched_bboxes, matched_logits, corners_list, refs_list,
        dn_out_bboxes, dn_out_logits, dn_corners_list, dn_refs_list,
        matched_hidden)``
        with the ``dn_*`` entries ``None`` / ``[]`` when there is no
        denoising segment.
        """
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
            # The hidden states live under the same prepended-denoising
            # layout as the per-layer outputs: only the matched segment is
            # exposed (the denoising segment's hidden states have no
            # consumer — its queries never enter the contrastive pool).
            matched_hidden = output[:, dn_num:]
        else:
            matched_bboxes, matched_logits = out_bboxes, out_logits
            dn_out_bboxes = dn_out_logits = None
            dn_corners_list = dn_refs_list = []
            matched_hidden = output

        return (
            matched_bboxes,
            matched_logits,
            corners_list,
            refs_list,
            dn_out_bboxes,
            dn_out_logits,
            dn_corners_list,
            dn_refs_list,
            matched_hidden,
        )
