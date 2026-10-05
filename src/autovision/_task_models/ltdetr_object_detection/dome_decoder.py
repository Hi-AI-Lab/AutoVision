# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Density-guided dynamic-query decoder (Dome-DETR PAQI port, doc 13 §4.3).

Subclasses :class:`DFINEv2Decoder` and replaces the fixed top-``num_queries``
proposal selection with the baseline Dome ``_get_decoder_input`` semantics
(``src/zoo/dome/dome_decoder.py:767-943``):

1. score every grid cell with the encoder heads and take the top
   ``max_num_select`` candidates;
2. the first ``min_num_select`` (highest-scoring) pass through unconditionally;
3. the rest are gated by the MWAS window mask — candidates whose anchor
   center falls into an unselected (low-density) window are dropped;
4. survivors get one per-class NMS pass whose per-candidate IoU threshold
   is modulated by the window-pooled density at the candidate center
   (``base + scale * density``): dense clusters keep more overlapping
   candidates, sparse regions are pruned aggressively;
5. per-image query counts end up data-dependent (``min ≤ n_b ≤ max``);
   the batch is zero-padded to the maximum and described by
   ``batch_queries_num`` / ``query_valid_mask``.

Deviations from the baseline (design doc 13 §5):

- D4: pooled-density lookups use consistent (y, x) indexing (the baseline
  transposes x/y at ``dome_decoder.py:847-850``, invisible on square
  inputs);
- D5: padded queries get their logits filled with -1e4 at the source, so
  neither the Hungarian matcher nor the decode path can pick them (the
  baseline only masks the classification loss).

The FDR refinement loop is inherited verbatim via
:meth:`DFINEv2Decoder._refine_layers`; this class only changes query
preparation, the denoising-group call (variable query count), and the
output contract (three extra keys).

Export: the density-gated selection is data-dependent (nonzero, per-class
loops) and cannot trace through ``torch.export`` — :meth:`convert_to_deploy`
flips the decoder back to the fixed top-``min_num_select`` baseline path
(design doc 13 §4.8).
"""

from __future__ import annotations

import torch

from autovision._task_models.object_detection_components.denoising import (
    box_cxcywh_to_xyxy,
    get_contrastive_denoising_training_group,
)
from autovision._task_models.object_detection_components.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision._task_models.object_detection_components.dynamic_nms import (
    per_class_dynamic_nms,
)

__all__ = ["DomeDFINEv2Decoder"]

#: Fill value for padded-query logits (D5): sigmoid(-1e4) == 0 exactly in
#: fp32, so padded queries contribute ~0 to VFL, are prohibitive in the
#: matcher's focal cost, and vanish under the decode threshold.
_PADDED_LOGIT = -1e4


class DomeDFINEv2Decoder(DFINEv2Decoder):
    """D-FINE decoder with density-guided dynamic query selection."""

    def __init__(
        self,
        *args,
        min_num_select: int = 300,
        max_num_select: int = 500,
        nms_iou_base: float = 0.4,
        nms_iou_density_scale: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < min_num_select <= max_num_select:
            raise ValueError(
                "Dome decoder needs 0 < min_num_select <= max_num_select, "
                f"got {min_num_select}/{max_num_select}."
            )
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select
        self.nms_iou_base = nms_iou_base
        self.nms_iou_density_scale = nms_iou_density_scale
        self.num_heads = self.decoder.layers[0].self_attn.num_heads
        self._static_export = False

    def convert_to_deploy(self) -> None:
        """Static-export mode: fixed top-``min_num_select`` proposals."""
        self._static_export = True

    # ── query selection ─────────────────────────────────────────────────
    def _get_decoder_input_dynamic(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        defe_info: dict,
    ):
        """Density-gated query selection (baseline Dome flow, D4/D5 fixes).

        Returns ``(content, ref_boxes_unact, batch_queries_num,
        query_valid_mask)`` with the batch padded to the per-batch maximum.
        """
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, device=memory.device, dtype=memory.dtype
        )
        if anchors.shape[0] == 1 and memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_logits = self.enc_score_head(output_memory)
        enc_coords_unact = self.enc_bbox_head(output_memory) + anchors

        # Candidate pool: top-max by max class logit (baseline "default"
        # query selection); fewer cells may exist on tiny inputs.
        topk = min(self.max_num_select, enc_logits.shape[1])
        enc_scores, enc_class_ids = enc_logits.max(-1)
        topk_ind = enc_scores.topk(topk, dim=-1).indices

        def _gather(t: torch.Tensor) -> torch.Tensor:
            return t.gather(
                1, topk_ind.unsqueeze(-1).expand(-1, -1, t.shape[-1])
            )

        mem_k = _gather(output_memory)
        # The NMS only reads per-candidate max score + class id — gather
        # those 1-D views instead of the full (B, K, C) logit matrix.
        scores_k = enc_scores.gather(1, topk_ind)
        class_k = enc_class_ids.gather(1, topk_ind)
        anchors_k = _gather(anchors)
        coords_k = _gather(enc_coords_unact)

        # Window gating: candidates beyond the first min_num_select must
        # fall into a density-selected window (consistent (y, x) — D4).
        window_mask = defe_info["window_mask"]  # (B, n_h, n_w) bool
        pooled = defe_info["density_pooled"]  # (B, 1, n_h, n_w) fp32
        n_h, n_w = window_mask.shape[1], window_mask.shape[2]
        anchor_x = anchors_k[..., 0].sigmoid()
        anchor_y = anchors_k[..., 1].sigmoid()
        window_y = (anchor_y * n_h).long().clamp(0, n_h - 1)
        window_x = (anchor_x * n_w).long().clamp(0, n_w - 1)
        batch_index = torch.arange(mem_k.shape[0], device=mem_k.device)
        gate = window_mask[batch_index[:, None], window_y, window_x]  # (B, K)
        min_num = min(self.min_num_select, topk)
        gate[:, :min_num] = True

        pooled_h, pooled_w = pooled.shape[-2:]
        kept_memory: list[torch.Tensor] = []
        kept_coords: list[torch.Tensor] = []
        queries_num: list[int] = []
        for b in range(mem_k.shape[0]):
            rest = gate[b, min_num:]
            mem_i = torch.cat([mem_k[b, :min_num], mem_k[b, min_num:][rest]], dim=0)
            scores_i = torch.cat(
                [scores_k[b, :min_num], scores_k[b, min_num:][rest]], dim=0
            )
            class_i = torch.cat(
                [class_k[b, :min_num], class_k[b, min_num:][rest]], dim=0
            )
            coords_i = torch.cat(
                [coords_k[b, :min_num], coords_k[b, min_num:][rest]], dim=0
            )

            # Density-modulated per-class NMS over the combined candidates;
            # the first min_num survive regardless (baseline semantics).
            proposals = coords_i.sigmoid()  # (n, 4) normalized cxcywh
            if proposals.shape[0] > min_num:
                center_y = proposals[:, 1]
                center_x = proposals[:, 0]
                density_y = (center_y * pooled_h).long().clamp(0, pooled_h - 1)
                density_x = (center_x * pooled_w).long().clamp(0, pooled_w - 1)
                density = pooled[b, 0, density_y, density_x].detach()
                iou_thresholds = (
                    self.nms_iou_base + self.nms_iou_density_scale * density
                )
                keep = per_class_dynamic_nms(
                    box_cxcywh_to_xyxy(proposals).detach(),
                    scores_i.detach(),
                    class_i.detach(),
                    iou_thresholds.detach(),
                )
                keep = keep[keep >= min_num]
                keep = torch.cat(
                    [
                        torch.arange(min_num, device=proposals.device),
                        keep,
                    ]
                )
                mem_i = mem_i[keep]
                coords_i = coords_i[keep]

            kept_memory.append(mem_i)
            kept_coords.append(coords_i)
            queries_num.append(int(mem_i.shape[0]))

        padded_queries = max(queries_num)
        content = output_memory.new_zeros(
            (mem_k.shape[0], padded_queries, mem_k.shape[-1])
        )
        ref_boxes_unact = coords_k.new_zeros((mem_k.shape[0], padded_queries, 4))
        query_valid_mask = torch.zeros(
            (mem_k.shape[0], padded_queries),
            dtype=torch.bool,
            device=mem_k.device,
        )
        for b in range(mem_k.shape[0]):
            n_b = queries_num[b]
            content[b, :n_b] = kept_memory[b]
            ref_boxes_unact[b, :n_b] = kept_coords[b]
            query_valid_mask[b, :n_b] = True

        return (
            content.detach(),
            ref_boxes_unact.detach(),
            queries_num,
            query_valid_mask,
        )

    def _padding_attention_mask(
        self,
        batch_queries_num: list[int],
        padded_queries: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Self-attention mask isolating padded queries (no denoising group)."""
        if all(n == padded_queries for n in batch_queries_num):
            return None
        mask = torch.zeros(
            (len(batch_queries_num) * self.num_heads, padded_queries, padded_queries),
            dtype=torch.bool,
            device=device,
        )
        for b, n_b in enumerate(batch_queries_num):
            if n_b >= padded_queries:
                continue
            head_start = b * self.num_heads
            head_end = (b + 1) * self.num_heads
            mask[head_start:head_end, :n_b, n_b:] = True
            mask[head_start:head_end, n_b:, :n_b] = True
        return mask

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, feats, targets=None, defe_info: dict | None = None):
        if self._static_export or defe_info is None:
            # Export mode / non-density path: verbatim baseline contract
            # (fixed top-num_queries proposals, no density involvement).
            return super().forward(feats, targets=targets)

        memory, spatial_shapes = self._get_encoder_input(feats)

        (
            init_ref_contents,
            init_ref_points_unact,
            batch_queries_num,
            query_valid_mask,
        ) = self._get_decoder_input_dynamic(memory, spatial_shapes, defe_info)
        padded_queries = max(batch_queries_num)

        denoising_logits = denoising_bbox_unact = None
        attn_mask = dn_meta = None
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    padded_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    batch_queries_num=batch_queries_num,
                    num_heads=self.num_heads,
                )
            )
            if denoising_bbox_unact is not None:
                init_ref_points_unact = torch.cat(
                    [denoising_bbox_unact, init_ref_points_unact], dim=1
                )
                init_ref_contents = torch.cat(
                    [denoising_logits, init_ref_contents], dim=1
                )
        if attn_mask is None:
            # No denoising group (zero-GT batch / eval): still isolate the
            # padded queries from the valid ones.
            attn_mask = self._padding_attention_mask(
                batch_queries_num, padded_queries, memory.device
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
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            attn_mask,
            dn_meta,
        )

        # D5: padded queries must never produce detections or be matched —
        # bury their class logits at the source (every decoder layer).
        if not all(n == padded_queries for n in batch_queries_num):
            invalid = (~query_valid_mask).unsqueeze(0).unsqueeze(-1)
            matched_logits = matched_logits.masked_fill(invalid, _PADDED_LOGIT)

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
            # Final-layer hidden states over the padded query set — the
            # per-object SSL head consumes them (design doc 14); padded
            # rows are filtered by the criterion (matcher + query mask).
            out["query_features"] = matched_hidden
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits, dn_out_bboxes
                )
                out["dn_meta"] = dn_meta
                out["dn_pred_corners_list"] = dn_corners_list
                out["dn_ref_boxes_list"] = dn_refs_list

        out["batch_queries_num"] = batch_queries_num
        out["query_valid_mask"] = query_valid_mask
        # Stamp the selection bounds so the criterion's count loss reads a
        # single source of truth (design doc 13 §4.1).
        defe_info.setdefault("min_num_select", self.min_num_select)
        defe_info.setdefault("max_num_select", self.max_num_select)
        out["defe_info"] = defe_info
        return out
