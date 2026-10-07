# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE criterion: RT-DETRv2 losses + FGL (0.15) + DDF (1.5) [+ SSL] [+ SIoU].

Runs ONE Hungarian matching pass per decoder layer (the D-FINE losses share
the classification/box assignments) and extends the RT-DETRv2 loss set
(VFL 1.0 + L1 5.0 + GIoU 2.0) with:

- ``loss_fgl``: fine-grained localization (see :func:`dfine_utils.fg_loss`);
- ``loss_ddf``: decoupled distillation focal — the final layer's side
  distributions teach every intermediate layer (aux and denoising), see
  :func:`dfine_utils.ddf_loss`;
- ``loss_ssl``: the per-object contrastive objective (design doc 14),
  computed ONCE at the final layer with the final layer's Hungarian match.
  Like the DeFE pair it carries no ``_aux_i`` / ``_dn_i`` suffixes; the
  weight-aggregation loop copies every ``weight_dict`` entry to the aux/dn
  suffixes, but the ``if key in losses`` filter drops the never-produced
  suffixed SSL keys.

``box_regression_iou="siou"`` (design doc 15) swaps the box loss's GIoU
term for SIoU (``loss_siou``) across every branch — main, ``_pre``,
``_dn_pre``, and the copied ``_aux_i`` / ``_dn_i`` / ``_enc_i`` suffixes.

Auxiliary layers are deep-supervised with factor 1.0 through the ``*_aux_i``
/ ``*_dn_i`` suffixes like the baseline; the dn branch is normalized by
``num_boxes × dn_num_group``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from autovision._distributed import reduce_dict
from autovision._task_models.object_detection_components.defe import (
    defe_count_loss,
    defe_density_loss,
)
from autovision._task_models.object_detection_components.denoising import (
    box_cxcywh_to_xyxy,
    get_cdn_matched_indices,
)
from autovision._task_models.object_detection_components.dfine_utils import (
    DEFAULT_REG_SCALE,
    bbox2distance,
    ddf_loss,
    fg_loss,
    fg_loss_from_targets,
    matched_pair_iou,
)
from autovision._task_models.object_detection_components.matcher import (
    indices_to_device,
)
from autovision._task_models.object_detection_components.rtdetrv2_criterion import (
    loss_boxes,
    loss_labels_vfl,
)
from autovision._task_models.object_detection_components.ssl_contrastive import (
    SSLParams,
    epps_pulley_sigreg,
    ssl_contrastive_loss,
)

__all__ = ["DFINECriterionV2"]


class DFINECriterionV2(nn.Module):
    """RT-DETRv2 criterion + distributional refinement losses (FGL/DDF).

    With ``defe_info`` (density-guided dynamic queries, design doc 13
    §4.5) two auxiliary DeFE losses join the set: the density-map loss
    and the object-count regression loss. With ``ssl_params`` the
    per-object contrastive ``loss_ssl`` joins it (design doc 14 §4.4).
    Both ride the same ``weight_dict`` mechanics as every other term —
    computed once, no ``_aux_i`` / ``_dn_i`` suffixes.

    Args:
        matcher: module able to compute the matching between proposals and targets
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        dn_fgl_loss_weight: FGL weight on the denoising branches (``_dn_*``).
            ``None`` follows the main ``loss_fgl`` weight (baseline parity).
            Dense tiny-object batches (e.g. VisDrone) push CDN to a single
            group of 2 × max-GT noised queries whose FGL residuals saturate
            the reg_max bins — the dn FGL term then dominates the total loss
            without converging; lower it there.
        ssl_params: per-object contrastive SSL contract (design doc 14); ``None``
            disables the objective entirely (no ``loss_ssl`` key, no head wiring).
        box_regression_iou: IoU term of the supervised box loss (design doc 15).
            ``"giou"`` (default) keeps the upstream ``loss_giou`` emission;
            ``"siou"`` replaces it with ``loss_siou`` (Scylla-IoU,
            arXiv:2205.12740) across every box-loss branch (main, ``_pre``,
            ``_dn_pre``, ``_aux_i``, ``_dn_i``, ``_enc_i``) — mutually exclusive
            emissions. The matcher's GIoU cost is intentionally untouched.
        siou_loss_coef: weight of ``loss_siou`` in siou mode; ``None`` (default)
            inherits the ``loss_giou`` weight so the swapped-in term keeps the
            anchor weight (E0 probe: loss-scale ratio 0.953 on VisDrone).
        siou_theta: shape-cost exponent of the SIoU loss (paper default 4).
    """

    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: dict[str, float] | None = None,
        reg_max: int = 32,
        reg_scale: float = DEFAULT_REG_SCALE,
        fgl_loss_weight: float = 0.15,
        dn_fgl_loss_weight: float | None = None,
        ddf_loss_weight: float = 1.5,
        defe_density_map_weight: float = 1.0,
        defe_count_loss_weight: float = 1.0,
        defe_underestimation_penalty: float = 0.3,
        defe_count_underestimation_weight: float = 2.0,
        ssl_params: SSLParams | None = None,
        box_regression_iou: Literal["giou", "siou"] = "giou",
        siou_loss_coef: float | None = None,
        siou_theta: float = 4.0,
    ) -> None:
        super().__init__()
        if box_regression_iou not in ("giou", "siou"):
            raise ValueError(
                f"Unknown box_regression_iou {box_regression_iou!r}; "
                "expected 'giou' or 'siou'."
            )
        self.matcher = matcher
        self.weight_dict = dict(
            weight_dict
            or {
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": fgl_loss_weight,
                "loss_ddf": ddf_loss_weight,
                "defe_density_loss": defe_density_map_weight,
                "defe_count_loss": defe_count_loss_weight,
            }
        )
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.fgl_loss_weight = fgl_loss_weight
        self.dn_fgl_loss_weight = dn_fgl_loss_weight
        self.ddf_loss_weight = ddf_loss_weight
        self.defe_density_map_weight = defe_density_map_weight
        self.defe_count_loss_weight = defe_count_loss_weight
        self.defe_underestimation_penalty = defe_underestimation_penalty
        self.defe_count_underestimation_weight = defe_count_underestimation_weight
        self.ssl_params = ssl_params
        if ssl_params is not None:
            self.weight_dict["loss_ssl"] = ssl_params.loss_coef
        if ssl_params is not None and ssl_params.sigreg_coef > 0:
            # SIGReg third leg (design doc 16, LeVJEPA R1-A1): own weight so
            # the raw statistic is scaled exactly once, independent of
            # loss_ssl.
            self.weight_dict["loss_sigreg"] = ssl_params.sigreg_coef
        # IoU term of the box loss (design doc 15): siou mode swaps the
        # weight_dict entry — the inheritance base is the PRE-pop loss_giou
        # weight (default dict: 2.0; an explicitly supplied weight_dict is
        # the same source) — then retires loss_giou so the dict matches the
        # mutually-exclusive emission set.
        self.box_regression_iou = box_regression_iou
        self.siou_theta = siou_theta
        self._iou_loss_key = (
            "loss_siou" if box_regression_iou == "siou" else "loss_giou"
        )
        if box_regression_iou == "siou":
            inherited = self.weight_dict.get("loss_giou", 2.0)
            self.weight_dict.pop("loss_giou", None)
            self.weight_dict["loss_siou"] = (
                inherited if siou_loss_coef is None else siou_loss_coef
            )
        # Written by the training wrapper each step; drives the
        # every_n_steps gate of the SSL objective (state lives outside the
        # forward call, mirroring the RF-DETR-SSL contract).
        self.ssl_step: int = 0
        # DDF pos/neg normalization counts cached from the aux pass for the
        # dn pass within the same forward (official semantics). Kept as
        # on-device scalars — float() here would sync the GPU every step.
        self._ddf_num_pos: torch.Tensor | None = None
        self._ddf_num_neg: torch.Tensor | None = None
        # Diagnostics from the most recent SSL evaluation (design doc 14
        # D10): matched-pool size and paired-pool size on this rank.
        self.last_ssl_stats: dict[str, float] | None = None

    def _loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Module-level box loss bound to this criterion's IoU term (doc 15)."""
        return loss_boxes(
            outputs,
            targets,
            indices,
            num_boxes,
            iou_term=self.box_regression_iou,
            siou_theta=self.siou_theta,
        )

    def forward(
        self,
        outputs_list: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        world_size: int = 1,
        dn_outputs: list[dict[str, torch.Tensor]] | None = None,
        dn_meta: dict | None = None,
        dn_pre_outputs: dict[str, torch.Tensor] | None = None,
        corners_list: list[torch.Tensor] | None = None,
        ref_boxes_list: list[torch.Tensor] | None = None,
        dn_corners_list: list[torch.Tensor] | None = None,
        dn_ref_boxes_list: list[torch.Tensor] | None = None,
        query_mask: torch.Tensor | None = None,
        pre_outputs: dict[str, torch.Tensor] | None = None,
        enc_aux_outputs: list[dict[str, torch.Tensor]] | None = None,
        enc_meta: dict | None = None,
        defe_info: dict | None = None,
        ssl_z1: torch.Tensor | None = None,
        ssl_z2: torch.Tensor | None = None,
        ssl_s1: torch.Tensor | None = None,
        ssl_s2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Route B (design doc 14 §4.10): label-free SSL-only pretraining.
        # The candidate pool needs no Hungarian match and no targets, so
        # the detection losses (and their matcher pass) are skipped
        # entirely; the contrastive objective IS the loss — in train AND
        # eval, so the validation loss is the held-out SSL loss (the
        # stage-1 model-selection signal). The stride gate does not apply:
        # there is no competing detection loss to interleave with.
        if (
            self.ssl_params is not None
            and self.ssl_params.instance_source == "encoder_candidates"
        ):
            if not (
                isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)
            ):
                raise ValueError(
                    "ssl instance_source='encoder_candidates' requires the model "
                    "outputs to carry ssl_z1/ssl_z2 candidate features; build the "
                    "task model with the same ssl config (design doc 14 §4.10)."
                )
            ssl_stats: dict[str, float] = {}
            ssl_only_losses: dict[str, torch.Tensor] = {
                "loss_ssl": ssl_contrastive_loss(
                    ssl_z1,
                    ssl_z2,
                    outputs_list[-1],
                    targets,
                    [],
                    self.ssl_params,
                    stats_sink=ssl_stats,
                )
            }
            self.last_ssl_stats = ssl_stats
            ssl_only_losses["loss"] = (
                ssl_only_losses["loss_ssl"] * self.weight_dict["loss_ssl"]
            )
            return reduce_dict(ssl_only_losses, average=False)

        num_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
        num_boxes = torch.tensor(
            float(num_boxes), device=outputs_list[-1]["pred_logits"].device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes /= torch.distributed.get_world_size()

        losses: dict[str, torch.Tensor] = {}
        num_entries = len(outputs_list)
        layer_indices: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        device = outputs_list[-1]["pred_logits"].device
        for output_index, outputs in enumerate(outputs_list):
            indices = indices_to_device(self.matcher(outputs, targets), device)
            layer_indices.append(indices)
            suffix = "" if output_index == num_entries - 1 else f"_aux_{output_index}"
            # VFL keeps each layer's own Hungarian assignment (official).
            losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                outputs, targets, indices, num_boxes, query_mask=query_mask
            )

        # Union matching across all decoded layers + the pre-output head
        # + the encoder proposals (official ``_get_go_indices``): box/GIoU
        # losses are computed ONCE on the final layer over the union; FGL
        # targets are cached once against the fixed initial anchor and
        # reused by every layer.
        aux_indices = layer_indices[:-1]
        pre_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if pre_outputs is not None:
            pre_indices = indices_to_device(self.matcher(pre_outputs, targets), device)
        enc_indices_list: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        if enc_aux_outputs:
            class_agnostic = bool(enc_meta.get("class_agnostic", False))
            enc_targets_list = (
                [
                    {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
                    for t in targets
                ]
                if class_agnostic
                else targets
            )
            for enc_outputs in enc_aux_outputs:
                enc_indices_list.append(
                    indices_to_device(self.matcher(enc_outputs, enc_targets_list), device)
                )
        indices_go = self._get_go_indices(
            layer_indices[-1],
            aux_indices + ([pre_indices] if pre_indices else []) + enc_indices_list,
        )
        num_boxes_go = torch.tensor(
            float(sum(int(src.numel()) for src, _ in indices_go)),
            device=num_boxes.device,
        ).clamp(min=1)

        if pre_outputs is not None:
            losses["loss_vfl_pre"] = loss_labels_vfl(
                pre_outputs, targets, pre_indices, num_boxes, query_mask=query_mask
            )
            # Box losses ride the union assignment (baseline: the pre head
            # is one more entry in the go-indices union for "boxes").
            for name, value in self._loss_boxes(
                pre_outputs, targets, indices_go, num_boxes_go
            ).items():
                losses[f"{name}_pre"] = value

        final_outputs = outputs_list[-1]
        for name, value in self._loss_boxes(
            final_outputs, targets, indices_go, num_boxes_go
        ).items():
            losses[name] = value

        # Cached FGL targets against the fixed initial anchor over the union
        # assignment — identical for every layer (official caching semantics).
        anchor = ref_boxes_list[-1] if ref_boxes_list is not None else None
        fgl_targets: dict | None = None
        if corners_list is not None and anchor is not None:
            fgl_targets = {}
            for batch_index, (src, dst) in enumerate(indices_go):
                if src.numel() == 0:
                    continue
                gt_xyxy = box_cxcywh_to_xyxy(
                    targets[batch_index]["boxes"][dst].to(anchor.device)
                )
                lab, w_r, w_l = bbox2distance(
                    anchor[batch_index, src].detach(),
                    gt_xyxy,
                    self.reg_max,
                    self.reg_scale,
                )
                fgl_targets[batch_index] = (src, dst, lab, w_r, w_l)

        if fgl_targets is not None:
            for output_index in range(num_entries):
                suffix = (
                    "" if output_index == num_entries - 1 else f"_aux_{output_index}"
                )
                layer_corners = corners_list[output_index]
                layer_boxes = outputs_list[output_index]["pred_boxes"]
                layer_loss = layer_corners.sum() * 0.0
                for batch_index, (src, dst, lab, w_r, w_l) in fgl_targets.items():
                    quality = matched_pair_iou(
                        layer_boxes[batch_index, src].detach(),
                        targets[batch_index]["boxes"][dst].to(layer_boxes.device),
                    )
                    layer_loss = layer_loss + fg_loss_from_targets(
                        layer_corners[batch_index, src].reshape(
                            src.numel(), 4, self.reg_max + 1
                        ),
                        lab,
                        w_r,
                        w_l,
                        quality,
                        avg_factor=1.0,
                    )
                losses[f"loss_fgl{suffix}"] = layer_loss / num_boxes_go.clamp(min=1)

        dn_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        dn_num_boxes: torch.Tensor | None = None
        if dn_outputs and dn_meta is not None:
            dn_indices = get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = torch.clamp(num_boxes * dn_meta["dn_num_group"], min=1)
            # The denoising segment of the pre-output head (baseline
            # ``dn_pre_outputs``) gets the same VFL + box pass as the
            # matched pre head, over the CDN assignment.
            if dn_pre_outputs is not None:
                for name, value in self._loss_boxes(
                    dn_pre_outputs, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}_dn_pre"] = value
                losses["loss_vfl_dn_pre"] = loss_labels_vfl(
                    dn_pre_outputs, targets, dn_indices, dn_num_boxes
                )
            for dn_index, dn_layer in enumerate(dn_outputs):
                suffix = f"_dn_{dn_index}"
                losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                    dn_layer, targets, dn_indices, dn_num_boxes
                )
                for name, value in self._loss_boxes(
                    dn_layer, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}{suffix}"] = value
                if dn_corners_list is not None and dn_ref_boxes_list is not None:
                    losses[f"loss_fgl{suffix}"] = fg_loss(
                        dn_corners_list[dn_index],
                        dn_ref_boxes_list[dn_index],
                        dn_layer["pred_boxes"],
                        targets,
                        dn_indices,
                        dn_num_boxes,
                        self.reg_max,
                        self.reg_scale,
                    )

        # DDF: the final layer's distributions teach every intermediate
        # layer over ALL queries (official full form — matched weight IoU,
        # unmatched weight teacher max prob; the num_pos/num_neg counts are
        # computed on this pass and reused by the dn branch).
        if (
            corners_list is not None
            and len(corners_list) >= 2
            and self.ddf_loss_weight > 0
        ):
            for output_index in range(num_entries - 1):
                loss_ddf, num_pos, num_neg = ddf_loss(
                    corners_list[output_index],
                    corners_list[-1],
                    outputs_list[output_index]["pred_logits"],
                    outputs_list[-1]["pred_logits"],
                    outputs_list[output_index]["pred_boxes"],
                    outputs_list[-1]["pred_boxes"],
                    indices_go,
                    self.reg_max,
                    targets=targets,
                )
                self._ddf_num_pos, self._ddf_num_neg = num_pos, num_neg
                losses[f"loss_ddf_aux_{output_index}"] = loss_ddf
            if dn_corners_list and dn_outputs and dn_indices is not None:
                for dn_index in range(len(dn_corners_list) - 1):
                    loss_ddf, _, _ = ddf_loss(
                        dn_corners_list[dn_index],
                        dn_corners_list[-1],
                        dn_outputs[dn_index]["pred_logits"],
                        dn_outputs[-1]["pred_logits"],
                        dn_outputs[dn_index]["pred_boxes"],
                        dn_outputs[-1]["pred_boxes"],
                        dn_indices,
                        self.reg_max,
                        targets=targets,
                        counts=(self._ddf_num_pos, self._ddf_num_neg),
                    )
                    losses[f"loss_ddf_dn_{dn_index}"] = loss_ddf

        # DeFE auxiliary losses (density-guided dynamic queries). The GT
        # heatmap is only present in training; min/max_num_select ride
        # defe_info so the decoder config stays the single source of truth.
        # The loss weights live exclusively in weight_dict (applied once
        # in the aggregation below) — no inline multiplication.
        if defe_info is not None and "gt_heatmap" in defe_info:
            losses["defe_density_loss"] = defe_density_loss(
                defe_info["density_map"],
                defe_info["gt_heatmap"],
                underestimation_penalty=self.defe_underestimation_penalty,
            )
            losses["defe_count_loss"] = defe_count_loss(
                defe_info["reg_value"],
                [int(t["boxes"].shape[0]) for t in targets],
                min_num_select=defe_info["min_num_select"],
                max_num_select=defe_info["max_num_select"],
                underestimation_weight=self.defe_count_underestimation_weight,
            )

        # Per-object contrastive SSL objective (design doc 14). Final-layer
        # only — the aux/dn branches below repeat every entry of the loss
        # set with suffixed keys, and their outputs carry no SSL features.
        # Eval mode never enters the contrastive path (no gather
        # collectives on the validation branch). A training-mode call with
        # ssl_params but missing features is a head/criterion configuration
        # mismatch — fail loud instead of silently training without SSL.
        if self.ssl_params is not None and self.training:
            if not (
                isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)
            ):
                raise ValueError(
                    "ssl_params is set but the model outputs carry no "
                    "ssl_z1/ssl_z2 features; build the task model with the "
                    "same ssl config so the SSLProjectionHead is attached "
                    "(design doc 14 §4.4)."
                )
            stride = self.ssl_params.every_n_steps
            if stride <= 1 or self.ssl_step % stride == 0:
                ssl_stats: dict[str, float] = {}
                losses["loss_ssl"] = ssl_contrastive_loss(
                    ssl_z1,
                    ssl_z2,
                    outputs_list[-1],
                    targets,
                    layer_indices[-1],
                    self.ssl_params,
                    query_mask=query_mask,
                    stats_sink=ssl_stats,
                )
                if self.ssl_params.sigreg_coef > 0:
                    if not (
                        isinstance(ssl_s1, torch.Tensor)
                        and isinstance(ssl_s2, torch.Tensor)
                    ):
                        raise ValueError(
                            "ssl sigreg_coef>0 requires the model outputs to "
                            "carry ssl_s1/ssl_s2 raw projector taps; build the "
                            "task model with the same ssl sigreg config "
                            "(design doc 16 R1-A1)."
                        )
                    # Per-view halves of the query pool, batch-flattened to
                    # (B*Q, D). The reference pools the group-0 match per
                    # view; here the regularizer covers the full query
                    # embedding distribution (keeps the unmatched-query
                    # half of the cloud Gaussian too). Each half is a
                    # per-sample mean, so no pairing and no gather.
                    losses["loss_sigreg"] = 0.5 * (
                        epps_pulley_sigreg(
                            ssl_s1.reshape(-1, ssl_s1.shape[-1]),
                            num_slices=self.ssl_params.sigreg_slices,
                            num_knots=self.ssl_params.sigreg_knots,
                        )
                        + epps_pulley_sigreg(
                            ssl_s2.reshape(-1, ssl_s2.shape[-1]),
                            num_slices=self.ssl_params.sigreg_slices,
                            num_knots=self.ssl_params.sigreg_knots,
                        )
                    )
                self.last_ssl_stats = ssl_stats
            else:
                # Stride miss: don't let the previous hit's diagnostics
                # leak into this step's log.
                self.last_ssl_stats = None

        # Zero-fill any NaN term (degenerate empty-GT / zero-area-box paths)
        # so one poisoned branch cannot NaN the total: nan_to_num's backward
        # routes zero gradient to the NaN input, so the term drops out of
        # the step instead of corrupting it. The engine's non-finite guard
        # still catches a NaN TOTAL — the failure mode this must not mask.
        losses = {
            key: torch.nan_to_num(value, nan=0.0) for key, value in losses.items()
        }
        # Encoder-proposal supervision (baseline enc_aux chain — see the
        # rtdetrv2 criterion for rationale): the enc top-k selection is
        # non-differentiable and the decoder inputs are detached, so
        # without this branch the proposal heads never receive a gradient.
        # Full VFL pass on its own match; box losses ride the union
        # assignment (baseline ``use_uni_set``).
        if enc_aux_outputs:
            class_agnostic = bool(enc_meta.get("class_agnostic", False))
            for enc_index, enc_outputs in enumerate(enc_aux_outputs):
                enc_targets = (
                    [
                        {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
                        for t in targets
                    ]
                    if class_agnostic
                    else targets
                )
                suffix = f"_enc_{enc_index}"
                losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                    enc_outputs,
                    enc_targets,
                    enc_indices_list[enc_index],
                    num_boxes,
                )
                for name, value in self._loss_boxes(
                    enc_outputs, enc_targets, indices_go, num_boxes_go
                ).items():
                    losses[f"{name}{suffix}"] = value

        weights = dict(self.weight_dict)
        iou_key = self._iou_loss_key
        if pre_outputs is not None:
            weights["loss_vfl_pre"] = self.weight_dict["loss_vfl"]
            weights["loss_bbox_pre"] = self.weight_dict["loss_bbox"]
            weights[f"{iou_key}_pre"] = self.weight_dict[iou_key]
        if dn_pre_outputs is not None:
            weights["loss_vfl_dn_pre"] = self.weight_dict["loss_vfl"]
            weights["loss_bbox_dn_pre"] = self.weight_dict["loss_bbox"]
            weights[f"{iou_key}_dn_pre"] = self.weight_dict[iou_key]
        for enc_index in range(len(enc_aux_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_enc_{enc_index}"] = value
        for output_index in range(num_entries - 1):
            for key, value in self.weight_dict.items():
                weights[f"{key}_aux_{output_index}"] = value
        for dn_index in range(len(dn_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_dn_{dn_index}"] = value
            if self.dn_fgl_loss_weight is not None:
                weights[f"loss_fgl_dn_{dn_index}"] = self.dn_fgl_loss_weight

        total = sum(
            losses[key] * weight for key, weight in weights.items() if key in losses
        )
        losses["loss"] = total
        return reduce_dict(losses, average=False)

    def _get_go_indices(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Union of the per-layer Hungarian assignments (official form).

        Accumulates every layer's (query, gt) pairs, keeps pairs seen in
        multiple layers first, and de-duplicates per query so each query
        appears once with its most-frequent GT.

        Vectorized form of the official sequential loop: unique rows sorted
        by count (descending, STABLE so equal counts keep the lexicographic
        order of ``torch.unique``), then the first occurrence of each query
        row — its highest-count assignment (earliest-lexicographic on
        ties). Rows are emitted in first-appearance order, matching the
        official dict-insertion order. No ``.item()`` round-trips: the
        per-row Python loop cost grew with VisDrone's union size (hundreds
        of rows per image, twice per training step).
        """
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]
        for ind in [
            torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices
        ]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            # Stable count-descending sort: equal counts keep torch.unique's
            # lexicographic order. A query row's pairs are NOT contiguous in
            # this order, so group by row first (stable keeps the
            # count-desc order within a row) — each group's first entry is
            # that row's highest-count assignment (earliest-lexicographic
            # on ties), exactly what the old sequential dict loop kept.
            unique_sorted = unique[torch.argsort(counts, descending=True, stable=True)]
            row_order = torch.argsort(unique_sorted[:, 0], stable=True)
            grouped_rows = unique_sorted[row_order, 0]
            group_mask = torch.ones(
                grouped_rows.shape[0], dtype=torch.bool, device=grouped_rows.device
            )
            group_mask[1:] = grouped_rows[1:] != grouped_rows[:-1]
            positions = row_order[group_mask.nonzero(as_tuple=True)[0]]
            # Ascending positions reproduce the old loop's visit (and
            # dict-insertion) order.
            positions = positions.sort().values
            results.append(
                (
                    unique_sorted[positions, 0].long(),
                    unique_sorted[positions, 1].long(),
                )
            )
        return results
