# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE criterion: RT-DETRv2 losses + FGL (0.15) + DDF (1.5) [+ SSL].

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

Auxiliary layers are deep-supervised with factor 1.0 through the ``*_aux_i``
/ ``*_dn_i`` suffixes like the baseline; the dn branch is normalized by
``num_boxes × dn_num_group``.
"""

from __future__ import annotations

import torch
from torch import nn

from autovision._distributed import reduce_dict
from autovision._task_models.object_detection_components.defe import (
    defe_count_loss,
    defe_density_loss,
)
from autovision._task_models.object_detection_components.denoising import (
    get_cdn_matched_indices,
)
from autovision._task_models.object_detection_components.dfine_utils import (
    ddf_loss,
    fg_loss,
)
from autovision._task_models.object_detection_components.rtdetrv2_criterion import (
    loss_boxes,
    loss_labels_vfl,
)
from autovision._task_models.object_detection_components.ssl_contrastive import (
    SSLParams,
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
        ssl_params: per-object contrastive SSL contract (design doc 14); ``None``
            disables the objective entirely (no ``loss_ssl`` key, no head wiring).
    """

    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: dict[str, float] | None = None,
        reg_max: int = 32,
        fgl_loss_weight: float = 0.15,
        ddf_loss_weight: float = 1.5,
        defe_density_map_weight: float = 1.0,
        defe_count_loss_weight: float = 1.0,
        defe_underestimation_penalty: float = 0.3,
        defe_count_underestimation_weight: float = 2.0,
        ssl_params: SSLParams | None = None,
    ) -> None:
        super().__init__()
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
        self.fgl_loss_weight = fgl_loss_weight
        self.ddf_loss_weight = ddf_loss_weight
        self.defe_density_map_weight = defe_density_map_weight
        self.defe_count_loss_weight = defe_count_loss_weight
        self.defe_underestimation_penalty = defe_underestimation_penalty
        self.defe_count_underestimation_weight = defe_count_underestimation_weight
        self.ssl_params = ssl_params
        if ssl_params is not None:
            self.weight_dict["loss_ssl"] = ssl_params.loss_coef
        # Written by the training wrapper each step; drives the
        # every_n_steps gate of the SSL objective (state lives outside the
        # forward call, mirroring the RF-DETR-SSL contract).
        self.ssl_step: int = 0
        # Diagnostics from the most recent SSL evaluation (design doc 14
        # D10): matched-pool size and paired-pool size on this rank.
        self.last_ssl_stats: dict[str, float] | None = None

    def forward(
        self,
        outputs_list: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        world_size: int = 1,
        dn_outputs: list[dict[str, torch.Tensor]] | None = None,
        dn_meta: dict | None = None,
        corners_list: list[torch.Tensor] | None = None,
        ref_boxes_list: list[torch.Tensor] | None = None,
        dn_corners_list: list[torch.Tensor] | None = None,
        dn_ref_boxes_list: list[torch.Tensor] | None = None,
        query_mask: torch.Tensor | None = None,
        defe_info: dict | None = None,
        ssl_z1: torch.Tensor | None = None,
        ssl_z2: torch.Tensor | None = None,
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
            if not (isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)):
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
        for output_index, outputs in enumerate(outputs_list):
            indices = self.matcher(outputs, targets)
            layer_indices.append(indices)
            suffix = "" if output_index == num_entries - 1 else f"_aux_{output_index}"
            losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                outputs, targets, indices, num_boxes, query_mask=query_mask
            )
            for name, value in loss_boxes(outputs, targets, indices, num_boxes).items():
                losses[f"{name}{suffix}"] = value
            if corners_list is not None and ref_boxes_list is not None:
                losses[f"loss_fgl{suffix}"] = fg_loss(
                    corners_list[output_index],
                    ref_boxes_list[output_index],
                    outputs["pred_boxes"],
                    targets,
                    indices,
                    num_boxes,
                    self.reg_max,
                )

        dn_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        dn_num_boxes: torch.Tensor | None = None
        if dn_outputs and dn_meta is not None:
            dn_indices = get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = torch.clamp(num_boxes * dn_meta["dn_num_group"], min=1)
            for dn_index, dn_layer in enumerate(dn_outputs):
                suffix = f"_dn_{dn_index}"
                losses[f"loss_vfl{suffix}"] = loss_labels_vfl(
                    dn_layer, targets, dn_indices, dn_num_boxes
                )
                for name, value in loss_boxes(
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
                    )

        # DDF: the final layer's distributions teach every intermediate layer.
        if corners_list is not None and len(corners_list) >= 2 and self.ddf_loss_weight > 0:
            for output_index in range(num_entries - 1):
                losses[f"loss_ddf_aux_{output_index}"] = ddf_loss(
                    corners_list[output_index],
                    corners_list[-1],
                    outputs_list[output_index]["pred_boxes"],
                    outputs_list[-1]["pred_boxes"],
                    layer_indices[output_index],
                    num_boxes,
                    self.reg_max,
                )
            if dn_corners_list and dn_outputs and dn_indices is not None:
                for dn_index in range(len(dn_corners_list) - 1):
                    losses[f"loss_ddf_dn_{dn_index}"] = ddf_loss(
                        dn_corners_list[dn_index],
                        dn_corners_list[-1],
                        dn_outputs[dn_index]["pred_boxes"],
                        dn_outputs[-1]["pred_boxes"],
                        dn_indices,
                        dn_num_boxes,
                        self.reg_max,
                    )

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
            if not (isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)):
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
                self.last_ssl_stats = ssl_stats
            else:
                # Stride miss: don't let the previous hit's diagnostics
                # leak into this step's log.
                self.last_ssl_stats = None

        weights = dict(self.weight_dict)
        for output_index in range(num_entries - 1):
            for key, value in self.weight_dict.items():
                weights[f"{key}_aux_{output_index}"] = value
        for dn_index in range(len(dn_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_dn_{dn_index}"] = value

        total = sum(
            losses[key] * weight for key, weight in weights.items() if key in losses
        )
        losses["loss"] = total
        return reduce_dict(losses, average=False)
