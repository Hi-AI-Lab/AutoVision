# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE criterion: RT-DETRv2 losses + FGL (0.15) + DDF (1.5).

Runs ONE Hungarian matching pass per decoder layer (the D-FINE losses share
the classification/box assignments) and extends the RT-DETRv2 loss set
(VFL 1.0 + L1 5.0 + GIoU 2.0) with:

- ``loss_fgl``: fine-grained localization (see :func:`dfine_utils.fg_loss`);
- ``loss_ddf``: decoupled distillation focal — the final layer's side
  distributions teach every intermediate layer (aux and denoising), see
  :func:`dfine_utils.ddf_loss`.

Auxiliary layers are deep-supervised with factor 1.0 through the ``*_aux_i``
/ ``*_dn_i`` suffixes like the baseline; the dn branch is normalized by
``num_boxes × dn_num_group``.
"""

from __future__ import annotations

import torch
from torch import nn

from autovision._distributed import reduce_dict
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

__all__ = ["DFINECriterionV2"]


class DFINECriterionV2(nn.Module):
    """RT-DETRv2 criterion + distributional refinement losses (FGL/DDF)."""

    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: dict[str, float] | None = None,
        reg_max: int = 32,
        fgl_loss_weight: float = 0.15,
        ddf_loss_weight: float = 1.5,
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
            }
        )
        self.reg_max = reg_max
        self.fgl_loss_weight = fgl_loss_weight
        self.ddf_loss_weight = ddf_loss_weight

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
    ) -> dict[str, torch.Tensor]:
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
                outputs, targets, indices, num_boxes
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
