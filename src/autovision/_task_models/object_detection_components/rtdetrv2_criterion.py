# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RT-DETRv2 criterion: Varifocal (VFL) classification + L1 + GIoU boxes.

Losses (baseline weight dict): ``loss_vfl 1.0 + loss_bbox 5.0 + loss_giou
2.0``; the box losses are normalized by the total number of target boxes
(averaged across ranks with ``all_reduce``). Auxiliary decoder layers are
weighted with factor 1.0 through the same loss names (``*_aux_{i}``), like
the baseline deep supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from autovision._distributed import reduce_dict
from autovision._task_models.object_detection_components.box_ops import (
    generalized_box_iou_loss,
)
from autovision._task_models.object_detection_components.denoising import (
    get_cdn_matched_indices,
)

__all__ = ["RTDETRCriterionV2", "VFL_LOSS_WEIGHT_DICT"]

VFL_LOSS_WEIGHT_DICT = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def loss_labels_vfl(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Varifocal loss with IoU-scored targets on the matched queries.

    Positives: BCE against the matched IoU score, weighted by
    ``target_score^gamma``. Negatives: BCE against 0, weighted by
    ``pred_score^gamma`` (the quality-focal formulation).

    ``query_mask``: optional (B, Q) bool marking *valid* queries. With
    density-guided dynamic queries the padded tail of every image is
    excluded from the classification loss (baseline Dome semantics —
    box losses need no mask, they only touch matched pairs).
    """
    logits = outputs["pred_logits"]  # (B, Q, C)
    batch_size, num_queries, num_classes = logits.shape
    # fp32 targets: IoU scores are fp32, and under bf16 autocast the
    # index_put_ above would otherwise clash with a bf16 destination.
    target_scores = torch.zeros_like(logits, dtype=torch.float32)
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        ious = _matched_iou(
            outputs["pred_boxes"][batch_index, src].detach(),
            targets[batch_index]["boxes"][dst].to(outputs["pred_boxes"].device),
        )
        classes = targets[batch_index]["labels"][dst].to(logits.device)
        target_scores[batch_index, src, classes] = ious
    pred_scores = logits.sigmoid()
    # Baseline RT-DETRv2 VFL: weight = alpha * pred^gamma * (1 - target) + target
    weight = alpha * pred_scores.detach().pow(gamma) * (1 - target_scores) + target_scores
    loss = F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none")
    loss = loss * weight
    if query_mask is not None:
        loss = loss * query_mask.to(loss.dtype).unsqueeze(-1)
    return loss.sum() / num_boxes.clamp(min=1)


def _matched_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """IoU of matched pairs (rows aligned)."""
    from torchvision.ops import box_iou

    pred_xyxy = _cxcywh_to_xyxy_unit(pred_boxes)
    target_xyxy = _cxcywh_to_xyxy_unit(target_boxes)
    ious = torch.diag(box_iou(pred_xyxy, target_xyxy))
    return ious.clamp(min=0.0)


def _cxcywh_to_xyxy_unit(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou

    return box_iou(_cxcywh_to_xyxy_unit(boxes1), _cxcywh_to_xyxy_unit(boxes2))


def loss_boxes(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """L1 + GIoU on matched pairs, normalized by num_boxes (rank-reduced)."""
    l1_total = outputs["pred_boxes"].sum() * 0.0
    giou_total = outputs["pred_boxes"].sum() * 0.0
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred = outputs["pred_boxes"][batch_index, src]
        target = targets[batch_index]["boxes"][dst]
        l1_total = l1_total + (pred - target).abs().sum()
        # generalized_box_iou_loss is PAIRWISE (N, M): taking .sum() over the
        # full matrix would add every pred-vs-every-other-GT cross term. The
        # matched pairs live on the diagonal.
        giou_matrix = generalized_box_iou_loss(
            _cxcywh_to_xyxy_unit(pred), _cxcywh_to_xyxy_unit(target)
        )
        giou_total = giou_total + giou_matrix.diagonal().sum()
    return {
        "loss_bbox": l1_total / num_boxes.clamp(min=1),
        "loss_giou": giou_total / num_boxes.clamp(min=1),
    }


class RTDETRCriterionV2(nn.Module):
    """Matches + computes the weighted RT-DETRv2 loss over all decoder outputs.

    When ``dn_outputs``/``dn_meta`` are provided (contrastive denoising
    training), each denoising layer is supervised against its *known*
    assignment (positive noised query ↔ its source GT; no matching needed),
    normalized by ``num_boxes × dn_num_group`` — baseline semantics.
    """

    def __init__(
        self, matcher: nn.Module, weight_dict: dict[str, float] | None = None
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = dict(weight_dict or VFL_LOSS_WEIGHT_DICT)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to the module-level VFL loss (subclass override seam)."""
        return loss_labels_vfl(outputs, targets, indices, num_boxes, query_mask=query_mask)

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Delegate to the module-level box loss (subclass override seam)."""
        return loss_boxes(outputs, targets, indices, num_boxes)

    def forward(
        self,
        outputs_list: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        world_size: int = 1,
        dn_outputs: list[dict[str, torch.Tensor]] | None = None,
        dn_meta: dict | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        num_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
        num_boxes = torch.tensor(float(num_boxes), device=outputs_list[-1]["pred_logits"].device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes /= torch.distributed.get_world_size()

        losses: dict[str, torch.Tensor] = {}
        num_entries = len(outputs_list)
        for output_index, outputs in enumerate(outputs_list):
            indices = self.matcher(outputs, targets)
            suffix = "" if output_index == num_entries - 1 else f"_aux_{output_index}"
            losses[f"loss_vfl{suffix}"] = self.loss_labels(
                outputs, targets, indices, num_boxes, query_mask=query_mask
            )
            for name, value in self.loss_boxes(outputs, targets, indices, num_boxes).items():
                losses[f"{name}{suffix}"] = value

        if dn_outputs and dn_meta is not None:
            dn_indices = get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = torch.clamp(
                num_boxes * dn_meta["dn_num_group"], min=1
            )
            for dn_index, dn_layer in enumerate(dn_outputs):
                suffix = f"_dn_{dn_index}"
                losses[f"loss_vfl{suffix}"] = self.loss_labels(
                    dn_layer, targets, dn_indices, dn_num_boxes
                )
                for name, value in self.loss_boxes(
                    dn_layer, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}{suffix}"] = value

        weights = dict(self.weight_dict)
        for output_index in range(num_entries - 1):
            for key, value in self.weight_dict.items():
                weights[f"{key}_aux_{output_index}"] = value
        for dn_index in range(len(dn_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_dn_{dn_index}"] = value
        total = sum(losses[key] * weight for key, weight in weights.items() if key in losses)
        losses["loss"] = total
        return reduce_dict(losses, average=False)
