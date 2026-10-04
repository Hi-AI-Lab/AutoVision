# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Hungarian matcher for DETR-family detectors.

Cost (baseline RT-DETRv2): ``2.0 * focal_cls + 5.0 * L1 + 2.0 * GIoU``.
The cost matrix is guarded with ``nan_to_num`` for AMP stability.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from autovision._task_models.object_detection_components.box_ops import (
    generalized_box_iou_loss,
)

__all__ = ["HungarianMatcher"]


@torch.no_grad()
def forward_match(
    matcher: "HungarianMatcher",
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute one-to-one assignments.

    ``outputs["pred_logits"]: (B, num_queries, num_classes)`` (raw),
    ``outputs["pred_boxes"]: (B, num_queries, 4)`` normalized cxcywh.
    ``targets[i]["boxes"]: (n_i, 4)`` cxcywh, ``targets[i]["labels"]: (n_i,)``.
    Returns per-image ``(query_indices, target_indices)`` tensors.
    """
    batch_logits = outputs["pred_logits"]
    batch_boxes = outputs["pred_boxes"]
    num_queries = batch_logits.shape[1]

    indices = []
    for batch_index in range(batch_logits.shape[0]):
        target = targets[batch_index]
        if target["boxes"].numel() == 0:
            indices.append(
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
            )
            continue
        target_boxes = target["boxes"].to(batch_boxes.device)
        target_labels = target["labels"].to(batch_logits.device)
        pred_logits = batch_logits[batch_index]  # (Q, C)
        pred_boxes = batch_boxes[batch_index]  # (Q, 4)

        # Focal-style classification cost (baseline alpha/gamma).
        pred_scores = pred_logits.sigmoid()
        negative_cost = (1 - matcher.alpha) * pred_scores.pow(matcher.gamma)
        positive_cost = matcher.alpha * (1 - pred_scores).pow(matcher.gamma)
        class_cost = torch.gather(
            positive_cost - negative_cost,
            1,
            target_labels[None].repeat(num_queries, 1),
        )  # (Q, n)

        l1_cost = torch.cdist(pred_boxes, target_boxes, p=1)  # (Q, n)
        giou_cost = generalized_box_iou_loss(
            pred_boxes, target_boxes
        )  # (Q, n)

        cost = (
            matcher.cost_class * class_cost
            + matcher.cost_bbox * l1_cost
            + matcher.cost_giou * giou_cost
        )
        cost = torch.nan_to_num(cost, nan=1.0)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append(
            (
                torch.as_tensor(rows, dtype=torch.int64),
                torch.as_tensor(cols, dtype=torch.int64),
            )
        )
    return indices


class HungarianMatcher(nn.Module):
    """One-to-one query-to-ground-truth assignment (DETREGiven)."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return forward_match(self, outputs, targets)
