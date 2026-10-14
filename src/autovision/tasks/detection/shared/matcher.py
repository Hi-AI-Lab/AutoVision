# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Hungarian matcher for DETR-family detectors.

Source lineage: the RT-DETRv2/D-FINE matching-cost design (Apache-2.0,
``Peterande/D-FINE`` @ ``956d170``), re-implemented for this framework;
ledger ``docs/provenance/LEDGER.md`` (P1-4).

Cost (baseline RT-DETRv2): ``2.0 * focal_cls + 5.0 * L1 + 2.0 * GIoU``.
The cost matrix is guarded with ``nan_to_num`` for AMP stability.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from autovision.geometry.box_ops import (
    clip_normalize_boxes,
    generalized_box_iou_loss,
)

__all__ = ["HungarianMatcher", "indices_to_device"]


def indices_to_device(
    indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Move per-image match indices to *device* (host→device, once).

    The matcher returns CPU tensors (scipy). Downstream losses index CUDA
    tensors with them — every such advanced-indexing op then pays a
    synchronous host→device copy. Hoisting once per matcher call makes
    every consumer (VFL/box/FGL/DDF loops) sync-free.
    """
    return [
        (src.to(device, non_blocking=True), dst.to(device, non_blocking=True))
        for src, dst in indices
    ]


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

    The cost matrices are built on the accelerator and moved to the host in
    ONE bulk transfer for the whole batch: a per-image ``.cpu()`` would
    stall the CUDA pipeline once per image (the criterion runs several
    matcher passes per training step), while scipy still assigns per image
    — a block-diagonal batched assignment would be O((Σn)³) instead of
    Σ O(n³). Assignments are bit-identical to the per-image transfer.
    """
    batch_logits = outputs["pred_logits"]
    batch_boxes = outputs["pred_boxes"]
    # Baseline guard: NaN/inf → 0 and clamp to the unit box before costs.
    batch_boxes = torch.nan_to_num(batch_boxes, nan=0.0, posinf=1.0, neginf=0.0).clamp(
        min=0.0, max=1.0
    )
    indices: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    costs: list[torch.Tensor] = []
    sizes: list[int] = []
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

        # Focal-style classification cost (baseline formula: the alpha/gamma
        # powers each scale a -log term — dropping the logs, as an earlier
        # revision did, changes the Hungarian assignment).
        pred_scores = pred_logits.sigmoid()
        tgt_scores = pred_scores[:, target_labels]  # (Q, n)
        neg_cost = (
            (1 - matcher.alpha)
            * tgt_scores.pow(matcher.gamma)
            * (-(1 - tgt_scores + 1e-8).log())
        )
        pos_cost = (
            matcher.alpha
            * (1 - tgt_scores).pow(matcher.gamma)
            * (-(tgt_scores + 1e-8).log())
        )
        class_cost = pos_cost - neg_cost  # (Q, n)

        # Baseline sanitizes the predicted boxes before the cost metrics
        # (line ~95): degenerate NaN/inf boxes get clamped costs instead
        # of every pair falling back to the NaN->1.0 guard.
        pred_boxes = clip_normalize_boxes(pred_boxes)
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
        costs.append(cost)
        sizes.append(cost.numel())
        indices.append(None)  # placeholder keeps batch ordering
    if costs:
        # shapes are static metadata — only this transfer waits on the GPU
        flat = torch.cat([cost.reshape(-1) for cost in costs]).cpu()
        for slot, cost_cpu in zip(
            (i for i, entry in enumerate(indices) if entry is None),
            flat.split(sizes),
        ):
            target = targets[slot]
            rows, cols = linear_sum_assignment(
                cost_cpu.reshape(-1, target["boxes"].shape[0]).numpy()
            )
            indices[slot] = (
                torch.as_tensor(rows, dtype=torch.int64),
                torch.as_tensor(cols, dtype=torch.int64),
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
