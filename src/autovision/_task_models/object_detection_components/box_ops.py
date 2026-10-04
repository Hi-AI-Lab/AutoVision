# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Bounding box utilities for detection tasks.

Internal representation throughout the framework: normalized
``(x_center, y_center, w, h)`` in [0, 1] (albumentations/YOLO convention).
"""

from __future__ import annotations

import torch
from torchvision.ops import box_convert

__all__ = [
    "box_cxcywh_norm_to_xyxy",
    "box_xyxy_to_cxcywh_norm",
    "generalized_box_iou_loss",
]


def box_cxcywh_norm_to_xyxy(
    boxes: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Normalized (cx, cy, w, h) -> absolute (x1, y1, x2, y2) pixels.

    ``size`` is (width, height) — the multiplication order matters.
    """
    w, h = size
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        dim=-1,
    )


def box_xyxy_to_cxcywh_norm(
    boxes: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Absolute (x1, y1, x2, y2) pixels -> normalized (cx, cy, w, h)."""
    w, h = size
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
    boxes = boxes / torch.tensor([w, h, w, h], device=boxes.device)
    return boxes


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Standard GIoU (Gilmer et al., CVPR 2019) on absolute xyxy boxes.

    Self-implemented: torchvision's version assumes integer-pixel boxes
    (+1 conventions) and returns -inf for identical unit-scale boxes.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-9)

    elt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    erb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    ewh = (erb - elt).clamp(min=0)
    area_c = ewh[..., 0] * ewh[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-9)


def generalized_box_iou_loss(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Pairwise GIoU loss (1 - GIoU, range [0, 2]) on absolute xyxy boxes.

    No upper clamp: GIoU's defining property is its gradient for
    non-overlapping pairs (1-GIoU saturating at 1 would zero it).
    """
    giou = _generalized_box_iou(boxes1, boxes2)
    loss = 1 - giou
    return torch.nan_to_num(loss, nan=1.0, posinf=2.0, neginf=-1.0)
