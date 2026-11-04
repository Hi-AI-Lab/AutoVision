# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Post-transform target-quality gates (design D3).

Geometric augmentations (the mosaic's whole-canvas affine, any future
rotate/crop op) can shear a legitimate target into garbage: a sliver a
fraction of its former area, an extreme aspect ratio, or a sub-pixel
strip. Left in the batch, those labels poison the loss far out of
proportion to their count. :func:`box_candidates` compares each target
before vs after the transform and marks the ones worth keeping; the
mosaic wiring applies it jointly to boxes and classes so the label rows
never desynchronize.

The thresholds are deliberately lenient (2 px, aspect 100, 10% area) —
this gate owns DEGENERATION, not size; the pipeline's
``SanitizeBoundingBoxes`` / ``min_bbox_size_px`` own the size floor.
"""

from __future__ import annotations

from typing import Final

import torch

__all__ = ["box_candidates"]

#: Denominator guard for zero-area rows (degenerate inputs fail via the
#: aspect-ratio clause, not by dividing by zero).
_EPS: Final[float] = 1e-16


def box_candidates(
    boxes_before: torch.Tensor,
    boxes_after: torch.Tensor,
    *,
    min_wh: float = 2.0,
    max_aspect_ratio: float = 100.0,
    min_area_ratio: float = 0.1,
) -> torch.Tensor:
    """Per-target keep mask comparing boxes across a geometric transform.

    Args:
        boxes_before: ``(N, 4)`` xyxy, pixel coordinates, pre-transform.
        boxes_after: ``(N, 4)`` xyxy in the SAME image space (post-transform).
        min_wh: Minimum width AND height (px) after the transform.
        max_aspect_ratio: Reject rows whose longer/shorter side ratio
            reaches this bound — checked on BOTH sides so an input that
            was already a sliver cannot "pass" by staying a sliver.
        min_area_ratio: Minimum ``area_after / area_before``.

    Returns:
        ``(N,)`` bool tensor; ``False`` for any row that degenerated,
        including rows that were degenerate before the transform.
    """
    before = boxes_before.detach().float()
    after = boxes_after.detach().float()
    width_before = (before[:, 2] - before[:, 0]).clamp_min(0.0)
    height_before = (before[:, 3] - before[:, 1]).clamp_min(0.0)
    width_after = (after[:, 2] - after[:, 0]).clamp_min(0.0)
    height_after = (after[:, 3] - after[:, 1]).clamp_min(0.0)

    aspect_before = torch.maximum(width_before, height_before) / (
        torch.minimum(width_before, height_before) + _EPS
    )
    aspect_after = torch.maximum(width_after, height_after) / (
        torch.minimum(width_after, height_after) + _EPS
    )
    area_ratio = (width_after * height_after) / (
        width_before * height_before + _EPS
    )

    return (
        (width_after > min_wh)
        & (height_after > min_wh)
        & (area_ratio > min_area_ratio)
        & (aspect_before < max_aspect_ratio)
        & (aspect_after < max_aspect_ratio)
    )
