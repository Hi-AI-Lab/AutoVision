# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""AV-DETR detection transform recipe.

Two things set the AV-DETR family apart from the generic RT-DETR recipe
(baseline ``ltdetr_transforms``):

- **Patch-grid arithmetic**: the image size is divisible by ``2 ×
  patch_size`` (the model args snap the default 640 -> 644 for patch-14
  backbones), and batch scale-jitter sizes stay on the same grid so the
  ViT patch count stays integral at every training resolution.
- **Mosaic** is part of the baseline recipe (prob 0.5; 0.75 for the
  EdgeCrafter "v2" variants) alongside the shared photometric /
  zoom-out / IoU-crop / flip / mixup set. Copy-paste blending
  (``copyblend``) exists upstream but is not ported yet — tracked as
  roadmap.
"""

from __future__ import annotations

import math
from typing import Any

from autovision.augment.detection.args import (
    RTDETRDetectionTransformArgs,
)
from autovision.types import ImageSizeTuple

from .config import resolve_avdetr_variant

__all__ = [
    "build_avdetr_train_transform_args",
    "avdetr_scale_jitter_sizes",
]


def avdetr_scale_jitter_sizes(
    image_size: ImageSizeTuple, divisor: int, total: int = 30
) -> list[int]:
    """Scale-jitter ladder aligned to ``2 × patch_size`` (baseline rule).

    Distinct sizes are the multiples of ``divisor`` covering ±25 % of the
    base size (for 640/32 that reproduces the stock 480..800 ladder); the
    base size then fills the list up to ``total`` entries so the base
    resolution dominates sampling, mirroring the baseline distribution.
    """
    base = image_size[0]
    k_min = math.floor(0.75 * base / divisor)
    k_max = math.ceil(1.25 * base / divisor)
    sizes = [k * divisor for k in range(k_min, k_max + 1)]
    sizes = [size for size in sizes if size > 0]
    if base not in sizes:
        sizes.append(base)
        sizes.sort()
    return sizes + [base] * max(0, total - len(sizes))


def build_avdetr_train_transform_args(
    image_size: ImageSizeTuple,
    backbone: str,
    overrides: dict[str, Any] | None = None,
) -> RTDETRDetectionTransformArgs:
    """AV-DETR train transform args: patch-aligned jitter + mosaic on."""
    _, spec = resolve_avdetr_variant(f"avdetr/{backbone}")
    mosaic_prob = 0.75 if spec.family == "ecvit" else 0.5
    payload: dict[str, Any] = {
        "image_size": tuple(image_size),
        "mosaic_prob": mosaic_prob,
        # Baseline AVDETRv2ObjectDetectionTrainTransformArgs: the ECViT
        # "v2" family lifts BOTH mixup and mosaic to 0.75 (the v1 family
        # keeps the 0.5 default).
        "mixup_prob": 0.75 if spec.family == "ecvit" else 0.5,
        "scale_jitter": {
            "sizes": avdetr_scale_jitter_sizes(tuple(image_size), 2 * spec.patch_size),
            "step_stop": "auto",
        },
        **(overrides or {}),
    }
    return RTDETRDetectionTransformArgs.model_validate(payload)
