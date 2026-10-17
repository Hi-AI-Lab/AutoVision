# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pure geometry math shared across augment / metrics / tasks layers.

No torch.nn modules and no framework imports beyond torch itself, so every
layer may import downward from here (design doc 30: this retires the two
``ALLOWED_UPWARD`` debts for box math).
"""

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
    generalized_box_iou_loss,
)

__all__ = [
    "box_cxcywh_norm_to_xyxy",
    "box_xyxy_to_cxcywh_norm",
    "generalized_box_iou_loss",
]
