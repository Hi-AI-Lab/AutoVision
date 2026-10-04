# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT panoptic segmentation (DINOv3 variant)."""

from autovision._task_models.dinov3_eomt_panoptic_segmentation.config import (
    Dinov3EomtPanopticSegmentationTrainArgs,
)
from autovision._task_models.dinov3_eomt_panoptic_segmentation.task_model import (
    Dinov3EomtPanopticSegmentation,
)
from autovision._task_models.dinov3_eomt_panoptic_segmentation.train_model import (
    Dinov3EomtPanopticSegmentationTrain,
)

__all__ = [
    "Dinov3EomtPanopticSegmentation",
    "Dinov3EomtPanopticSegmentationTrain",
    "Dinov3EomtPanopticSegmentationTrainArgs",
]
