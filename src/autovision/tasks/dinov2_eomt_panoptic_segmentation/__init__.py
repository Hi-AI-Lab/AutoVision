# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT panoptic segmentation (DINOv2 variant): encoder-only mask
transformer decoding queries into per-pixel (class, segment) maps."""

from autovision.tasks.dinov2_eomt_panoptic_segmentation.config import (
    Dinov2EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.dinov2_eomt_panoptic_segmentation.task_model import (
    Dinov2EomtPanopticSegmentation,
)
from autovision.tasks.dinov2_eomt_panoptic_segmentation.train_model import (
    Dinov2EomtPanopticSegmentationTrain,
)

__all__ = [
    "Dinov2EomtPanopticSegmentation",
    "Dinov2EomtPanopticSegmentationTrain",
    "Dinov2EomtPanopticSegmentationTrainArgs",
]
