# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT instance segmentation (DINOv3 variant)."""

from autovision._task_models.dinov3_eomt_instance_segmentation.config import (
    Dinov3EomtInstanceSegmentationTrainArgs,
)
from autovision._task_models.dinov3_eomt_instance_segmentation.task_model import (
    Dinov3EomtInstanceSegmentation,
)
from autovision._task_models.dinov3_eomt_instance_segmentation.train_model import (
    Dinov3EomtInstanceSegmentationTrain,
)

__all__ = [
    "Dinov3EomtInstanceSegmentation",
    "Dinov3EomtInstanceSegmentationTrain",
    "Dinov3EomtInstanceSegmentationTrainArgs",
]
