# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT instance segmentation (DINOv2 variant): encoder-only mask
transformer with class-agnostic queries and Hungarian mask matching."""

from autovision._task_models.dinov2_eomt_instance_segmentation.config import (
    Dinov2EomtInstanceSegmentationTrainArgs,
)
from autovision._task_models.dinov2_eomt_instance_segmentation.task_model import (
    Dinov2EomtInstanceSegmentation,
)
from autovision._task_models.dinov2_eomt_instance_segmentation.train_model import (
    Dinov2EomtInstanceSegmentationTrain,
)

__all__ = [
    "Dinov2EomtInstanceSegmentation",
    "Dinov2EomtInstanceSegmentationTrain",
    "Dinov2EomtInstanceSegmentationTrainArgs",
]
