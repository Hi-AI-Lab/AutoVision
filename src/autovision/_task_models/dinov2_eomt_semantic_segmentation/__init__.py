# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT semantic segmentation (DINOv2 variant): encoder-only mask
transformer trained end to end via the HF implementation."""

from autovision._task_models.dinov2_eomt_semantic_segmentation.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)
from autovision._task_models.dinov2_eomt_semantic_segmentation.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision._task_models.dinov2_eomt_semantic_segmentation.train_model import (
    Dinov2EomtSemanticSegmentationTrain,
)

__all__ = [
    "Dinov2EomtSemanticSegmentation",
    "Dinov2EomtSemanticSegmentationTrain",
    "Dinov2EomtSemanticSegmentationTrainArgs",
]
