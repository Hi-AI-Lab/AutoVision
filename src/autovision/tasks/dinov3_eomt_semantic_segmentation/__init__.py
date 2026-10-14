# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT semantic segmentation (DINOv3 variant): encoder-only mask
transformer backed by the HF ``EomtDinov3`` implementation."""

from autovision.tasks.dinov3_eomt_semantic_segmentation.config import (
    Dinov3EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.dinov3_eomt_semantic_segmentation.task_model import (
    Dinov3EomtSemanticSegmentation,
)
from autovision.tasks.dinov3_eomt_semantic_segmentation.train_model import (
    Dinov3EomtSemanticSegmentationTrain,
)

__all__ = [
    "Dinov3EomtSemanticSegmentation",
    "Dinov3EomtSemanticSegmentationTrain",
    "Dinov3EomtSemanticSegmentationTrainArgs",
]
