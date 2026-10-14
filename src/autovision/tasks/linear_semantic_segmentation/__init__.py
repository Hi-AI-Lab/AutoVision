# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Linear semantic segmentation task model package."""

from autovision.tasks.linear_semantic_segmentation.config import (
    LinearSemanticSegmentationTrainArgs,
)
from autovision.tasks.linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)
from autovision.tasks.linear_semantic_segmentation.train_model import (
    LinearSemanticSegmentationTrain,
)

__all__ = [
    "LinearSemanticSegmentation",
    "LinearSemanticSegmentationTrain",
    "LinearSemanticSegmentationTrainArgs",
]
