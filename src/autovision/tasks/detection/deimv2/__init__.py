# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DEIMv2 object detection task model (HF transformers backend)."""

from autovision.tasks.detection.deimv2.config import (
    DEIMV2_MODEL_REGISTRY,
    Deimv2ObjectDetectionTrainArgs,
    resolve_deimv2_variant,
)
from autovision.tasks.detection.deimv2.task_model import (
    Deimv2ObjectDetection,
)
from autovision.tasks.detection.deimv2.train_model import (
    Deimv2ObjectDetectionTrain,
)

__all__ = [
    "DEIMV2_MODEL_REGISTRY",
    "Deimv2ObjectDetection",
    "Deimv2ObjectDetectionTrain",
    "Deimv2ObjectDetectionTrainArgs",
    "resolve_deimv2_variant",
]
