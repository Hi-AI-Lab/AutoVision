# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DEIMv2 object detection task model (HF transformers backend)."""

from autovision._task_models.deimv2_object_detection.config import (
    DEIMV2_MODEL_REGISTRY,
    Deimv2ObjectDetectionTrainArgs,
    resolve_deimv2_variant,
)
from autovision._task_models.deimv2_object_detection.task_model import (
    Deimv2ObjectDetection,
)
from autovision._task_models.deimv2_object_detection.train_model import (
    Deimv2ObjectDetectionTrain,
)

__all__ = [
    "DEIMV2_MODEL_REGISTRY",
    "Deimv2ObjectDetection",
    "Deimv2ObjectDetectionTrain",
    "Deimv2ObjectDetectionTrainArgs",
    "resolve_deimv2_variant",
]
