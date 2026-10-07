# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RF-DETR object detection task model (HF transformers backend)."""

from autovision._task_models.rfdetr_object_detection.config import (
    RFDETRObjectDetectionTrainArgs,
    resolve_rfdetr_variant,
)
from autovision._task_models.rfdetr_object_detection.task_model import (
    RFDETRObjectDetection,
)
from autovision._task_models.rfdetr_object_detection.train_model import (
    RFDETRObjectDetectionTrain,
)

__all__ = [
    "RFDETRObjectDetection",
    "RFDETRObjectDetectionTrain",
    "RFDETRObjectDetectionTrainArgs",
    "resolve_rfdetr_variant",
]
