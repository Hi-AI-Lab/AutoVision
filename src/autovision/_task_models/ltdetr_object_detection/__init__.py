# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""LT-DETR object detection family (SSL-backbone detection finetuning)."""

from autovision._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
    LTDETRObjectDetectionTrainArgs,
    LTDETRVariantSpec,
    list_ltdetr_model_names,
    resolve_ltdetr_variant,
)
from autovision._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from autovision._task_models.ltdetr_object_detection.train_model import (
    LTDETRObjectDetectionTrain,
)

__all__ = [
    "LTDETR_MODEL_REGISTRY",
    "LTDETRObjectDetection",
    "LTDETRObjectDetectionTrain",
    "LTDETRObjectDetectionTrainArgs",
    "LTDETRVariantSpec",
    "list_ltdetr_model_names",
    "resolve_ltdetr_variant",
]
