# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""LT-DETR object detection family (SSL-backbone detection finetuning)."""

from autovision._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
    LTDETRDensityArgs,
    LTDETRObjectDetectionTrainArgs,
    LTDETRSSLArgs,
    LTDETRVariantSpec,
    list_ltdetr_model_names,
    resolve_ltdetr_variant,
)
from autovision._task_models.ltdetr_object_detection.dome_decoder import (
    DomeDFINEv2Decoder,
)
from autovision._task_models.ltdetr_object_detection.dome_encoder import (
    DomeHybridEncoder,
)
from autovision._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from autovision._task_models.ltdetr_object_detection.train_model import (
    LTDETRObjectDetectionTrain,
)

__all__ = [
    "LTDETR_MODEL_REGISTRY",
    "LTDETRDensityArgs",
    "DomeDFINEv2Decoder",
    "DomeHybridEncoder",
    "LTDETRObjectDetection",
    "LTDETRObjectDetectionTrain",
    "LTDETRObjectDetectionTrainArgs",
    "LTDETRSSLArgs",
    "LTDETRVariantSpec",
    "list_ltdetr_model_names",
    "resolve_ltdetr_variant",
]
