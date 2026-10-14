# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""AV-DETR object detection family (SSL-backbone detection finetuning)."""

from autovision.tasks.detection.avdetr.config import (
    AVDETR_MODEL_REGISTRY,
    AVDETRDensityArgs,
    AVDETRDistillArgs,
    AVDETRObjectDetectionTrainArgs,
    AVDETROpenVocabArgs,
    AVDETRSSLArgs,
    AVDETRTokenDropArgs,
    AVDETRVariantSpec,
    avdetr_default_batch_size,
    list_avdetr_model_names,
    resolve_avdetr_variant,
)
from autovision.tasks.detection.avdetr.dome_decoder import (
    DomeDFINEv2Decoder,
)
from autovision.tasks.detection.avdetr.dome_encoder import (
    DomeHybridEncoder,
)
from autovision.tasks.detection.avdetr.task_model import (
    AVDETRObjectDetection,
)
from autovision.tasks.detection.avdetr.train_model import (
    AVDETRObjectDetectionTrain,
)

__all__ = [
    "AVDETR_MODEL_REGISTRY",
    "AVDETRDensityArgs",
    "AVDETRDistillArgs",
    "DomeDFINEv2Decoder",
    "DomeHybridEncoder",
    "AVDETRObjectDetection",
    "AVDETRObjectDetectionTrain",
    "AVDETRObjectDetectionTrainArgs",
    "AVDETROpenVocabArgs",
    "AVDETRSSLArgs",
    "AVDETRTokenDropArgs",
    "AVDETRVariantSpec",
    "list_avdetr_model_names",
    "avdetr_default_batch_size",
    "resolve_avdetr_variant",
]
