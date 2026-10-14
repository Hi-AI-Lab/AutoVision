# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""FSDC-DETR training wrapper (design doc 22 §4.6).

Inherits the entire DEIMv2 training stack unchanged — the HF criterion is
consumed inside the forward, the optimizer buckets split parameters by
name fragments (``.conv_encoder.backbone.`` → DINOv3 LR; the native
``spatial_tuning_adapter``/``dbfsaf_blocks``/``pan_sfsff`` subtrees and
the FSC encoder fall into the "other" bucket at full detector LR, which
is exactly the doc 22 §4.6 grouping: the ViT is the only backbone-LR
part, every frequency module is a randomly-initialized connector), EMA
and evaluation are family-agnostic. Only the task-model construction is
overridden.
"""

from __future__ import annotations

from autovision.tasks.detection.deimv2.train_model import (
    Deimv2ObjectDetectionTrain,
)
from autovision.tasks.detection.fsdc.config import (
    FsdcObjectDetectionTrainArgs,
)
from autovision.tasks.detection.fsdc.task_model import (
    FsdcObjectDetection,
)

__all__ = ["FsdcObjectDetectionTrain"]


class FsdcObjectDetectionTrain(Deimv2ObjectDetectionTrain):
    """DEIMv2 training stack + native FSDC task model."""

    def _build_task_model(
        self, model_args: FsdcObjectDetectionTrainArgs
    ) -> FsdcObjectDetection:
        return FsdcObjectDetection(
            num_classes=model_args.num_classes,
            model=model_args.model,
            image_size=model_args.image_size,
            num_select=model_args.num_select,
            load_weights=model_args.load_weights,
            fsdc=model_args.fsdc,
        )
