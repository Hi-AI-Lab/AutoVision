# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RT-DETR oriented object detection family (YOLO-OBB training)."""

from autovision._task_models.rtdetr_oriented_object_detection.config import (
    RTDETROrientedObjectDetectionTrainArgs,
)
from autovision._task_models.rtdetr_oriented_object_detection.task_model import (
    RTDETROrientedObjectDetection,
)
from autovision._task_models.rtdetr_oriented_object_detection.train_model import (
    RTDETROrientedObjectDetectionTrain,
)

__all__ = [
    "RTDETROrientedObjectDetection",
    "RTDETROrientedObjectDetectionTrain",
    "RTDETROrientedObjectDetectionTrainArgs",
]
