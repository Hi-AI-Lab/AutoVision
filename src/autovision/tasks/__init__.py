# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Task model layer: bases + detection family."""

from autovision.tasks.detection.rtdetr_object_detection import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = [
    "BaseModelOutput",
    "ModelInputSpec",
    "RTDETRObjectDetection",
    "RTDETRObjectDetectionTrain",
    "RTDETRObjectDetectionTrainArgs",
    "TaskNetwork",
    "TaskStepResult",
    "TensorSpec",
    "TrainModel",
]
