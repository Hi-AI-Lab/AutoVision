# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Task model layer: bases + detection family."""

from autovision._task_models.rtdetr_object_detection import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision._task_models.task_model import TaskModel
from autovision._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)
from autovision._task_models.train_model import TaskStepResult, TrainModel

__all__ = [
    "BaseModelOutput",
    "ModelInputSpec",
    "RTDETRObjectDetection",
    "RTDETRObjectDetectionTrain",
    "RTDETRObjectDetectionTrainArgs",
    "TaskModel",
    "TaskStepResult",
    "TensorSpec",
    "TrainModel",
]
