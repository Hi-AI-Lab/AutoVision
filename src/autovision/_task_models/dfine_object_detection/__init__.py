# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE object detection family (distributional box refinement)."""

from autovision._task_models.dfine_object_detection.config import (
    DFINEObjectDetectionTrainArgs,
)
from autovision._task_models.dfine_object_detection.task_model import (
    DFINEObjectDetection,
)
from autovision._task_models.dfine_object_detection.train_model import (
    DFINEObjectDetectionTrain,
)

__all__ = [
    "DFINEObjectDetection",
    "DFINEObjectDetectionTrain",
    "DFINEObjectDetectionTrainArgs",
]
