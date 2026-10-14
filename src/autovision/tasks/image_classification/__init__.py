# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Image classification task model package."""

from autovision.tasks.image_classification.config import (
    ImageClassificationTrainArgs,
)
from autovision.tasks.image_classification.task_model import (
    ImageClassification,
)
from autovision.tasks.image_classification.train_model import (
    ImageClassificationTrain,
)

__all__ = [
    "ImageClassification",
    "ImageClassificationTrain",
    "ImageClassificationTrainArgs",
]
