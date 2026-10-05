# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Detection transform family (torchvision v2 based, step scheduled)."""

from autovision._transforms.detection_transforms.args import (
    OrientedObjectDetectionTransformArgs,
    RTDETRDetectionTransformArgs,
    ScaleJitterArgs,
)
from autovision._transforms.detection_transforms.collate import (
    RTDETRDetectionCollateFunction,
    TwoViewDetectionCollateFunction,
)
from autovision._transforms.detection_transforms.oriented_transform import (
    OrientedObjectDetectionCollateFunction,
    OrientedObjectDetectionTransform,
    build_oriented_val_transform_args,
)
from autovision._transforms.detection_transforms.schedule import (
    StepSchedule,
    resolve_detection_step_schedule,
)
from autovision._transforms.detection_transforms.transform import (
    RTDETRDetectionTransform,
    StepScheduledCompose,
    build_val_transform_args,
)

__all__ = [
    "OrientedObjectDetectionCollateFunction",
    "OrientedObjectDetectionTransform",
    "OrientedObjectDetectionTransformArgs",
    "RTDETRDetectionCollateFunction",
    "RTDETRDetectionTransform",
    "RTDETRDetectionTransformArgs",
    "ScaleJitterArgs",
    "StepSchedule",
    "StepScheduledCompose",
    "TwoViewDetectionCollateFunction",
    "build_oriented_val_transform_args",
    "build_val_transform_args",
    "resolve_detection_step_schedule",
]
