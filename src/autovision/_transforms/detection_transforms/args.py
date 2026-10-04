# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Detection transform arguments with step-window policies."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autovision._configs.base import PydanticConfig
from autovision._transforms.task_transform import TaskTransformArgs
from autovision._transforms.transform_args import NormalizeArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "OrientedObjectDetectionTransformArgs",
    "RTDETRDetectionTransformArgs",
    "ScaleJitterArgs",
]


class ScaleJitterArgs(PydanticConfig):
    sizes: list[int]
    prob: float = 1.0
    #: Step at which batch scale jitter turns OFF (the baseline disables it
    #: for the final no-augmentation window). ``None`` keeps it on for the
    #: whole run; ``"auto"`` resolves to the shared step schedule.
    step_stop: int | None | str = None


class OrientedObjectDetectionTransformArgs(TaskTransformArgs):
    """Oriented (YOLO-OBB) detection recipe.

    Rotated-box pipeline: flips mirror corners, resize scales corners,
    polygons are sanitized by px² shoelace area. IoU-crop / zoom-out /
    scale-jitter / mixup are omitted — the torchvision v2 implementations
    behind them are axis-aligned-box-only.
    """

    image_size: ImageSizeTuple = (640, 640)
    photometric_distort_prob: float | None = 0.5
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    min_bbox_size_px: float = 4.0


class RTDETRDetectionTransformArgs(TaskTransformArgs):
    """Detection recipe (train defaults; the val recipe disables them)."""

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort_prob: float | None = 0.5
    random_zoom_out_prob: float | None = 0.5
    random_iou_crop_prob: float | None = 0.8
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=lambda: ScaleJitterArgs(
            sizes=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        )
    )
    mixup_prob: float | None = 0.5
    mixup_step_start: int | str = "auto"
    mixup_step_stop: int | None | str = "auto"
    #: Mosaic (4-tile sample-level paste; off by default — the LT-DETR
    #: family enables it per the baseline recipe). Mosaic samples skip the
    #: zoom-out and IoU-crop ops for that step.
    mosaic_prob: float | None = None
    mosaic_output_size: int = 320
    mosaic_step_start: int | str = "auto"
    mosaic_step_stop: int | None | str = "auto"
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    min_bbox_size_px: float = 4.0

    # Step windows (auto: resolved from the training schedule).
    photometric_step_start: int | str = "auto"
    photometric_step_stop: int | None | str = "auto"
    zoom_out_step_start: int | str = "auto"
    zoom_out_step_stop: int | None | str = "auto"
    iou_crop_step_start: int | str = "auto"
    iou_crop_step_stop: int | None | str = "auto"

    def resolve_auto(self, **kwargs) -> None:
        num_channels = kwargs.get("num_channels")
        if self.image_size == "auto":
            default = kwargs.get("default_image_size", (640, 640))
            self.image_size = tuple(default)  # type: ignore[assignment]
        if self.num_channels == "auto":
            self.num_channels = (
                num_channels if num_channels is not None else len(self.normalize.mean)
            )

    def resolve_step_schedule(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision._transforms.detection_transforms.schedule import (
            resolve_detection_step_schedule,
        )

        steps_per_epoch = max(1, train_num_batches // max(1, gradient_accumulation_steps))
        schedule = resolve_detection_step_schedule(total_steps, steps_per_epoch)
        for op, start_attr, stop_attr in (
            ("photometric", "photometric_step_start", "photometric_step_stop"),
            ("zoom_out", "zoom_out_step_start", "zoom_out_step_stop"),
            ("iou_crop", "iou_crop_step_start", "iou_crop_step_stop"),
            ("mixup", "mixup_step_start", "mixup_step_stop"),
            ("mosaic", "mosaic_step_start", "mosaic_step_stop"),
        ):
            start, stop = schedule.window(op)
            if getattr(self, start_attr) == "auto":
                setattr(self, start_attr, start)
            if getattr(self, stop_attr) == "auto":
                setattr(self, stop_attr, stop)
        if self.scale_jitter is not None and self.scale_jitter.step_stop == "auto":
            self.scale_jitter.step_stop = schedule.stop

    def resolve_incompatible(self) -> None:
        if self.num_channels != 3:
            self.photometric_distort_prob = None
