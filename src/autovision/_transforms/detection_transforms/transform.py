# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Step-scheduled detection transform pipeline (torchvision v2 based).

The active augmentation set is recomputed per step from the step windows;
each distinct activation tuple gets its own cached v2 Compose, so stepping
is free. Because active state changes must reach already-spawned dataloader
workers, the transform reports when the dataloader must be reinitialized.

Mosaic (opt-in via ``mosaic_prob``; the LT-DETR family enables it) pastes
four cached samples onto a 2x2 canvas at the sample level; a mosaic sample
then skips the zoom-out and IoU-crop ops (baseline behavior) but keeps
photometric distort, flips, and the resize.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from autovision._transforms.transform_args import NormalizeArgs
from autovision.types import ImageSizeTuple

__all__ = ["RTDETRDetectionTransform", "StepScheduledCompose"]


class StepScheduledCompose:
    """Caches one v2 Compose per activation tuple and switches per step.

    The key is ``(mosaic_sample, photometric, zoom_out, iou_crop)``;
    ``mosaic_sample=True`` means THIS sample came from the mosaic canvas and
    must skip zoom-out / IoU-crop.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.step = 0
        self._compose_cache: dict[tuple[bool, bool, bool, bool], v2.Compose] = {}
        self._active: tuple[bool, bool, bool, bool] | None = None
        self._reinit_pending = False

    def _build(self, active: tuple[bool, bool, bool, bool]) -> v2.Compose:
        args = self.args
        _, photometric, zoom_out, iou_crop = active
        ops: list[Any] = []
        if photometric:
            ops.append(v2.RandomPhotometricDistort(p=args.photometric_distort_prob))
        if zoom_out:
            ops.append(v2.RandomZoomOut(fill=0.0, p=args.random_zoom_out_prob))
        if iou_crop:
            # torchvision's RandomIoUCrop has no probability parameter.
            ops.append(
                v2.RandomApply([v2.RandomIoUCrop()], p=args.random_iou_crop_prob)
            )
        if args.horizontal_flip_prob > 0:
            ops.append(v2.RandomHorizontalFlip(p=args.horizontal_flip_prob))
        if args.vertical_flip_prob > 0:
            ops.append(v2.RandomVerticalFlip(p=args.vertical_flip_prob))
        ops.append(v2.Resize(args.image_size, antialias=True))
        ops.append(v2.SanitizeBoundingBoxes(min_size=args.min_bbox_size_px))
        ops.append(v2.ToDtype(torch.float32, scale=True))
        ops.append(
            v2.Normalize(mean=list(args.normalize.mean), std=list(args.normalize.std))
        )
        return v2.Compose(ops)

    def _active_at(self, step: int) -> tuple[bool, bool, bool, bool]:
        args = self.args

        def is_active(prob: float | None, start, stop) -> bool:
            return (
                prob is not None
                and prob > 0
                and start != "auto"
                and step >= int(start)
                and (stop is None or stop == "auto" or step < int(stop))
            )

        return (
            is_active(
                getattr(args, "mosaic_prob", None),
                getattr(args, "mosaic_step_start", None),
                getattr(args, "mosaic_step_stop", None),
            ),
            is_active(args.photometric_distort_prob, args.photometric_step_start, args.photometric_step_stop),
            is_active(args.random_zoom_out_prob, args.zoom_out_step_start, args.zoom_out_step_stop),
            is_active(args.random_iou_crop_prob, args.iou_crop_step_start, args.iou_crop_step_stop),
        )

    def set_step(self, step: int) -> None:
        self.step = step
        active = self._active_at(step)
        if self._active is not None and active != self._active:
            self._reinit_pending = True
        self._active = active

    def compose_for(self, mosaic_sample: bool) -> v2.Compose:
        if self._active is None:
            self._active = self._active_at(self.step)
        mosaic_active, photometric, zoom_out, iou_crop = self._active
        if mosaic_sample:
            # Mosaic samples keep photometric distort and flips; only the
            # zoom-out / IoU-crop pair is skipped (baseline behavior).
            key = (True, photometric, False, False)
        else:
            key = (False, photometric, zoom_out, iou_crop)
        if key not in self._compose_cache:
            self._compose_cache[key] = self._build(key)
        return self._compose_cache[key]

    def mosaic_active(self) -> bool:
        """Whether the mosaic window is on at the current step (the per-
        sample coin flip happens in the transform)."""
        if self._active is None:
            self._active = self._active_at(self.step)
        return self._active[0]

    def requires_dataloader_reinitialization(self) -> bool:
        return self._reinit_pending

    def mark_dataloader_as_reinitialized(self) -> None:
        self._reinit_pending = False


class RTDETRDetectionTransform:
    """Per-sample detection transform with step scheduling and mosaic.

    Mosaic keeps a rolling cache of recent raw samples (per dataloader
    worker, mirroring the baseline's in-transform cache) and, on a coin
    flip inside its step window, pastes the current sample plus three
    cached ones onto a 2x2 ``mosaic_output_size``-tile canvas.
    """

    #: Rolling cache depth (baseline MosaicArgs.max_cached_images).
    MOSAIC_CACHE_SIZE = 50

    def __init__(self, transform_args) -> None:
        self.transform_args = transform_args
        self.compose = StepScheduledCompose(transform_args)
        self._mosaic_cache: deque[dict[str, Any]] = deque(maxlen=self.MOSAIC_CACHE_SIZE)

    def set_step(self, step: int) -> None:
        self.compose.set_step(step)

    def requires_dataloader_reinitialization(self) -> bool:
        return self.compose.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self.compose.mark_dataloader_as_reinitialized()

    # ── mosaic ──────────────────────────────────────────────────────────
    def _mosaic_hit(self) -> bool:
        args = self.transform_args
        prob = getattr(args, "mosaic_prob", None)
        if not self.compose.mosaic_active() or random.random() >= (prob or 0.0):
            return False
        # 3 donors besides the current sample, which is never drawn twice.
        return len(self._mosaic_cache) >= 4

    def _build_mosaic(self, item: dict[str, Any]) -> dict[str, Any]:
        args = self.transform_args
        tile = int(args.mosaic_output_size)
        # The current sample was appended last; exclude it from the draw so
        # no tile duplicates (baseline mosaics compose four distinct images).
        donors = random.sample(list(self._mosaic_cache)[:-1], 3)
        tiles = [self._clone_sample(item), *donors]
        random.shuffle(tiles)

        canvas = torch.zeros(
            (3, 2 * tile, 2 * tile), dtype=tiles[0]["image"].dtype
        )
        boxes_out: list[torch.Tensor] = []
        classes_out: list[torch.Tensor] = []
        for index, sample in enumerate(tiles):
            row, col = divmod(index, 2)
            y0, x0 = row * tile, col * tile
            image: torch.Tensor = sample["image"]
            _, height, width = image.shape
            canvas[:, y0 : y0 + tile, x0 : x0 + tile] = TF.resize(
                image, [tile, tile], antialias=True
            )
            boxes: torch.Tensor = tv_tensors.BoundingBoxes.as_subclass(
                sample["bboxes"], torch.Tensor
            )
            if boxes.numel() == 0:
                continue
            scaled = boxes.clone()
            scaled[:, [0, 2]] *= tile / width
            scaled[:, [1, 3]] *= tile / height
            scaled[:, [0, 2]] += x0
            scaled[:, [1, 3]] += y0
            scaled[:, [0, 2]] = scaled[:, [0, 2]].clamp(0.0, 2 * tile)
            scaled[:, [1, 3]] = scaled[:, [1, 3]].clamp(0.0, 2 * tile)
            keep = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
            if keep.any():
                boxes_out.append(scaled[keep])
                classes_out.append(sample["classes"][keep])

        if boxes_out:
            merged = tv_tensors.BoundingBoxes(
                torch.cat(boxes_out, dim=0), format="XYXY", canvas_size=(2 * tile, 2 * tile)
            )
            merged_classes = torch.cat(classes_out, dim=0)
        else:
            merged = tv_tensors.BoundingBoxes(
                torch.zeros(0, 4), format="XYXY", canvas_size=(2 * tile, 2 * tile)
            )
            merged_classes = torch.zeros(0, dtype=torch.int64)
        return {
            "image_path": item["image_path"],
            "image": canvas,
            "bboxes": merged,
            "classes": merged_classes,
            "original_size": item["original_size"],
        }

    @staticmethod
    def _clone_sample(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": item["image"].clone(),
            "bboxes": item["bboxes"].clone(),
            "classes": item["classes"].clone(),
        }

    # ── pipeline ────────────────────────────────────────────────────────
    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if getattr(self.transform_args, "mosaic_prob", None) is not None:
            # Keep a rolling cache of raw samples for the mosaic canvas;
            # skipped entirely when mosaic is off (no per-sample copies).
            self._mosaic_cache.append(self._clone_sample(item))
        mosaic_sample = self._mosaic_hit()
        if mosaic_sample:
            item = self._build_mosaic(item)
        image: torch.Tensor = item["image"]
        boxes: tv_tensors.BoundingBoxes = item["bboxes"]
        transformed = self.compose.compose_for(mosaic_sample)(
            {"image": image, "boxes": boxes, "labels": item["classes"]}
        )
        image_out = transformed["image"]
        boxes_out: tv_tensors.BoundingBoxes = transformed["boxes"]
        labels_out = transformed["labels"]
        # Back to normalized cxcywh for the criterion.
        _, height, width = image_out.shape
        boxes_out = boxes_out.to(torch.float32)
        cx = (boxes_out[:, 0] + boxes_out[:, 2]) / 2 / width
        cy = (boxes_out[:, 1] + boxes_out[:, 3]) / 2 / height
        w = (boxes_out[:, 2] - boxes_out[:, 0]) / width
        h = (boxes_out[:, 3] - boxes_out[:, 1]) / height
        return {
            "image_path": item["image_path"],
            "image": image_out,
            "bboxes": torch.stack([cx, cy, w, h], dim=-1),
            "classes": labels_out,
            "original_size": item["original_size"],
        }


def build_val_transform_args(train_args):
    """Disable all scheduled augmentations for the val transform.

    The small-box filter is also relaxed to the torchvision minimum: at
    eval time, dropping sub-threshold objects with the *training* threshold
    would silently delete tiny GT (dense aerial datasets) and distort the
    mAP; only sub-pixel degenerate boxes are removed.
    """
    val_args = train_args.model_copy(deep=True)
    val_args.photometric_distort_prob = None
    val_args.random_zoom_out_prob = None
    val_args.random_iou_crop_prob = None
    val_args.mosaic_prob = None
    val_args.horizontal_flip_prob = 0.0
    val_args.vertical_flip_prob = 0.0
    val_args.scale_jitter = None
    val_args.min_bbox_size_px = 1.0
    return val_args
