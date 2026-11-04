# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection dataset over mmap rows + task transform."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torchvision import tv_tensors

from autovision.data.file_helpers import open_image
from autovision.data.scan_cache import scan_manifest
from autovision.data.task_dataset import TaskDataset
from autovision.runtime.env import Env
from autovision.types import PathLike

__all__ = ["ObjectDetectionDataset"]


class ObjectDetectionDataset(TaskDataset):
    """Yields detection items; boxes are normalized cxcywh on entry and
    converted to absolute-pixel XYXY ``tv_tensors.BoundingBoxes`` for the
    geometric transform pipeline.

    This dataset reads rows straight from the COCO/YOLO manifest (there is
    no separate per-image scan stage upstream of ``__getitem__``), so the
    scan cache (design doc §4.8 D8) attaches to a file-manifest validation
    layer run here: every referenced image is probed for existence and
    decodability with the same decoder ``__getitem__`` uses, and its
    annotation payload is checked for parseability and box/label agreement.
    Verdicts (never parsed annotations) are cached next to the dataset and
    reused when the file stats and task parameters are unchanged. The scan
    never raises and never changes item semantics — it only adds a census
    (``self.scan_report`` / ``self.scan_summary``) and accelerates repeated
    startups.
    """

    def __init__(
        self,
        dataset_args: Any,
        image_info: Sequence[dict[str, Any]],
        transform: Any = None,
        num_channels: int = 3,
        use_scan_cache: bool = True,
    ) -> None:
        super().__init__(dataset_args, image_info, transform)
        self.num_channels = num_channels
        self.scan_report: dict[str, Any] | None = None
        self.scan_summary: str | None = None
        if use_scan_cache and Env.SCAN_CACHE.value:
            self._run_scan_cache()

    def _scan_cache_params(self) -> dict[str, Any]:
        """Task parameters that affect scan verdicts (part of the fingerprint)."""
        return {
            "task": "object_detection",
            "num_channels": self.num_channels,
            "skip_if_annotations_missing": bool(
                getattr(self.dataset_args, "skip_if_annotations_missing", False)
            ),
            "ignore_classes": sorted(
                getattr(self.dataset_args, "ignore_classes", None) or []
            ),
        }

    def _run_scan_cache(self) -> None:
        """File-manifest validation pass, cached under the dataset image root."""
        rows = [row for row in self.image_info if isinstance(row, dict)]
        first_path = rows[0].get("image_path") if rows else None
        if first_path is None:
            return
        root = Path(str(first_path)).parent
        report, _from_cache = scan_manifest(
            root,
            rows,
            task_params=self._scan_cache_params(),
            num_channels=self.num_channels,
        )
        self.scan_report = report
        self.scan_summary = str(report.get("summary", ""))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.image_info[index]
        if isinstance(row, dict) and "image_path" in row:
            image_path: PathLike = row["image_path"]
            bboxes = json.loads(row["bboxes"])
            class_labels = json.loads(row["class_labels"])
        else:
            raise TypeError(f"Unsupported image info row: {type(row).__name__}")
        image = open_image(image_path, num_channels=self.num_channels)
        height, width = image.shape[-2:]
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(class_labels, dtype=torch.int64)
        if boxes.numel() > 0:
            # normalized cxcywh -> absolute xyxy for torchvision transforms.
            cx, cy, w, h = boxes.unbind(-1)
            boxes_xyxy = torch.stack(
                [(cx - w / 2) * width, (cy - h / 2) * height,
                 (cx + w / 2) * width, (cy + h / 2) * height],
                dim=-1,
            )
            boxes_tv = tv_tensors.BoundingBoxes(
                boxes_xyxy, format="XYXY", canvas_size=(height, width)
            )
        else:
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros(0, 4), format="XYXY", canvas_size=(height, width)
            )
        item = {
            "image_path": str(image_path),
            "image": image,
            "bboxes": boxes_tv,
            "classes": labels,
            "original_size": (width, height),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
