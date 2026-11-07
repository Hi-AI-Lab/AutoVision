# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO-OBB (oriented) object detection dataset.

Ultralytics YOLO-OBB layout: ``<split>/images/...`` with per-image label
files of ``class x1 y1 x2 y2 x3 y3 x4 y4`` lines — normalized *ordered
corner* coordinates (xyxyxyxy). The 8-component flow is kept end to end
(dataset → oriented transform → collate); OBB detection heads that consume
it are scheduled separately (the geometry core lives in
``object_detection_components/obb.py``).
"""

from __future__ import annotations

import json
from typing import Any

import torch

from autovision.data.file_helpers import open_image
from autovision.data.label_helpers import contiguous_class_id_map
from autovision.data.object_detection_dataset import ObjectDetectionDataset
from autovision.data.yolo_object_detection_dataset import (
    _LABEL_ENCODING,
    IMAGE_SUFFIXES,
    YOLOObjectDetectionDataArgs,
    _MalformedLabelLines,
    yolo_images_to_labels_path,
)
from autovision.errors import AutoVisionError

__all__ = [
    "OrientedObjectDetectionDataset",
    "YOLOOrientedObjectDetectionDataArgs",
]


class YOLOOrientedObjectDetectionDataArgs(YOLOObjectDetectionDataArgs):
    """YOLO-OBB data args: identical layout, 8-component corner labels."""

    format: str = "yolo-obb"

    def list_image_info(self, split: str = "train"):  # type: ignore[override]
        images_dir = getattr(self, split)
        labels_dir = yolo_images_to_labels_path(images_dir)
        included = set(self.names) - set(self.ignore_classes or [])
        internal = contiguous_class_id_map(sorted(included))
        ignored = set(self.ignore_classes or [])
        malformed = _MalformedLabelLines(expected_fields=9)

        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image_path.relative_to(images_dir)
            label_path = (labels_dir / relative).with_suffix(".txt")
            boxes: list[list[float]] = []
            labels: list[int] = []
            if label_path.exists():
                try:
                    lines = label_path.read_text(encoding=_LABEL_ENCODING).splitlines()
                except UnicodeDecodeError as error:
                    # A binary/GBK label file used to die with a bare
                    # UnicodeDecodeError and no file context (DATA-21).
                    raise AutoVisionError(
                        f"YOLO label file '{label_path}' is not valid "
                        f"UTF-8 (utf-8-sig): {error}. Re-export the labels "
                        "in UTF-8."
                    ) from error
                for line_number, line in enumerate(lines, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 9:
                        malformed.record(label_path, line_number)
                        continue
                    try:
                        class_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        # Same hot-path guard as the detection loader (DATA-06).
                        malformed.record(label_path, line_number)
                        continue
                    if class_id in ignored:
                        continue  # deliberate ignore_classes drop — silent
                    if class_id not in internal:
                        malformed.record_unknown(label_path, line_number, class_id)
                        continue
                    boxes.append(coords)
                    labels.append(internal[class_id])
            elif self.skip_if_label_file_missing and split == "train":
                continue
            yield {
                "image_path": str(image_path),
                "bboxes": json.dumps(boxes),
                "class_labels": json.dumps(labels),
            }
        malformed.warn_once(split)


class OrientedObjectDetectionDataset(ObjectDetectionDataset):
    """Yields oriented detection items.

    Boxes enter as normalized xyxyxyxy corner polygons (N, 8) and are
    converted to absolute-pixel corners for the geometric transform
    pipeline (plain tensors — ``tv_tensors.BoundingBoxes`` cannot represent
    rotated boxes). Output keys match the detection dataset; ``bboxes`` is
    an (N, 8) absolute-pixel corner tensor before the transform and
    normalized corners after it.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.image_info[index]
        if not (isinstance(row, dict) and "image_path" in row):
            raise TypeError(f"Unsupported image info row: {type(row).__name__}")
        image_path = row["image_path"]
        bboxes = json.loads(row["bboxes"])
        class_labels = json.loads(row["class_labels"])
        image = open_image(image_path, num_channels=self.num_channels)
        height, width = image.shape[-2:]
        polys = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 8)
        polys[:, 0::2] *= width
        polys[:, 1::2] *= height
        item = {
            "image_path": str(image_path),
            "image": image,
            "bboxes": polys,
            "classes": torch.as_tensor(class_labels, dtype=torch.int64),
            "original_size": (width, height),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
