# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO instance segmentation dataset (in-memory, eval-oriented).

Reads a COCO json (annotations carry ``segmentation`` polygons or RLE),
converts each instance to a binary mask via pycocotools, and yields
normalized boxes + per-instance masks. The internal class ids are
contiguous (insertion order of the train categories), matching the
detection dataset conventions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["COCOInstanceSegmentationDataset"]

logger = logging.getLogger("autovision")


class COCOInstanceSegmentationDataset(Dataset):
    """COCO instance dataset: (image, bboxes, classes, masks) per row.

    ``masks`` is a (K, H, W) float tensor with one binary mask per
    instance; ``bboxes`` is (K, 4) normalized cxcywh on the original frame.
    """

    def __init__(
        self,
        annotations: Path,
        images_dir: Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        from torchvision.transforms.v2 import functional as TF

        self._tf = TF
        with open(annotations) as file:
            payload = json.load(file)
        images_dir = images_dir or Path(annotations).parent
        categories = sorted(int(c["id"]) for c in payload.get("categories", []))
        self._user_to_internal = {c: i for i, c in enumerate(categories)}
        self._images_dir = images_dir
        self.transform = transform

        image_meta = {
            int(image["id"]): (image["file_name"], int(image["width"]), int(image["height"]))
            for image in payload.get("images", [])
        }
        rows: list[dict[str, Any]] = []
        malformed_polygons = 0
        dropped_instances = 0
        for image_id, (file_name, width, height) in image_meta.items():
            boxes: list[list[float]] = []
            classes: list[int] = []
            masks: list[np.ndarray] = []
            for annotation in payload.get("annotations", []):
                if int(annotation["image_id"]) != image_id:
                    continue
                if int(annotation["category_id"]) not in self._user_to_internal:
                    continue
                mask, fell_back = self._decode_mask(annotation, height, width)
                if mask is None:
                    # Malformed geometry with no usable bbox either — the
                    # instance cannot be represented, skip it entirely.
                    dropped_instances += 1
                    continue
                malformed_polygons += int(fell_back)
                boxes.append([float(v) for v in annotation["bbox"]])
                classes.append(self._user_to_internal[int(annotation["category_id"])])
                masks.append(mask)
            rows.append(
                {
                    "image_path": str(images_dir / file_name),
                    "size": (width, height),
                    "bboxes": boxes,
                    "classes": classes,
                    "masks": masks,
                }
            )
        if malformed_polygons:
            logger.warning(
                "%d malformed segmentation polygon(s) in '%s' fell back to "
                "their bounding boxes (design doc D19).",
                malformed_polygons,
                annotations,
            )
        if dropped_instances:
            logger.warning(
                "%d annotation(s) in '%s' had no usable geometry "
                "(malformed polygon and no bbox) and were dropped.",
                dropped_instances,
                annotations,
            )
        self._rows = rows

    @staticmethod
    def _decode_mask(
        annotation: dict[str, Any], height: int, width: int
    ) -> tuple[np.ndarray | None, bool]:
        """Binary mask for one annotation, with polygon sanitization (D19).

        Returns ``(mask, fell_back)``; ``mask`` is ``None`` when the
        annotation has no usable geometry at all. Malformed polygons
        (odd length, < 3 points, non-finite, non-numeric) are replaced by
        the annotation's bbox rectangle so the instance survives —
        ``frPyObjects`` would otherwise raise and abort dataset
        construction over one bad row.
        """
        from pycocotools import mask as mask_utils

        from autovision.geometry.polygons import is_valid_polygon, polygon_from_bbox

        segmentation = annotation.get("segmentation")
        bbox = annotation.get("bbox")
        # Bbox fallback prepared once; None when the bbox itself is unusable
        # (a NaN rectangle would only move the crash into the rasterizer).
        fallback_polygon = (
            polygon_from_bbox(bbox) if bbox is not None else None
        )
        if isinstance(segmentation, dict):  # RLE
            rle = segmentation
            if isinstance(rle.get("counts"), str):
                # Compressed RLE (the json form crowd regions use) decodes
                # directly; frPyObjects would choke on the string counts.
                return mask_utils.decode(rle).astype(bool), False
            return mask_utils.decode(
                mask_utils.frPyObjects(rle, height, width)
            ).astype(bool), False
        if not isinstance(segmentation, (list, tuple)):
            segmentation = []
        # Flat single polygon ([x1, y1, ...]) vs nested list of polygons.
        if segmentation and not isinstance(segmentation[0], (list, tuple)):
            polygons = [segmentation]
        else:
            polygons = list(segmentation)
        fell_back = False
        fixed: list[list[float]] = []
        for polygon in polygons:
            if is_valid_polygon(polygon):
                fixed.append([float(v) for v in polygon])
                continue
            fell_back = True
            if fallback_polygon is not None:
                fixed.append(fallback_polygon.reshape(-1).tolist())
        if not fixed and fallback_polygon is not None:
            # Empty or all-malformed polygon list: the bbox rectangle keeps
            # the instance alive as one polygon.
            fixed = [fallback_polygon.reshape(-1).tolist()]
            fell_back = True
        if not fixed:
            return None, True
        rles = mask_utils.frPyObjects(fixed, height, width)
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool), fell_back

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from autovision.data.file_helpers import open_image

        row = self._rows[index]
        width, height = row["size"]
        image = open_image(row["image_path"]).float() / 255.0  # (3, H, W)
        if image.shape[-2:] != (height, width):
            image = self._tf.resize(image, [height, width], antialias=True)
        masks = (
            torch.as_tensor(np.stack(row["masks"]), dtype=torch.float)
            if row["masks"]
            else torch.zeros(0, height, width)
        )
        boxes = torch.as_tensor(row["bboxes"], dtype=torch.float).reshape(-1, 4)
        boxes = torch.stack(
            [
                (boxes[:, 0] + boxes[:, 2] / 2) / width,
                (boxes[:, 1] + boxes[:, 3] / 2) / height,
                boxes[:, 2] / width,
                boxes[:, 3] / height,
            ],
            dim=-1,
        )
        classes = torch.as_tensor(row["classes"], dtype=torch.long)
        if self.transform is not None:
            image, masks, boxes, classes = self.transform(image, masks, boxes, classes)
        return {
            "image": image,
            "masks": masks,
            "bboxes": boxes,
            "classes": classes,
            "original_size": (int(height), int(width)),
        }
