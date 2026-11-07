# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO instance segmentation dataset (eval-oriented, lazy mask decode).

Reads a COCO json (annotations carry ``segmentation`` polygons or RLE),
keeps each instance as its (compact) RLE, and decodes to binary masks
per ``__getitem__`` (DATA-04): the eager variant materialized every mask
as a bool (H, W) array at construction time, which is hundreds of GB on
COCO train2017-scale data. The internal class ids are contiguous
(insertion order of the train categories), matching the detection
dataset conventions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import (
    load_annotation_json,
    required_field,
    resolve_under,
)
from autovision.errors import AutoVisionError

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
        payload = load_annotation_json(annotations, kind="COCO annotations")
        images_dir = images_dir or Path(annotations).parent
        categories = sorted(
            int(
                required_field(
                    c, "id", source=annotations, entry="categories entry"
                )
            )
            for c in payload.get("categories", [])
        )
        self._user_to_internal = {c: i for i, c in enumerate(categories)}
        self._images_dir = images_dir
        self.transform = transform

        image_meta: dict[int, tuple[str, int, int]] = {}
        for image in payload.get("images", []):
            image_id = int(
                required_field(image, "id", source=annotations, entry="images entry")
            )
            file_name = str(
                required_field(
                    image, "file_name", source=annotations, entry="images entry"
                )
            )
            if image_id in image_meta:
                # Detection raises on duplicate ids (H3-2b); silently
                # overwriting the first entry here would drop every
                # annotation that references it (DATA-08).
                raise AutoVisionError(
                    f"COCO annotations '{annotations}': duplicate image id "
                    f"{image_id} (file names '{image_meta[image_id][0]}' and "
                    f"'{file_name}') — image ids must be unique."
                )
            width = int(required_field(image, "width", source=annotations, entry="images entry"))
            height = int(required_field(image, "height", source=annotations, entry="images entry"))
            if width <= 0 or height <= 0:
                # Mask RLE encoding and the cxcywh normalization divide by
                # both; a non-positive declared size used to die on a bare
                # ZeroDivisionError with no file context (DATA-20).
                raise AutoVisionError(
                    f"COCO annotations '{annotations}' image '{file_name}' "
                    f"(id {image_id}) declares a non-positive size: "
                    f"width={width}, height={height} — both must be > 0."
                )
            image_meta[image_id] = (file_name, width, height)
        # Pre-group annotations by image once (H3-2c): the naive
        # images × annotations double scan made val2017-scale construction
        # O(minutes); appending in JSON order keeps per-image annotation
        # order identical to the old scan.
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[
                int(
                    required_field(
                        annotation,
                        "image_id",
                        source=annotations,
                        entry="annotations entry",
                    )
                )
            ].append(annotation)
        rows: list[dict[str, Any]] = []
        malformed_polygons = 0
        dropped_instances = 0
        for image_id, (file_name, width, height) in image_meta.items():
            boxes: list[list[float]] = []
            classes: list[int] = []
            rles: list[dict[str, Any]] = []
            for annotation in annotations_by_image.get(image_id, []):
                if (
                    int(
                        required_field(
                            annotation,
                            "category_id",
                            source=annotations,
                            entry=f"annotations entry (image_id {image_id})",
                        )
                    )
                    not in self._user_to_internal
                ):
                    continue
                rle, fell_back = self._mask_geometry(annotation, height, width)
                if rle is None:
                    # Malformed geometry with no usable bbox either — the
                    # instance cannot be represented, skip it entirely.
                    dropped_instances += 1
                    continue
                malformed_polygons += int(fell_back)
                bbox = required_field(
                    annotation,
                    "bbox",
                    source=annotations,
                    entry=f"annotations entry (image_id {image_id})",
                )
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    # A wrong-length bbox used to flow into the row tensors
                    # (or die on a bare unpacking) without naming the file
                    # it came from (H3-2e).
                    raise AutoVisionError(
                        f"COCO annotations '{annotations}' annotation "
                        f"{annotation.get('id')} (image_id {image_id}) has a "
                        f"malformed 'bbox': expected 4 values [x, y, w, h], "
                        f"got "
                        f"{len(bbox) if isinstance(bbox, (list, tuple)) else type(bbox).__name__}."
                    )
                boxes.append([float(v) for v in bbox])
                classes.append(self._user_to_internal[int(annotation["category_id"])])
                rles.append(rle)
            rows.append(
                {
                    "image_path": str(resolve_under(images_dir, file_name)),
                    "size": (width, height),
                    "bboxes": boxes,
                    "classes": classes,
                    "rles": rles,
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
        orphan_drops = sum(
            len(rows)
            for image_id, rows in annotations_by_image.items()
            if image_id not in image_meta
        )
        if orphan_drops:
            # Orphan annotations (image_id not in 'images') were silently
            # ignored; the detection loader counts and reports them
            # (DATA-29) — parity here too (review P3-9).
            logger.warning(
                "%d annotation(s) in '%s' reference image ids that are not "
                "declared in 'images' — dropped.",
                orphan_drops,
                annotations,
            )
        self._rows = rows

    @staticmethod
    def _mask_geometry(
        annotation: dict[str, Any], height: int, width: int
    ) -> tuple[dict[str, Any] | None, bool]:
        """RLE for one annotation, with polygon sanitization (D19, DATA-04).

        Returns ``(rle, fell_back)``; ``rle`` is ``None`` when the
        annotation has no usable geometry at all. Malformed polygons
        (odd length, < 3 points, non-finite, non-numeric) are replaced by
        the annotation's bbox rectangle so the instance survives —
        ``frPyObjects`` would otherwise raise and abort dataset
        construction over one bad row. Rows keep the compact RLE instead
        of the decoded (H, W) array: the eager variant held every mask in
        memory, which is hundreds of GB at COCO train2017 scale.
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
                return dict(rle), False
            # Uncompressed RLE (int counts): frPyObjects returns one
            # compressed RLE — decode/merge semantics match the eager path
            # exactly (merge expects a LIST of RLEs and would choke here).
            return mask_utils.frPyObjects(rle, height, width), False
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
        return mask_utils.merge(rles), fell_back

    @staticmethod
    def _decode_mask(
        annotation: dict[str, Any], height: int, width: int
    ) -> tuple[np.ndarray | None, bool]:
        """Decoded bool mask — the reference oracle over ``_mask_geometry``.

        The loader stores RLE (DATA-04); regression tests compare decoded
        masks against the pre-lazy eager semantics through this wrapper.
        """
        from pycocotools import mask as mask_utils

        rle, fell_back = COCOInstanceSegmentationDataset._mask_geometry(
            annotation, height, width
        )
        if rle is None:
            return None, fell_back
        return mask_utils.decode(rle).astype(bool), fell_back

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from pycocotools import mask as mask_utils

        from autovision.data.file_helpers import open_image

        row = self._rows[index]
        width, height = row["size"]
        image = open_image(row["image_path"]).float() / 255.0  # (3, H, W)
        if image.shape[-2:] != (height, width):
            image = self._tf.resize(image, [height, width], antialias=True)
        decoded: list[np.ndarray] = []
        for rle in row["rles"]:
            try:
                decoded.append(mask_utils.decode(rle).astype(bool))
            except Exception as error:
                # A corrupt RLE only surfaces at decode time now that rows
                # are lazy (DATA-04) — name the row instead of a bare
                # pycocotools traceback.
                raise AutoVisionError(
                    f"Instance mask for row {index} (image "
                    f"'{row['image_path']}') could not be decoded from its "
                    f"RLE: {error}."
                ) from error
        masks = (
            torch.as_tensor(np.stack(decoded), dtype=torch.float)
            if decoded
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
