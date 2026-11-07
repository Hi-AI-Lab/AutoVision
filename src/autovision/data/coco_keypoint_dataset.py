# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO keypoint (person) dataset plane for the GTR pose family (doc 31 §11 S5b).

Mirrors the official ``CocoPoseDetection`` + ``ConvertCocoKeypointsToTarget``
semantics:

- train split keeps only images with at least one annotated keypoint
  (``sum(num_keypoints) > 0``); the val split keeps **all** images — images
  without persons are true negatives under the COCO protocol;
- per-annotation filters (both splits): drop ``iscrowd == 1`` persons and
  persons with ``num_keypoints == 0``; person boxes clamp to the frame and
  degenerate boxes are dropped;
- targets stay in **absolute original-frame pixels** here (keypoints
  ``(N, K, 3)`` ``[x, y, v]`` with ``v`` in ``{0, 1, 2}``, boxes xyxy, areas
  px²); the family transform (``tasks/pose_estimation/gtr/transforms.py``)
  resizes/flips and converts to the normalized contract the criterion
  consumes (``PoseNormalize`` order: zero invisible coords, ``v = 1`` for
  labelled-visible, divide by the resized frame).

The evaluator consumes the raw annotation file separately (official AP
ground truth is the unfiltered JSON — pycocotools applies the keypoint
ignore rules itself); this dataset's ``index`` field is the position in
the sorted-image-id enumeration, which is how predictions map back to GT.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import (
    load_annotation_json,
    open_image,
    required_field,
    resolve_under,
)
from autovision.errors import AutoVisionError
from autovision.types import PathLike

__all__ = ["CocoKeypointDataset"]


class CocoKeypointDataset(Dataset):
    """COCO ``person_keypoints`` dataset (GTR pose data plane).

    ``train=True`` applies the official train-split image filter; both
    splits apply the official annotation filter. Every item carries the
    original-frame targets, ``original_size`` (w, h), the COCO ``image_id``
    and the ``index`` position used to line predictions up with the
    evaluator's ground truth.
    """

    def __init__(
        self,
        annotations: PathLike,
        images: PathLike,
        transform: Callable[[dict], dict] | None = None,
        train: bool = False,
        num_body_points: int = 17,
    ) -> None:
        self.annotations_path = Path(annotations)
        self.images_dir = Path(images)
        self.train_mode = bool(train)
        self.num_body_points = int(num_body_points)
        self.transform = transform
        self.coco: dict[str, Any] = load_annotation_json(
            self.annotations_path, kind="COCO person_keypoints annotations"
        )
        self.categories: dict[int, str] = {
            int(
                required_field(
                    category, "id", source=self.annotations_path, entry="categories entry"
                )
            ): str(
                required_field(
                    category, "name", source=self.annotations_path, entry="categories entry"
                )
            )
            for category in self.coco.get("categories", [])
        }
        self._images: dict[int, dict[str, Any]] = {}
        for image in self.coco.get("images", []):
            image_id = int(
                required_field(image, "id", source=self.annotations_path, entry="images entry")
            )
            if image_id in self._images:
                # Detection raises on duplicate ids (H3-2b); silently
                # overwriting the first entry here would drop every
                # annotation that references it (DATA-08).
                raise AutoVisionError(
                    f"COCO annotations '{self.annotations_path}': duplicate "
                    f"image id {image_id} (file names "
                    f"'{self._images[image_id].get('file_name')}' and "
                    f"'{image.get('file_name')}') — image ids must be unique."
                )
            self._images[image_id] = image
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in self.coco.get("annotations", []):
            anns_by_image[
                int(
                    required_field(
                        annotation,
                        "image_id",
                        source=self.annotations_path,
                        entry="annotations entry",
                    )
                )
            ].append(annotation)
        self._anns_by_image = anns_by_image

        sorted_ids = sorted(self._images)
        if self.train_mode:
            # Official CocoPoseDetection: keep only images with at least one
            # annotated keypoint (images with no annotations or with every
            # person at num_keypoints == 0 carry no training signal).
            self.image_ids = [
                image_id
                for image_id in sorted_ids
                if sum(
                    int(annotation.get("num_keypoints", 0))
                    for annotation in anns_by_image.get(image_id, [])
                )
                > 0
            ]
        else:
            self.image_ids = sorted_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        info = self._images[image_id]
        file_name = str(
            required_field(
                info, "file_name", source=self.annotations_path, entry="images entry"
            )
        )
        image = open_image(resolve_under(self.images_dir, file_name))
        height, width = image.shape[-2:]

        keypoints: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        labels: list[int] = []
        areas: list[float] = []
        for annotation in self._anns_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            if int(annotation.get("num_keypoints", 0)) == 0:
                continue
            kp = torch.as_tensor(
                required_field(
                    annotation,
                    "keypoints",
                    source=self.annotations_path,
                    entry=f"annotations entry (image_id {image_id})",
                ),
                dtype=torch.float32,
            )
            kp = kp.reshape(-1, 3)
            if kp.shape[0] != self.num_body_points:
                raise AutoVisionError(
                    f"COCO annotations '{self.annotations_path}' annotation "
                    f"{annotation.get('id')} carries {kp.shape[0]} keypoints; "
                    f"this family was built for {self.num_body_points}."
                )
            bbox = required_field(
                annotation,
                "bbox",
                source=self.annotations_path,
                entry=f"annotations entry (image_id {image_id})",
            )
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                # A wrong-length bbox used to die on the unpacking below with
                # a bare ValueError that never named the annotation file (H3-2e).
                raise AutoVisionError(
                    f"COCO annotations '{self.annotations_path}' annotation "
                    f"{annotation.get('id')} (image_id {image_id}) has a "
                    f"malformed 'bbox': expected 4 values [x, y, w, h], got "
                    f"{len(bbox) if isinstance(bbox, (list, tuple)) else type(bbox).__name__}."
                )
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + w, float(width)), min(y + h, float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            keypoints.append(kp)
            boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
            labels.append(
                int(
                    required_field(
                        annotation,
                        "category_id",
                        source=self.annotations_path,
                        entry=f"annotations entry (image_id {image_id})",
                    )
                )
            )
            areas.append(float(annotation.get("area", w * h)))

        if keypoints:
            keypoints_t = torch.stack(keypoints)
            boxes_t = torch.stack(boxes)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
        else:
            keypoints_t = torch.zeros(0, self.num_body_points, 3)
            boxes_t = torch.zeros(0, 4)
            labels_t = torch.zeros(0, dtype=torch.int64)
            areas_t = torch.zeros(0)

        item: dict[str, Any] = {
            "image": image,  # uint8 (C, H, W), original frame
            "keypoints": keypoints_t,  # (N, K, 3) absolute px, v in {0, 1, 2}
            "bboxes": boxes_t,  # (N, 4) xyxy absolute px
            "classes": labels_t,  # COCO category ids (person = 1)
            "area": areas_t,  # (N,) px²
            "original_size": (width, height),
            "image_id": image_id,
            "index": index,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
