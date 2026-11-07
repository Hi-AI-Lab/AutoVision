# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO object detection dataset.

Rows are built once from the COCO json and serialized into the mmap cache;
the internal representation is normalized ``cxcywh``. The class list comes
from the **train** annotation file's ``categories`` (like the baseline).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autovision.data.file_helpers import (
    load_annotation_json,
    required_field,
    resolve_under,
)
from autovision.data.label_helpers import contiguous_class_id_map
from autovision.errors import AutoVisionError

__all__ = ["COCOObjectDetectionDataArgs"]

logger = logging.getLogger("autovision")


class SplitArgs(BaseModel):
    annotations: Path
    images: Path | None = None

    model_config = ConfigDict(strict=False)


class COCOObjectDetectionDataArgs(BaseModel):
    format: str = "coco"
    train: SplitArgs
    val: SplitArgs
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False
    data_config_file: Path | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(strict=False, extra="forbid")

    def resolve_data_paths(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            return
        for split in (self.train, self.val):
            for name in ("annotations", "images"):
                path: Path | None = getattr(split, name)
                if path is not None and not path.is_absolute():
                    setattr(split, name, base_dir / path)

    def _classes(self) -> dict[int, str]:
        path = self.train.annotations
        annotations = load_annotation_json(path, kind="COCO annotations")
        return {
            int(required_field(c, "id", source=path, entry="categories entry")): str(
                required_field(c, "name", source=path, entry="categories entry")
            )
            for c in annotations.get("categories", [])
        }

    def classes(self) -> dict[int, str]:
        """Class id -> name (internal contiguous ids)."""
        user_classes = self._classes()
        included = _included_class_ids(user_classes, self.ignore_classes)
        internal = contiguous_class_id_map(sorted(included))
        return {internal[class_id]: user_classes[class_id] for class_id in sorted(included)}

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                self.train.annotations.resolve(),
                self.train.annotations.stat().st_mtime,
                self.train.images,
                sorted(self.ignore_classes or []),
                self.skip_if_annotations_missing,
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                self.val.annotations.resolve(),
                self.val.annotations.stat().st_mtime,
                self.val.images,
                sorted(self.ignore_classes or []),
                self.skip_if_annotations_missing,
            )
        )

    def list_image_info(self, split: str = "train") -> Iterator[dict[str, Any]]:
        split_args: SplitArgs = getattr(self, split)
        annotations_path = Path(split_args.annotations)
        annotations = load_annotation_json(
            annotations_path, kind="COCO annotations"
        )
        images_dir = split_args.images or annotations_path.parent

        image_id_to_file: dict[int, tuple[str, int, int]] = {}
        image_id_to_name: dict[int, str] = {}
        for image in annotations.get("images", []):
            image_id = int(
                required_field(image, "id", source=annotations_path, entry="images entry")
            )
            file_name = str(
                required_field(
                    image, "file_name", source=annotations_path, entry="images entry"
                )
            )
            width = int(
                required_field(image, "width", source=annotations_path, entry="images entry")
            )
            height = int(
                required_field(image, "height", source=annotations_path, entry="images entry")
            )
            if width <= 0 or height <= 0:
                # The cxcywh normalization divides by both; a zero or
                # negative declared size used to die on a bare
                # ZeroDivisionError with no file context (DATA-20).
                raise AutoVisionError(
                    f"COCO annotations '{split_args.annotations}' image "
                    f"'{file_name}' (id {image_id}) declares a non-positive "
                    f"size: width={width}, height={height} — both must be > 0."
                )
            candidate = resolve_under(Path(images_dir), file_name)
            first_file_name = image_id_to_name.get(image_id)
            if first_file_name is not None:
                # A duplicate id would silently overwrite the first image's
                # row and mis-place every annotation of the loser (H3-2b).
                raise AutoVisionError(
                    f"COCO annotations '{split_args.annotations}' declare "
                    f"image id {image_id} twice: '{first_file_name}' and "
                    f"'{file_name}'; duplicate image ids would silently "
                    "overwrite each other."
                )
            image_id_to_name[image_id] = file_name
            image_id_to_file[image_id] = (str(candidate), width, height)
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in annotations.get("annotations", []):
            annotations_by_image.setdefault(
                int(
                    required_field(
                        annotation,
                        "image_id",
                        source=annotations_path,
                        entry="annotations entry",
                    )
                ),
                [],
            ).append(annotation)

        user_classes = self._classes()
        included = _included_class_ids(user_classes, self.ignore_classes)
        internal = contiguous_class_id_map(sorted(included))
        undeclared_drops = 0

        for image_id, (image_path, width, height) in image_id_to_file.items():
            image_annotations = annotations_by_image.get(image_id, [])
            boxes: list[list[float]] = []
            labels: list[int] = []
            iscrowds: list[int] = []
            for annotation in image_annotations:
                class_id = int(
                    required_field(
                        annotation,
                        "category_id",
                        source=annotations_path,
                        entry=f"annotations entry (image_id {image_id})",
                    )
                )
                if class_id not in internal:
                    if class_id not in user_classes:
                        # Undeclared in 'categories': a broken export, not a
                        # deliberate ignore_classes drop — count and report
                        # it like merge_coco does (H3-2b).
                        undeclared_drops += 1
                    continue
                bbox = required_field(
                    annotation,
                    "bbox",
                    source=annotations_path,
                    entry=f"annotations entry (image_id {image_id})",
                )
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    # A wrong-length bbox used to die on the unpacking below
                    # with a bare ValueError that never named the file (H3-2e).
                    raise AutoVisionError(
                        f"COCO annotations '{annotations_path}' annotation "
                        f"{annotation.get('id')} (image_id {image_id}) has a "
                        f"malformed 'bbox': expected 4 values [x, y, w, h], "
                        f"got "
                        f"{len(bbox) if isinstance(bbox, (list, tuple)) else type(bbox).__name__}."
                    )
                x, y, w, h = (float(v) for v in bbox)  # absolute pixels xywh
                boxes.append(
                    [
                        (x + w / 2) / width,
                        (y + h / 2) / height,
                        w / width,
                        h / height,
                    ]
                )
                labels.append(internal[class_id])
                iscrowds.append(int(annotation.get("iscrowd", 0)))
            if not boxes and self.skip_if_annotations_missing and split == "train":
                continue
            yield {
                "image_path": image_path,
                "bboxes": json.dumps(boxes),
                "class_labels": json.dumps(labels),
                # Official-protocol ignore regions (consumed by the mAP
                # evaluator; training treats every box as a regular target).
                "iscrowds": json.dumps(iscrowds),
            }
        orphan_drops = sum(
            len(rows)
            for image_id, rows in annotations_by_image.items()
            if image_id not in image_id_to_file
        )
        if undeclared_drops:
            logger.warning(
                "'%s' contains %d annotation(s) whose category_id has no "
                "declared class name (train 'categories') — dropped.",
                split_args.annotations,
                undeclared_drops,
            )
        if orphan_drops:
            # Orphan annotations (image_id not in 'images') were silently
            # ignored while merge_coco reported them (DATA-29).
            logger.warning(
                "'%s' contains %d annotation(s) referencing image ids that "
                "are not declared in 'images' — dropped (merge_coco reports "
                "the same condition).",
                split_args.annotations,
                orphan_drops,
            )


def _included_class_ids(
    user_classes: dict[int, str], ignore_classes: set[int] | None
) -> set[int]:
    included = set(user_classes)
    if ignore_classes:
        unknown = set(ignore_classes) - set(user_classes)
        if unknown:
            # An ignore_classes entry that names no declared class silently
            # no-ops; a typo would silently KEEP the class (DATA-30).
            _warn_unknown_ignore_classes(tuple(sorted(unknown)))
        included -= set(ignore_classes)
    return included


#: Dedup memory for the DATA-30 warning, keyed by the unknown-id tuple.
_WARNED_UNKNOWN_IGNORE_CLASSES: set[tuple[int, ...]] = set()


def _warn_unknown_ignore_classes(unknown: tuple[int, ...]) -> None:
    if unknown not in _WARNED_UNKNOWN_IGNORE_CLASSES:
        _WARNED_UNKNOWN_IGNORE_CLASSES.add(unknown)
        logger.warning(
            "ignore_classes names class id(s) %s that are not declared in "
            "the train 'categories' — they no-op (typo?); the remaining "
            "ignore entries apply as declared.",
            list(unknown),
        )
