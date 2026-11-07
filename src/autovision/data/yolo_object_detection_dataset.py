# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO object detection dataset.

Ultralytics layout: ``<split>/images/...`` with labels mirrored under a
``labels`` directory (the first ``images`` path part is replaced). Labels
are per-image ``class cx cy w h`` text files in normalized cxcywh.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autovision.data.label_helpers import contiguous_class_id_map
from autovision.errors import AutoVisionError

__all__ = ["YOLOObjectDetectionDataArgs", "yolo_images_to_labels_path"]

logger = logging.getLogger("autovision")

#: Labels are read with ``utf-8-sig`` so UTF-8 files — including the BOM
#: some Windows annotation tools prepend — parse identically (H3-2f). The
#: training hot path stays lenient about coordinate VALUES on purpose:
#: the offline quality audit (``autovision.data.quality``) is the
#: validation surface for those.
_LABEL_ENCODING = "utf-8-sig"

#: Image formats accepted as YOLO rows. Single source for both the row
#: enumeration and the mmap fingerprint digest so they can never drift
#: apart (DATA-03).
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class _MalformedLabelLines:
    """Counts dropped YOLO label lines across one split load (H3-2a, DATA-06/07).

    0.6.0 silently dropped lines whose field count did not match the
    format, crashed on field-count-legal lines with garbage values (a bare
    ``ValueError`` bypassing this counter), and dropped unknown-class lines
    without a trace. The skipping stays (deprecation protocol, AGENTS.md
    §3.4) but the load counts every dropped line — malformed (wrong field
    count, non-integer class id, non-float coordinate) and unknown-class
    separately — and reports once at the end with the offending file and
    line number; the malformed default flips to raising in 0.7.0. Lines
    naming an ``ignore_classes`` id stay silent: dropping them is the
    documented contract.
    """

    def __init__(self, expected_fields: int) -> None:
        self._expected_fields = expected_fields
        self._total = 0
        self._unknown_total = 0
        self._files: set[Path] = set()
        self._unknown_files: set[Path] = set()
        self._first: tuple[Path, int] | None = None
        self._first_unknown: tuple[Path, int, int] | None = None

    def record(self, label_path: Path, line_number: int) -> None:
        """Record one malformed line (1-based ``line_number`` in its file)."""
        self._total += 1
        self._files.add(label_path)
        if self._first is None:
            self._first = (label_path, line_number)

    def record_unknown(self, label_path: Path, line_number: int, class_id: int) -> None:
        """Record one line naming a class id absent from names (DATA-07)."""
        self._unknown_total += 1
        self._unknown_files.add(label_path)
        if self._first_unknown is None:
            self._first_unknown = (label_path, line_number, class_id)

    def warn_once(self, split: str) -> None:
        """Emit the end-of-load warnings; no-op when nothing was dropped."""
        if self._total and self._first is not None:
            first_path, first_line = self._first
            logger.warning(
                "Skipped %d malformed YOLO label line(s) across %d file(s) in the "
                "'%s' split (each line must have %d whitespace-separated fields: "
                "an integer class id followed by float coordinates); "
                "first offender: '%s' line %d. Skipping malformed lines is "
                "deprecated in 0.6.1 and will raise by default in 0.7.0.",
                self._total,
                len(self._files),
                split,
                self._expected_fields,
                first_path,
                first_line,
            )
        if self._unknown_total and self._first_unknown is not None:
            path, line_number, class_id = self._first_unknown
            logger.warning(
                "Skipped %d YOLO label line(s) naming class ids that are in "
                "neither the declared names nor ignore_classes, across %d "
                "file(s) in the '%s' split; first offender: '%s' line %d "
                "(class id %d). Declare the class in names, add it to "
                "ignore_classes, or fix the annotation (DATA-07).",
                self._unknown_total,
                len(self._unknown_files),
                split,
                path,
                line_number,
                class_id,
            )


def yolo_images_to_labels_path(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


class YOLOObjectDetectionDataArgs(BaseModel):
    format: str = "yolo"
    path: Path
    train: Path
    val: Path
    names: dict[int, str]  # required: no auto inference, like the baseline
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False
    data_config_file: Path | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(strict=False, extra="forbid")

    @model_validator(mode="after")
    def _validate_split_layout(self) -> "YOLOObjectDetectionDataArgs":
        for split in (self.train, self.val):
            if "images" not in split.parts:
                raise ValueError(
                    f"YOLO split path '{split}' must contain an 'images' part."
                )
        return self

    def resolve_data_paths(self, base_dir: Path | None = None) -> None:
        # Relative splits anchor to `path` (Ultralytics layout) even when
        # invoked with a dict payload (base_dir is only set for config
        # files, where it anchors `path` itself first).
        if base_dir is not None and not self.path.is_absolute():
            self.path = base_dir / self.path
        if not self.path.is_absolute():
            return
        for name in ("train", "val"):
            path: Path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, self.path / path)

    def classes(self) -> dict[int, str]:
        included = set(self.names) - set(self.ignore_classes or [])
        internal = contiguous_class_id_map(sorted(included))
        return {internal[class_id]: self.names[class_id] for class_id in sorted(included)}

    def _image_manifest_digest(self, images_dir: Path) -> str:
        """sha256 over the sorted image manifest (relpath, size, mtime_ns).

        The mmap row set IS the image enumeration, so adding or deleting an
        image must invalidate the row cache even when no label file changes
        (DATA-03). The suffix filter mirrors ``list_image_info`` exactly.
        """
        digest = hashlib.sha256()
        count = 0
        for child in sorted(
            candidate
            for candidate in images_dir.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        ):
            stat = child.stat()
            digest.update(
                f"{child.relative_to(images_dir).as_posix()}|"
                f"{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
            )
            count += 1
        return f"{count}:{digest.hexdigest()}"

    def _mmap_hash(self, split: Path) -> str:
        return str(
            (
                (self.path / split).resolve(),
                tuple(sorted(self.names.items())),
                sorted(self.ignore_classes or []),
                self.skip_if_label_file_missing,
                self._image_manifest_digest(split),
            )
        )

    def train_data_mmap_hash(self) -> str:
        return self._mmap_hash(self.train)

    def val_data_mmap_hash(self) -> str:
        return self._mmap_hash(self.val)

    def list_image_info(self, split: str = "train") -> Iterator[dict[str, Any]]:
        images_dir = getattr(self, split)
        labels_dir = yolo_images_to_labels_path(images_dir)
        included = set(self.names) - set(self.ignore_classes or [])
        internal = contiguous_class_id_map(sorted(included))
        ignored = set(self.ignore_classes or [])
        malformed = _MalformedLabelLines(expected_fields=5)

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
                    if len(parts) != 5:
                        malformed.record(label_path, line_number)
                        continue
                    try:
                        class_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        # A field-count-legal line with garbage values used
                        # to crash the hot path with a bare ValueError and
                        # no file/line context, bypassing this counter
                        # entirely (DATA-06) — it now rides the same
                        # deprecation channel as malformed field counts.
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
