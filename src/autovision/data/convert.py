# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO <-> COCO detection annotation conversion (filesystem-only).

Both directions keep images in place; only annotation artifacts are
written. COCO category ids are arbitrary; the YOLO side uses 0-based
contiguous ids assigned by sorted COCO category id (the same convention
the data layer's class mapping documents).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["coco_to_yolo", "yolo_to_coco"]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sorted_categories(
    categories: Sequence[Mapping[str, Any]] | Mapping[int, str] | Sequence[str],
) -> list[tuple[int, str]]:
    """Normalize the accepted ``names`` shapes to sorted (id, name) pairs."""
    if isinstance(categories, Mapping):
        return sorted((int(k), str(v)) for k, v in categories.items())
    return [(index + 1, str(name)) for index, name in enumerate(categories)]


def _categories_from_payload(payload: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Read a COCO json ``categories`` list into sorted (id, name) pairs."""
    return sorted(
        (int(c["id"]), str(c["name"])) for c in payload.get("categories", [])
    )


def yolo_to_coco(
    images_dir: Path | str,
    names: Mapping[int, str] | Sequence[str],
    out_json: Path | str,
    labels_dir: Path | str | None = None,
) -> Path:
    """Convert a YOLO layout (images + per-image label txts) to COCO json.

    ``names`` maps YOLO class indices to names (dict) or is a list whose
    positions are the indices. Label lines are ``cls cx cy w h`` with
    boxes normalized to [0, 1]; missing label files yield images with no
    annotations (valid for a background-heavy split).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else images_dir.parent / "labels"
    category_pairs = _sorted_categories(names)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    for image_id, image_path in enumerate(image_paths, start=1):
        from PIL import Image

        with Image.open(image_path) as handle:
            width, height = handle.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"'{label_path}': expected 'cls cx cy w h', got '{line}'"
                )
            class_index = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_pairs[class_index][0],
                    "bbox": [x, y, w * width, h * height],
                    "area": w * width * h * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for category_id, name in category_pairs
        ],
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def coco_to_yolo(
    annotations_json: Path | str,
    out_labels_dir: Path | str,
    names_out: Path | str | None = None,
) -> Path:
    """Convert a COCO detection json to YOLO label txts.

    Writes one ``<image_stem>.txt`` per annotated image with
    ``cls cx cy w h`` lines (normalized, YOLO class index = position of
    the COCO category id in sorted order). ``names_out`` (if given)
    receives one class name per line in that same order. Image ids with
    no annotations produce no file, mirroring ultralytics-style layouts.
    """
    payload = json.loads(Path(annotations_json).read_text())
    category_pairs = _categories_from_payload(payload)
    id_to_index = {category_id: index for index, (category_id, _) in enumerate(category_pairs)}

    sizes = {
        int(image["id"]): (int(image["width"]), int(image["height"]))
        for image in payload.get("images", [])
    }
    stems = {
        int(image["id"]): Path(image["file_name"]).stem
        for image in payload.get("images", [])
    }
    lines_by_image: dict[int, list[str]] = {}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in sizes:
            continue
        x, y, w, h = (float(v) for v in annotation["bbox"])
        width, height = sizes[image_id]
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        class_index = id_to_index[int(annotation["category_id"])]
        lines_by_image.setdefault(image_id, []).append(
            f"{class_index} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"
        )

    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    for image_id, lines in lines_by_image.items():
        (out_labels_dir / f"{stems[image_id]}.txt").write_text(
            "\n".join(lines) + "\n"
        )
    if names_out is not None:
        Path(names_out).write_text(
            "\n".join(name for _, name in category_pairs) + "\n"
        )
    return out_labels_dir
