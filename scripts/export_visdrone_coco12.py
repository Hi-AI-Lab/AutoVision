# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Export VisDrone as a 12-class COCO annotation set (official-Dome parity).

The official Dome-S-VisDrone run evaluates over TWELVE categories — the
usual ten objects plus ``ignored regions`` (id 0) and ``others`` (id 11)
from the raw VisDrone label files. Our 10-class export drops the two
non-object categories, so headline mAP numbers are not directly
comparable with the official logs. This script converts the raw
per-image label txts into COCO json with all twelve categories (COCO ids
1-12, VisDrone id + 1), matching the official ``annotations_coco``
layout.

Usage:

    .venv/bin/python scripts/export_visdrone_coco12.py

Writes ``annotations_coco12/visdrone_train.json`` and
``annotations_coco12/visdrone_val.json`` next to the existing 10-class
export. Point a training arm's ``data`` at the new directory (the
pipeline infers ``num_classes = 12`` from the annotations).
"""

from __future__ import annotations

import json
from pathlib import Path

VISDRONE = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone"
)

#: Official VisDrone category table (raw label id -> name), COCO id = id + 1.
CATEGORIES = [
    (0, "ignored regions"),
    (1, "pedestrian"),
    (2, "people"),
    (3, "bicycle"),
    (4, "car"),
    (5, "van"),
    (6, "truck"),
    (7, "tricycle"),
    (8, "awning-tricycle"),
    (9, "bus"),
    (10, "motor"),
    (11, "others"),
]


def export_split(split_dir: Path, images_root: Path, out_file: Path) -> None:
    annotation_dirs = sorted((split_dir / "annotations").glob("*.txt"))
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for image_id, txt_file in enumerate(annotation_dirs, start=1):
        stem = txt_file.stem
        image_path = images_root / f"{stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"image missing for {txt_file.name}")
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size
        images.append(
            {"id": image_id, "file_name": f"{stem}.jpg", "width": width, "height": height}
        )
        for line in txt_file.read_text().splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 8:
                continue
            x, y, w, h = (float(parts[i]) for i in range(4))
            category = int(parts[5])
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": int(parts[4]),
                    "truncation": int(parts[6]),
                    "occlusion": int(parts[7]),
                }
            )
            annotation_id += 1
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": coco_id, "name": name} for coco_id, name in CATEGORIES
                ],
            }
        )
    )
    print(f"{out_file.name}: {len(images)} images, {len(annotations)} annotations")


def main() -> None:
    for split in ("train", "val"):
        export_split(
            VISDRONE / f"VisDrone2019-DET-{split}",
            VISDRONE / f"VisDrone2019-DET-{split}" / "images",
            VISDRONE / "annotations_coco12" / f"visdrone_{split}.json",
        )


if __name__ == "__main__":
    main()
