# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Gate G1 (design doc 17 §6): window-droppability statistics for token dropping.

Reads a COCO annotation file directly (no model, no GPU) and replays the
training-time mask construction at the training resolution: boxes are
scaled to the square canvas, the margin-dilated box-overlapping windows
are protected (box-aware eraser semantics, doc 16), and a ``keep_ratio``
subset of the background windows survives. Reports the per-image kept /
dropped-window distribution — the droppability numbers that decide whether
in-backbone token dropping is a speed+augmentation feature (G1 > 25%
expected dropped windows) or degrades to an augmentation-only ablation
(< 10%), per design doc 17 §6.

    python scripts/visdrone_token_drop_stats.py \
        --annotations /path/visdrone_train.json --resolution 800 --window 32
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from autovision.augment.detection.box_aware_eraser import (
    protected_cell_mask,
    sample_erase_mask,
)


def _coco_boxes_norm(
    annotations: list[dict], src_w: float, src_h: float, dst: int
) -> list[list[float]]:
    """COCO absolute xywh → normalized cxcywh on the square training canvas.

    Mirrors the training transform's resize: the image is distorted onto
    the square ``dst`` canvas, and normalized box coordinates are
    scale-invariant under that resize — so the normalized values come
    straight from the source geometry (the ``dst`` argument only documents
    the canvas the grid masks are built for).
    """
    out: list[list[float]] = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / src_w
        cy = (y + h / 2) / src_h
        out.append([cx, cy, w / src_w, h / src_h])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window-droppability statistics (doc 17 gate G1)."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone"
            "/annotations/visdrone_train.json"
        ),
        help="COCO annotation json (default: VisDrone train).",
    )
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--keep-ratio", type=float, default=0.4)
    parser.add_argument(
        "--max-images", type=int, default=0, help="0 = all images."
    )
    namespace = parser.parse_args()

    payload = json.loads(namespace.annotations.read_text())
    images = payload["images"]
    if namespace.max_images:
        images = images[: namespace.max_images]
    boxes_by_image: dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        boxes_by_image.setdefault(ann["image_id"], []).append(ann)

    res, window = namespace.resolution, namespace.window
    if res % window:
        raise SystemExit(f"resolution {res} not divisible by window {window}")

    # Statistics that mirror training_step's mask construction: a sample
    # whose draw keeps every window runs dense (None mask), so its windows
    # are NOT droppable in practice either.
    droppable_fractions: list[float] = []
    images_fully_protected = 0
    images_no_boxes = 0
    total_windows = 0
    dropped_windows = 0
    for image in images:
        annotations = boxes_by_image.get(image["id"], [])
        boxes_norm = _coco_boxes_norm(
            annotations, image["width"], image["height"], res
        )
        if not boxes_norm:
            images_no_boxes += 1
            continue
        boxes = torch.tensor(boxes_norm, dtype=torch.float32)
        protected = protected_cell_mask(
            boxes, res, res, cell=window, padding=namespace.padding
        )
        keep = ~sample_erase_mask(protected, 1.0 - namespace.keep_ratio)
        total_windows += keep.numel()
        dropped = int((~keep).sum())
        dropped_windows += dropped
        if bool(keep.all()):
            images_fully_protected += 1
        else:
            droppable_fractions.append(dropped / keep.numel())

    count = len(droppable_fractions)
    annotated = count + images_fully_protected
    print("=" * 64)
    print(f"annotations file : {namespace.annotations.name}")
    print(
        f"canvas {res}px, window {window}px "
        f"({res // window}x{res // window} grid), padding {namespace.padding}, "
        f"keep_ratio {namespace.keep_ratio}"
    )
    print(f"images           : {len(images)} total, {annotated} with boxes, "
          f"{images_no_boxes} without")
    print(f"fully-protected  : {images_fully_protected} "
          f"({images_fully_protected / max(annotated, 1):.1%}) — run dense")
    if count:
        mean = statistics.mean(droppable_fractions)
        median = statistics.median(droppable_fractions)
        ordered = sorted(droppable_fractions)
        p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        print(f"droppable/image  : mean {mean:.1%}, median {median:.1%}, "
              f"p90 {p90:.1%}")
    expected = dropped_windows / max(total_windows, 1)
    print(f"expected dropped : {expected:.1%} of all windows "
          f"(G1 gate: <10% aug-only, >25% speed+aug)")
    print("=" * 64)


if __name__ == "__main__":
    main()
