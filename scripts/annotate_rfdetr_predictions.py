#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotate an image with the prepared Apache-2.0 RF-DETR weights.

Loads each ``rfdetr/rf-detr-*/weights.pt`` through
:func:`autovision.load_task_model`, runs the task model's predict pipeline,
and writes annotated images plus a JSON detection dump.

Usage::

    python scripts/annotate_rfdetr_predictions.py \
        --image /path/to/baidu.jpg \
        --out ~/Research/Model/AutoVision/test_predictions/rfdetr_baidu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from annotate_dfine_predictions import (  # noqa: E402
    _annotate,
    _montage,
)
from PIL import Image  # noqa: E402

from autovision import load_task_model  # noqa: E402

VARIANT_SIZES = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "base": 560,
    "large": 704,
}


def _detections(output: dict, id2label: dict[int, str]) -> list[dict]:
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    boxes = output["bboxes"].detach().cpu().tolist()
    detections = []
    for label, score, box in zip(labels, scores, boxes):
        label = int(label)
        if label == 0:  # RF-DETR's N/A slot, not a COCO object
            continue
        detections.append(
            {
                "label": label,
                "name": str(id2label.get(label, label)),
                "score": float(score),
                "box": [float(value) for value in box],
            }
        )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True, help="input image")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/test_predictions/rfdetr_baidu",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["nano", "small", "medium", "base", "large"],
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/apache_detection/rfdetr",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    source.save(args.out / "source.jpg", quality=92)
    all_detections: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    panels: list[tuple[str, Image.Image]] = [("source", source)]

    for variant in args.variants:
        weights = args.store / f"rf-detr-{variant}" / "weights.pt"
        if not weights.is_file():
            raise SystemExit(f"missing weights: {weights}")
        model = load_task_model(weights, device=args.device)
        output = model.predict(source, threshold=args.threshold)
        # The hermetic rebuild does not carry the upstream id2label table;
        # read the official mapping shipped next to weights.pt.
        config_path = weights.parent / "config.json"
        raw_id2label = json.loads(config_path.read_text(encoding="utf-8"))["id2label"]
        id2label = {int(key): value for key, value in raw_id2label.items()}
        detections = _detections(output, id2label)
        key = f"RF-DETR-{variant.upper()}"
        all_detections[key] = detections
        counts: dict[str, int] = {}
        for det in detections:
            counts[det["name"]] = counts.get(det["name"], 0) + 1
        top_classes = sorted(counts, key=lambda name: -counts[name])[:8]
        summary[key] = {
            "detections": len(detections),
            "top_classes": top_classes,
            "class_counts": counts,
            "top_scores": [round(det["score"], 3) for det in detections[:10]],
        }
        size = VARIANT_SIZES[variant]
        title = (
            f"RF-DETR-{variant.upper()}  {size}x{size}  "
            f"threshold {args.threshold:.2f}  {len(detections)} detections"
        )
        annotated = _annotate(source, detections, title)
        path = args.out / f"RF-DETR-{variant.upper()}.jpg"
        annotated.save(path, quality=92)
        panels.append((f"RF-DETR-{variant.upper()}", annotated))
        print(f"{variant}: {len(detections)} detections -> {path}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _montage(panels, args.out / "comparison.jpg", columns=2)
    (args.out / "detections.json").write_text(
        json.dumps(all_detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# RF-DETR Baidu image predictions",
        "",
        f"Image: `{args.image}` ({source.width}x{source.height})",
        f"Threshold: {args.threshold}",
        "",
        "| Model | Detections | Top classes |",
        "|---|---:|---|",
    ]
    for key, entry in summary.items():
        lines.append(
            f"| {key} | {entry['detections']} | {', '.join(entry['top_classes'])} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
