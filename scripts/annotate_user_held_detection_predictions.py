#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotate an image with the user-held DINOv3 COCO detectors.

These seven checkpoints are the restructured user-held detectors in
``~/Research/Model/AutoVision/detection``. They are **not redistributable**
(DINOv3 License + a third-party head with no declared license); this script
only produces local prediction images/JSON for inspection.

Usage::

    python scripts/annotate_user_held_detection_predictions.py \
        --image /path/to/baidu.jpg \
        --out ~/Research/Model/AutoVision/test_predictions/user_held_detection_baidu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from annotate_dfine_predictions import _annotate, _montage  # noqa: E402
from PIL import Image  # noqa: E402

from autovision.tasks.detection.detr.coco_checkpoint import (  # noqa: E402
    load_baseline_coco_checkpoint,
)
from autovision.tasks.detection.detr.task_model import (  # noqa: E402
    DETRObjectDetection,
)

#: variant dir -> (current model key, extra build overrides)
VARIANTS = {
    "dinov3-vitt16": ("dinov3-vitt16-notpretrained", {}),
    "dinov3-vitt16plus": ("dinov3-vitt16plus-notpretrained", {}),
    "dinov3-vits16": ("dinov3-vits16-notpretrained", {}),
    "dinov3-convnext-tiny": (
        "dinov3-convnext-tiny",
        {"dec_layers": 6, "decoder_hidden_dim": 256},
    ),
    "dinov3-convnext-small": (
        "dinov3-convnext-small",
        {"dec_layers": 6, "decoder_hidden_dim": 256},
    ),
    "dinov3-convnext-base": (
        "dinov3-convnext-base",
        {"dec_layers": 6, "decoder_hidden_dim": 256},
    ),
    "dinov3-convnext-large": (
        "dinov3-convnext-large",
        {"dec_layers": 6, "decoder_hidden_dim": 256},
    ),
}


def _detections(output: dict, classes: dict[int, str]) -> list[dict]:
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    boxes = output["bboxes"].detach().cpu().tolist()
    detections = []
    for label, score, box in zip(labels, scores, boxes):
        label = int(label)
        detections.append(
            {
                "label": label,
                "name": classes.get(label, str(label)),
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
        default=Path.home()
        / "Research/Model/AutoVision/test_predictions/user_held_detection_baidu",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/detection",
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
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant}")
        model_key, overrides = VARIANTS[variant]
        weights = args.store / variant / "weights.pt"
        if not weights.is_file():
            raise SystemExit(f"missing weights: {weights}")
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        init_args = payload.get("model_init_args", {})
        raw_classes = init_args.get("classes", {})
        classes = {int(key): str(value) for key, value in raw_classes.items()}
        model = DETRObjectDetection(
            num_classes=80,
            model_name=model_key,
            image_size=(640, 640),
            backbone_pretrained=False,
            load_weights=False,
            **overrides,
        ).eval()
        load_summary = load_baseline_coco_checkpoint(model, weights)
        model.to(args.device)
        output = model.predict(source, threshold=args.threshold)
        detections = _detections(output, classes)
        key = variant.upper()
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
            "load": load_summary,
        }
        title = (
            f"{key}  640x640  threshold {args.threshold:.2f}  "
            f"{len(detections)} detections"
        )
        annotated = _annotate(source, detections, title)
        path = args.out / f"{key}.jpg"
        annotated.save(path, quality=92)
        panels.append((key, annotated))
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
        "# User-held DINOv3 detector predictions",
        "",
        f"Image: `{args.image}` ({source.width}x{source.height})",
        f"Threshold: {args.threshold}",
        "",
        "| Model | Detections | Top classes | Load summary |",
        "|---|---:|---|---|",
    ]
    for key, entry in summary.items():
        load = ", ".join(f"{k}={v}" for k, v in entry["load"].items())
        lines.append(
            f"| {key} | {entry['detections']} | "
            f"{', '.join(entry['top_classes'])} | {load} |"
        )
    lines += [
        "",
        "> These weights are user-held: DINOv3 License + a third-party head",
        "> with no declared license. Do not redistribute or use commercially.",
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
