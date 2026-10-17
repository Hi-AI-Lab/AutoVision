#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end detection self-training loop (roadmap C1, §2.3).

One pass of the pseudo-labeling closed loop, producing an evaluable model:

    teacher (.pt)  --generate_pseudo_labels-->  COCO pseudo labels
                   --(+ optional labeled train set, merged)-->
                   student training (train_object_detection)
                   --> exported model
                   --val on the labeled val split-->  metrics (the read)

Usage::

    python scripts/self_train_detect.py \
        --teacher runs/teacher/exported_models/exported_best.pt \
        --images /data/unlabeled_pool \
        --val-data coco_val.yaml \
        --out runs/self_train_r1 \
        --student rtdetr/resnet18 --steps 2000 --batch-size 8 \
        --threshold 0.5 [--sahi] [--labeled coco_train.yaml]

``--val-data`` is the labeled validation spec (yaml path or a dict literal)
whose ``val`` entry carries the annotations json + images dir; its category
list is what the pseudo labels are written against (the teacher's class
list). Writes into ``--out``: ``pseudo/train.json`` (+ ``pseudo/meta.json``),
the merged train json when ``--labeled`` is given, the student run
directory, and ``report.json`` with the full round record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

import autovision  # noqa: E402
from autovision.data.convert import merge_coco  # noqa: E402
from autovision.tasks.detection.shared.pseudo_label import (  # noqa: E402
    generate_pseudo_labels,
)


def _load_data_spec(value: str) -> dict[str, Any]:
    if value.strip().startswith("{"):
        return json.loads(value)
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"data spec not found: '{value}'")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"data spec must be a mapping: '{value}'")
    return loaded


def _categories_of(spec: dict[str, Any], split: str) -> dict[int, str]:
    entry = spec.get(split)
    if not isinstance(entry, dict) or "annotations" not in entry:
        raise SystemExit(f"the data spec needs a '{split}' entry with annotations.")
    payload = json.loads(Path(entry["annotations"]).read_text())
    return {int(c["id"]): str(c["name"]) for c in payload.get("categories", [])}


def _absolutize(annotations_json: Path, images_root: Path, out_json: Path) -> Path:
    """Rewrite ``file_name``s to absolute paths (for cross-root merging)."""
    payload = json.loads(annotations_json.read_text())
    for image in payload.get("images", []):
        name = Path(str(image["file_name"]))
        image["file_name"] = (
            name.as_posix() if name.is_absolute() else (images_root / name).resolve().as_posix()
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", required=True, help="exported teacher .pt (or registry key path)")
    parser.add_argument("--images", required=True, help="unlabeled image directory")
    parser.add_argument("--val-data", required=True, help="labeled val spec (yaml path or json literal)")
    parser.add_argument("--out", required=True, help="round output directory")
    parser.add_argument("--student", default="rtdetr/resnet18", help="student model key")
    parser.add_argument("--labeled", default=None, help="optional labeled train spec to merge in")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class-thresholds", default=None,
                        help='per-class overrides, e.g. \'{"person": 0.7}\'')
    parser.add_argument("--sahi", action="store_true", help="SAHI slicing for the teacher pass")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--pseudo-batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--steps", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--devices", default="auto", help="train/accelerator devices (auto/cpu/1/...)")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--model-args", default=None, help="student model_args as a json literal")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_spec = _load_data_spec(args.val_data)
    categories = _categories_of(val_spec, "val")

    # ── 1. pseudo labels from the teacher ───────────────────────────────
    teacher = autovision.load_task_model(args.teacher, device=None)
    pseudo = generate_pseudo_labels(
        teacher,
        args.images,
        out_dir / "pseudo" / "train.json",
        threshold=args.threshold,
        class_thresholds=json.loads(args.class_thresholds) if args.class_thresholds else None,
        categories=categories,
        sahi=args.sahi,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.pseudo_batch_size,
        max_images=args.max_images,
    )
    (out_dir / "pseudo" / "meta.json").write_text(
        json.dumps(pseudo.to_dict(), indent=2)
    )
    print(f"[pseudo] {pseudo.annotations} labels over {pseudo.images} images "
          f"(threshold {args.threshold}, sahi={args.sahi}) -> {pseudo.out_json}")
    if pseudo.annotations == 0:
        # An empty-annotation train json trains silently (all images are
        # skipped as background-free) — refuse instead of shipping a
        # nonsense student.
        raise SystemExit(
            f"No pseudo labels above threshold {args.threshold} over "
            f"{pseudo.images} images — lower --threshold, check the teacher, "
            "or use --class-thresholds. Refusing to train on empty labels."
        )

    # ── 2. student train spec: pseudo (+ optional labeled, merged) ──────
    train_annotations = pseudo.out_json
    if args.labeled:
        labeled_spec = _load_data_spec(args.labeled)
        labeled_entry = labeled_spec.get("train")
        if not isinstance(labeled_entry, dict) or "annotations" not in labeled_entry:
            raise SystemExit("--labeled needs a 'train' entry with annotations.")
        labeled_abs = _absolutize(
            Path(labeled_entry["annotations"]),
            Path(labeled_entry.get("images") or Path(labeled_entry["annotations"]).parent),
            out_dir / "labeled_absolute.json",
        )
        train_annotations = merge_coco(
            [labeled_abs, pseudo.out_json], out_dir / "train_merged.json"
        )
        print(f"[merge] labeled + pseudo -> {train_annotations}")
    train_spec = {
        "format": "coco",
        "train": {"annotations": str(train_annotations), "images": str(Path(args.images).resolve())},
        "val": val_spec["val"],
    }

    # ── 3. train the student ────────────────────────────────────────────
    student_kwargs: dict[str, Any] = {}
    if args.model_args:
        student_kwargs["model_args"] = json.loads(args.model_args)
    result = autovision.train_object_detection(
        out=out_dir / "student",
        data=train_spec,
        model=args.student,
        steps=args.steps,
        batch_size=args.batch_size,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        **student_kwargs,
    )
    print(f"[train] student finished: {result.steps_run} steps -> "
          f"{result.best_exported_model_path}")

    # ── 4. evaluate the produced model (the round's read) ───────────────
    student = autovision.load_model(result.best_exported_model_path, device=None)
    metrics = student.val(data=val_spec, split="val")
    print("[val] " + " | ".join(f"{key} {value:.4f}" for key, value in metrics.items()))

    report = {
        "teacher": str(args.teacher),
        "student": args.student,
        "pseudo": pseudo.to_dict(),
        "labeled_merged": bool(args.labeled),
        "train_annotations": str(train_annotations),
        "steps": result.steps_run,
        "exported_model": str(result.best_exported_model_path),
        "val_metrics": {key: float(value) for key, value in metrics.items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[report] {out_dir / 'report.json'}")
    print(f"[model] 可评测模型: {result.best_exported_model_path}")


if __name__ == "__main__":
    main()
