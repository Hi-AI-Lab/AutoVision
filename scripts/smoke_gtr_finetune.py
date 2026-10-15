#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR smoke fine-tune: official ``gtr_s_coco`` weights -> ``detr/gtrvit-s``.

Loads the FULL official GTR-S checkpoint (EMA): backbone via
``extract_gtr_backbone`` (``backbone_weights``), encoder+decoder via the
AVWarmStart assembly (``source_format="gtr"``, 100% decoder coverage per
the 2026-10-16 gold-check), then runs a short COCO fine-tune with a
500-image val subset for the trajectory (full-val runs cost ~7 min each;
the subset keeps the smoke interpretable, it is NOT the publishable
number).

Gates fail loudly (no silent partial runs):

- CUDA device present;
- COCO 2017 present (default ``~/Research/Project/Dino-Dome-Research/dataset/coco``);
- the ``gtr_s_coco`` asset resolvable (local weight registry, MIT).

Usage::

    python scripts/smoke_gtr_finetune.py --steps 1000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_COCO_ROOT = Path.home() / "Research/Project/Dino-Dome-Research/dataset/coco"


def _die(message: str) -> None:
    print(f"[gate] {message}", file=sys.stderr)
    sys.exit(1)


def _write_val_subset(val_annotations: Path, out: Path, count: int) -> Path:
    """First ``count`` val images (with annotations) as a standalone json."""
    if out.exists():
        return out
    with open(val_annotations) as fh:
        payload = json.load(fh)
    keep = {image["id"] for image in payload["images"][:count]}
    subset = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "categories": payload["categories"],
        "images": [image for image in payload["images"] if image["id"] in keep],
        "annotations": [
            annotation
            for annotation in payload["annotations"]
            if annotation["image_id"] in keep
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-subset", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("runs/gtr_smoke"))
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("runs/gtr_smoke_assets"),
        help="warm-start / backbone / val-subset artifacts (kept out of the "
        "run dir so the trainer's empty-dir guard does not conflict)",
    )
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run in --out (steps = total target)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        _die("CUDA device required (the GTR GLA backbone is a Triton kernel).")
    train_annotations = args.coco_root / "annotations/instances_train2017.json"
    val_annotations = args.coco_root / "annotations/instances_val2017.json"
    if not train_annotations.exists() or not val_annotations.exists():
        _die(f"COCO 2017 annotations not found under '{args.coco_root}'.")

    from autovision.registry.gtr_assets import resolve_gtr_asset

    try:
        checkpoint = resolve_gtr_asset("gtr_s_coco").path
    except Exception as error:  # noqa: BLE001 - gate: fail loud with context
        _die(f"gtr_s_coco asset not resolvable: {error}")
    print(f"[asset] GTR-S checkpoint: {checkpoint}")

    # ── encoder+decoder warm start (AVWarmStart, MIT provenance) ────────
    warm_start = args.warm_start or (
        args.assets / "warm_start" / "gtrvit_s_official.pt"
    )
    if not warm_start.exists():
        from autovision.tasks.detection.detr.dfine_official_warm_start import (
            build_dfine_official_warm_start,
        )

        print("[assemble] building gtrvit-s warm start from the official EMA ...")
        report = build_dfine_official_warm_start(
            "gtrvit-s",
            out=warm_start,
            checkpoint=checkpoint,
            backbone_pretrained=False,
            license_id="MIT",
            origin="https://huggingface.co/Phoenix8125/GTR",
            source_format="gtr",
        )
        print(f"[assemble] {report}")
        if report["decoder_coverage"] < 1.0 or report["skipped_shape"] != 0:
            _die(
                "warm-start assembly is not complete "
                f"(coverage={report['decoder_coverage']}, "
                f"skipped={report['skipped_shape']}) — refusing to smoke on a "
                "partial load."
            )

    # ── backbone weights (official ViT-SpatialSwiGLU state) ─────────────
    backbone_file = args.assets / "backbone" / "gtr_s_backbone.pth"
    if not backbone_file.exists():
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )

        backbone_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(extract_gtr_backbone(load_gtr_teacher(checkpoint)), backbone_file)
        print(f"[assemble] backbone state -> {backbone_file}")

    # ── 500-image val subset (trajectory instrument) ────────────────────
    val_subset = _write_val_subset(
        val_annotations,
        args.assets / "val_subset" / "instances_val_subset.json",
        args.val_subset,
    )

    from autovision import train_object_detection

    data = {
        "format": "coco",
        "train": {
            "annotations": str(train_annotations),
            "images": str(args.coco_root / "images/train2017"),
        },
        "val": {
            "annotations": str(val_subset),
            "images": str(args.coco_root / "images/val2017"),
        },
    }
    print(
        f"[train] detr/gtrvit-s full-official-weights fine-tune: "
        f"{args.steps} steps, bs={args.batch_size}, val={args.val_subset} imgs"
    )
    result = train_object_detection(
        out=args.out,
        data=data,
        model="detr/gtrvit-s",
        steps=args.steps,
        batch_size=args.batch_size,
        resume_interrupted=args.resume,
        model_args={
            "warm_start": warm_start,
            "backbone_weights": backbone_file,
            "image_size": (640, 640),
            # Flat criterion loss paths (doc 26 W2): 2.8x step time, <1e-6
            # per-loss parity.
            "criterion_flat_losses": True,
            # EMA seeds from the (official) starting weights; a short smoke
            # must not be dominated by a cold EMA window.
            "ema_warmup_steps": min(100, max(1, args.steps // 10)),
        },
        trainer_args={
            "val_every_num_steps": max(1, args.steps // 4),
        },
    )
    best = result.best_metrics.get("val_metric/map")
    print(f"[done] steps={result.steps_run} best val_metric/map={best}")
    print(f"[done] exported: {result.best_exported_model_path}")


if __name__ == "__main__":
    main()
