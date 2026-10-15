#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""N-tier spectrum recipe (roadmap design doc 29 P1 / decision D1b).

Two modes:

- ``--mode baseline`` (default): supervised 12ep-equivalent baseline for
  ``detr/n`` — the reference anchor the distillation gain is measured
  against (D1b 增益锚).
- ``--mode distill``: cross-model detection distillation. **Not wired
  yet** — design doc 23 is the blueprint (teacher = RF-DETR-n /
  dfine_s). This script refuses with a pointer instead of silently
  training something else; wiring the teacher into the detection train
  path is a P1 remaining item.

Gates fail loudly: CUDA device, COCO 2017 root.

Usage::

    python scripts/train_spectrum_n_distill.py --epochs 12
    python scripts/train_spectrum_n_distill.py --steps 500 --out runs/spectrum_n_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_COCO_ROOT = Path.home() / "Research/Project/Dino-Dome-Research/dataset/coco"


def _die(message: str) -> None:
    print(f"[gate] {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--mode", choices=("baseline", "distill"), default="baseline")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/spectrum_n"))
    args = parser.parse_args()

    import torch

    if args.mode == "distill":
        _die(
            "distill mode: the GTR teacher bundle is ready "
            "(registry/gtr_assets.py, sha256-pinned, EMA semantics; "
            "gtr/gtr_teacher.py) but the teacher FORWARD still requires "
            "the GLA inference backend — roadmap doc 31 G2 (fla extras + "
            "re-ported vit_adapter line). State-dict level teacher usage "
            "(assembly into detr/gtr-s) is available today. Or run "
            "--mode baseline for the D1b anchor."
        )
    if not torch.cuda.is_available():
        _die("CUDA device required for the N-tier baseline run.")
    if not (args.coco_root / "annotations/instances_train2017.json").exists():
        _die(f"COCO 2017 not found under '{args.coco_root}'.")

    from autovision import train_object_detection

    data = {
        "format": "coco",
        "train": {
            "annotations": str(args.coco_root / "annotations/instances_train2017.json"),
            "images": str(args.coco_root / "images/train2017"),
        },
        "val": {
            "annotations": str(args.coco_root / "annotations/instances_val2017.json"),
            "images": str(args.coco_root / "images/val2017"),
        },
    }
    steps = args.steps or args.epochs * 3697  # 118287 imgs / 32 batch
    print(f"[train] detr/n supervised baseline (D1b anchor): {steps} steps")
    result = train_object_detection(
        out=args.out,
        data=data,
        model="detr/n",
        steps=steps,
        batch_size="auto",
        model_args={
            "image_size": (640, 640),
            # Flat (fully vectorized) criterion loss paths — see
            # finetune_spectrum_s.py (measured 2.8x at bs16/640, <1e-6
            # per-loss parity; new recipes opt in explicitly, doc 26 W2).
            "criterion_flat_losses": True,
        },
        trainer_args={"val_every_num_steps": max(1, steps // args.epochs)},
    )
    print(f"[done] best val_metric/map: {result.best_metrics.get('val_metric/map')}")
    print(f"[done] exported: {result.best_exported_model_path}")


if __name__ == "__main__":
    main()
