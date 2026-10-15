#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""S-tier spectrum recipe: assemble ``detr/s`` from the Apache-2.0 D-FINE-S
asset, then a short COCO finetune (roadmap design doc 29 P1, D1 route =
assembly-finetune).

Gates fail loudly (no silent partial runs):

- CUDA device present;
- COCO 2017 present (default ``~/Research/Project/Dino-Dome-Research/dataset/coco``);
- the ``d-fine/dfine_s_coco`` checkpoint resolvable (the warm-start asset
  is built automatically on first run — widths are dfine_s-compatible by
  construction, decoder coverage 1.0).

Usage::

    python scripts/finetune_spectrum_s.py --epochs 6
    python scripts/finetune_spectrum_s.py --steps 500 --out runs/spectrum_s_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_COCO_ROOT = Path.home() / "Research/Project/Dino-Dome-Research/dataset/coco"
CACHE_CANDIDATES = (
    Path.home() / ".cache/autovision/d-fine_dfine_s_coco.pth",
    Path.home() / ".cache/autovision/downloadable_models/d-fine/dfine_s_coco.pt",
)


def _die(message: str) -> None:
    print(f"[gate] {message}", file=sys.stderr)
    sys.exit(1)


def _resolve_dfine_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            _die(f"--checkpoint does not exist: {path}")
        return path
    for candidate in CACHE_CANDIDATES:
        if candidate.exists():
            return candidate
    _die(
        "d-fine/dfine_s_coco not found in the weight cache. Resolve the "
        "registered asset once (network + mirrors per README), or pass "
        "--checkpoint."
    )
    raise SystemExit(1)  # unreachable; helps type checkers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=None,
                        help="override epoch-based step count")
    parser.add_argument("--out", type=Path, default=Path("runs/spectrum_s"))
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="reuse an existing AVWarmStart asset")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        _die("CUDA device required for the S-tier finetune (this recipe is "
             "not meaningful on CPU).")
    if not (args.coco_root / "annotations/instances_train2017.json").exists():
        _die(f"COCO 2017 not found under '{args.coco_root}'.")

    from autovision.tasks.detection.detr.dfine_official_warm_start import (
        build_dfine_official_warm_start,
    )

    warm_start = args.warm_start
    if warm_start is None:
        warm_start = args.out / "warm_start" / "detr_s_hgnetv2b1_dfine_s.pt"
        if not warm_start.exists():
            checkpoint = _resolve_dfine_checkpoint(
                str(args.checkpoint) if args.checkpoint else None
            )
            print(f"[assemble] building warm-start from {checkpoint.name} ...")
            report = build_dfine_official_warm_start(
                "hgnetv2-b1",
                out=warm_start,
                checkpoint=checkpoint,
                backbone_pretrained=True,
            )
            print(f"[assemble] done: {report}")

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
    print(f"[train] detr/s warm-start finetune: {steps} steps -> {args.out}")
    result = train_object_detection(
        out=args.out,
        data=data,
        model="detr/s",
        steps=steps,
        batch_size="auto",
        model_args={
            "warm_start": warm_start,
            "image_size": (640, 640),
            # Flat (fully vectorized) criterion loss paths: measured 2.8x
            # step-time at bs16/640 with <1e-6 per-loss parity (doc 26 W2;
            # off by default only for comparability with pre-2026-10-15
            # runs — new recipes opt in explicitly).
            "criterion_flat_losses": True,
        },
        trainer_args={"val_every_num_steps": max(1, steps // args.epochs)},
    )
    print(f"[done] best val_metric/map: {result.best_metrics.get('val_metric/map')}")
    print(f"[done] exported: {result.best_exported_model_path}")


if __name__ == "__main__":
    main()
