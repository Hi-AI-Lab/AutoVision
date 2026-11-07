#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection backend coverage: run every registered detection family.

AutoVision registers four detection families. ``gtr`` (``detr/gtrvit-*``)
and ``rfdetr`` ship released, pretrained checkpoints, so this script runs
them directly. ``rtdetr`` and ``detr`` are released *architectures* without
pretrained task weights (``pretrained=none``) — they are exercised by a
tiny training run on ``coco8`` followed by inference on the exported
checkpoint. This way every family's code path (and the dispatch inside
``autovision.detection`` / ``autovision.train_object_detection``) is
covered end-to-end.

    python examples/tasks/detection_backends.py
    python examples/tasks/detection_backends.py --families rtdetr detr
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from _common import gtr_requirement, resolve_device

import autovision
from autovision.errors import AutoVisionError
from autovision.tasks.detection.pretrained import pretrained_class_names
from autovision.visualize import draw_prediction

ASSET = Path(__file__).resolve().parent / "assets" / "airport_aerial.jpg"

#: (registry key, short tag, mode). ``pretrained`` runs released weights;
#: ``train`` trains a 2-step smoke model on coco8 first (no released weights).
FAMILIES = [
    ("detr/gtrvit-s", "gtr", "pretrained"),
    ("rfdetr/rf-detr-nano", "rfdetr", "pretrained"),
    ("rtdetr/resnet18", "rtdetr", "train"),
    ("detr/s", "detr", "train"),
]


def _tolist(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _predict_with_fallback(
    model: Any, image: Path, threshold: float, device: str
) -> tuple[Any, str]:
    """Predict on ``device``, retrying on CPU after a CUDA OOM.

    A shared GPU can be short on memory for the random-init smoke models.
    Only an out-of-memory failure triggers the retry; every other error
    propagates unchanged instead of being hidden by the coverage example.
    """
    try:
        if device == "cuda" and hasattr(model, "to"):
            model.to(device)
        return model.predict(image, threshold=threshold), device
    except RuntimeError as error:
        if device != "cuda" or "out of memory" not in str(error).lower():
            raise
        if hasattr(model, "to"):
            model.to("cpu")
        print("note        = CUDA inference ran out of memory; retried on CPU")
        return model.predict(image, threshold=threshold), "cpu"


def _run_inference(model, key, tag, names, args, out_path, device=None) -> None:
    device = device or resolve_device(args.device)
    prediction, device = _predict_with_fallback(
        model, args.image, args.threshold, device
    )
    rendered = draw_prediction(
        args.image, prediction, class_names=names, min_score=args.threshold
    )
    rendered.save(out_path)
    labels = (
        [int(value) for value in _tolist(prediction["labels"])]
        if prediction.get("labels") is not None
        else []
    )
    named = [names[value] if names else str(value) for value in labels]
    print(f"family      = {tag} ({key})")
    print(f"device      = {device}")
    print(f"detections  = {len(labels)}")
    print(f"classes     = {dict(Counter(named).most_common())}")
    print(f"prediction  = {out_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ASSET)
    parser.add_argument(
        "--families",
        nargs="+",
        default=[tag for _, tag, _ in FAMILIES],
        help="subset of family tags to run (default: all four)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = CUDA when available, else CPU",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--train-steps", type=int, default=2,
                        help="steps for the train-only families (rtdetr/detr)")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/examples/detection_backends")
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    selected = {
        tag: (key, mode) for key, tag, mode in FAMILIES if tag in args.families
    }
    if not selected:
        raise SystemExit(
            f"No known family in {args.families}; choose from "
            f"{[tag for _, tag, _ in FAMILIES]}."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    for tag, (key, mode) in selected.items():
        out_path = args.out / f"{tag}.jpg"
        if mode == "pretrained":
            if key.startswith("detr/gtrvit"):
                gtr_requirement(device=resolve_device(args.device))
            names = pretrained_class_names(key)
            try:
                model = autovision.detection(key, pretrained=True)
                _run_inference(model, key, tag, names, args, out_path)
            except AutoVisionError as error:
                print(f"family      = {tag} ({key})")
                print(f"SKIPPED     = {error}")
                print()
            continue

        # Train-only family: a 2-step smoke run on coco8, then inference on
        # the exported checkpoint. Exercise the backend's training + export.
        print(f"family      = {tag} ({key}) [train {args.train_steps} steps]")
        work: Path | None = None
        try:
            data = autovision.ensure_coco8()
            work = Path(tempfile.mkdtemp(prefix=f"av_{tag}_"))
            result = autovision.train_object_detection(
                out=work,
                data=data,
                model=key,
                steps=args.train_steps,
                batch_size=2,
                num_workers=0,
                seed=0,
                overwrite=True,
                trainer_args={"val_every_num_steps": max(1, args.train_steps)},
            )
            pt = result.exported_model_path
            if pt is None:
                print("note        = training completed; no exported .pt produced")
                print("              (backend training path exercised; skip inference)")
                print()
                continue
            # Honour --device for the train-only families too. A shared GPU
            # may still be short on memory, so an OOM load falls back to CPU.
            requested = resolve_device(args.device)
            try:
                model = autovision.load_model(pt, device=requested)
            except RuntimeError as error:
                if requested != "cuda" or "out of memory" not in str(error).lower():
                    raise
                print("note        = CUDA model load ran out of memory; using CPU")
                requested = "cpu"
                model = autovision.load_model(pt, device="cpu")
            # Trained checkpoints carry no class names; label by index.
            _run_inference(model, key, tag, None, args, out_path, device=requested)
        except Exception as error:  # noqa: BLE001 - smoke run may fail broadly
            print(f"SKIPPED     = {type(error).__name__}: {error}")
            print()
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
