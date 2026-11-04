#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented detection inference: load weights -> rotated boxes -> annotate.

No training. By default this loads the official MIT ``gtrobb_s_dota``
checkpoint and runs it on the bundled aerial airport image. Pass
``--weights path/to/exported.pt`` to use a model from
``oriented_detection_train.py`` instead.
``--onnx`` runs the forward through an exported ONNX artifact (any
device; export one from a training run with
``oriented_detection_train.py --export-onnx runs/gtrobb.onnx``).
"""

from __future__ import annotations

from pathlib import Path

from _common import (
    AIRPORT_ASSET,
    _dota_names,
    build_inference_parser,
    gtr_eager_device_guard,
    gtr_requirement,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Oriented object detection inference (GTR OBB / exported .pt)",
        default_out="runs/examples/oriented_detection",
        default_image=AIRPORT_ASSET,
    )
    parser.add_argument("--checkpoint", default="gtrobb_s_dota")
    parser.add_argument("--variant", choices=("s", "x"), default="s")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help=(
            "exported .onnx artifact: run the forward through it (any "
            "device; skips the checkpoint download — the network only "
            "supplies pre/post-processing)"
        ),
    )
    args = parser.parse_args()

    # ONNX mode keeps the network CPU-resident (the session runs on the
    # CPU EP), so the eager-only CUDA requirements do not apply.
    device = "cpu" if args.onnx is not None else resolve_device(args.device)
    if args.weights is not None:
        model = autovision.load_model(args.weights, device=device)
        if args.onnx is None:
            gtr_eager_device_guard(model.network, device=device)
        names = load_class_names(args.class_names) or _dota_names()
    else:
        gtr_requirement(need_cuda=not args.onnx, device=device)
        from autovision.tasks.oriented_detection.gtr import GTRROBBOrientedDetection

        model = GTRROBBOrientedDetection(
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(1024, 1024),
            load_checkpoint=not args.onnx,
        )
        model.eval().to(device)
        names = _dota_names()
    predict_and_annotate(model, args, names, title="oriented_detection")


if __name__ == "__main__":
    main()
