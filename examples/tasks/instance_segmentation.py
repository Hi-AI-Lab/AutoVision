#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Instance segmentation inference: boxes + masks -> annotate.

No training. By default this loads the official MIT ``gtrseg_s_coco``
checkpoint and runs it on the bundled street image. Pass
``--weights path/to/exported.pt`` to use a model from
``instance_segmentation_train.py`` (including EOMT checkpoints).
``--onnx`` runs the forward through an exported ONNX artifact (any
device; export one from a training run with
``instance_segmentation_train.py --export-onnx runs/gtrseg.onnx``).
"""

from __future__ import annotations

from pathlib import Path

from _common import (
    _coco_names,
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
        "Instance segmentation inference (GTR seg / exported .pt)",
        default_out="runs/examples/instance_segmentation",
    )
    parser.add_argument("--checkpoint", default="gtrseg_s_coco")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
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
        names = load_class_names(args.class_names) or _coco_names()
    else:
        gtr_requirement(need_cuda=not args.onnx, device=device)
        from autovision.tasks.instance_segmentation.gtr import GTRSegObjectDetection

        model = GTRSegObjectDetection(
            num_classes=80,
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(640, 640),
            load_checkpoint=not args.onnx,
        )
        model.eval().to(device)
        names = _coco_names()
    predict_and_annotate(model, args, names, title="instance_segmentation")


if __name__ == "__main__":
    main()
