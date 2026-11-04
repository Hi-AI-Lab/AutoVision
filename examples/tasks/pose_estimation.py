#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pose estimation inference: keypoints -> COCO skeleton overlay.

No training. By default this loads the official MIT ``gtrpose_s_coco``
checkpoint and runs it on the bundled street image. Pass
``--weights path/to/exported.pt`` to use a model from
``pose_estimation_train.py``.
``--onnx`` runs the forward through an exported ONNX artifact (any
device; export one from a training run with
``pose_estimation_train.py --export-onnx runs/gtrpose.onnx``).
"""

from __future__ import annotations

from pathlib import Path

from _common import (
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
        "Pose estimation inference (GTR pose / exported .pt)",
        default_out="runs/examples/pose_estimation",
    )
    parser.add_argument("--checkpoint", default="gtrpose_s_coco")
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
        names = load_class_names(args.class_names) or {0: "person", 1: "person"}
    else:
        gtr_requirement(need_cuda=not args.onnx, device=device)
        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        model = GTRPoseEstimation(
            num_classes=2,
            checkpoint=args.checkpoint,
            variant=args.variant,
            image_size=(640, 640),
            load_checkpoint=not args.onnx,
        )
        model.eval().to(device)
        names = {0: "person", 1: "person"}
    predict_and_annotate(model, args, names, title="pose_estimation")


if __name__ == "__main__":
    main()
