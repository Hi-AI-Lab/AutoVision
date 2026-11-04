#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Panoptic segmentation inference: panoptic map -> colorized overlay.

There is no released panoptic task checkpoint in the catalog. Pass an
exported ``.pt`` from ``panoptic_segmentation_train.py``:

    python examples/tasks/panoptic_segmentation.py \
        --weights runs/examples/panoptic_segmentation_train/exported_models/exported_last.pt
``--onnx`` runs the forward through an ONNX artifact exported from
the trained checkpoint (any device; pair with the same ``--weights``
so the network can supply pre/post-processing).
"""

from __future__ import annotations

from pathlib import Path

from _common import (
    build_inference_parser,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Panoptic segmentation inference (exported .pt)",
        default_out="runs/examples/panoptic_segmentation",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help=(
            "exported .onnx artifact: run the forward through it (any "
            "device; still needs --weights for the pre/post-processing "
            "network)"
        ),
    )
    args = parser.parse_args()

    if args.weights is None:
        raise SystemExit(
            "Panoptic has no released task checkpoint. Train once with "
            "`python examples/tasks/panoptic_segmentation_train.py` and pass "
            "--weights .../exported_models/exported_last.pt."
        )
    device = "cpu" if args.onnx is not None else resolve_device(args.device)
    model = autovision.load_model(args.weights, device=device)
    names = load_class_names(args.class_names)
    predict_and_annotate(model, args, names, title="panoptic_segmentation")


if __name__ == "__main__":
    main()
