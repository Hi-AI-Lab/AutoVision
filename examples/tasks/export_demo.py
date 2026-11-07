#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Export a trained checkpoint to every deployment format.

The per-task examples only exercise ONNX export. This script takes an
exported ``.pt`` (from any ``<task>_train.py`` run) and produces ONNX,
TensorRT (``.engine``), OpenVINO (``.xml``) and INT8 ONNX artifacts via
the parity-gated export layer, so those code paths are covered too.

    python examples/tasks/export_demo.py \
        --weights runs/examples/classification_train/exported_models/exported_last.pt
    python examples/tasks/export_demo.py --weights <det.pt> --int8 --calib-dir assets

Note (verified coverage boundaries): classification ``.pt`` exports cleanly to
all four formats. For detection (``gtr`` family) the TensorRT build fails the
parity gate (``assert_outputs_close_sum`` — GLA unrolled-graph numerical
drift); that is a framework limitation surfaced by this example, not an example
bug. The OpenVINO / ONNX / INT8 paths run end to end — the detection
postprocess now accepts the flat ``list`` the compiled runners (OpenVINO /
TensorRT) return, in the exported output-port order ``[logits, boxes]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _common import export_task_to_onnx

from autovision.cli.export import export_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="exported .pt from a <task>_train.py run",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/examples/export_demo")
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for TensorRT building (auto = CUDA when available)",
    )
    parser.add_argument(
        "--int8", action="store_true", help="also build an INT8 ONNX via static PTQ"
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="image folder for INT8 calibration (default: assets/)",
    )
    parser.add_argument(
        "--trt-precision",
        choices=("fp32", "fp16"),
        default="fp16",
        help="TensorRT build precision",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: '{args.weights}'")

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # 1) ONNX (parity-gated). Reproduced here for a complete artifact set; the
    #    per-task examples already cover the same call.
    onnx_path = args.out / "model.onnx"
    try:
        export_task_to_onnx(args.weights, onnx_path)
        print(f"onnx        = {onnx_path}")
    except SystemExit as error:
        print(f"onnx        = SKIPPED: {error}")

    # 2) TensorRT (.engine) — needs CUDA + the tensorrt package.
    if device == "cuda":
        try:
            engine = export_model(
                args.weights,
                args.out / "model.engine",
                device="cuda",
                precision=args.trt_precision,
            )
            print(f"tensorrt    = {engine}")
        except Exception as error:  # noqa: BLE001 - export may refuse (parity gate)
            print(f"tensorrt    = SKIPPED: {type(error).__name__}: {error}")
    else:
        print("tensorrt    = SKIPPED (needs CUDA)")

    # 3) OpenVINO (.xml) — needs the openvino package (CPU IR).
    try:
        openvino = export_model(args.weights, args.out / "model.xml", device="cpu")
        print(f"openvino    = {openvino}")
    except Exception as error:  # noqa: BLE001 - export may refuse (parity gate)
        print(f"openvino    = SKIPPED: {type(error).__name__}: {error}")

    # 4) INT8 ONNX (static PTQ) — needs an image folder for calibration.
    if args.int8:
        calib = args.calib_dir or (Path(__file__).resolve().parent / "assets")
        try:
            from autovision.export import image_folder_calibration_reader
            from autovision.tasks.task_model_helpers import load_task_model

            network = load_task_model(args.weights, device="cpu")
            reader = image_folder_calibration_reader(network, calib, max_samples=8)
            int8 = network.export_onnx_int8(
                args.out / "model_int8.onnx", calibration_reader=reader
            )
            print(f"int8_onnx   = {int8}")
        except Exception as error:  # noqa: BLE001 - calibration can fail broadly
            print(f"int8_onnx   = SKIPPED: {error}")


if __name__ == "__main__":
    main()
