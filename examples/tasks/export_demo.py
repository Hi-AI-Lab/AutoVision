#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Export a trained checkpoint to every deployment format.

The per-task examples only exercise ONNX export. This script takes an
exported ``.pt`` (from any ``<task>_train.py`` run) and produces ONNX,
TensorRT (``.engine``), OpenVINO (``.xml``) and INT8 ONNX artifacts via
the parity-gated export layer, so those code paths are covered too.

    python examples/tasks/export_demo.py \
        --weights runs/examples/classification_train/exported_models/exported_last.pt
    python examples/tasks/export_demo.py --weights <det.pt> --int8 --allow-skips

Coverage boundaries: a classification ``.pt`` exports cleanly to all four
formats. GTR detection checkpoints can be refused by the parity gates —
TensorRT 11.4 miscompiles the unrolled GLA graph and the OpenVINO IR can
drift numerically (see ``docs/export.md``). Such refusals are reported as
``SKIPPED`` and make the script exit non-zero, unless ``--allow-skips``
accepts a known-unhealthy backend. Compiled-backend *inference*
(``--onnx`` / ``--openvino`` / ``--engine``) is unaffected: the detection
postprocess accepts the flat ``list`` a compiled runner returns.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from _common import export_task_to_onnx

from autovision.cli.export import export_model

_LABEL_WIDTH = 12


def main(argv: Sequence[str] | None = None) -> int:
    """Export every target, reporting skipped gates through the exit code."""
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
        "--allow-skips",
        action="store_true",
        help=(
            "exit 0 even when a parity gate refuses a target; use only for "
            "known-unhealthy backend/weight combinations"
        ),
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
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"Weights not found: '{args.weights}'", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    skipped: list[str] = []

    def ok(label: str, artifact: Path) -> None:
        """Report a target only when the artifact is really on disk."""
        if not artifact.is_file():
            skipped.append(
                f"{label}: export reported success but '{artifact}' is missing"
            )
            print(
                f"{label:<{_LABEL_WIDTH}}= SKIPPED: artifact missing at {artifact}"
            )
            return
        produced.append(label)
        print(f"{label:<{_LABEL_WIDTH}}= {artifact}")

    def fail(label: str, error: BaseException) -> None:
        """Record a backend/parity refusal instead of hiding it."""
        reason = str(error) or type(error).__name__
        skipped.append(f"{label}: {type(error).__name__}: {reason}")
        print(f"{label:<{_LABEL_WIDTH}}= SKIPPED: {type(error).__name__}: {reason}")

    # 1) ONNX (parity-gated). Reproduced here for a complete artifact set; the
    #    per-task examples already cover the same call.
    try:
        onnx_path = Path(export_task_to_onnx(args.weights, args.out / "model.onnx"))
        ok("onnx", onnx_path)
    except (SystemExit, Exception) as error:
        fail("onnx", error)

    # 2) TensorRT (.engine) — needs CUDA + the tensorrt package. An explicit
    #    --device cpu is a user choice, not a failed export.
    if device == "cuda":
        try:
            engine = export_model(
                args.weights,
                args.out / "model.engine",
                device="cuda",
                precision=args.trt_precision,
            )
            ok("tensorrt", Path(engine))
        except Exception as error:  # noqa: BLE001 - export may refuse (parity gate)
            fail("tensorrt", error)
    else:
        print(f"{'tensorrt':<{_LABEL_WIDTH}}= SKIPPED (needs CUDA; --device cpu)")

    # 3) OpenVINO (.xml) — needs the openvino package (CPU IR).
    try:
        openvino = export_model(args.weights, args.out / "model.xml", device="cpu")
        ok("openvino", Path(openvino))
    except Exception as error:  # noqa: BLE001 - export may refuse (parity gate)
        fail("openvino", error)

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
            ok("int8_onnx", Path(int8))
        except Exception as error:  # noqa: BLE001 - calibration can fail broadly
            fail("int8_onnx", error)

    print(f"{'summary':<{_LABEL_WIDTH}}= {len(produced)} exported, {len(skipped)} skipped")
    if skipped and not args.allow_skips:
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
        print(
            "Export skipped one or more targets; inspect the reasons above. "
            "Pass --allow-skips only when the backend is a known-unhealthy "
            "combination (e.g. GTR detection on TensorRT 11.x).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
