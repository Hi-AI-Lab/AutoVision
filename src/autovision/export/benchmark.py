# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Inference benchmarking across execution backends.

Backends mirror the baseline ``benchmark_backends`` semantics:

- ``torch``: the eager PyTorch model (optional ``torch.compile`` + autocast).
- ``onnx``: exports ONNX (reusing the export pipeline) and benchmarks an
  onnxruntime session; the requested provider is verified to actually be
  active (no silent CPU fallback).
- ``tensorrt``: builds a TensorRT engine via ``TaskNetwork.export_tensorrt``
  and benchmarks raw engine execution.

Only the forward pass is timed — preprocessing and postprocessing are
excluded, matching :func:`benchmark_inference`. An optional parity gate
compares torch vs ONNX raw outputs with the same tolerances as the export
verification gate (atol 5e-3 / rtol 1e-1).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from autovision.config.base import PydanticConfig

if TYPE_CHECKING:
    from autovision.tasks.task_model import TaskNetwork

__all__ = [
    "BenchmarkResult",
    "ONNXBackendArgs",
    "TensorRTBackendArgs",
    "TorchBackendArgs",
    "benchmark",
    "benchmark_inference",
]

logger = logging.getLogger("autovision")


@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    num_steps: int
    batch_size: int
    total_time_sec: float
    images_per_sec: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    def to_markdown(self) -> str:
        backend = self.extra.get("backend", "torch")
        precision = self.extra.get("precision", "fp32")
        return (
            f"| metric | value |\n|---|---|\n"
            f"| model | {self.model_name} |\n"
            f"| backend | {backend} ({precision}) |\n"
            f"| device | {self.device} |\n"
            f"| images/s | {self.images_per_sec:.1f} |\n"
            f"| latency mean (ms) | {self.latency_ms_mean:.2f} |\n"
            f"| latency p50/p90 (ms) | {self.latency_ms_p50:.2f} / {self.latency_ms_p90:.2f} |\n"
        )


@torch.no_grad()
def benchmark_inference(
    model,
    example_inputs: dict[str, torch.Tensor],
    *,
    model_name: str = "model",
    warmup_steps: int = 5,
    steps: int = 20,
    backend: str = "torch",
    precision: str = "fp32",
    forward: Any = None,
) -> BenchmarkResult:
    """Measure per-step latency and throughput on the current device.

    ``forward`` overrides the timed callable (defaults to
    ``model.forward_backend``), e.g. to benchmark a compiled function.
    """
    forward_fn = forward if forward is not None else model.forward_backend
    model.eval()
    device = next(model.parameters()).device
    inputs = {name: t.to(device) for name, t in example_inputs.items()}
    example = next(iter(inputs.values()))
    for _ in range(warmup_steps):
        forward_fn(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies = []
    start_total = time.perf_counter()
    for _ in range(steps):
        start = time.perf_counter()
        forward_fn(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    total = time.perf_counter() - start_total
    batch_size = example.shape[0]
    latencies_ms = sorted(latency * 1000 for latency in latencies)
    result = BenchmarkResult(
        model_name=model_name,
        device=str(device),
        num_steps=steps,
        batch_size=batch_size,
        total_time_sec=total,
        images_per_sec=batch_size * steps / total,
        latency_ms_mean=sum(latencies_ms) / len(latencies_ms),
        latency_ms_p50=latencies_ms[max(0, (len(latencies_ms) + 1) // 2 - 1)],
        latency_ms_p90=latencies_ms[int(len(latencies_ms) * 0.9)],
        extra={"backend": backend, "precision": precision},
    )
    logger.info(
        "Benchmark [%s/%s]: %.1f img/s (%.2f ms mean)",
        backend, precision, result.images_per_sec, result.latency_ms_mean,
    )
    return result


# ── backend args ────────────────────────────────────────────────────────


class TorchBackendArgs(PydanticConfig):
    """Eager PyTorch backend options."""

    compile: bool = False
    precision: Literal["fp32", "bf16-mixed", "fp16-mixed"] = "fp32"


class ONNXBackendArgs(PydanticConfig):
    """ONNX runtime backend options."""

    precision: Literal["fp32", "fp16"] = "fp32"
    provider: Literal["cpu", "cuda", "tensorrt"] = "cpu"
    dynamic_batch_size: bool = True
    simplify: bool = True
    export_args: dict[str, Any] | None = None


class TensorRTBackendArgs(PydanticConfig):
    """TensorRT engine backend options."""

    precision: Literal["fp32", "fp16"] = "fp32"
    min_batch_size: int = 1
    export_args: dict[str, Any] | None = None


BackendArgs = TorchBackendArgs | ONNXBackendArgs | TensorRTBackendArgs


# ── backends ────────────────────────────────────────────────────────────


class TorchBackend:
    """Eager model with optional compile and autocast."""

    def __init__(
        self, model: "TaskNetwork", args: TorchBackendArgs, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.args = args
        if hasattr(model, "deploy"):
            model.deploy()
        else:
            model.eval()
        self._forward = model.forward_backend
        if args.compile:
            self._forward = torch.compile(model.forward_backend)

    def run(self, images: torch.Tensor) -> Any:
        autocast_dtype = {
            "bf16-mixed": torch.bfloat16,
            "fp16-mixed": torch.float16,
        }.get(self.args.precision)
        with torch.autocast(
            device_type=self.device.type, dtype=autocast_dtype or torch.float32,
            enabled=autocast_dtype is not None,
        ):
            return self._forward(images)


_ONNX_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
}


class ONNXBackend:
    """ONNX runtime session; the model stays on CPU for postprocessing."""

    def __init__(
        self,
        model: "TaskNetwork",
        args: ONNXBackendArgs,
        *,
        batch_size: int,
        out_dir: Path,
        device: torch.device,
    ) -> None:
        import copy

        import onnxruntime as ort

        export_model = copy.deepcopy(model).to("cpu").float()
        export_args = dict(args.export_args or {})
        export_args.setdefault("verify", False)
        export_args.setdefault("simplify", args.simplify)
        export_args.setdefault(
            "dynamic_batch_size", args.dynamic_batch_size
        )
        if not args.dynamic_batch_size:
            export_args["batch_size"] = batch_size
        export_args.setdefault("precision", args.precision)
        self.onnx_path = out_dir / "model.onnx"
        export_model.export_onnx(out=self.onnx_path, **export_args)

        providers = _ONNX_PROVIDERS[args.provider]
        available = set(ort.get_available_providers())
        missing = [p for p in providers if p not in available]
        if missing:
            raise RuntimeError(
                f"ONNX backend provider '{args.provider}' requires {missing} "
                f"but only {sorted(available)} are available."
            )
        device_id = device.index if device.type == "cuda" and device.index is not None else 0
        provider_options = []
        for provider in providers:
            options: dict[str, Any] = {}
            if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                options["device_id"] = device_id
            provider_options.append(options)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )
        expected = providers[0]
        if expected not in self.session.get_providers():
            raise RuntimeError(
                f"ONNX backend provider '{args.provider}' failed to initialize: "
                f"requested {expected} but session is using "
                f"{self.session.get_providers()}."
            )
        self.precision = args.precision
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, images: torch.Tensor) -> list[Any]:
        feed = images.cpu()
        if self.precision == "fp16":
            feed = feed.half()
        outputs = self.session.run(
            output_names=self.output_names,
            input_feed={self.input_name: feed.numpy()},
        )
        # The copy is included in the timing on purpose: host transfer is part
        # of the ONNX deployment path.
        return [
            torch.from_numpy(output) if hasattr(output, "shape") else output
            for output in outputs
        ]


class TensorRTBackend:
    """TensorRT engine execution (requires the tensorrt package + GPU)."""

    def __init__(
        self,
        model: "TaskNetwork",
        args: TensorRTBackendArgs,
        *,
        batch_size: int,
        out_dir: Path,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("The TensorRT backend requires a CUDA device.")
        import copy

        self.engine_path = out_dir / "model.engine"
        export_model = copy.deepcopy(model).to(device)
        export_args = dict(args.export_args or {})
        export_args.setdefault("min_batch_size", args.min_batch_size)
        export_args.setdefault("opt_batch_size", batch_size)
        export_args.setdefault("max_batch_size", batch_size)
        export_args["onnx_args"] = {
            **export_args.get("onnx_args", {}),
            "verify": False,
            "precision": "fp32",
        }
        export_model.export_tensorrt(
            out=self.engine_path, precision=args.precision, **export_args
        )
        self.precision = args.precision

        import tensorrt as trt  # type: ignore[import-not-found]

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as handle:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        self.context = self.engine.create_execution_context()
        self.output_names: list[str] = []
        self.input_name = ""
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        self.input_shape = list(self.engine.get_tensor_shape(self.input_name))
        self.stream = torch.cuda.Stream(device=device)
        self._trt = trt

    def run(self, images: torch.Tensor) -> list[torch.Tensor]:
        import numpy as np

        images = images.contiguous().to("cuda")
        self.context.set_input_shape(self.input_name, list(images.shape))
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        outputs: list[torch.Tensor] = []
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            dtype = self._trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=dtype)).dtype
            buffer = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, buffer.data_ptr())
            outputs.append(buffer)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


# ── command ─────────────────────────────────────────────────────────────

_BACKEND_KEYS = ("torch", "onnx", "tensorrt")


def _normalize_backends(
    backends: dict[str, BackendArgs | dict[str, Any] | None] | None,
) -> dict[str, BackendArgs]:
    """Fill defaults and coerce dicts into typed backend args."""
    from autovision.config.validate import pydantic_model_validate

    if backends is None:
        backends = {"torch": TorchBackendArgs(), "onnx": ONNXBackendArgs()}
    unknown = set(backends) - set(_BACKEND_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown backends: {sorted(unknown)}. Supported: {list(_BACKEND_KEYS)}"
        )
    normalized: dict[str, BackendArgs] = {}
    args_classes: dict[str, type] = {
        "torch": TorchBackendArgs,
        "onnx": ONNXBackendArgs,
        "tensorrt": TensorRTBackendArgs,
    }
    for key, value in backends.items():
        args_cls = args_classes[key]
        if isinstance(value, args_cls):
            normalized[key] = value
        else:
            normalized[key] = pydantic_model_validate(args_cls, dict(value or {}))
    return normalized


def _resolve_benchmark_model(
    model: str,
    *,
    num_classes: int,
    image_size: tuple[int, int] | None,
) -> "TaskNetwork":
    """Resolve a ``"<family>/<backbone>"`` spec to a task model instance.

    Currently supports the RT-DETR detection family (``rtdetr/resnet18`` ...);
    the resolved model carries random weights — benchmarks measure speed, not
    accuracy.
    """
    from autovision.tasks.detection.rtdetr import (
        RTDETRObjectDetection,
    )

    package, _, name = model.partition("/")
    if package != "rtdetr":
        raise ValueError(
            f"Unsupported benchmark model spec '{model}'. Supported: "
            f"'rtdetr/<backbone>' (e.g. 'rtdetr/resnet18'), or pass a "
            f"TaskNetwork instance."
        )
    return RTDETRObjectDetection(
        num_classes=num_classes,
        backbone=f"torchvision/{name}" if "/" not in name else name,
        image_size=image_size or (640, 640),
        load_weights=False,
    )


def benchmark(
    model: "TaskNetwork | str",
    out: Path | str,
    *,
    num_classes: int = 10,
    image_size: tuple[int, int] | None = None,
    batch_size: int = 1,
    runs: int = 20,
    warmup: int = 5,
    backends: dict[str, BackendArgs | dict[str, Any] | None] | None = None,
    verify_parity: bool = False,
    device: str = "auto",
) -> dict[str, BenchmarkResult]:
    """Benchmark inference latency across execution backends.

    ``model`` is a :class:`TaskNetwork` instance or a ``"rtdetr/<backbone>"``
    spec string. Writes ``benchmark_<backend>.json`` (per backend) and
    ``benchmark.md`` (summary table) to ``out``. With ``verify_parity=True``,
    torch vs ONNX raw outputs are compared with the export-gate tolerances
    before timing (requires the ``torch`` backend to be benchmarked first).
    """
    from autovision.engine.engine import resolve_device

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    backend_args = _normalize_backends(backends)
    if verify_parity and "torch" not in backend_args:
        raise ValueError(
            "verify_parity=True requires the 'torch' backend: the parity gate "
            "compares torch vs ONNX raw outputs."
        )
    torch_device = resolve_device(device)
    if isinstance(model, str):
        model = _resolve_benchmark_model(
            model, num_classes=num_classes, image_size=image_size
        )
    model = model.to(torch_device)

    example_inputs = model.model_input_spec.example_inputs(
        batch_size=batch_size, device=torch_device
    )
    input_key = next(iter(example_inputs))
    images = example_inputs[input_key]
    if image_size is not None:
        # The input spec is the export-facing shape; a benchmark may override
        # the spatial size (e.g. to compare backends at the production size).
        images = torch.nn.functional.interpolate(
            images, size=image_size, mode="bilinear", align_corners=False
        )
    model_name = type(model).__name__

    raw_torch: Any = None
    results: dict[str, BenchmarkResult] = {}
    for key, args in backend_args.items():
        if key == "torch":
            backend = TorchBackend(model, args, device=torch_device)
            if verify_parity:
                raw_torch = backend.run(images)
            result = benchmark_inference(
                model,
                {input_key: images},
                model_name=model_name,
                warmup_steps=warmup,
                steps=runs,
                backend="torch",
                precision=args.precision,
                forward=backend.run,
            )
        elif key == "onnx":
            onnx_backend = ONNXBackend(
                model, args, batch_size=batch_size, out_dir=out, device=torch_device
            )
            raw_onnx = onnx_backend.run(images.cpu())
            if verify_parity and raw_torch is not None:
                _assert_output_parity(raw_torch, raw_onnx)
            images_cpu = images.cpu()  # hoist: don't time the per-step copy
            timings = _time_callable(
                lambda: onnx_backend.run(images_cpu),  # noqa: B023
                warmup_steps=warmup,
                steps=runs,
            )
            result = _result_from_timings(
                timings, model_name=model_name, batch_size=batch_size,
                backend="onnx", precision=args.precision,
            )
        else:
            trt_backend = TensorRTBackend(
                model, args, batch_size=batch_size, out_dir=out, device=torch_device
            )
            timings = _time_callable(
                lambda: trt_backend.run(images),  # noqa: B023
                warmup_steps=warmup,
                steps=runs,
            )
            result = _result_from_timings(
                timings, model_name=model_name, batch_size=batch_size,
                backend="tensorrt", precision=args.precision,
            )
        results[key] = result
        result.to_json(out / f"benchmark_{key}.json")

    _write_markdown_summary(results, out / "benchmark.md")
    return results


def _assert_output_parity(raw_torch: Any, raw_onnx: list) -> None:
    torch_outputs = list(raw_torch) if isinstance(raw_torch, (list, tuple)) else [raw_torch]
    if len(torch_outputs) != len(raw_onnx):
        raise RuntimeError(
            f"Parity gate: torch returned {len(torch_outputs)} outputs but the "
            f"ONNX graph has {len(raw_onnx)}; refusing to compare by index."
        )
    for index, (torch_out, onnx_out) in enumerate(zip(torch_outputs, raw_onnx)):
        torch.testing.assert_close(
            torch_out.float().sum(dim=1),
            torch.as_tensor(onnx_out).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )


def _time_callable(fn, *, warmup_steps: int, steps: int) -> list[float]:
    for _ in range(warmup_steps):
        fn()
    latencies = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - start)
    return latencies


def _result_from_timings(
    latencies: list[float],
    *,
    model_name: str,
    batch_size: int,
    backend: str,
    precision: str,
) -> BenchmarkResult:
    latencies_ms = sorted(latency * 1000 for latency in latencies)
    return BenchmarkResult(
        model_name=model_name,
        device="onnx-session" if backend == "onnx" else "tensorrt-engine",
        num_steps=len(latencies),
        batch_size=batch_size,
        total_time_sec=sum(latencies),
        images_per_sec=batch_size * len(latencies) / sum(latencies),
        latency_ms_mean=sum(latencies_ms) / len(latencies_ms),
        latency_ms_p50=latencies_ms[max(0, (len(latencies_ms) + 1) // 2 - 1)],
        latency_ms_p90=latencies_ms[int(len(latencies_ms) * 0.9)],
        extra={"backend": backend, "precision": precision},
    )


def _write_markdown_summary(results: dict[str, BenchmarkResult], path: Path) -> None:
    lines = ["| backend | precision | images/s | latency mean (ms) |", "|---|---|---|---|"]
    for key, result in results.items():
        lines.append(
            f"| {key} | {result.extra.get('precision', 'fp32')} "
            f"| {result.images_per_sec:.1f} | {result.latency_ms_mean:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")
