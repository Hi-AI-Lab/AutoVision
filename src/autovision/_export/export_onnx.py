# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""ONNX export pipeline (Path A: torch.export -> dynamo ONNX -> gates).

Pipeline: eval/fp32 -> deploy hooks -> ``torch.export`` (strict, batch-dim
dynamic) -> ``torch.onnx.export(dynamo=True)`` -> optional FP16 conversion
with a per-model fp32 op blacklist -> onnxslim -> value-info repair ->
metadata -> **verification gate** (onnx.checker + onnxruntime parity with
atol 5e-3 / rtol 1e-1 on dim-1 sums).

int8 PTQ (``export_onnx_int8``) reuses this pipeline for the FP32 graph and
post-processes it with ``onnxruntime.quantization.quantize_static`` (QDQ,
calibrated) — see ``autovision._export.quantization``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn

from autovision._version import __version__
from autovision._license import LICENSE_INFO

__all__ = ["ONNXExportMixin", "ONNXExportPrecisionPolicy"]

# Default int8 parity-gate tolerances: int8 activation quantization is
# expected to shift logits; the gate only catches structural breakage.
INT8_PARITY_ATOL = 2.0
INT8_PARITY_RTOL = 0.5

logger = logging.getLogger("autovision")


class ONNXExportPrecisionPolicy:
    """Per-model ops that must stay FP32 in an FP16 export."""

    def __init__(self, fp32_onnx_op_types: tuple[str, ...] = ()) -> None:
        self.fp32_onnx_op_types = tuple(fp32_onnx_op_types)


class ONNXExportMixin(nn.Module):
    """Mixin for :class:`TaskModel` subclasses that can export ONNX."""

    onnx_export_precision_policy = ONNXExportPrecisionPolicy()

    @property
    def model_input_spec(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def verify_onnx_export_outputs(
        self, torch_outputs: Any, onnx_outputs: list
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def onnx_export_metadata(self) -> dict[str, str]:
        metadata = {
            "autovision_version": __version__,
            "license_info": LICENSE_INFO,
        }
        model_name = self.init_args.get("model_name") or self.init_args.get("backbone")
        if model_name:
            metadata["model_name"] = str(model_name)
        normalize = getattr(self, "normalize", None)
        if normalize is not None:
            metadata["image_normalize"] = json.dumps(normalize.to_dict(), sort_keys=True)
        if hasattr(self, "num_classes"):
            metadata["num_classes"] = str(self.num_classes)
        return metadata

    @torch.no_grad()
    def export_onnx(
        self,
        out: Path | str,
        *,
        precision: str = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export the model to ONNX with verification gates."""
        import onnx

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got '{precision}'")
        if precision == "fp16" and not simplify:
            raise ValueError("fp16 export requires simplify=True.")

        # 1. trace in fp32 (fp16 conversion happens on the graph afterwards).
        module: nn.Module = self
        module.eval()
        module.to(torch.float32)
        if hasattr(self, "deploy"):
            self.deploy()

        # 2. torch.export (strict) with batch-dim dynamic shapes.
        from torch.export import Dim

        spec = self.model_input_spec
        batch_dim = Dim("batch_size", min=1, max=128) if dynamic_batch_size else batch_size
        example_name, example_tensor = next(iter(spec.example_inputs(
            batch_size=2 if dynamic_batch_size else batch_size
        ).items()))
        # The TaskModel forward contract is ``forward(self, x: Tensor)``.
        dynamic_shapes = {"x": (batch_dim, None, None, None)}
        program = torch.export.export(
            module, args=(example_tensor,), dynamic_shapes=dynamic_shapes,
            strict=True,
        )

        # 3. dynamo ONNX export (graph inputs get the spec names).
        export_kwargs: dict[str, Any] = {"opset_version": opset_version}
        export_kwargs.update(format_args or {})
        try:
            torch.onnx.export(
                program,
                args=(),
                f=str(out),
                input_names=list(spec.input_specs),
                output_names=self._onnx_output_names(example_tensor),
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                **export_kwargs,
            )
        except Exception as exc:
            # torch 2.4/2.5 dynamo exporter gaps (e.g. unregistered
            # ``_native_batch_norm_legit_no_training`` decompositions, tuple
            # attrs in fp16 paths) break models that the legacy TorchScript
            # exporter handles fine; the legacy path is the supported
            # fallback across the declared torch>=2.4 range.
            logger.warning(
                "dynamo ONNX export failed (%s: %s); retrying with the "
                "legacy TorchScript exporter.", type(exc).__name__, exc,
            )
            legacy_kwargs = {
                key: value
                for key, value in export_kwargs.items()
                if key not in ("dynamo", "dynamic_shapes", "dynamic_axes")
            }
            dynamic_axes = (
                {input_names[0]: {0: "batch_size"}} if dynamic_batch_size else None
            ) if (input_names := list(spec.input_specs)) else None
            torch.onnx.export(
                module,
                (example_tensor,),
                f=str(out),
                input_names=input_names,
                output_names=self._onnx_output_names(example_tensor),
                dynamo=False,
                dynamic_axes=dynamic_axes,
                **legacy_kwargs,
            )

        # 4. FP16 conversion with the op blacklist.
        if precision == "fp16":
            self._apply_fp16(out)

        # 5. onnxslim (constant folding skipped to avoid model inflation).
        if simplify:
            import onnxslim

            onnxslim.slim(str(out), output=str(out), skip_optimizations=["constant_folding"])

        # 6. repair value_info (dynamo exporter can leave wrong rank hints).
        _repair_value_info(out)

        # 7. metadata.
        _write_onnx_metadata(out, self.onnx_export_metadata())

        # 8. verification gate.
        if verify:
            self._verify_onnx(out, example_name, example_tensor, precision)
        logger.info("Exported ONNX model to '%s' (%s).", out, precision)
        return out

    # ── internals ───────────────────────────────────────────────────────
    def _onnx_output_names(self, example_tensor: torch.Tensor) -> list[str]:
        import dataclasses

        sample = self.forward_backend(example_tensor)
        if dataclasses.is_dataclass(sample):
            return [f.name for f in dataclasses.fields(sample)]
        if isinstance(sample, dict):
            return list(sample)
        return [f"output_{i}" for i in range(len(sample) if isinstance(sample, (list, tuple)) else 1)]

    def _apply_fp16(self, out: Path) -> None:
        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(out))
        op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            self.onnx_export_precision_policy.fp32_onnx_op_types
        )
        model = ort_float16.convert_float_to_float16(model, op_block_list=op_block_list)
        _remove_duplicate_casts(model)
        _fix_topological_order(model)
        onnx.save(model, str(out))

    def _verify_onnx(
        self, out: Path, example_name: str, example_tensor: torch.Tensor,
        precision: str,
    ) -> None:
        import copy

        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)
        if precision == "fp16" and "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.warning("FP16 parity check skipped: no CUDA execution provider.")
            return

        reference = copy.deepcopy(self).cpu().float().eval()
        if hasattr(reference, "deploy"):
            reference.deploy()
        torch_outputs = reference.forward_backend(example_tensor.float().cpu())
        if not isinstance(torch_outputs, (list, tuple)):
            import dataclasses
            if dataclasses.is_dataclass(torch_outputs):
                torch_outputs = [getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)]
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        feeds = {}
        for session_input in session.get_inputs():
            tensor = example_tensor.float().cpu()
            if session_input.type == "tensor(float16)":
                tensor = tensor.half()
            feeds[session_input.name] = tensor.numpy()
        onnx_outputs = session.run(None, feeds)
        self.verify_onnx_export_outputs(torch_outputs, onnx_outputs)


    def export_onnx_int8(
        self,
        out: Path | str,
        *,
        calibration_data: Iterable[dict[str, torch.Tensor]] | None = None,
        calibration_reader: Any | None = None,
        quantization_config: Any | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        parity_atol: float = INT8_PARITY_ATOL,
        parity_rtol: float = INT8_PARITY_RTOL,
        keep_fp32_reference: bool = False,
    ) -> Path:
        """Export a QDQ int8 ONNX model via static PTQ (§12.2 extension).

        The FP32 ONNX is produced by the standard dynamo pipeline
        (:meth:`export_onnx`), then post-processed with
        ``onnxruntime.quantization.quantize_static`` using the calibration
        samples. Provide either ``calibration_data`` (an iterable of
        ``{input_name: Tensor}`` dicts) or a prebuilt ORT
        ``calibration_reader`` — the parity gate replays the first
        calibration sample against the FP32 graph.
        """
        from autovision._export.quantization import (
            Int8CalibrationReader,
            quantize_onnx_int8,
        )

        if calibration_data is None and calibration_reader is None:
            raise ValueError(
                "int8 PTQ requires calibration data: pass calibration_data "
                "(iterable of {input_name: Tensor}) or calibration_reader."
            )
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 1. FP32 reference graph (passes the standard FP32 gate itself).
        fp32_path = out.parent / f"{out.stem}.fp32.onnx"
        self.export_onnx(
            fp32_path,
            precision="fp32",
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
        )

        # 2. Static quantization (QDQ) with calibration.
        reader = calibration_reader or Int8CalibrationReader(
            calibration_data, expected_input_names=list(self.model_input_spec.input_specs)
        )
        quantize_onnx_int8(fp32_path, out, reader, quantization_config)
        num_calibration_samples = getattr(reader, "num_samples", None)

        # 3. int8 parity gate: int8-ONNX vs FP32-ONNX on dim-1 sums.
        if verify:
            self._verify_int8_parity(fp32_path, out, reader, parity_atol, parity_rtol)

        # 4. Quantization metadata.
        from autovision._export.quantization import _int8_metadata_payload, Int8QuantizationConfig

        config = quantization_config or Int8QuantizationConfig()
        _write_onnx_metadata(
            out,
            {
                "int8_quantization": _int8_metadata_payload(config, num_calibration_samples),
                "int8_parity_atol": str(parity_atol),
                "int8_parity_rtol": str(parity_rtol),
            },
        )

        # 5. FP32 reference bookkeeping.
        if keep_fp32_reference:
            logger.info("Kept FP32 reference graph at '%s'.", fp32_path)
        else:
            fp32_path.unlink(missing_ok=True)
        logger.info("Exported int8 ONNX model to '%s'.", out)
        return out

    def _verify_int8_parity(
        self,
        fp32_path: Path,
        int8_path: Path,
        reader: Any,
        parity_atol: float,
        parity_rtol: float,
    ) -> None:
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(int8_path), full_check=True)
        # The reader was consumed by quantize_static; rewind it (supported by
        # Int8CalibrationReader) to replay the first calibration sample.
        rewind = getattr(reader, "rewind", None)
        if rewind is not None:
            rewind()
        sample = reader.get_next()
        if sample is None:
            raise ValueError(
                "int8 parity gate has no calibration sample to replay; the "
                "calibration reader must support replaying samples (use "
                "Int8CalibrationReader) or provide at least two passes."
            )
        sessions = {}
        for label, path in (("fp32", fp32_path), ("int8", int8_path)):
            sessions[label] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        fp32_outputs = sessions["fp32"].run(None, sample)
        int8_outputs = sessions["int8"].run(None, sample)
        for index, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            if fp32_out.shape != int8_out.shape:
                raise AssertionError(
                    f"int8 parity gate: output {index} shape mismatch "
                    f"{int8_out.shape} vs FP32 {fp32_out.shape}."
                )
            fp32_sum = torch.as_tensor(fp32_out).float().sum(dim=1)
            int8_sum = torch.as_tensor(int8_out).float().sum(dim=1)
            torch.testing.assert_close(
                int8_sum, fp32_sum,
                atol=parity_atol, rtol=parity_rtol, equal_nan=True,
                check_device=False, check_dtype=False,
            )

    def export_tensorrt(
        self,
        out: Path | str,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "fp32",
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        fp32_attention_scores: bool = False,
        strongly_typed: bool = False,
        onnx_args: dict[str, Any] | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> Path:
        """Build a TensorRT engine from this model (design §12.3).

        Requires the ``tensorrt`` package (not a framework dependency) and a
        CUDA-capable machine for the engine build. The intermediate ONNX is
        produced by :meth:`export_onnx` — see
        :func:`autovision._export.tensorrt_helpers.export_onnx_to_tensorrt`
        for the engine-build semantics.
        """
        from autovision._export.tensorrt_helpers import export_onnx_to_tensorrt

        return export_onnx_to_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=next(self.parameters()).dtype,
            onnx_args=onnx_args,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            fp32_attention_scores=fp32_attention_scores,
            strongly_typed=strongly_typed,
            verbose=verbose,
            debug=debug,
        )


def _repair_value_info(out: Path) -> None:
    import onnx

    model = onnx.load(str(out))
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    onnx.save(model, str(out))


def _write_onnx_metadata(out: Path, metadata: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(out))
    existing = {prop.key: prop.value for prop in model.metadata_props}
    existing.update(metadata)
    del model.metadata_props[:]
    for key, value in existing.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(out))


def _remove_duplicate_casts(model) -> None:
    """Restore SSA form after FP16 conversion: the ORT float16 converter can
    emit several Cast nodes sharing one output name; keep the first."""
    import copy

    graph = model.graph
    seen_outputs: set[str] = set()
    survivors = []
    for node in graph.node:
        if node.output and node.output[0] in seen_outputs:
            continue  # duplicate-name node; its output name stays served
        seen_outputs.update(node.output)
        survivors.append(copy.deepcopy(node))
    if len(survivors) != len(graph.node):
        graph.ClearField("node")
        for node in survivors:
            graph.node.append(node)


def _fix_topological_order(model) -> None:
    """Kahn topological sort of graph nodes (FP16 cast insertion can leave
    nodes out of order, which onnx.checker rejects)."""
    import collections
    import copy

    graph = model.graph
    nodes = list(graph.node)
    produced_by: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            produced_by[output] = index
    dependents: dict[int, list[int]] = collections.defaultdict(list)
    indegree: dict[int, int] = {}
    for index, node in enumerate(nodes):
        deps = {
            produced_by[tensor]
            for tensor in node.input
            if tensor in produced_by and produced_by[tensor] != index
        }
        indegree[index] = len(deps)
        for dep in deps:
            dependents[dep].append(index)

    ready = collections.deque(sorted(i for i, count in indegree.items() if count == 0))
    order: list[int] = []
    while ready:
        index = ready.popleft()
        order.append(index)
        for dependent in sorted(dependents.get(index, ())):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    # Cycles (rare) keep their original relative order at the end.
    order.extend(sorted(set(range(len(nodes))) - set(order)))

    # Deep-copy BEFORE clearing: protobuf list views are invalidated by
    # ClearField.
    ordered_nodes = [copy.deepcopy(nodes[index]) for index in order]
    graph.ClearField("node")
    for node in ordered_nodes:
        graph.node.append(node)
