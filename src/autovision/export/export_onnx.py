# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ONNX export pipeline (Path A: torch.export -> dynamo ONNX -> gates).

Pipeline: eval/fp32 -> deploy hooks -> ``torch.export`` (strict, batch-dim
dynamic) -> ``torch.onnx.export(dynamo=True)`` -> optional FP16 conversion
with a per-model fp32 op blacklist -> onnxslim -> value-info repair ->
metadata -> **verification gate** (onnx.checker + onnxruntime parity with
atol 5e-3 / rtol 1e-1 on dim-1 sums).

int8 PTQ (``export_onnx_int8``) reuses this pipeline for the FP32 graph and
post-processes it with ``onnxruntime.quantization.quantize_static`` (QDQ,
calibrated) — see ``autovision.export.quantization``.
"""

from __future__ import annotations

import glob
import inspect
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
from torch import nn

from autovision.errors import AutoVisionError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = ["ONNXExportMixin", "ONNXExportPrecisionPolicy"]

# Default int8 parity-gate tolerances: int8 activation quantization is
# expected to shift logits; the gate only catches structural breakage.
INT8_PARITY_ATOL = 2.0
INT8_PARITY_RTOL = 0.5

#: Exported graphs below this size are treated as truncated (design D14b):
#: a header-only "success" fails the export instead of shipping a graph
#: with no weights.
# EXPORT audit P3-12: the dynamo batch Dim's documented ceiling; ORT
# rejects a larger batch at load time with a shape error.
_MAX_DYNAMIC_BATCH = 128
# EXPORT-10: external data beyond this size cannot be inlined by the
# later repair/metadata steps (protobuf limit) — fail fast instead.
_MAX_INLINE_EXTERNAL_DATA = 2 * (1 << 30)

_MIN_ARTIFACT_BYTES = 4096

logger = logging.getLogger("autovision")


def _spec_dynamic_axes(spec: Any, input_name: str) -> dict[int, Any]:
    """Explicit axis markers from ``ModelInputSpec.input_dynamic_shapes``.

    Returns ``{axis_index: marker}`` for the input's declared axes in rank
    order (None entries are static). The legacy exporter only uses the axis
    indices; the dynamo path converts markers via :func:`_dim_from_marker`.
    """
    declared = getattr(spec, "input_dynamic_shapes", None) or {}
    axes = declared.get(input_name)
    if not axes:
        return {}
    return {
        index: marker for index, marker in enumerate(axes) if marker is not None
    }


def _dim_from_marker(
    marker: Any, axis: int, cache: dict[str, Any] | None = None
) -> Any:
    """``torch.export.Dim`` for one dynamic-axis marker.

    Accepted forms: a ready ``Dim`` (used as-is), an axis NAME (str →
    ``Dim(name, min=1)``), or a ``(min, max)`` tuple. NOTE: the name form
    gets a ``min=1`` bound, which a model with internal size guards (e.g.
    feature pyramids asserting ``size // divisor >= 1``) will REJECT with
    ``Constraints violated``; use the tuple form (or the
    ``dynamic_spatial`` flag, whose bounds derive from the example) there.

    ``cache`` (H3-4e) is a per-export dict: two markers with the same NAME
    must resolve to the SAME ``Dim`` object — torch.export treats a fresh
    ``Dim`` with a reused name as a second unrelated axis (equal-sizes
    constraint lost, or an outright export error). Scope is one export
    call; a module-level cache would leak bounds across exports.
    """
    from torch.export import Dim

    if isinstance(marker, Dim):
        return marker
    if isinstance(marker, str):
        if cache is not None and marker in cache:
            return cache[marker]
        dim = Dim(marker, min=1)
        if cache is not None:
            cache[marker] = dim
        return dim
    if isinstance(marker, (tuple, list)) and len(marker) == 2:
        low, high = marker
        return Dim(f"dynamic_axis{axis}", min=int(low), max=int(high))
    raise ValueError(
        f"input_dynamic_shapes marker {marker!r} on axis {axis} is not a "
        "Dim / axis name / (min, max) tuple."
    )


def _drop_onnx_artifacts(*paths: Path) -> None:
    """Remove ONNX files with their external-data sidecars (dynamo writes
    ``<name>.onnx.data`` for large graphs — 100MB-class litter otherwise)."""
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".data").unlink(missing_ok=True)


def _move_onnx_artifacts(source: Path, target: Path) -> None:
    """Move an ONNX graph, its external-data sidecar, and the decode
    contract sidecar in one rename sequence (EXPORT-09).

    The spec (when present at the staged path) moves FIRST — after the
    graph rename a reader sees either old-graph+old-spec or
    new-graph+new-spec; publishing the graph before the spec exposed a
    window where the new graph paired with the PREVIOUS run's decode
    contract.
    """
    source_spec = Path(str(source) + ".postprocess.json")
    target_spec = Path(str(target) + ".postprocess.json")
    spec_backup: bytes | None = None
    if source_spec.is_file():
        # EXPORT-09 (review P2-4): publish the spec first, keep the OLD
        # spec's bytes, and restore them if the graph rename fails — a
        # failed export must not leave the new decode contract paired with
        # the OLD graph.
        spec_backup = target_spec.read_bytes() if target_spec.is_file() else None
        os.replace(source_spec, target_spec)
    try:
        os.replace(source, target)
    except BaseException:
        if spec_backup is not None:
            target_spec.write_bytes(spec_backup)
        elif source_spec.is_file():
            # The old graph had NO spec: remove the newly published one.
            target_spec.unlink(missing_ok=True)
        raise
    source_data = source.with_suffix(source.suffix + ".data")
    if source_data.exists():
        os.replace(source_data, target.with_suffix(target.suffix + ".data"))


def _sweep_orphan_external_data(out: Path) -> None:
    """Delete the dynamo exporter's ``<out>.data`` sidecar when the final
    graph no longer references external data.

    torch's dynamo exporter externalizes all initializers, then onnxslim
    inlines them back into the graph; the success path
    (:func:`_move_onnx_artifacts`) still moves the sidecar next to the final
    artifact, where it is dead weight (tens of MB per artifact) that a
    deployer packaging ``model.onnx*`` would ship by accident. Graphs that
    genuinely keep external initializers (e.g. beyond the ~2GB protobuf
    limit) keep their sidecar.
    """
    data = out.with_suffix(out.suffix + ".data")
    if not data.exists():
        return
    import onnx
    from onnx import TensorProto

    # load_external_data=False keeps the graph's external references intact
    # (the default loader materializes them and clears the data_location
    # marker) and skips reading gigabytes of weights just to check flags.
    model = onnx.load(str(out), load_external_data=False)
    external = any(
        tensor.data_location == TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    )
    if not external:
        data.unlink(missing_ok=True)


def _dynamic_export_error(exc: BaseException) -> str:
    """Actionable message for a failed dynamic-spatial dynamo export.

    ``torch.export``/``torch.onnx.export(dynamo=True)`` raise deep dynamo
    internals; the two failure classes users hit are a missing/misbehaving
    dynamo toolchain and a model that branches in PYTHON on the input size.
    """
    hint = ""
    if "Constraints violated" in str(exc):
        # A trace that generated an input-size guard cannot hold across a
        # whole dynamic range: such backbones branch or guard in PYTHON on
        # the input size — the transformers ViT embeddings switch between
        # "native size" and "interpolate" on the patch count, and the HF
        # ConvNeXt stage grid guards its divisor relations.
        hint = (
            "Cause: this backbone branches/guards in PYTHON on the INPUT "
            "SIZE (e.g. the transformers ViT patch-count branch or HF "
            "stage-grid guards), which strict export cannot make symbolic — "
            "dynamic spatial export is not available for it. "
        )
    return (
        hint
        + "Dynamic spatial axes require the dynamo export path "
        "(torch.export + torch.onnx.export(dynamo=True)); it failed with "
        f"{type(exc).__name__}: {exc}. Export with dynamic_spatial=False "
        "(static H/W), use a backbone whose graph traces symbolically "
        "(e.g. rtdetr/resnet18, the HGNetV2 tiers), or verify the "
        "'onnxscript' package is installed (part of the 'export' extra)."
    )


def _forward_with_export_hooks(network: Any, example: Any) -> Any:
    """``network.forward_backend`` with the export-safe module swaps active.

    The GTR routes' eager forward runs fla's CUDA-only Triton GLA kernels,
    so a parity reference that must run on CPU (the OpenVINO gate deep-copies
    to CPU) replays the trace-time swapped path instead — numerically the
    exported graph's own semantics. Networks without the duck-typed hook
    (and hook-less modules) forward unchanged.
    """
    prepare = getattr(network, "_prepare_onnx_export_modules", None)
    if not callable(prepare):
        return network.forward_backend(example)
    prepared = prepare()
    try:
        return network.forward_backend(example)
    finally:
        network._restore_onnx_export_modules(prepared)


def _eager_reference_device_bound(network: Any) -> bool:
    """True when the eager forward cannot run where the model currently
    sits: a CPU-resident network with a ``cpu_eager_blocker`` module (the
    duck-typed GTR GLA flag)."""
    parameter = next(network.parameters(), None)
    if parameter is None or parameter.device.type != "cpu":
        return False
    return any(
        getattr(module, "cpu_eager_blocker", None) for module in network.modules()
    )


def _parity_session_providers(available: Sequence[str]) -> list[str]:
    """Execution providers for the parity-gate session (H1-8).

    The fp16 gate's warning promises "re-verify on a CUDA-enabled
    onnxruntime build for exact parity": when such a build is installed, the
    session actually routes through the CUDA execution provider so the fp16
    graph is gated on the device numerics it targets. The CUDA EP is listed
    first so onnxruntime cannot silently fall back to the CPU EP.
    """
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _warn_tensorrt_version_advisory(network: Any) -> None:
    """Warn when a module declares a TensorRT-version advisory and the
    installed TensorRT falls in the known-bad range (the GTR pure-ONNX
    graph miscompiles on TensorRT 11.x — docs/known_limits.md).

    Soft on purpose: ``verify=True`` still fail-closes a wrong-numerics
    engine after the build; this warning exists so ``verify=False`` builds
    are not silent and the user knows before spending the build time.
    """
    advisory = next(
        (
            message
            for message in (
                getattr(module, "trt_version_advisory", None)
                for module in network.modules()
            )
            if message
        ),
        None,
    )
    if advisory is None:
        return
    try:
        import tensorrt
    except ImportError:
        # Nothing to advise on yet; the engine build raises its own
        # actionable error about the missing package.
        return
    try:
        major = int(tensorrt.__version__.split(".")[0])
    except (AttributeError, ValueError):
        return
    if major < 11:
        return
    logger.warning(
        "TensorRT %s: %s Keep verify=True (the engine parity gate deletes "
        "an engine whose numerics drift); deploy on TensorRT 10.x or via "
        "the upstream GLA plugin route (docs/export.md, "
        "docs/known_limits.md).",
        tensorrt.__version__,
        advisory,
    )


#: Hooks already told (H3-4c) that their gate runs at fp32 tolerances.
_FP32_TOLERANCE_WARNED: set[str] = set()


def _hook_accepts_precision(hook: Any) -> bool:
    """Whether a verify hook consumes the ``precision`` keyword (H3-4c).

    Only a named parameter or ``**kwargs`` counts: the task-family hooks
    declare the parameter but half of them ignore it (fp32-tolerance
    bodies) — a signature alone cannot see that, so the central fallback
    can only promise correctness for hooks WITHOUT the parameter.
    """
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        return True  # not introspectable: assume compatible, stay silent
    if "precision" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _call_verify_hook(
    network: Any, torch_outputs: Any, onnx_outputs: list, *, precision: str
) -> None:
    """Invoke the task's verify hook with its precision contract (H3-4c).

    Hooks that accept ``precision`` get it; hooks that do not are called
    without it (legacy plugins keep working) and warned about ONCE: their
    gate then runs at the fp32 tolerance band even for an fp16 export,
    which is the loose-tolerance direction users must know about.
    """
    hook = network.verify_onnx_export_outputs
    if precision == "fp16" and not _hook_accepts_precision(hook):
        marker = f"{type(network).__module__}.{type(network).__qualname__}"
        if marker not in _FP32_TOLERANCE_WARNED:
            _FP32_TOLERANCE_WARNED.add(marker)
            logger.warning(
                "Verify hook %s.verify_onnx_export_outputs does not accept "
                "a 'precision' argument; the fp16 export is gated at the "
                "fp32 tolerance band (atol 5e-3 / rtol 1e-1) and may be "
                "too tight for real fp16 numerics. Update the hook to "
                "accept precision and branch its tolerances.",
                marker,
            )
        hook(torch_outputs, onnx_outputs)
        return
    hook(torch_outputs, onnx_outputs, precision=precision)


class ONNXExportPrecisionPolicy:
    """Per-model ops that must stay FP32 in an FP16 export."""

    def __init__(self, fp32_onnx_op_types: tuple[str, ...] = ()) -> None:
        self.fp32_onnx_op_types = tuple(fp32_onnx_op_types)


class ONNXExportMixin(nn.Module):
    """Mixin for :class:`TaskNetwork` subclasses that can export ONNX."""

    onnx_export_precision_policy = ONNXExportPrecisionPolicy()

    #: Non-``None`` blocks export with this message. Submodules declare it
    #: when their forward contains ops the pipeline cannot lower (e.g. the
    #: GTR GLA custom CUDA kernels) — the guard scans the whole module tree
    #: so any composition containing such a submodule is covered.
    export_blocker: str | None = None

    @property
    def model_input_spec(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def verify_onnx_export_outputs(
        self, torch_outputs: Any, onnx_outputs: list, *, precision: str = "fp32"
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _export_blocker(self) -> str | None:
        """First non-``None`` ``export_blocker`` in the module tree."""
        if self.export_blocker is not None:
            return self.export_blocker
        for module in self.modules():
            blocker = getattr(module, "export_blocker", None)
            if blocker is not None:
                return blocker
        return None

    def _prepare_onnx_export_modules(self) -> list[nn.Module]:
        """Duck-typed module-tree hook (same idiom as ``_export_blocker``):
        call ``prepare_onnx_export()`` on every submodule that declares it.

        Submodules whose eager forward runs ops the tracer cannot lower (the
        GTR GLA Triton/CUDA kernels, the encoder's CUDA-stream fork/join,
        einsum in the segmentation head) swap in an ONNX-lowerable path for
        the duration of one trace and undo it in
        :meth:`_restore_onnx_export_modules`. Called AFTER ``deploy()`` so a
        deploy-time ``convert_to_deploy`` cannot clobber the swapped state.
        """
        prepared: list[nn.Module] = []
        for module in self.modules():
            prepare = getattr(module, "prepare_onnx_export", None)
            if callable(prepare):
                prepare()
                prepared.append(module)
        return prepared

    @staticmethod
    def _restore_onnx_export_modules(prepared: list[nn.Module]) -> None:
        """Undo :meth:`_prepare_onnx_export_modules` (never raises past the
        real export error: each hook is a plain state flip)."""
        for module in prepared:
            module.restore_after_export()

    def onnx_export_metadata(self) -> dict[str, str]:
        """Artifact identity (design D1): the schema lives in
        ``artifact_metadata.build_metadata`` so the ONNX in-graph metadata
        and the OpenVINO/TensorRT sidecars carry exactly the same keys."""
        from autovision.export.artifact_metadata import build_metadata

        return build_metadata(self)

    @torch.no_grad()
    def export_onnx(
        self,
        out: Path | str,
        *,
        precision: str = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        dynamic_spatial: bool = False,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        write_spec: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export to a temp graph, verify, then atomically replace ``out``.

        A failed export never leaves a partial graph at the target path (the
        previous good artifact survives), and the postprocess sidecar is only
        written after the graph has passed its gates.
        """
        final_out = Path(out)
        final_out.parent.mkdir(parents=True, exist_ok=True)
        # Hygiene: sweep temp graphs orphaned by a killed exporter. The
        # pattern ends in ".tmp<suffix>" so a kept FP32 reference
        # ("{stem}.fp32.onnx") can never match.
        sweep_stale_temp_files(
            final_out.parent,
            f"{glob.escape(final_out.stem)}"
            f".*.tmp{glob.escape(final_out.suffix or '.onnx')}",
        )
        # No leading dot: ONNX external-data locations starting with '..'
        # are rejected by onnx.load as path traversal.
        tmp_out = final_out.with_name(
            f"{final_out.stem}.{uuid.uuid4().hex}.tmp{final_out.suffix or '.onnx'}"
        )
        try:
            self._export_onnx_to_path(
                tmp_out,
                precision=precision,
                batch_size=batch_size,
                dynamic_batch_size=dynamic_batch_size,
                dynamic_spatial=dynamic_spatial,
                opset_version=opset_version,
                simplify=simplify,
                verify=verify,
                # EXPORT-09: the spec stages beside the graph so
                # _move_onnx_artifacts publishes the pair in one window.
                # P1-2: forward the CALLER's write_spec — the hardcoded True
                # ignored the TRT/OV/int8 wrappers' write_spec=False and
                # littered intermediate artifacts with decode contracts.
                write_spec=write_spec,
                format_args=format_args,
            )
            _move_onnx_artifacts(tmp_out, final_out)
        except Exception:
            Path(str(tmp_out) + ".postprocess.json").unlink(missing_ok=True)
            _drop_onnx_artifacts(tmp_out)
            raise
        return final_out

    @torch.no_grad()
    def _export_onnx_to_path(
        self,
        out: Path | str,
        *,
        precision: str = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        dynamic_spatial: bool = False,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        write_spec: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export the model to ONNX with verification gates.

        ``write_spec=False`` is used by the TensorRT/OpenVINO wrappers for
        their intermediate graph: only the final artifact gets a
        ``.postprocess.json`` sidecar.

        ``dynamic_spatial=True`` marks the H/W axes dynamic (roadmap D2):
        the graph then accepts any input size the network itself supports
        (e.g. multiples of the variant's patch/stride divisor — the caller
        resizes). ``model_input_spec.input_dynamic_shapes`` markers are
        honored on top: a non-``None`` entry (a ``torch.export.Dim``, an
        axis name, or a ``(min, max)`` tuple) marks that axis dynamic. The
        parity gate then verifies a SECOND spatial size (torch vs ONNX
        Runtime) so a graph that merely traces cannot pass.
        """

        blocker = self._export_blocker()
        if blocker is not None:
            raise AutoVisionError(blocker)
        if dynamic_spatial:
            # Duck-typed module-tree scan (same idiom as ``export_blocker``):
            # models whose graphs are inherently fixed-shape (the GTR GLA
            # recurrence unrolls for the variant's token count) refuse here
            # instead of grinding through symbolic-shape tracing.
            for submodule in self.modules():
                blocker = getattr(submodule, "dynamic_spatial_blocker", None)
                if blocker:
                    raise AutoVisionError(blocker)

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
        # The export-prep hooks must come after deploy(): a convert_to_deploy
        # may re-enable eager-only fast paths (e.g. the encoder's parallel
        # stages) that the hook then turns back off for the trace.
        prepared = self._prepare_onnx_export_modules()
        try:
            return self._trace_and_write_onnx(
                out, module,
                precision=precision,
                batch_size=batch_size,
                dynamic_batch_size=dynamic_batch_size,
                dynamic_spatial=dynamic_spatial,
                opset_version=opset_version,
                simplify=simplify,
                verify=verify,
                write_spec=write_spec,
                format_args=format_args,
            )
        finally:
            self._restore_onnx_export_modules(prepared)

    def _trace_and_write_onnx(
        self,
        out: Path,
        module: nn.Module,
        *,
        precision: str,
        batch_size: int,
        dynamic_batch_size: bool,
        dynamic_spatial: bool,
        opset_version: int | None,
        simplify: bool,
        verify: bool,
        write_spec: bool,
        format_args: dict[str, Any] | None,
    ) -> Path:
        """Trace ``module`` (already deployed, export hooks prepared) and write
        the gated ONNX artifact to ``out`` — steps 2-8 of the export pipeline."""

        # 2. torch.export (strict) with batch-dim dynamic shapes.
        from torch.export import Dim

        spec = self.model_input_spec
        # EXPORT audit P3-12: the dynamic batch ceiling is a documented
        # export contract — ORT rejects batch > 128 at load time with a
        # shape error; keep the limit visible at the definition.
        batch_dim = Dim("batch_size", min=1, max=_MAX_DYNAMIC_BATCH) if dynamic_batch_size else batch_size
        # Trace on the model's own device: a CUDA-resident model (the normal
        # case right after training) would otherwise hit a device mismatch
        # against a CPU example input inside the strict export.
        parameter = next(module.parameters(), None)
        example_device = (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        example_name, example_tensor = next(iter(spec.example_inputs(
            batch_size=2 if dynamic_batch_size else batch_size, device=example_device
        ).items()))
        # Warm one eager forward BEFORE tracing: some backbones run one-time
        # setup on their FIRST eager call that dynamo cannot trace — the
        # transformers ViT family installs its output-capturing hooks under a
        # ``threading.Lock``, and dynamo cannot trace ``with <lock>``, so the
        # strict trace fails outright for every HF-backed variant. After the
        # warm call the hooks are already installed and the trace takes the
        # fast path. (Numerically transparent: the hooks only record.)
        module(example_tensor)
        # The TaskNetwork forward contract is ``forward(self, x: Tensor)``.
        # dim_cache (H3-4e): same-name axis markers must share one Dim.
        dim_cache: dict[str, Any] = {}
        dynamic_shapes = {"x": self._onnx_dynamic_shapes(
            spec, example_name, batch_dim, dynamic_batch_size, dynamic_spatial,
            example_tensor=example_tensor, dim_cache=dim_cache,
        )}
        dynamic_axes_requested = dynamic_spatial or any(
            axis != 0 for axis in _spec_dynamic_axes(spec, example_name)
        )
        program = None
        trace_error: Exception | None = None
        try:
            program = torch.export.export(
                module, args=(example_tensor,), dynamic_shapes=dynamic_shapes,
                strict=True,
            )
        except Exception as exc:  # noqa: BLE001 - routed below
            trace_error = exc
        if program is None and dynamic_axes_requested:
            # The legacy TorchScript exporter bakes the traced spatial size
            # into the graph: it would produce a "dynamic" graph whose sizes
            # silently break elsewhere, so dynamic EXPORT requires the dynamo
            # path outright (no legacy fallback).
            assert trace_error is not None
            raise AutoVisionError(_dynamic_export_error(trace_error)) from trace_error
        if program is None:
            # A strict-trace failure with STATIC axes is not fatal: the legacy
            # TorchScript exporter handles some graphs dynamo cannot (the
            # documented supported fallback across the torch>=2.4 range).
            logger.warning(
                "torch.export strict trace failed (%s: %s); retrying with "
                "the legacy TorchScript exporter.",
                type(trace_error).__name__, trace_error,
            )
            self._legacy_onnx_export(
                module, example_tensor, out, spec, example_name, format_args,
                opset_version, dynamic_batch_size,
            )
        else:
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
                if dynamic_axes_requested:
                    raise AutoVisionError(_dynamic_export_error(exc)) from exc
                # torch 2.4/2.5 dynamo exporter gaps (e.g. unregistered
                # ``_native_batch_norm_legit_no_training`` decompositions,
                # tuple attrs in fp16 paths) break models that the legacy
                # TorchScript exporter handles fine; the legacy path is the
                # supported fallback across the declared torch>=2.4 range.
                logger.warning(
                    "dynamo ONNX export failed (%s: %s); retrying with the "
                    "legacy TorchScript exporter.", type(exc).__name__, exc,
                )
                self._legacy_onnx_export(
                    module, example_tensor, out, spec, example_name, format_args,
                    opset_version, dynamic_batch_size,
                )

        # 4. FP16 conversion with the op blacklist.
        if precision == "fp16":
            self._apply_fp16(out)

        # 5. onnxslim (constant folding skipped to avoid model inflation).
        if simplify:
            import onnxslim

            onnxslim.slim(str(out), output=str(out), skip_optimizations=["constant_folding"])

        # EXPORT-10: the dynamo exporter externalizes ALL initializers at
        # trace time (normal); onnxslim inlines them back — UNLESS the graph
        # is beyond the ~2GB protobuf limit, where the value_info repair and
        # metadata write (both `onnx.load` without the flag) would die
        # opaquely deep inside the pipeline. Gate on the actual .data size
        # AFTER onnxslim: a >2GB sidecar cannot be inlined by any later step.
        data_sidecar = out.with_suffix(out.suffix + ".data")
        if data_sidecar.is_file() and data_sidecar.stat().st_size > _MAX_INLINE_EXTERNAL_DATA:
            raise AutoVisionError(
                f"'{out.name}' carries {data_sidecar.stat().st_size / 1e9:.1f}GB "
                "of external initializers (beyond the ~2GB protobuf limit): "
                "the value-info repair and metadata write would inline the "
                "whole model and fail opaquely. Export in-graph (smaller "
                "model / lower precision) or split the graph."
            )

        # 6. repair value_info (dynamo exporter can leave wrong rank hints).
        _repair_value_info(out)

        # 7. verification gate FIRST (EXPORT-02): the verify strength
        # ("full" / fp16-no-CUDA "smoke-only" / "unverified") is part of the
        # artifact's identity and must land in the metadata — with the gate
        # after the metadata write, a smoke-only graph could never be
        # distinguished from a fully-parity-verified one by anything reading
        # the artifact. Review P2-1: verify=False (throwaway debug builds,
        # per the docstring) must NOT claim "full" — VerifyStrengthGate
        # would wave it through.
        verify_strength = "unverified"
        if verify:
            verify_strength = self._verify_onnx(
                out, example_name, example_tensor, precision,
                dynamic_spatial=dynamic_spatial,
            )

        # 8. metadata (the toolchain record carries the graph's REAL opset:
        # the export argument may be None and the exporter's default won).
        metadata = self.onnx_export_metadata()
        effective_opset = _graph_opset_version(out)
        if effective_opset is not None:
            from autovision.export.artifact_metadata import record_onnx_opset

            record_onnx_opset(metadata, effective_opset)
        metadata["autovision_verify_strength"] = verify_strength
        _write_onnx_metadata(out, metadata)
        logger.info(
            "Artifact identity: task=%s, num_classes=%s, image_size=%s, "
            "verify=%s.",
            metadata.get("autovision_task"),
            metadata.get("num_classes"),
            metadata.get("autovision_image_size"),
            verify_strength,
        )
        # Gross-truncation guard (parity owns numerics; this owns the
        # "wrote only a header" accident). A real exported graph with its
        # weights is never this small.
        if out.stat().st_size < _MIN_ARTIFACT_BYTES:
            raise AutoVisionError(
                f"Exported ONNX artifact '{out.name}' is only "
                f"{out.stat().st_size} bytes (< {_MIN_ARTIFACT_BYTES}); the "
                "graph looks truncated. Re-run the export."
            )
        # onnxslim re-inlined the dynamo exporter's external data: the sidecar
        # is litter unless the graph genuinely keeps external initializers.
        _sweep_orphan_external_data(out)
        logger.info("Exported ONNX model to '%s' (%s).", out, precision)
        if write_spec:
            # EXPORT-09: publish the decode contract BEFORE the graph is
            # renamed into place — a concurrent reader then either sees
            # old-graph+old-spec or new-graph+new-spec. The old order (graph
            # first) exposed a window where the new graph paired with the
            # PREVIOUS run's decode contract.
            _maybe_write_postprocess_spec(self, out)
        return out

    # ── internals ───────────────────────────────────────────────────────
    def _legacy_onnx_export(
        self,
        module: nn.Module,
        example_tensor: torch.Tensor,
        out: Path,
        spec: Any,
        example_name: str,
        format_args: dict[str, Any] | None,
        opset_version: int | None,
        dynamic_batch_size: bool,
    ) -> None:
        """Legacy TorchScript export (the documented dynamo fallback).

        Only reached with STATIC spatial axes (the caller rejects it when
        dynamic axes were requested): TorchScript traces the example's sizes
        into the graph, so a "dynamic" legacy graph would be a lie.
        """
        export_kwargs: dict[str, Any] = {"opset_version": opset_version}
        export_kwargs.update(format_args or {})
        legacy_kwargs = {
            key: value
            for key, value in export_kwargs.items()
            if key not in ("dynamo", "dynamic_shapes", "dynamic_axes")
        }
        input_names = list(spec.input_specs)
        axes: dict[int, str] = {}
        # Mirror ``_onnx_dynamic_shapes``: the batch axis is governed by the
        # FLAG only (a marker on axis 0 is warned about and dropped there), and
        # spatial axes never reach this path (the caller rejects legacy export
        # when any dynamic axis was requested — the TorchScript graph would
        # bake the traced size in).
        if dynamic_batch_size:
            axes[0] = "batch_size"
        for axis, marker in _spec_dynamic_axes(spec, example_name).items():
            if axis == 0:
                continue
            axes[axis] = marker if isinstance(marker, str) else f"dynamic_axis{axis}"
        dynamic_axes = {input_names[0]: axes} if (axes and input_names) else None
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

    def _onnx_dynamic_shapes(
        self,
        spec: Any,
        example_name: str,
        batch_dim: Any,
        dynamic_batch_size: bool,
        dynamic_spatial: bool,
        *,
        example_tensor: torch.Tensor | None = None,
        dim_cache: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Per-axis dynamic markers for the image input.

        Batch follows ``dynamic_batch_size``; H/W are dynamic when
        ``dynamic_spatial`` is set; ``spec.input_dynamic_shapes`` entries
        mark additional/other axes explicitly. The H/W bounds are derived
        from the EXAMPLE size (``min = min(64, example)``): a fixed lower
        bound can undercut a model's own internal guard (e.g. feature
        pyramids assert ``size // divisor >= 1``), and torch.export then
        refuses to export because the Dim range admits violating values.
        """
        from torch.export import Dim

        rank = 4  # the TaskNetwork image contract is NCHW
        axes: list[Any] = [None] * rank
        axes[0] = batch_dim if dynamic_batch_size else None
        if dynamic_spatial:
            examples = (
                tuple(int(v) for v in example_tensor.shape[2:4])
                if example_tensor is not None else (640, 640)
            )
            axes[2] = Dim("height", min=min(64, examples[0]), max=8192)
            axes[3] = Dim("width", min=min(64, examples[1]), max=8192)
        for axis, marker in _spec_dynamic_axes(spec, example_name).items():
            if axis >= rank:
                raise ValueError(
                    f"input_dynamic_shapes for '{example_name}' marks axis "
                    f"{axis}, but the image input has rank {rank}."
                )
            if axis == 0:
                if not dynamic_batch_size:
                    logger.warning(
                        "input_dynamic_shapes declares a dynamic batch axis, "
                        "but dynamic_batch_size=False; the batch axis stays "
                        "STATIC."
                    )
                continue  # batch is governed by dynamic_batch_size
            axes[axis] = _dim_from_marker(marker, axis, cache=dim_cache)
        return tuple(axes)

    def _onnx_output_names(self, example_tensor: torch.Tensor) -> list[str]:
        import dataclasses

        sample = self.forward_backend(example_tensor)
        if dataclasses.is_dataclass(sample):
            return [f.name for f in dataclasses.fields(sample)]
        if isinstance(sample, dict):
            return list(sample)
        return [f"output_{i}" for i in range(len(sample) if isinstance(sample, (list, tuple)) else 1)]

    def _fp16_op_block_list(self) -> list[str]:
        """Ops the fp16 conversion must keep in fp32: ORT's default block list,
        the model's precision policy, and any submodule-declared pins.

        Submodules may declare ``onnx_fp32_op_types`` (a tuple of op-type
        strings; the same duck-typed module-tree scan as ``export_blocker``)
        when their internals must stay fp32 in an otherwise-fp16 graph —
        e.g. the GTR GLA keeps its unnormalized fp32 state, and the
        converter's dtype propagation needs MatMul/Resize pinned to stay
        consistent across that boundary.
        """
        from onnxruntime.transformers import float16 as ort_float16

        block = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            self.onnx_export_precision_policy.fp32_onnx_op_types
        )
        for module in self.modules():
            extra = getattr(module, "onnx_fp32_op_types", None)
            if extra:
                block.extend(extra)
        return block

    def _apply_fp16(self, out: Path) -> None:
        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(out))
        model = ort_float16.convert_float_to_float16(
            model, op_block_list=self._fp16_op_block_list()
        )
        _remove_duplicate_casts(model)
        _fix_topological_order(model)
        onnx.save(model, str(out))

    def _verify_onnx(
        self, out: Path, example_name: str, example_tensor: torch.Tensor,
        precision: str,
        dynamic_spatial: bool = False,
    ) -> str:
        """Run the parity gate; return the machine-readable verify strength.

        EXPORT-02: ``"full"`` (element-wise parity ran) vs
        ``"smoke-only"`` (fp16 without a CUDA execution provider — the
        execute-and-finiteness gate). The strength lands in the artifact
        metadata so downstream consumers can tell whether verify=True ever
        compared numbers on this machine instead of guessing from a
        warning lost in the build log.
        """
        import copy

        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)
        available_providers = ort.get_available_providers()
        if precision == "fp16" and "CUDAExecutionProvider" not in available_providers:
            # ORT's CPU EP runs fp16 graphs through internal fp32 casts, but
            # the per-model verify hooks are tuned for the CUDA-EP numerics
            # these graphs target — element-wise parity would false-positive.
            # Run the runtime smoke gate instead of shipping an export whose
            # verify=True checked nothing numerical; exact parity still needs
            # a CUDA-enabled onnxruntime build.
            logger.warning(
                "FP16 export: no CUDA execution provider (installed "
                "providers: %s) — exact numerical parity unavailable, "
                "running the execute-and-finiteness smoke gate instead. "
                "Re-verify on a CUDA-enabled onnxruntime build (e.g. pip "
                "install onnxruntime-gpu) for exact parity.",
                ", ".join(available_providers),
            )
            self._smoke_verify_fp16(out, example_tensor)
            return "smoke-only"

        providers = _parity_session_providers(available_providers)
        if providers[0] == "CUDAExecutionProvider":
            logger.info(
                "ONNX parity gate: CUDA execution provider active "
                "(available: %s; onnxruntime %s) — the graph is gated on "
                "device numerics.",
                ", ".join(available_providers),
                ort.__version__,
            )

        reference = copy.deepcopy(self).cpu().float().eval()
        if hasattr(reference, "deploy"):
            reference.deploy()
        # The reference must replay the EXPORTED graph's semantics, not the
        # eager CUDA-only kernels: run it through the same module swaps the
        # trace used. Hooking it explicitly (not by deepcopy-inheritance
        # from the trace window) keeps the gate correct wherever it runs,
        # including a post-export re-verify on a hooks-restored model.
        prepare = getattr(reference, "_prepare_onnx_export_modules", None)
        prepared = prepare() if callable(prepare) else []
        try:
            session = ort.InferenceSession(str(out), providers=providers)

            def _compare(tensor: torch.Tensor) -> None:
                torch_outputs = reference.forward_backend(tensor.float().cpu())
                if not isinstance(torch_outputs, (list, tuple)):
                    import dataclasses
                    if dataclasses.is_dataclass(torch_outputs):
                        torch_outputs = [getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)]
                session_inputs = session.get_inputs()
                if len(session_inputs) > 1:
                    # H3-4f: feeding every input the SAME tensor "verifies"
                    # nothing for the second+ inputs (their real distributions
                    # differ); the TaskNetwork export contract is a single
                    # image tensor, so this is unreachable for supported
                    # exports — refuse loudly instead of pretending.
                    raise AutoVisionError(
                        "暂不支持多输入对拍: the parity gate feeds one "
                        f"tensor, but the graph declares {len(session_inputs)} "
                        "inputs ("
                        + ", ".join(item.name for item in session_inputs)
                        + "). Export with verify=False or extend the gate "
                        "for this multi-input graph."
                    )
                feeds = {}
                for session_input in session_inputs:
                    feed = tensor.float().cpu()
                    if session_input.type == "tensor(float16)":
                        feed = feed.half()
                    feeds[session_input.name] = feed.numpy()
                onnx_outputs = session.run(None, feeds)
                _call_verify_hook(self, torch_outputs, onnx_outputs, precision=precision)

            # The comparison input is SEEDED RANDOM, not the spec's all-zero
            # example: on a zero image a DETR's query selection has tie-broken
            # grid coordinates (many equal scores), so element-wise near-ties
            # between torch and the backend flip top-k boxes and the gate turns
            # flaky on a perfectly valid graph (largest observed: box sums
            # differing by ~0.5 on a 640px two-class model). Tracing still uses
            # the spec example.
            generator = torch.Generator().manual_seed(0)
            compare_tensor = torch.rand(
                example_tensor.shape, generator=generator, dtype=example_tensor.dtype
            )
            _compare(compare_tensor)
            if dynamic_spatial:
                # A graph that merely traced at one size can still be wrong at
                # another; the gate runs a SECOND spatial size (torch vs ORT at
                # the same size — the "does the dynamic graph stay accurate"
                # question, independent of the variant's resize conventions).
                # The probe scales DOWN for large examples (the common
                # production need) and UP for small ones (a downscale would
                # undercut feature-pyramid floors).
                _, _, height, width = example_tensor.shape
                factor = (2, 3) if min(height, width) >= 128 else (3, 2)
                probe_height = max(1, height * factor[0] // factor[1])
                probe_width = max(1, width * factor[0] // factor[1])
                # EXPORT-05: the probe input is SEEDED like the first
                # compare input (same tie-flip argument the first input's
                # comment derives — an unseeded probe made the gate flaky
                # at the second size, exactly the failure mode the seeding
                # exists to prevent).
                probe_generator = torch.Generator().manual_seed(0)
                probe = torch.rand(
                    (example_tensor.shape[0], example_tensor.shape[1],
                     probe_height, probe_width),
                    generator=probe_generator,
                    dtype=example_tensor.dtype,
                )
                logger.info(
                    "Dynamic-spatial parity probe at %dx%d (example %dx%d).",
                    probe_width, probe_height, width, height,
                )
                _compare(probe)
        finally:
            self._restore_onnx_export_modules(prepared)
        return "full"  # EXPORT-02: the element-wise gate ran end to end

    def _smoke_verify_fp16(self, out: Path, example_tensor: torch.Tensor) -> None:
        """Runtime smoke gate for fp16 graphs on builds without a CUDA EP.

        Executes the graph on the CPU EP with the same seeded-random compare
        input the parity gate uses and requires every output to be finite:
        a mixed-dtype conversion artifact or a saturated fp16 segment fails
        here instead of silently shipping an unverified graph. Exact
        numerical parity stays on the CUDA-EP path in :meth:`_verify_onnx`.
        """
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        generator = torch.Generator().manual_seed(0)
        compare = torch.rand(
            example_tensor.shape, generator=generator, dtype=example_tensor.dtype
        ).numpy()
        feeds = {}
        for session_input in session.get_inputs():
            feed = compare
            if session_input.type == "tensor(float16)":
                feed = feed.astype(np.float16)
            feeds[session_input.name] = feed
        outputs = session.run(None, feeds)
        for index, output in enumerate(outputs):
            if not np.isfinite(output).all():
                raise AssertionError(
                    f"fp16 smoke gate: output {index} carries non-finite "
                    "values — the fp16 conversion corrupted numerics."
                )

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
        from autovision.export.quantization import (
            Int8CalibrationReader,
            quantize_onnx_int8,
        )

        if calibration_data is None and calibration_reader is None:
            raise ValueError(
                "int8 PTQ requires calibration data: pass calibration_data "
                "(an image path/dir or an iterable of {input_name: Tensor}) "
                "or calibration_reader."
            )
        if calibration_data is not None:
            from autovision.export.quantization import resolve_calibration_samples

            calibration_data = resolve_calibration_samples(self, calibration_data)
        final_out = Path(out)
        final_out.parent.mkdir(parents=True, exist_ok=True)
        # Hygiene: only the int8 chain's own temp names (".int8.tmp…") —
        # a kept "{stem}.fp32.onnx" reference must never match.
        stem = glob.escape(final_out.stem)
        sweep_stale_temp_files(final_out.parent, f"{stem}.*.int8.tmp.onnx")
        sweep_stale_temp_files(final_out.parent, f"{stem}.*.int8.tmp.fp32.onnx")
        tmp_out = final_out.with_name(
            f"{final_out.stem}.{uuid.uuid4().hex}.int8.tmp.onnx"
        )

        # 1. FP32 reference graph (passes the standard FP32 gate itself).
        fp32_path = tmp_out.parent / f"{tmp_out.stem}.fp32.onnx"
        try:
            self.export_onnx(
                fp32_path,
                precision="fp32",
                opset_version=opset_version,
                simplify=simplify,
                verify=verify,
                write_spec=False,
            )

            # 2. Static quantization (QDQ) with calibration.
            reader = calibration_reader or Int8CalibrationReader(
                calibration_data, expected_input_names=list(self.model_input_spec.input_specs)
            )
            quantize_onnx_int8(fp32_path, tmp_out, reader, quantization_config)
            # The reader's sample count is a property of the QUANTIZATION
            # pass; capture it BEFORE the parity gate replays a sample
            # (its rewind+get_next would inflate the count otherwise).
            num_calibration_samples = getattr(reader, "num_samples", None)

            # 3. int8 parity gate: int8-ONNX vs FP32-ONNX on dim-1 sums.
            if verify:
                self._verify_int8_parity(fp32_path, tmp_out, reader, parity_atol, parity_rtol)
        except Exception:
            # A failed int8 chain must not leave a partial graph or delete a
            # pre-existing good export at the target path.
            _drop_onnx_artifacts(tmp_out, fp32_path)
            raise

        # 4. Quantization metadata (written before the atomic publish).
        from autovision.export.quantization import (
            Int8QuantizationConfig,
            _int8_metadata_payload,
        )

        config = quantization_config or Int8QuantizationConfig()
        _write_onnx_metadata(
            tmp_out,
            {
                "int8_quantization": _int8_metadata_payload(config, num_calibration_samples),
                "int8_parity_atol": str(parity_atol),
                "int8_parity_rtol": str(parity_rtol),
            },
        )
        _move_onnx_artifacts(tmp_out, final_out)

        # 5. FP32 reference bookkeeping (with the dynamo .data sidecar).
        if keep_fp32_reference:
            kept_fp32 = final_out.parent / f"{final_out.stem}.fp32.onnx"
            _move_onnx_artifacts(fp32_path, kept_fp32)
            logger.info("Kept FP32 reference graph at '%s'.", kept_fp32)
        else:
            _drop_onnx_artifacts(fp32_path)
        logger.info("Exported int8 ONNX model to '%s'.", final_out)
        _maybe_write_postprocess_spec(self, final_out)
        return final_out

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
        if len(fp32_outputs) != len(int8_outputs):
            # EXPORT-06: a silent zip truncation would compare only the
            # shared prefix and pass a structurally broken quantized graph
            # (the TRT gate raises on the same condition).
            raise AssertionError(
                f"int8 parity gate: output count mismatch — "
                f"{len(int8_outputs)} int8 output(s) vs {len(fp32_outputs)} "
                "FP32 output(s); the quantized graph is structurally broken."
            )
        for index, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            if fp32_out.shape != int8_out.shape:
                raise AssertionError(
                    f"int8 parity gate: output {index} shape mismatch "
                    f"{int8_out.shape} vs FP32 {fp32_out.shape}."
                )
            fp32_sum = torch.as_tensor(fp32_out).float().sum(dim=1)
            int8_sum = torch.as_tensor(int8_out).float().sum(dim=1)
            # equal_nan=False (EXPORT-06): a NaN in the FP32 REFERENCE means
            # the source model or calibration is broken — letting NaN==NaN
            # pass would bless a broken calibration with a green gate.
            torch.testing.assert_close(
                int8_sum, fp32_sum,
                atol=parity_atol, rtol=parity_rtol, equal_nan=False,
                check_device=False, check_dtype=False,
            )

    def export_tensorrt(
        self,
        out: Path | str,
        *,
        precision: Literal["auto", "fp32", "fp16", "int8"] = "fp32",
        calibration_data: Iterable[dict[str, torch.Tensor]] | None = None,
        calibration_reader: Any | None = None,
        quantization_config: Any | None = None,
        keep_int8_onnx: bool = False,
        keep_staged_onnx: bool = False,
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        fp32_attention_scores: bool = False,
        strongly_typed: bool = False,
        onnx_args: dict[str, Any] | None = None,
        verbose: bool = False,
        debug: bool = False,
        verify: bool = True,
    ) -> Path:
        """Build a TensorRT engine from this model (design §12.3).

        Requires the ``tensorrt`` package (not a framework dependency) and a
        CUDA-capable machine for the engine build. The intermediate ONNX is
        produced by :meth:`export_onnx` and removed once the engine built
        and passed its gates (H3-4b: the staged graph never ran the parity
        gate — deploying it because it happens to sit next to the verified
        engine is exactly the accident this avoids). ``keep_staged_onnx=True``
        preserves it for TensorRT-side debugging.

        ``precision="int8"`` (roadmap D4) quantizes first — the QDQ ONNX
        from :meth:`export_onnx_int8` (its int8-vs-fp32 parity gate runs
        here) is staged and built into a strongly-typed engine, so the
        graph's Q/DQ nodes carry the precision contract; ``calibration_data``
        / ``calibration_reader`` are required. ``calibration_data`` accepts
        an image path/directory as well as sample dicts.
        ``keep_int8_onnx=True`` keeps the SANITIZED QDQ graph (the
        TensorRT-side int32 fold mutates it in place).

        ``verify=True`` runs a seeded eager-vs-engine parity check against
        the staged engine BEFORE publication (final tensors, tolerances
        chosen by the engine's RESOLVED precision — H3-4a: ``auto`` reads
        the build decision back instead of assuming fp16); a mismatch drops
        the staged engine and raises while any previous engine at the target
        path stays untouched. Set ``verify=False`` only for throwaway debug
        builds.
        """
        _warn_tensorrt_version_advisory(self)
        from autovision.export.tensorrt_helpers import export_onnx_to_tensorrt

        # H3-4a: the helper reports the build decision point's resolved
        # precision here; the gate reads it after the build returns.
        resolved_precision_report: dict[str, str] = {}

        def _verify_engine(engine_path: Path) -> None:
            """Eager-vs-engine parity gate against the STAGED engine (fail
            closed: drop the staged engine on mismatch). The gate runs before
            publication, so a numerically drifting engine never overwrites
            the artifact at the published target (H1-6)."""
            if not verify:
                return
            from autovision.export.tensorrt_helpers import (
                TensorRTEngineRunner,
                verify_engine_outputs,
            )

            spec = self.model_input_spec
            parameter = next(self.parameters(), None)
            example_device = (
                parameter.device if parameter is not None else torch.device("cpu")
            )
            sample = next(
                iter(
                    spec.example_inputs(
                        batch_size=min_batch_size, device=example_device
                    ).values()
                )
            )
            generator = torch.Generator().manual_seed(0)
            compare = torch.rand(
                sample.shape, generator=generator, dtype=sample.dtype
            ).to(sample.device)
            resolved_precision = resolved_precision_report.get("precision")
            if resolved_precision is None:
                # The helper pre-dating H3-4a reporting (e.g. a mocked
                # build): keep the documented fallback — auto assumed the
                # fp16 band, explicit precisions resolve to themselves.
                resolved_precision = precision if precision != "auto" else "fp16"
            atol, rtol = {
                "fp32": (5e-3, 1e-2),
                "fp16": (5e-2, 5e-2),
                "int8": (2.0, 0.5),
            }[resolved_precision]
            with torch.no_grad():
                # The eager reference runs where the model sits. A CPU-resident
                # GTR model cannot run its fla kernels there — fall back to the
                # export-safe swapped path (the gate then checks engine-vs-
                # exported-graph numerics, the strongest CPU check available —
                # a CUDA-resident model keeps the eager-vs-engine gate).
                if _eager_reference_device_bound(self):
                    logger.warning(
                        "TensorRT parity gate: eager forward unavailable on the "
                        "model's device (cpu); comparing the engine against the "
                        "export-safe reference path instead."
                    )
                    expected = _forward_with_export_hooks(self, compare)
                else:
                    expected = self.forward_backend(compare)
            runner_device = (
                str(compare.device)
                if compare.device.type == "cuda"
                else "cuda"
            )
            runner = TensorRTEngineRunner(engine_path, device=runner_device)
            actual = runner.run(compare.cpu())
            try:
                # Detection outputs can permute tied queries between
                # backends; compare the FINAL boxes/scores (label + IoU
                # matching) instead of raw query order. This is the
                # deployment question and avoids false failures when two
                # anchors tie under fp16.
                used_e2e = False
                image_size = getattr(self, "image_size", None)
                postprocess = getattr(self, "postprocess", None)
                if (
                    resolved_precision in ("fp32", "fp16")
                    and image_size is not None
                    and callable(postprocess)
                ):
                    size = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)
                    metadata = [
                        {"orig_w": size, "orig_h": size}
                        for _ in range(int(compare.shape[0]))
                    ]
                    try:
                        reference_records = postprocess(
                            expected, metadata, threshold=0.0
                        )
                        engine_records = postprocess(
                            actual, metadata, threshold=0.0
                        )
                    except (AutoVisionError, TypeError, AttributeError, KeyError):
                        pass
                    else:
                        if (
                            reference_records
                            and isinstance(reference_records[0], dict)
                            and "bboxes" in reference_records[0]
                        ):
                            from autovision.export.e2e_parity import (
                                compare_detection_records,
                            )

                            report = compare_detection_records(
                                reference_records,
                                engine_records,
                                box_tolerance_px={
                                    "fp32": 1.0,
                                    "fp16": 1.0,
                                    "int8": 5.0,
                                }[resolved_precision],
                                score_tolerance={
                                    "fp32": 1e-2,
                                    "fp16": 5e-2,
                                    "int8": 0.5,
                                }[resolved_precision],
                            )
                            if not report.passed:
                                raise RuntimeError(
                                    "TensorRT end-to-end parity failed: "
                                    f"{report.to_dict()}"
                                )
                            used_e2e = True
                if not used_e2e:
                    if resolved_precision == "int8":
                        # int8 accepts a looser tensor-level contract; the
                        # full-AP regression gate lives in measure_int8_accuracy.
                        verify_engine_outputs(
                            engine_path,
                            expected,
                            compare.cpu(),
                            atol=atol,
                            rtol=rtol,
                        )
                    else:
                        verifier = getattr(self, "verify_onnx_export_outputs", None)
                        if callable(verifier):
                            verifier(expected, actual)
                        else:
                            verify_engine_outputs(
                                engine_path,
                                expected,
                                compare.cpu(),
                                atol=atol,
                                rtol=rtol,
                            )
            except Exception:
                Path(engine_path).unlink(missing_ok=True)
                raise

        # The intermediate ONNX is not the delivered artifact: only the
        # engine gets a postprocess sidecar.
        effective_onnx_args = {**(onnx_args or {}), "write_spec": False}

        # H1-6: build and gate the engine under a STAGED name, publish with
        # os.replace only after the gate passes. The ".tmp-<pid>" marker is
        # the atomic-write convention shared with checkpoint writes, so a
        # killed exporter's orphans stay sweepable by fs_utils' mtime guard.
        final_out = Path(out)
        final_out.parent.mkdir(parents=True, exist_ok=True)
        sweep_stale_temp_files(
            final_out.parent, glob.escape(final_out.name) + ".tmp-*"
        )
        tmp_engine = final_out.with_name(f"{final_out.name}.tmp-{os.getpid()}")

        if precision == "int8":
            if calibration_data is None and calibration_reader is None:
                raise ValueError(
                    "precision='int8' requires calibration: pass "
                    "calibration_data (iterable of {input_name: Tensor}) or "
                    "calibration_reader."
                )
            int8_onnx = final_out.with_suffix(".int8.onnx")

            def _drop(path: Path) -> None:
                for candidate in (path, path.with_suffix(path.suffix + ".data")):
                    candidate.unlink(missing_ok=True)

            try:
                self.export_onnx_int8(
                    int8_onnx,
                    calibration_data=calibration_data,
                    calibration_reader=calibration_reader,
                    quantization_config=quantization_config,
                    verify=True,
                )
            except Exception:
                # The QDQ chain stages an fp32 reference (and the dynamo
                # exporter external-data sidecar) next to the target; failed
                # gates must not leave 100MB-class litter behind.
                _drop(int8_onnx.with_suffix(".fp32.onnx"))
                _drop(int8_onnx)
                raise
            try:
                engine = export_onnx_to_tensorrt(
                    export_onnx_fn=self.export_onnx,
                    out=tmp_engine,
                    precision="int8",
                    model_dtype=next(self.parameters()).dtype,
                    onnx_args=effective_onnx_args,
                    min_batch_size=min_batch_size,
                    opt_batch_size=opt_batch_size,
                    max_batch_size=max_batch_size,
                    fp32_attention_scores=fp32_attention_scores,
                    strongly_typed=True,
                    verbose=verbose,
                    debug=debug,
                    pre_staged_onnx=int8_onnx,
                    resolved_precision_out=resolved_precision_report,
                )
                _verify_engine(engine)
            except BaseException:
                tmp_engine.unlink(missing_ok=True)
                raise
            finally:
                _drop(int8_onnx.with_suffix(".fp32.onnx"))
                if not keep_int8_onnx:
                    _drop(int8_onnx)
            # EXPORT-09: same sidecar-first ordering as the fp32/fp16 path
            # (review P1-1: one snapshot before any publish).
            sidecar_paths = (
                final_out.with_suffix(".meta.json"),
                Path(str(final_out) + ".postprocess.json"),
            )
            int8_backup: dict[Path, bytes] = {
                path: path.read_bytes() for path in sidecar_paths if path.is_file()
            }
            try:
                _write_engine_identity(self, final_out, precision)
                _maybe_write_postprocess_spec(self, final_out)
                os.replace(tmp_engine, final_out)
            except BaseException:
                for path, payload in int8_backup.items():
                    path.write_bytes(payload)
                for path in sidecar_paths:
                    if path not in int8_backup:
                        path.unlink(missing_ok=True)
                tmp_engine.unlink(missing_ok=True)
                raise
            if not keep_int8_onnx:
                # export_onnx_int8 wrote the staged ONNX sidecar; the engine
                # carries its own, so do not leave a dangling contract.
                Path(str(int8_onnx) + ".postprocess.json").unlink(missing_ok=True)
            return final_out

        try:
            engine = export_onnx_to_tensorrt(
                export_onnx_fn=self.export_onnx,
                out=tmp_engine,
                precision=precision,
                model_dtype=next(self.parameters()).dtype,
                onnx_args=effective_onnx_args,
                min_batch_size=min_batch_size,
                opt_batch_size=opt_batch_size,
                max_batch_size=max_batch_size,
                fp32_attention_scores=fp32_attention_scores,
                strongly_typed=strongly_typed,
                verbose=verbose,
                debug=debug,
                resolved_precision_out=resolved_precision_report,
                keep_staged_onnx=keep_staged_onnx,
            )
            _verify_engine(engine)
        except BaseException:
            # Build or gate failure: only the staged engine is dropped; the
            # published target (if any) keeps its previous bytes.
            tmp_engine.unlink(missing_ok=True)
            raise
        # EXPORT-09: identity + decode-contract sidecars publish to the
        # LOADER's names (with_suffix of the final artifact) BEFORE the
        # engine rename, with the previous sidecars backed up and restored
        # on failure — after the rename a reader sees either
        # old-engine+old-sidecars or new-engine+new-sidecars. The old order
        # (engine first, sidecars after) let a fresh engine pair with the
        # PREVIOUS run's identity sidecar during the window.
        # Review P1-1: the backup snapshot must be taken ONCE, BEFORE any
        # publish — per-iteration re-snapshots read the NEW bytes written
        # by the previous publish and overwrite the old identity in the
        # backup, so a failure mid-sequence restored NEW sidecars onto the
        # OLD engine (the exact state this block exists to prevent).
        sidecar_paths = (
            final_out.with_suffix(".meta.json"),
            Path(str(final_out) + ".postprocess.json"),
        )
        backup: dict[Path, bytes] = {
            path: path.read_bytes() for path in sidecar_paths if path.is_file()
        }
        try:
            _write_engine_identity(self, final_out, precision)
            _maybe_write_postprocess_spec(self, final_out)
            os.replace(tmp_engine, final_out)
        except BaseException:
            # Restore the previous sidecars so the OLD engine keeps its OLD
            # identity; sidecars absent from the backup were newly created
            # by this failed export and are removed.
            for path, payload in backup.items():
                path.write_bytes(payload)
            for path in sidecar_paths:
                if path not in backup:
                    path.unlink(missing_ok=True)
            tmp_engine.unlink(missing_ok=True)
            raise
        return final_out

    def export_openvino(
        self,
        out: "Path | str",
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        device: str = "CPU",
        onnx_args: dict[str, Any] | None = None,
        verify: bool = True,
        keep_onnx: bool = False,
    ) -> Path:
        """Export an OpenVINO IR (roadmap D3).

        ``out`` is the ``.xml`` path or a directory (``<dir>/model.xml``).
        The IR converts from this model's own ONNX export (its fp32/fp16
        gates run first), and with ``verify`` the COMPILED OpenVINO model is
        compared against torch on the example input via
        ``verify_onnx_export_outputs`` before the artifact is accepted.
        Requires the ``openvino`` package (``pip install
        'autovision[openvino]'``).
        """
        from autovision.export.openvino_helpers import export_onnx_to_openvino

        xml_path = export_onnx_to_openvino(
            out=out,
            export_onnx_fn=self.export_onnx,
            model=self,
            precision=precision,
            device=device,
            onnx_args={**(onnx_args or {}), "write_spec": False},
            verify=verify,
            keep_onnx=keep_onnx,
        )
        _maybe_write_postprocess_spec(self, xml_path)
        return xml_path


def _write_engine_identity(network: Any, engine: Path, precision: str) -> None:
    """Write the TensorRT engine's identity sidecar (design D1).

    Engines cannot carry in-graph metadata; the ``<engine>.meta.json``
    sidecar mirrors the OpenVINO ``<xml>.json`` (identity + backend and
    precision) so ``InferenceEngine`` can verify network-vs-artifact
    identity the same way for every backend.
    """
    from autovision.export.artifact_metadata import write_artifact_metadata

    write_artifact_metadata(
        network,
        engine,
        backend="tensorrt",
        extra={"backend": "tensorrt", "precision": precision},
    )


def _maybe_write_postprocess_spec(network: Any, artifact: Path) -> None:
    """Write the decode contract only for task families that have one.

    Segmentation / pose / depth / classification exports must keep working;
    they simply have no DETR top-k contract to serialize. The explicit
    ``postprocess_spec_family`` marker distinguishes the DETR detection task
    models from other heads that also carry ``image_size``/``num_classes``
    or ``num_top_queries`` (OBB / instance segmentation / PicoDet).

    The sidecar is staged and published with ``os.replace`` like the
    artifact itself (H1-6): a killed exporter must not leave a partial
    decode contract next to a good graph/engine/IR.
    """
    family = getattr(network, "postprocess_spec_family", None)
    if family is None:
        logger.debug("No postprocess spec for %s.", type(network).__name__)
        return
    from autovision.export.postprocess import write_postprocess_spec

    try:
        artifact = Path(artifact)
        target = Path(str(artifact) + ".postprocess.json")
        # write_postprocess_spec appends ".postprocess.json" to the artifact
        # path it is handed, so the sidecar is staged under the artifact's
        # ".tmp-<pid>" marker (keeping the suffix intact for the rename —
        # and keeping the staged name sweepable, like the artifact tmps).
        staged = artifact.with_name(f"{artifact.name}.tmp-{os.getpid()}")
        staged_sidecar = Path(str(staged) + ".postprocess.json")
        try:
            write_postprocess_spec(network, staged, family=family)
            os.replace(staged_sidecar, target)
        finally:
            staged_sidecar.unlink(missing_ok=True)
    except NotImplementedError:
        logger.debug("No postprocess spec for %s.", type(network).__name__)


def _repair_value_info(out: Path) -> None:
    import onnx

    model = onnx.load(str(out))
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    onnx.save(model, str(out))


def _graph_opset_version(out: Path) -> int | None:
    """The opset the written graph actually targets (H3-4d).

    ``opset_version=None`` lets the exporter pick its default; reading the
    graph back records what really shipped instead of guessing the default.
    """
    import onnx

    try:
        model = onnx.load(str(out), load_external_data=False)
    except Exception:  # noqa: BLE001 - metadata is best-effort, parity owns failure
        return None
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return None


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
