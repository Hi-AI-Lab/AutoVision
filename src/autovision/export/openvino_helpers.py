# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""OpenVINO backend: ONNX -> IR conversion + runtime parity gate (D3).

OpenVINO is the CPU-first deployment target (Apache-2.0, pip-installable,
large installed base). The IR is produced from the framework's OWN ONNX
export, so the ONNX-side gates carry over (fp32/fp16 with the op
blacklist); this module adds the OpenVINO-specific steps: convert the ONNX
graph to IR (:func:`openvino.convert_model`), compile it on the requested
device, and — before the artifact is accepted — verify torch vs OpenVINO
outputs on the model's example input through the same
``verify_onnx_export_outputs`` hook every other exporter uses.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pathlib
from typing import Any

from autovision.errors import AutoVisionError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = [
    "OpenVINOEngineRunner",
    "export_onnx_to_openvino",
]

logger = logging.getLogger("autovision")


def _import_openvino():
    try:
        import openvino
    except ImportError as error:  # pragma: no cover - exercised via the gate path
        raise AutoVisionError(
            "The 'openvino' package is required for the OpenVINO backend. "
            "Install it with: pip install 'autovision[openvino]'."
        ) from error
    return openvino


def _resolve_ir_path(out: "str | os.PathLike[str]") -> pathlib.Path:
    """``out`` is the ``.xml`` path; anything else is a directory target
    (``<dir>/model.xml``) — including extensionless paths, but NOT
    ``*.onnx``-like file names, which would silently become a directory."""
    target = pathlib.Path(os.fspath(out))
    if target.suffix.lower() != ".xml":
        if target.suffix and not target.is_dir():
            raise ValueError(
                f"OpenVINO IR target '{target.name}' is neither a .xml file "
                "nor a directory; pass '<dir>' or '<dir>/model.xml'."
            )
        target = target / "model.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _verify_openvino_parity(
    openvino: Any, model: Any, xml_path: pathlib.Path, device: str,
    precision: str = "fp32",
) -> None:
    """Torch vs the COMPILED OpenVINO model on the example input, through the
    model's own export-parity hook — an artifact that fails is not accepted."""
    import copy
    import dataclasses

    compiled = openvino.Core().compile_model(str(xml_path), device)
    reference = copy.deepcopy(model).cpu().float().eval()
    if hasattr(reference, "deploy"):
        reference.deploy()
    example_inputs = model.model_input_spec.example_inputs(batch_size=2)
    input_name = next(iter(example_inputs))
    example = example_inputs[input_name].cpu().float()
    # The reference replays the trace-time swapped path, not the eager
    # kernels: the IR is compiled from the swapped-path ONNX, and the
    # reference is forced onto CPU where device-bound eager kernels (the
    # GTR GLA fla/Triton ops) cannot run at all.
    from autovision.export.export_onnx import _forward_with_export_hooks

    torch_outputs = _forward_with_export_hooks(reference, example)
    if not isinstance(torch_outputs, (list, tuple)) and dataclasses.is_dataclass(torch_outputs):
        torch_outputs = [
            getattr(torch_outputs, f.name) for f in dataclasses.fields(torch_outputs)
        ]
    result = compiled(example.numpy())
    openvino_outputs = [result[port] for port in compiled.outputs]
    model.verify_onnx_export_outputs(torch_outputs, openvino_outputs, precision=precision)
    logger.info("OpenVINO parity gate passed on device '%s'.", device)


def export_onnx_to_openvino(
    *,
    out: "str | os.PathLike[str]",
    export_onnx_fn: Any,
    model: Any,
    precision: str = "fp32",
    device: str = "CPU",
    onnx_args: dict[str, Any] | None = None,
    verify: bool = True,
    keep_onnx: bool = False,
) -> pathlib.Path:
    """Convert ``model``'s ONNX export to an OpenVINO IR at ``out``.

    ``out`` is the ``.xml`` path or a directory (then ``<dir>/model.xml``;
    the ``.bin`` weights land beside it). The intermediate ONNX is staged
    under a ``.tmp-<pid>`` name and removed after the call; only an explicit
    ``keep_onnx=True`` publishes it as ``<stem>.onnx`` (replacing any file
    at that path, and without a decode sidecar). With ``keep_onnx=False``
    staging keeps a pre-existing ``<stem>.onnx`` (a user's own artifact)
    untouched. Metadata rides in a ``<stem>.json`` sidecar (IR
    cannot carry ONNX metadata); with ``verify`` the torch reference and
    the COMPILED OpenVINO model are compared on the example input via
    ``model.verify_onnx_export_outputs``. The IR pair is staged under temp
    names and published with ``os.replace`` only after the parity gate
    passes — a gate failure drops the staged pair and leaves any previous
    IR at the target untouched.
    """
    openvino = _import_openvino()
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got '{precision}'")

    xml_path = _resolve_ir_path(out)
    bin_path = xml_path.with_suffix(".bin")
    published_onnx = xml_path.with_suffix(".onnx")

    # H1-7: the IR pair is staged under ".tmp-<pid>" names and published
    # with os.replace only after the parity gate passes. The marker sits
    # BEFORE the ".xml" extension (not appended): ``save_model`` and the
    # OpenVINO runtime resolve the weights file by replacing that suffix,
    # so the staged pair must remain a valid xml/bin pair. The names stay
    # compatible with fs_utils' mtime-guarded orphan sweep.
    stem_glob = glob.escape(xml_path.stem)
    sweep_stale_temp_files(xml_path.parent, f"{stem_glob}.tmp-*.xml")
    sweep_stale_temp_files(xml_path.parent, f"{stem_glob}.tmp-*.bin")
    sweep_stale_temp_files(xml_path.parent, f"{stem_glob}.tmp-*.onnx*")
    sweep_stale_temp_files(xml_path.parent, f"{stem_glob}.json.tmp-*")
    pid = os.getpid()
    tmp_xml = xml_path.with_name(f"{xml_path.stem}.tmp-{pid}.xml")
    tmp_bin = tmp_xml.with_suffix(".bin")
    # EXPORT-09: the intermediate graph shares the IR's temp marker. Using
    # ``published_onnx`` here silently overwrote — and then deleted — a
    # user's ONNX artifact sitting next to the requested IR.
    staged_onnx = xml_path.with_name(f"{xml_path.stem}.tmp-{pid}.onnx")

    def _drop_ir_tmp() -> None:
        tmp_xml.unlink(missing_ok=True)
        tmp_bin.unlink(missing_ok=True)

    def _drop_intermediate() -> None:
        """Remove only the files this call staged, never ``published_onnx``."""
        for candidate in (
            staged_onnx,
            staged_onnx.with_suffix(".onnx.data"),
            pathlib.Path(str(staged_onnx) + ".postprocess.json"),
        ):
            candidate.unlink(missing_ok=True)

    def _publish_intermediate() -> None:
        """Keep the staged graph under the stable ``<stem>.onnx`` name."""
        os.replace(staged_onnx, published_onnx)
        data = staged_onnx.with_suffix(".onnx.data")
        if data.exists():
            os.replace(data, published_onnx.with_suffix(".onnx.data"))

    export_kwargs: dict[str, Any] = dict(onnx_args or {})
    export_kwargs.setdefault("precision", precision)
    try:
        export_onnx_fn(staged_onnx, **export_kwargs)
        ov_model = openvino.convert_model(str(staged_onnx))
        openvino.save_model(ov_model, str(tmp_xml))
        if verify:
            _verify_openvino_parity(openvino, model, tmp_xml, device, precision)
    except BaseException:
        # Gate or conversion failure: the staged pair is dropped and the
        # published target keeps whatever was there before — the same
        # "previous good artifact survives a failed gate" semantics the ONNX
        # exporter provides.
        _drop_ir_tmp()
        _drop_intermediate()
        raise

    # EXPORT-09 publish order: identity json → weights bin → entry-point
    # xml LAST. The xml is the file every consumer opens first, so after
    # the xml rename a reader sees a complete (json, bin, xml) triple; the
    # old order (bin/xml first, json last) exposed a window where the new
    # IR paired with the PREVIOUS run's identity json — the worst pairing
    # possible, since the json IS the identity gate's input.
    metadata = {
        **model.onnx_export_metadata(),
        "backend": "openvino",
        "precision": precision,
        "device": device,
    }
    json_path = xml_path.with_suffix(".json")
    tmp_json = json_path.with_name(json_path.name + f".tmp-{pid}")
    # EXPORT-09 (review P2-4): back up the OLD identity json so a failure in
    # the bin/xml renames restores it — the old IR must not pair with the
    # NEW identity json.
    json_backup: bytes | None = (
        json_path.read_bytes() if json_path.is_file() else None
    )
    try:
        tmp_json.write_text(json.dumps(metadata, indent=2))
        os.replace(tmp_json, json_path)
        os.replace(tmp_bin, bin_path)
        os.replace(tmp_xml, xml_path)
    except BaseException:
        if json_backup is not None:
            json_path.write_bytes(json_backup)
        else:
            json_path.unlink(missing_ok=True)
        _drop_ir_tmp()
        _drop_intermediate()
        raise
    finally:
        tmp_json.unlink(missing_ok=True)

    # Publish the optional ONNX graph only after the IR triple is in place:
    # a failed IR publish must never replace a user's existing ONNX graph.
    # The kept graph carries no decode sidecar — the IR owns that contract.
    try:
        if keep_onnx:
            _publish_intermediate()
    finally:
        _drop_intermediate()
    logger.info("Published OpenVINO IR to '%s'.", xml_path)
    return xml_path


class OpenVINOEngineRunner:
    """Compiled OpenVINO model with the framework's input/output ordering.

    ``run(images)`` takes the same CPU tensors as the other backends and
    returns framework-ordered outputs, so parity comparisons and timing
    harnesses stay backend-agnostic.
    """

    def __init__(self, ir_path: "str | os.PathLike[str]", device: str = "CPU") -> None:
        openvino = _import_openvino()
        self.ir_path = pathlib.Path(os.fspath(ir_path))
        if not self.ir_path.exists():
            raise FileNotFoundError(f"OpenVINO IR not found: '{self.ir_path}'.")
        self.device = device
        self.core = openvino.Core()
        self.compiled = self.core.compile_model(str(self.ir_path), device)

    def run(self, images: Any) -> list[Any]:
        import torch

        result = self.compiled(images.detach().cpu().numpy())
        return [torch.as_tensor(result[port]) for port in self.compiled.outputs]
