# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""TensorRT engine building from the ONNX export pipeline.

``export_onnx_to_tensorrt`` first produces an ONNX file (reusing
``TaskModel.export_onnx``), then parses it with the TensorRT ONNX parser and
serializes an engine. Engine-build semantics (design doc §12.3, mirroring the
baseline):

- EXPLICIT_BATCH network, optional STRONGLY_TYPED (precisions taken from the
  ONNX graph; mutually exclusive with ``fp32_attention_scores``).
- Input name fixed to ``"images"``; spatial dims must be static — only the
  batch dimension gets an optimization profile (min/opt/max).
- 1 GB workspace; TF32 cleared to keep mixed-precision paths stable.
- ``precision="fp16"`` sets the FP16 builder flag plus
  OBEY_PRECISION_CONSTRAINTS so per-layer FP32 overrides are respected.
- ``fp32_attention_scores=True`` pins all Softmax layers and the MatMul
  layers feeding them to FP32 (attention scores are the known FP16
  overflow hotspot).

TensorRT is not a framework dependency: a missing installation raises a
``ModuleNotFoundError`` with install guidance. All argument validation
happens *before* the import so misconfiguration is reportable anywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

import torch

__all__ = ["export_onnx_to_tensorrt"]

logger = logging.getLogger("autovision")

_TRT_INSTALL_GUIDE = (
    "TensorRT is required, but is not installed. Install TensorRT for your "
    "system following NVIDIA's guide: "
    "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html "
    "(on CUDA 12.x systems, `pip install tensorrt-cu12` often works)."
)


def export_onnx_to_tensorrt(
    *,
    export_onnx_fn: Callable[..., Any],
    out: Path | str,
    precision: Literal["auto", "fp32", "fp16"] = "fp32",
    model_dtype: torch.dtype = torch.float32,
    onnx_args: dict[str, Any] | None = None,
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    fp32_attention_scores: bool = False,
    strongly_typed: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> Path:
    """Build a TensorRT engine at ``out`` from the model's ONNX export.

    ``export_onnx_fn`` is typically ``model.export_onnx``; keyword arguments
    in ``onnx_args`` are forwarded to it. The intermediate ONNX file is
    written next to the engine (same stem, ``.onnx`` suffix) unless
    ``onnx_args["out"]`` says otherwise.
    """
    if not (min_batch_size <= opt_batch_size <= max_batch_size):
        raise ValueError(
            "Batch sizes must satisfy: min_batch_size <= opt_batch_size <= "
            f"max_batch_size (got {min_batch_size}/{opt_batch_size}/{max_batch_size})."
        )
    if strongly_typed and fp32_attention_scores:
        raise ValueError(
            "strongly_typed and fp32_attention_scores are mutually exclusive: "
            "a strongly-typed network derives all precisions from the ONNX "
            "graph and does not allow the per-layer overrides used by "
            "fp32_attention_scores."
        )

    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:  # pragma: no cover - trt absent in CI
        raise ModuleNotFoundError(_TRT_INSTALL_GUIDE) from error

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, trt_logger)

    onnx_args = dict(onnx_args or {})
    onnx_path = Path(onnx_args.pop("out", Path(out).with_suffix(".onnx")))
    onnx_args.setdefault("verify", False)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_args = {"out": onnx_path, **onnx_args}
    export_onnx_fn(**export_args)

    if debug:  # pragma: no cover - requires onnx + trt debugging workflow
        _debug_mark_all_layers_as_outputs(onnx_path)

    logger.info("Loading ONNX file from %s", onnx_path)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for error_index in range(parser.num_errors):
            logger.error("%s", parser.get_error(error_index))
        raise RuntimeError(
            f"Failed to parse ONNX file '{onnx_path}' with TensorRT."
        )

    if fp32_attention_scores:
        _force_fp32_for_attention_scores(network)

    images_input = None
    for index in range(network.num_inputs):
        network_input = network.get_input(index)
        if network_input.name == "images":
            images_input = network_input
            break
    if images_input is None:
        raise RuntimeError(
            "Could not find the 'images' input in the ONNX network; the "
            "TensorRT path expects the standard export input spec."
        )
    _, channels, height, width = (int(dim) for dim in images_input.shape)
    if height == -1 or width == -1:
        raise ValueError(
            "Dynamic image height and width are not supported by the "
            "TensorRT export; export the ONNX model with a static image size."
        )
    logger.info("Detected input shape: (N, %d, %d, %d)", channels, height, width)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # TF32 can silently change numerics in mixed-precision paths.
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    if strongly_typed:
        # Precisions come from the ONNX graph's dtypes and Cast nodes; the
        # FP16/precision-constraint flags do not apply to such networks.
        logger.info(
            "Strongly-typed network: precision taken from the ONNX graph."
        )
    elif precision == "fp16" or (
        precision == "auto" and model_dtype == torch.float16
    ):
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # Ensure layer.precision overrides are respected, not merely
            # preferred.
            if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            logger.info("FP16 optimization enabled.")
        else:
            logger.warning("FP16 not supported on this platform; using FP32.")

    if debug:  # pragma: no cover - requires trt debugging workflow
        config.set_flag(trt.BuilderFlag.DEBUG)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(min_batch_size, channels, height, width),
        opt=(opt_batch_size, channels, height, width),
        max=(max_batch_size, channels, height, width),
    )
    for index in range(network.num_inputs):
        network_input = network.get_input(index)
        if network_input.name == "images":
            continue
        shape = tuple(int(dim) for dim in network_input.shape)
        if -1 not in shape:
            continue
        if any(dim == -1 for dim in shape[1:]):
            raise ValueError(
                f"Dynamic non-batch dimensions are not supported for input "
                f"'{network_input.name}'."
            )
        profile.set_shape(
            network_input.name,
            min=(min_batch_size, *shape[1:]),
            opt=(opt_batch_size, *shape[1:]),
            max=(max_batch_size, *shape[1:]),
        )
    config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine (this can take several minutes)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine building failed.")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        handle.write(engine)
    logger.info("TensorRT engine saved to '%s'.", out)
    return out


def _force_fp32_for_attention_scores(net) -> None:
    """Pin Softmax layers (and the MatMuls feeding them) to FP32."""
    import tensorrt as trt  # type: ignore[import-not-found]

    io_tensors = {
        *(net.get_input(index) for index in range(net.num_inputs)),
        *(net.get_output(index) for index in range(net.num_outputs)),
    }
    softmax_inputs: set[str] = set()
    for index in range(net.num_layers):
        layer = net.get_layer(index)
        if "Softmax" in layer.name:
            layer_input = layer.get_input(0)
            if layer_input is not None:
                softmax_inputs.add(layer_input.name)

    forced_softmax = 0
    forced_matmul = 0
    for index in range(net.num_layers):
        layer = net.get_layer(index)
        if "Softmax" in layer.name:
            layer.precision = trt.DataType.FLOAT
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if output in io_tensors:
                    output.dtype = trt.DataType.FLOAT
            forced_softmax += 1
            continue
        if "MatMul" in layer.name:
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if output is not None and output.name in softmax_inputs:
                    layer.precision = trt.DataType.FLOAT
                    if output in io_tensors:
                        output.dtype = trt.DataType.FLOAT
                    forced_matmul += 1
                    break
    logger.info(
        "Forced FP32 on %d Softmax layers and %d attention-score MatMul layers.",
        forced_softmax, forced_matmul,
    )


def _debug_mark_all_layers_as_outputs(out: Path) -> None:
    """Mark every intermediate tensor as a graph output for polygraphy runs."""
    import onnx
    from onnx import helper, shape_inference

    model = onnx.load(str(out))
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    existing = {output.name for output in graph.output}
    value_info = {
        tensor.name: tensor
        for tensor in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    for node in graph.node:
        for tensor_name in node.output:
            if not tensor_name or tensor_name in existing or tensor_name not in value_info:
                continue
            tensor_type = value_info[tensor_name].type.tensor_type
            graph.output.append(
                helper.make_tensor_value_info(tensor_name, tensor_type, None)
            )
            existing.add(tensor_name)
    onnx.save(model, str(out))
