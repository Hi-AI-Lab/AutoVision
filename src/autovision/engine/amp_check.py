# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pre-flight AMP numeric probe (design doc D16).

``precision="16-mixed"``/``"bf16-mixed"`` on a GPU whose driver/stack
mishandles autocast silently degrades training before the in-training
NaN guard could see it: the guard is the countermeasure, this probe is
the pre-flight check. The model runs twice on the same inputs — fp32 and
autocast — and every float output tensor must agree within ``atol``.
The probe uses the caller's own model; no external weights are
downloaded (NOTICE clean-room discipline). Probe failure always
degrades to fp32 (fail-closed) and never aborts the run.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn import Module

from autovision.runtime.accelerator import supports_amp

__all__ = ["check_amp_support", "resolve_amp_precision"]

logger = logging.getLogger("autovision")


def _flatten_outputs(obj: Any, out: list[torch.Tensor]) -> None:
    """Collect float tensors from tensor/dict/sequence/object outputs."""
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            out.append(obj.detach())
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _flatten_outputs(value, out)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _flatten_outputs(value, out)
        return
    state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        for value in state.values():
            _flatten_outputs(value, out)


def _model_device(model: Module) -> torch.device:
    """The device the model's weights live on (probe inputs follow it)."""
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _unwrap_task_model(model: Module) -> Module:
    """The network the probe must actually forward.

    Engine call sites pass the TrainModel wrapper, whose ``forward`` is
    the training_step contract (batch dict + step index). The probe feeds
    a plain ``forward(x: Tensor)`` — only the inner TaskNetwork accepts
    that, so probing the wrapper directly would crash and fail the run
    closed into fp32 on every family.
    """
    getter = getattr(model, "get_task_model", None)
    task_model = getter() if callable(getter) else model
    return task_model if isinstance(task_model, Module) else model


def _probe_batch(
    model: Module, batch: torch.Tensor | None
) -> torch.Tensor | None:
    """Probe inputs: the caller's tensor, or the model's declared input spec.

    The TaskNetwork forward contract is ``forward(x: Tensor)``, so only
    the first declared spec tensor is used (batch 2 to cover
    batch-dependent ops). Inputs move to the model's own parameter device
    — on a CPU-only host with a CUDA-targeted probe this keeps the
    forward runnable instead of failing on a device mismatch.
    """
    target = _model_device(model)
    if batch is not None:
        return batch.to(target)
    spec = getattr(model, "model_input_spec", None)
    if spec is not None and hasattr(spec, "example_inputs"):
        inputs = spec.example_inputs(batch_size=2, device=target)
        for value in inputs.values():
            if torch.is_tensor(value):
                return value
    return None


def check_amp_support(
    model: Module,
    *,
    device: torch.device,
    batch: torch.Tensor | None = None,
    atol: float = 0.5,
    dtype: torch.dtype | None = None,
) -> bool | None:
    """Probe whether autocast outputs track fp32 on this device/model.

    Args:
        model: The module that will actually train — the TrainModel
            wrapper or the bare TaskNetwork (the probe unwraps via
            ``get_task_model`` when present).
        device: Target training device; non-CUDA/NPU devices return False.
            The forward itself runs on the model's current weights device.
        batch: Optional probe input; defaults to the model's declared
            ``model_input_spec`` example tensor.
        atol: Absolute tolerance of the fp32-vs-autocast comparison.
        dtype: Autocast dtype under test; defaults to float16. Callers
            training in bf16-mixed must pass bfloat16 so the verdict is
            made about the dtype actually trained with.

    Returns:
        True when autocast is safe, False when it is demonstrably unsafe
        (CPU/MPS, divergent outputs, probe exceptions), and ``None`` when
        the probe is INCONCLUSIVE — no probe inputs available, or the
        model declared itself probe-incompatible (``amp_probe = "skip"``;
        a stack whose cross-pathway variance already exceeds ``atol``
        makes the comparison uninformative) — in which case the caller
        keeps the requested precision instead of silently downgrading.
        The probe runs in eval mode (previous mode restored); train-mode
        dropout noise would otherwise fail the comparison spuriously.
    """
    if device.type not in ("cuda", "npu"):
        logger.info(
            "AMP probe skipped: device '%s' has no autocast support.",
            device.type,
        )
        return False
    if not isinstance(device, torch.device):
        # Duck-typed stand-ins (tests) normalize for the autocast context.
        device = torch.device(device.type)
    if not any(p.is_floating_point() for p in model.parameters()):
        return False
    probe_model = _unwrap_task_model(model)
    if getattr(probe_model, "amp_probe", None) == "skip":
        logger.info(
            "AMP probe declined by model declaration (amp_probe='skip'); "
            "keeping the requested precision."
        )
        return None
    inputs = _probe_batch(probe_model, batch)
    if inputs is None:
        logger.warning(
            "AMP probe inconclusive: no probe inputs (pass batch= or give "
            "the model a model_input_spec); precision request is kept."
        )
        return None

    if device.type == "cuda" and not torch.cuda.is_available():
        # torch silently disables autocast("cuda") on hosts without CUDA,
        # so the probe below would compare fp32 against fp32 and report
        # "supported" from a vacuous tie. Fail closed instead: a cuda-like
        # device on a CUDA-less host has no autocast training path.
        logger.warning(
            "AMP probe skipped: device is cuda-like but CUDA is "
            "unavailable on this host; autocast support is False."
        )
        return False

    was_training = probe_model.training
    probe_model.eval()
    autocast_dtype = dtype if dtype is not None else torch.float16
    try:
        with torch.no_grad():
            fp32_outputs = probe_model(inputs)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                amp_outputs = probe_model(inputs)
    except Exception as exc:  # noqa: BLE001 - any probe failure is soft
        logger.warning(
            "AMP probe raised %s: %s; falling back to fp32.",
            type(exc).__name__,
            exc,
        )
        return False
    finally:
        probe_model.train(was_training)

    fp32_tensors: list[torch.Tensor] = []
    amp_tensors: list[torch.Tensor] = []
    _flatten_outputs(fp32_outputs, fp32_tensors)
    _flatten_outputs(amp_outputs, amp_tensors)
    if not fp32_tensors or len(fp32_tensors) != len(amp_tensors):
        logger.warning(
            "AMP probe saw %d vs %d float output tensors; falling back "
            "to fp32.",
            len(fp32_tensors),
            len(amp_tensors),
        )
        return False
    try:
        ok = all(
            torch.allclose(a.float(), b.float(), atol=atol)
            for a, b in zip(fp32_tensors, amp_tensors)
        )
    except Exception as exc:  # noqa: BLE001 - shape mismatch etc. is soft
        logger.warning(
            "AMP probe comparison failed (%s); falling back to fp32.", exc
        )
        return False
    if not ok:
        logger.warning(
            "AMP probe detected fp32/autocast divergence beyond atol=%s; "
            "falling back to fp32.",
            atol,
        )
    return ok


def resolve_amp_precision(
    model: Module,
    *,
    device: torch.device,
    precision: str,
    amp_check: bool = True,
) -> str:
    """Effective precision after the device-support and probe gates.

    Args:
        model: The module that will actually train.
        device: Target training device.
        precision: Requested precision (``32-true``/``bf16-mixed``/
            ``16-mixed``/``auto`` resolution happens upstream).
        amp_check: False skips the numeric probe (device-support gate
            still applies).

    Returns:
        The precision to train with — request unchanged, or ``32-true``
        when mixed precision is unsupported/fails the probe. Under DDP the
        rank-0 decision is broadcast as int8 (gloo cannot broadcast bool),
        so every rank trains with the same precision.
    """
    if precision not in ("bf16-mixed", "16-mixed"):
        return precision
    if not isinstance(device, torch.device):
        device = torch.device(device.type)
    if not supports_amp(device.type):
        logger.warning("Mixed precision requested without CUDA/NPU; using fp32.")
        return "32-true"
    if not amp_check:
        return precision
    supported = check_amp_support(
        model,
        device=device,
        dtype=torch.bfloat16 if precision == "bf16-mixed" else torch.float16,
    )
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        # Three-state int8 (True=1, False=0, inconclusive=-1) — bool cannot
        # ride gloo and a two-value encoding would silently turn an
        # inconclusive probe into a downgrade. On the model's device so
        # NCCL sees a CUDA tensor where that backend is active.
        flag = torch.tensor(
            {True: 1, False: 0, None: -1}[supported],
            dtype=torch.int8,
            device=_model_device(model),
        )
        torch.distributed.broadcast(flag, src=0)
        decoded = int(flag.item())
        supported = {1: True, 0: False, -1: None}.get(decoded)
    if supported is None:
        logger.info("AMP probe inconclusive; keeping %s.", precision)
        return precision
    if not supported:
        logger.warning(
            "AMP pre-flight probe failed on %s; training in fp32 instead.",
            device,
        )
        return "32-true"
    return precision
