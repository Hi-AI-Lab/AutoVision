# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model inventory helpers: parameter layers and GFLOPs (design doc D23c).

``model_info`` returns the parameter accounting the fit banner and model
cards share. The FLOPs estimate uses ``thop`` when installed (optional
extra ``profiling``) and degrades to ``None`` otherwise — the banner
prints "n/a" instead of pretending the model was profiled.
"""

from __future__ import annotations

import logging

import torch
from torch.nn import Module

__all__ = ["estimate_gflops", "model_info"]

logger = logging.getLogger("autovision")


def model_info(model: Module) -> dict[str, int]:
    """Parameter accounting: totals, trainable, frozen, leaf modules.

    Args:
        model: Any ``nn.Module`` (task network, pretraining recipe).

    Returns:
        ``{"params_total", "params_trainable", "params_frozen",
        "modules"}`` — parameter counts are element counts; ``modules``
        counts leaf modules (no children of their own).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    leaves = sum(1 for m in model.modules() if len(m._modules) == 0)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_frozen": int(total - trainable),
        "modules": int(leaves),
    }


def estimate_gflops(model: Module, example: torch.Tensor | None = None) -> float | None:
    """Estimated GFLOPs for one forward, or ``None`` without ``thop``.

    ``None`` means "not measurable" (no thop, no probe input); callers that
    surface the number omit the line rather than printing a fake zero.

    Args:
        model: Module to profile (any device; the profile runs on the
            example tensor's device).
        example: Probe input; defaults to the model's declared
            ``model_input_spec`` example (batch 1). Without either, the
            estimate is not computable.

    Returns:
        GFLOPs (billions of multiply-accumulate operations) for one
        forward, or ``None`` when ``thop`` is missing or the profile
        fails — callers render that as "n/a" rather than a fake zero.
    """
    if example is None:
        spec = getattr(model, "model_input_spec", None)
        if spec is None or not hasattr(spec, "example_inputs"):
            return None
        # Probe on the weights' device: a CUDA-resident model cannot run a
        # CPU example (the banner calls this AFTER method.to(device)).
        parameter = next(model.parameters(), None)
        target = parameter.device if parameter is not None else None
        inputs = spec.example_inputs(batch_size=1, device=target)
        example = next(iter(inputs.values()), None)
    if example is None:
        return None
    try:
        import thop  # type: ignore[import-not-found]
    except ImportError:
        return None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            flops = thop.profile(
                model, inputs=(example,), verbose=False
            )[0]  # type: ignore[union-attr, operator]
    except Exception as exc:  # noqa: BLE001 - profiling is best-effort
        logger.debug("GFLOPs profile failed (%s); reporting n/a.", exc)
        return None
    finally:
        model.train(was_training)
    return float(flops) / 1e9


def format_param_count(count: int) -> str:
    """Human parameter count for banners (``21.3M`` / ``1.4B``)."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
