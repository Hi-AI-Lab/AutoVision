# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Generic Conv+BatchNorm fusion for inference-time graphs (design doc D23d).

Task modules keep their own reparameterization paths (``ConvNormLayer_fuse``
and friends); this module is the *generic* fallback for models built from
plain ``nn.Sequential(conv, bn)`` blocks — e.g. externally supplied
backbones — so export/serving can fold batch norms without per-task work.

Fusion is validated numerically: :func:`assert_fused_parity` checks
eval-mode outputs before vs after within tolerance — the same fail-loud
spirit as the export parity gates, applied to a graph rewrite. Call it
after :func:`fuse_conv_bn_model`; the rewrite itself does not self-check.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Module,
    Sequential,
)

__all__ = ["assert_fused_parity", "fuse_conv_bn_model", "fuse_conv_bn_pair"]

logger = logging.getLogger("autovision")

_CONV_TYPES = (Conv1d, Conv2d, Conv3d)
_NORM_TYPES = (BatchNorm1d, BatchNorm2d, BatchNorm3d)

#: A folded BN may carry a small residual in float math; the parity check
#: rides on relative tolerance plus an absolute floor for near-zero outputs.
_PARITY_ATOL = 1e-5
_PARITY_RTOL = 1e-4


#: A conv type pairs with the batch norm of the same spatial rank.
_CONV_TO_NORM: dict[type[Module], type[Module]] = {
    Conv1d: BatchNorm1d,
    Conv2d: BatchNorm2d,
    Conv3d: BatchNorm3d,
}


def fuse_conv_bn_pair(conv: Module, norm: Module) -> Module:
    """Fold one conv+norm pair into a bias-carrying conv (eval semantics).

    Args:
        conv: A ``Conv1d/2d/3d`` (bias present or absent).
        norm: The same-rank ``BatchNorm{1,2,3}d`` that immediately follows.

    Returns:
        A new conv of the same class whose weight/bias encode the folded
        composition ``bn(conv(x))`` in eval mode. A QAT ``amax`` rescale
        (design note) is intentionally out of scope here — QAT graphs must
        be fused by their own quantizer-aware path.

    Raises:
        ValueError: If the pair types do not match in spatial rank.
    """
    expected_norm = _CONV_TO_NORM.get(type(conv))
    if expected_norm is None or not isinstance(norm, expected_norm):
        raise ValueError(
            "fuse_conv_bn_pair expects a Conv* + same-rank BatchNorm* "
            f"pair, got {type(conv).__name__} + {type(norm).__name__}."
        )
    if norm.weight is None or norm.running_mean is None:  # type: ignore[union-attr]
        raise ValueError(
            "fuse_conv_bn_pair needs an affine BatchNorm with running "
            "stats (affine=False / track_running_stats=False cannot fold)."
        )
    conv_w = conv.weight  # type: ignore[union-attr]
    conv_b = getattr(conv, "bias", None)
    bn_w = norm.weight  # type: ignore[union-attr]
    bn_b = norm.bias  # type: ignore[union-attr]
    running_mean = norm.running_mean  # type: ignore[union-attr]
    running_var = norm.running_var  # type: ignore[union-attr]
    eps = float(norm.eps)  # type: ignore[union-attr]

    with torch.no_grad():
        # Folded weight = W * diag(gamma / std): the scale rides the OUT
        # channel axis, so it reshapes to (out, 1, ..., 1) over weight.ndim-1.
        scale = (bn_w / torch.sqrt(running_var + eps)).reshape(
            -1, *([1] * (conv_w.ndim - 1))
        )
        folded_w = conv_w * scale
        folded_b = bn_b - running_mean * scale.reshape(-1)
        if conv_b is not None:
            folded_b = folded_b + conv_b
        fused = type(conv)(
            conv.in_channels,  # type: ignore[union-attr]
            conv.out_channels,  # type: ignore[union-attr]
            kernel_size=conv.kernel_size,  # type: ignore[union-attr]
            stride=conv.stride,  # type: ignore[union-attr]
            padding=conv.padding,  # type: ignore[union-attr]
            dilation=conv.dilation,  # type: ignore[union-attr]
            groups=conv.groups,  # type: ignore[union-attr]
            bias=True,
            padding_mode=conv.padding_mode,  # type: ignore[union-attr]
            device=folded_w.device,
            dtype=folded_w.dtype,
        )
        fused.weight.data.copy_(folded_w)
        fused.bias.data.copy_(folded_b)
    return fused


def fuse_conv_bn_model(model: Module) -> Module:
    """Return a copy of ``model`` with ``Sequential(conv, bn)`` blocks folded.

    Args:
        model: Any module. Only direct ``nn.Sequential`` runs of
            conv→bn are rewritten (a conservative pattern — attribute-held
            pairs behind custom modules belong to the task's own
            reparameterization path). Nested Sequentials recurse.

    Returns:
        A deep-copied model with fused blocks. The input model is not
        modified; run this at export/serving time on an eval model.
    """
    fused = copy.deepcopy(model)
    fused.eval()  # fusion is eval-mode semantics; train-mode blocks skip
    _replace_in_module(fused)
    return fused


def _replace_in_module(module: Module) -> None:
    """Rewrite eligible ``Sequential`` children in place on the copy."""
    for name, child in list(module.named_children()):
        if isinstance(child, Sequential):
            fused_run = _fuse_sequential_run(child)
            if fused_run is not None:
                setattr(module, name, fused_run)
                continue
        _replace_in_module(child)


def _fuse_sequential_run(run: Sequential) -> Sequential | None:
    """Fold every conv→bn adjacency inside one ``Sequential`` (or None)."""
    children = list(run.children())
    if len(children) < 2 or not any(
        isinstance(a, _CONV_TYPES) and isinstance(b, _NORM_TYPES)
        for a, b in zip(children, children[1:])
    ):
        return None
    rebuilt: list[Module] = []
    index = 0
    while index < len(children):
        current = children[index]
        follower = children[index + 1] if index + 1 < len(children) else None
        if (
            isinstance(current, _CONV_TYPES)
            and isinstance(follower, _NORM_TYPES)
            and current.training is False
        ):
            rebuilt.append(fuse_conv_bn_pair(current, follower))
            index += 2
        else:
            rebuilt.append(current)
            index += 1
    return Sequential(*rebuilt)


def assert_fused_parity(
    original: Module, fused: Module, example: torch.Tensor, *,
    atol: float = _PARITY_ATOL, rtol: float = _PARITY_RTOL,
) -> None:
    """Assert eval-mode outputs match before/after fusion; raise otherwise.

    Args:
        original: The unfused model.
        fused: The fused copy.
        example: Probe input (single tensor, matching the model's forward).
        atol/rtol: Comparison tolerances.

    Raises:
        AutoVisionError: When any output tensor diverges beyond tolerance.
    """
    from autovision.errors import AutoVisionError

    states = [m.training for m in (original, fused)]
    try:
        for module in (original, fused):
            module.eval()
        with torch.no_grad():
            reference = original(example)
            candidate = fused(example)
        reference_tensors = _float_tensors(reference)
        candidate_tensors = _float_tensors(candidate)
        if len(reference_tensors) != len(candidate_tensors):
            raise AutoVisionError(
                "Conv+BN fusion changed the output structure "
                f"({len(reference_tensors)} -> {len(candidate_tensors)} float "
                "tensors); refusing a structurally drifted graph."
            )
        for tensor_a, tensor_b in zip(reference_tensors, candidate_tensors):
            if not torch.allclose(tensor_a, tensor_b, atol=atol, rtol=rtol):
                raise AutoVisionError(
                    "Conv+BN fusion changed outputs beyond tolerance "
                    f"(max |Δ| = {float((tensor_a - tensor_b).abs().max()):.3e}); "
                    "refusing to hand over a numerically drifted graph."
                )
    finally:
        for module, state in zip((original, fused), states):
            module.train(state)


def _float_tensors(output: Any) -> Sequence[torch.Tensor]:
    """Float tensors from tensor/tuple/list outputs."""
    if torch.is_tensor(output):
        return [output] if output.is_floating_point() else []
    if isinstance(output, (list, tuple)):
        flattened: list[torch.Tensor] = []
        for item in output:
            flattened.extend(_float_tensors(item))
        return flattened
    return []
