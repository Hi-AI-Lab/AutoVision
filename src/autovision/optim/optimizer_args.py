# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Optimizer configuration models and the optimizer registry."""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field

from autovision.config.base import PydanticConfig
from autovision.optim.optimizer_type import OptimizerType

__all__ = [
    "AdamWArgs",
    "LARSArgs",
    "OptimizerArgs",
    "SGDArgs",
    "get_optimizer_args_cls",
]

ParamsT = Any  # iterable of tensors or param-group dicts

# Weight decay applied when ``weight_decay="auto"`` resolves per architecture.
_WEIGHT_DECAY_TRANSFORMER = 0.04
_WEIGHT_DECAY_CONV_BATCHNORM = 1e-6


class OptimizerArgs(PydanticConfig):
    """Base class for optimizer configs."""

    def type(self) -> OptimizerType:
        raise NotImplementedError

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> torch.optim.Optimizer:
        """Instantiate the optimizer; ``lr`` is scaled by ``lr_scale``."""
        raise NotImplementedError

    def resolve_auto(self, wrapped_model: Any = None) -> None:
        """Resolve any ``"auto"`` fields (default: nothing to resolve)."""


def _is_probably_conv(module: Any) -> bool:
    """Heuristic architecture guess from module class names (fallback when the
    model wrapper does not implement ``architecture_info``)."""
    names = " ".join(type(m).__name__.lower() for m in module.modules())
    has_attention = "attn" in names or "attention" in names
    if "layernorm" in names and has_attention:
        return False
    if "layernorm" in names and "conv" in names:
        return False
    if "conv" in names:
        return True
    return False


def _resolve_auto_weight_decay(wrapped_model: Any) -> float:
    """Resolve ``weight_decay="auto"`` from the model architecture."""
    info_fn = getattr(wrapped_model, "architecture_info", None)
    if callable(info_fn):
        info = info_fn()
        model_type: str = info["model_type"]
        norm_type: str = info["norm_type"]
        if model_type in ("transformer", "hybrid"):
            return _WEIGHT_DECAY_TRANSFORMER
        if model_type == "convolutional" and norm_type == "layernorm":
            return _WEIGHT_DECAY_TRANSFORMER
        if model_type == "convolutional" and norm_type == "batchnorm":
            return _WEIGHT_DECAY_CONV_BATCHNORM
        raise ValueError(
            f"Cannot resolve weight_decay='auto' for architecture "
            f"model_type={model_type!r}, norm_type={norm_type!r}. "
            f"Set weight_decay explicitly."
        )
    if wrapped_model is not None and _is_probably_conv(wrapped_model):
        return _WEIGHT_DECAY_CONV_BATCHNORM
    return _WEIGHT_DECAY_TRANSFORMER


class AdamWArgs(OptimizerArgs):
    lr: float = 0.001
    betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    eps: float = 1e-8
    weight_decay: float | Literal["auto"] = 0.01

    def type(self) -> OptimizerType:
        return OptimizerType.ADAMW

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> torch.optim.Optimizer:
        kwargs = self.model_dump()
        kwargs["lr"] = self.lr * lr_scale
        return torch.optim.AdamW(params=params, **kwargs)

    def resolve_auto(self, wrapped_model: Any = None) -> None:
        if self.weight_decay == "auto":
            self.weight_decay = _resolve_auto_weight_decay(wrapped_model)


class SGDArgs(OptimizerArgs):
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4

    def type(self) -> OptimizerType:
        return OptimizerType.SGD

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> torch.optim.Optimizer:
        kwargs = self.model_dump()
        kwargs["lr"] = self.lr * lr_scale
        return torch.optim.SGD(params=params, **kwargs)


class LARSArgs(OptimizerArgs):
    lr: float = 0.3
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    trust_coefficient: float = 0.001
    eps: float = 1e-8

    def type(self) -> OptimizerType:
        return OptimizerType.LARS

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> torch.optim.Optimizer:
        from autovision.optim.lars import LARS

        kwargs = self.model_dump()
        kwargs["lr"] = self.lr * lr_scale
        return LARS(params=params, **kwargs)


# String -> config class registry ("adamw" -> AdamWArgs, ...).
_OPTIM_TYPE_TO_ARGS: dict[str, type[OptimizerArgs]] = {
    OptimizerType.ADAMW.value: AdamWArgs,
    OptimizerType.SGD.value: SGDArgs,
    OptimizerType.LARS.value: LARSArgs,
}


def get_optimizer_args_cls(optim_type: str) -> type[OptimizerArgs]:
    """Look up the optimizer args class for an optimizer type string."""
    if optim_type == "adamw8bit":
        from autovision.optim.adamw8bit_args import AdamW8bitArgs

        return AdamW8bitArgs
    try:
        return _OPTIM_TYPE_TO_ARGS[OptimizerType(optim_type).value]
    except ValueError:
        valid = ", ".join(sorted(_OPTIM_TYPE_TO_ARGS))
        raise ValueError(
            f"Unknown optimizer type: '{optim_type}'. Valid types: {valid}"
        ) from None
