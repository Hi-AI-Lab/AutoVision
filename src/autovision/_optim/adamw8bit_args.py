# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""8-bit AdamW via bitsandbytes (extras: ``autovision[bitsandbytes]``)."""

from __future__ import annotations

import torch
from pydantic import Field

from autovision._optim.optimizer_args import AdamWArgs

__all__ = ["AdamW8bitArgs"]


class AdamW8bitArgs(AdamWArgs):
    """Same defaults as :class:`AdamWArgs`; requires bitsandbytes at build
    time (lazily imported)."""

    betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)

    def type(self):
        from autovision._optimizer_type import OptimizerType

        return OptimizerType.ADAMW8BIT

    def get_optimizer(self, params, lr_scale: float) -> torch.optim.Optimizer:
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as error:
            raise ImportError(
                "8-bit AdamW requires the 'bitsandbytes' package: "
                "pip install 'autovision[bitsandbytes]'"
            ) from error
        kwargs = self.model_dump()
        kwargs["lr"] = self.lr * lr_scale
        return AdamW8bit(params=params, **kwargs)
