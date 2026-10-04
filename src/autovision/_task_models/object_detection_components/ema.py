# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Exponential moving average of model weights (baseline semantics).

``decay=0.9999`` with a linear warmup from 0 over ``warmup_steps``; updated
once per optimizer step via :meth:`update`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import nn

__all__ = ["ModelEMA"]

logger = logging.getLogger("autovision")


class ModelEMA:
    """Maintains exponential moving averages of model parameters/buffers."""

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 2000
    ) -> None:
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates)) if (
            self.updates <= self.warmup_steps
        ) else self.decay
        model_state: dict[str, Any] = model.state_dict()
        ema_state = self.model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key]
            if source.device != value.device:
                source = source.detach().to(value.device)
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.updates = state_dict.get("updates", 0)
