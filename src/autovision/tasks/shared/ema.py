# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Exponential moving average of model weights (baseline semantics).

Source lineage: D-FINE ``src/optim/ema.py`` (Apache-2.0, ``Peterande/D-FINE`` @ ``956d170``),
rewritten by AutoVision; provenance ledger entry (P1-4).

``decay=0.9999`` with a linear warmup from 0 over ``warmup_steps``; updated
once per optimizer step via :meth:`update`.
"""

from __future__ import annotations

import copy
import logging
import math
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
        if self.warmup_steps > 0:
            # Baseline exponential ramp: at step k the effective decay is
            # decay * (1 - exp(-k / warmup_steps)) — a SLOW approach (at
            # k = warmup the effective decay is still ~63% of target), so
            # the EMA tracks recent weights early instead of averaging in
            # initial ones.
            decay = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        else:
            decay = self.decay
        model_state: dict[str, Any] = model.state_dict()
        ema_state = self.model.state_dict()
        seen_storages: set[int] = set()
        float_targets: list[torch.Tensor] = []
        float_sources: list[torch.Tensor] = []
        other_targets: list[torch.Tensor] = []
        other_sources: list[torch.Tensor] = []
        for key, value in ema_state.items():
            if value.numel() == 0:
                continue
            source = model_state[key]
            if source.device != value.device:
                source = source.detach().to(value.device)
            if value.data_ptr() in seen_storages:
                # Aliased storage — the same module exposed under several
                # state_dict keys (the AV-DETR task model registers the
                # backbone both as ``backbone_wrapper`` and inside
                # ``backbone``). The foreach update must land ONCE per
                # unique tensor: applied per key, the effective update
                # becomes ``t <- d^2*t + 2*(1-d)*s`` and the aliased
                # module's EMA inflates uniformly toward ``2/(1+d)`` times
                # the model — invisible at decay 0.995, fatal at 0.9999
                # whose long warmup lingers at d ~ 0.3-0.7.
                continue
            seen_storages.add(value.data_ptr())
            if value.dtype.is_floating_point:
                float_targets.append(value)
                float_sources.append(source)
            else:
                other_targets.append(value)
                other_sources.append(source)
        # Foreach batches the update into a handful of kernel launches
        # instead of two per tensor (hundreds of launches on a detector).
        # The foreach ops reject empty tensor lists (a float-only or
        # non-float-only model leaves one bucket empty).
        if float_targets:
            torch._foreach_mul_(float_targets, decay)
            torch._foreach_add_(float_targets, float_sources, alpha=1.0 - decay)
        if other_targets:
            torch._foreach_copy_(other_targets, other_sources)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.updates = state_dict.get("updates", 0)

    @torch.no_grad()
    def repair_from(self, model: nn.Module) -> int:
        """Resync non-finite EMA tensors from live weights (design D33).

        The EMA is a running average: a single non-finite update poisons it
        for the rest of the run, while the live model may already have
        recovered — every checkpoint written afterwards would carry the NaN.
        Only tensors that are non-finite in the EMA AND finite in the live
        model are repaired (a live tensor with no clean source is left
        alone and stays subject to the D20 serialization clamp).

        Returns the number of repaired tensors; callers log the count.
        Requires a full (non-sharded) ``model.state_dict()`` — the engine
        skips this pass under FSDP.
        """
        model_state = model.state_dict()
        ema_state = self.model.state_dict()
        floating = [
            value
            for value in ema_state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        ]
        if floating and bool(
            torch.isfinite(
                torch.stack([value.abs().amax() for value in floating])
            ).all()
        ):
            # Healthy fast path: amax propagates NaN and keeps inf, so one
            # stacked reduction replaces a per-tensor device sync on every
            # checkpoint save.
            return 0
        repaired = 0
        for key, value in ema_state.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            if bool(torch.isfinite(value).all()):
                continue
            source = model_state.get(key)
            if source is None or not torch.is_tensor(source):
                continue
            if value.shape != source.shape or not bool(
                torch.isfinite(source).all()
            ):
                continue
            logger.warning(
                "EMA tensor '%s' is non-finite; resyncing from live weights.",
                key,
            )
            value.copy_(source.to(value.device))
            repaired += 1
        return repaired
