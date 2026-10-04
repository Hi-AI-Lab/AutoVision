# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""NaN capture: a self-contained replay snapshot on the first instability.

On a non-finite loss/gradient the monitor saves model weights, optimizer
state, the captured microbatches, and RNG states into a single file, then
raises :class:`NaNDetectedError`. The snapshot replays deterministically via
:func:`load_nan_capture`.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from autovision._accelerator import npu_available

__all__ = ["NaNCaptureMonitor", "NaNDetectedError", "load_nan_capture"]

logger = logging.getLogger("autovision")


class NaNDetectedError(RuntimeError):
    """Raised after a NaN snapshot has been saved."""


class NaNCaptureMonitor:
    """Watches losses/gradients and snapshots the training state on failure.

    Usage::

        monitor = NaNCaptureMonitor(model, out_dir, max_microbatches=21)
        monitor.capture_batch(batch)          # inside the training loop
        ...
        monitor.check_loss(loss)              # after computing the loss
        monitor.check_gradients(model)        # before optimizer.step()
    """

    def __init__(
        self, model: nn.Module, out_dir: Path | str,
        max_microbatches: int = 21, model_class_path: str = "",
        model_init_args: dict | None = None,
    ) -> None:
        self.model = model
        self.out_dir = Path(out_dir)
        self.max_microbatches = max_microbatches
        self.model_class_path = model_class_path
        self.model_init_args = model_init_args or {}
        self._microbatches: deque = deque(maxlen=max_microbatches)
        self._last_optimizer_state: dict | None = None

    def capture_batch(self, batch: dict) -> None:
        detached = {
            key: self._detach(value) for key, value in batch.items()
        }
        self._microbatches.append(detached)

    @staticmethod
    def _detach(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, list):
            return [NaNCaptureMonitor._detach(v) for v in value]
        if isinstance(value, tuple):
            return tuple(NaNCaptureMonitor._detach(v) for v in value)
        return value

    def set_optimizer_state(self, optimizer_state: dict) -> None:
        self._last_optimizer_state = optimizer_state

    @staticmethod
    def _has_non_finite(tensor: torch.Tensor) -> bool:
        return tensor.is_floating_point() and not torch.isfinite(tensor).all()

    def check_loss(self, loss: torch.Tensor) -> None:
        if self._has_non_finite(loss.detach()):
            self._save_and_raise("loss")

    def check_gradients(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.grad is not None and self._has_non_finite(param.grad):
                self._save_and_raise(f"gradient:{name}")

    def _save_and_raise(self, trigger: str) -> None:
        snapshot_dir = self.out_dir / "nan_capture"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / "nan_capture.pt"
        snapshot = {
            "trigger": trigger,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._last_optimizer_state,
            "model_class_path": self.model_class_path,
            "model_init_args": self.model_init_args,
            "microbatches": list(self._microbatches),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else torch.npu.get_rng_state_all()
                if npu_available()
                else None
            ),
            "numpy_rng_state": np.random.get_state(),
        }
        torch.save(snapshot, path)
        logger.error("NaN captured (%s); snapshot saved to '%s'.", trigger, path)
        raise NaNDetectedError(
            f"Non-finite values detected ({trigger}). A replay snapshot was "
            f"saved to '{path}'. Check for data issues or lower the learning "
            f"rate."
        )


def load_nan_capture(path: Path | str) -> dict[str, Any]:
    """Load a NaN snapshot for deterministic replay."""
    return torch.load(path, map_location="cpu", weights_only=False)
