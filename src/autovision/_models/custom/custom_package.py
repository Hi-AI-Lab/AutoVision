# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Custom package: last resort for user-provided ``ModelWrapper`` instances.

This package lists no models: it accepts any object implementing the
``ModelWrapper`` protocol (duck-typed) and exports its state dict. It must
always be registered LAST so named packages are preferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from autovision._models.model_wrapper import (
    missing_model_wrapper_attrs,
)
from autovision._models.package import BasePackage

__all__ = ["CustomPackage"]


class CustomPackage(BasePackage):
    """Last-resort package for user-provided ``ModelWrapper`` instances."""

    name = "custom"

    @classmethod
    def is_supported_model(cls, model: Any) -> bool:
        return (
            not isinstance(model, str)
            and not missing_model_wrapper_attrs(model)
        )

    @classmethod
    def export_model(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        state_dict = (
            model.state_dict()
            if isinstance(model, nn.Module)
            else model.get_model().state_dict()
        )
        torch.save(state_dict, out)
        if log_example:
            class_name = type(model).__name__
            print(
                f"Exported custom model '{class_name}' state dict to '{out}'. "
                f"Rebuild the '{class_name}' instance and load the state dict "
                f"to reuse it."
            )
