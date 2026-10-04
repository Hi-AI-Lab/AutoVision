# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EdgeCrafter (ECViT) package registration.

Model names: ``edgecrafter/{ecvitt,ecvittplus,ecvits,ecvitsplus}``. Builds
are local (random init) — the published EdgeCrafter checkpoints are not on
HuggingFace; load one via the task model's ``backbone_weights`` argument
(weights download URLs are kept in
:attr:`autovision._models.ecvit.ecvit.ECVIT_PRETRAINED_URLS`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autovision._models.ecvit.ecvit import (
    ECVIT_PRESETS,
    ECViTModelWrapper,
)
from autovision._models.package import Package

__all__ = ["ECVIT_PACKAGE", "EcviTPackage"]


class EcviTPackage(Package):
    name = "edgecrafter"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return sorted(f"{cls.name}/{name}" for name in ECVIT_PRESETS)

    @classmethod
    def is_supported_model(cls, model: Any) -> bool:
        return isinstance(model, ECViTModelWrapper)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> ECViTModelWrapper:
        # load_weights refers to package-published checkpoints; ECViT has
        # none in this package, so builds are always local-init.
        return ECViTModelWrapper(
            model_name,
            num_input_channels=num_input_channels,
            **(model_args or {}),
        )

    @classmethod
    def get_model_wrapper(cls, model: Any) -> Any:
        if isinstance(model, ECViTModelWrapper):
            return model
        raise ValueError(
            f"Cannot wrap {type(model).__name__} as an ECViT model wrapper."
        )

    @classmethod
    def export_model(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapper = cls.get_model_wrapper(model)
        torch.save(wrapper.get_model().state_dict(), out)
        if log_example:
            print(
                f"Exported ECViT model to '{out}'. Rebuild with "
                f"autovision.get_wrapped_model('<edgecrafter model name>') "
                f"and load_state_dict(torch.load('{out}'))."
            )


ECVIT_PACKAGE = EcviTPackage()

from autovision._models.package_helpers import register_package  # noqa: E402

register_package(ECVIT_PACKAGE)
