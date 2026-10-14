# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""timm package: every model in the timm registry, any input channel count.

Models are built **untrained by default**; pretrained weights require an
explicit ``model_args={"pretrained": true}`` (or ``"checkpoint_path"``) and
network access. timm itself is an optional extra: without it, the model list
is empty instead of erroring.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn

from autovision.encoders.package import Package
from autovision.encoders.timm.timm_wrapper import TIMMModelWrapper

__all__ = ["TIMM_PACKAGE", "TimmPackage"]

# ViT-family models reject variable input sizes unless this flag is set.
_DYNAMIC_INPUT_PREFIXES = ("vit", "eva", "deit")


class TimmPackage(Package):
    name = "timm"

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            import timm
        except ImportError:
            return []
        return [f"{cls.name}/{name}" for name in timm.list_models()]

    @classmethod
    def is_supported_model(cls, model: Any) -> bool:
        if isinstance(model, TIMMModelWrapper):
            return True
        raw = model.get_model() if isinstance(model, TIMMModelWrapper) else model
        try:
            class_hierarchy = inspect.getmro(type(raw))
        except TypeError:  # pragma: no cover - non-class objects
            return False
        return any(cls_.__module__.split(".")[0] == "timm" for cls_ in class_hierarchy)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        try:
            import timm
        except ImportError as error:
            raise ValueError(
                f"Cannot create model 'timm/{model_name}' because timm is not "
                f"installed. Install it with: pip install 'autovision[timm]'."
            ) from error
        args: dict[str, Any] = {
            "pretrained": False,
            "in_chans": num_input_channels,
        }
        if model_name.startswith(_DYNAMIC_INPUT_PREFIXES):
            # Allow inputs whose size differs from the training size (needed
            # by multi-crop methods and the embed command).
            args["dynamic_img_size"] = True
        if model_args:
            args.update(model_args)
        if not load_weights:
            args["pretrained"] = False
            args["checkpoint_path"] = None
        return timm.create_model(model_name, **args)

    @classmethod
    def get_model_wrapper(cls, model: nn.Module) -> TIMMModelWrapper:
        if isinstance(model, TIMMModelWrapper):
            return model
        return TIMMModelWrapper(model)

    @classmethod
    def export_model(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = model.get_model() if isinstance(model, TIMMModelWrapper) else model
        if not cls.is_supported_model(raw):
            raise ValueError(
                f"The timm package only exports timm models, got "
                f"'{type(raw).__name__}'."
            )
        torch.save(raw.state_dict(), out)
        if log_example:
            architecture = ""
            pretrained_cfg = getattr(raw, "pretrained_cfg", None)
            if isinstance(pretrained_cfg, dict):
                architecture = str(pretrained_cfg.get("architecture", ""))
            print(
                f"Exported timm model to '{out}'. Load with: "
                f"timm.create_model(model_name='{architecture}', "
                f"checkpoint_path='{out}')."
            )


TIMM_PACKAGE = TimmPackage()

from autovision.encoders.package_helpers import register_package  # noqa: E402

register_package(TIMM_PACKAGE)
