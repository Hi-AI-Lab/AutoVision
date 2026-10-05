# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv3 package, backed by HuggingFace ``transformers``.

Per framework policy, pretrained DINOv3 usage prefers ``transformers`` over
vendored code. transformers gained DINOv3 support relatively recently, so
this package is **gated**: it auto-enables when the installed transformers
exposes the DINOv3 classes and degrades gracefully (empty model list,
informative error) otherwise — no vendored fallback.

Model names mirror the baseline registry: ``vits16(+plus)``, ``vitb16``,
``vitl16(+plus alias vith16plus)``, ``vit7b16``, ``convnext-*``, SAT
variants, each with a ``-notpretrained`` offline variant.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autovision._models.dinov3.dinov3_wrapper import (
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)
from autovision._models.package import Package
from autovision._models.weight_verification import load_pretrained_verified

__all__ = ["DINOV3_PACKAGE", "Dinov3Package"]

_PATCH_SIZE = 16
_IMAGE_SIZE = 224

# model name -> (HF checkpoint, architecture, config kwargs)
_MODEL_INFO: dict[str, dict[str, Any]] = {
    "vits16": dict(
        hf="facebook/dinov3-vits16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
    ),
    "vits16plus": dict(
        hf="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        arch="vit",
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
    ),
    "vitb16": dict(
        hf="facebook/dinov3-vitb16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "vitl16": dict(
        hf="facebook/dinov3-vitl16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "vith16plus": dict(
        hf="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        arch="vit",
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
    ),
    "vit7b16": dict(
        hf="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=4096,
        num_hidden_layers=40,
        num_attention_heads=16,
    ),
    "vitl16-sat493m": dict(
        hf="facebook/dinov3-vitl16-pretrain-sat493m",
        arch="vit",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "vit7b16-sat493m": dict(
        hf="facebook/dinov3-vit7b16-pretrain-sat493m",
        arch="vit",
        hidden_size=4096,
        num_hidden_layers=40,
        num_attention_heads=16,
    ),
    "convnext-tiny": dict(
        hf="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[96, 192, 384, 768],
    ),
    "convnext-small": dict(
        hf="facebook/dinov3-convnext-small-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[96, 192, 384, 768],
    ),
    "convnext-base": dict(
        hf="facebook/dinov3-convnext-base-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[128, 256, 512, 1024],
    ),
    "convnext-large": dict(
        hf="facebook/dinov3-convnext-large-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[192, 384, 768, 1536],
    ),
}


def _dinov3_classes() -> dict[str, Any] | None:
    """Live detection of DINOv3 support in the installed transformers.

    Returns the class map or ``None`` when transformers is missing or too
    old. Reads module attributes (not imports) so the check stays honest
    across transformers versions with differing class names.
    """
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        return None
    # transformers exports the DINOv3 classes as ``DINOv3*`` (both the 4.x
    # lazy exports and 5.x); the ``Dinov3*`` spellings are defensive
    # fallbacks for intermediate layouts.
    vit_config = (
        getattr(transformers, "DINOv3ViTConfig", None)
        or getattr(transformers, "Dinov3ViTConfig", None)
        or getattr(transformers, "Dinov3Config", None)
    )
    vit_model = (
        getattr(transformers, "DINOv3ViTModel", None)
        or getattr(transformers, "Dinov3ViTModel", None)
        or getattr(transformers, "Dinov3Model", None)
    )
    convnext_config = (
        getattr(transformers, "DINOv3ConvNextConfig", None)
        or getattr(transformers, "Dinov3ConvNextConfig", None)
    )
    convnext_model = (
        getattr(transformers, "DINOv3ConvNextModel", None)
        or getattr(transformers, "Dinov3ConvNextModel", None)
    )
    if None in (vit_config, vit_model, convnext_config, convnext_model):
        return None
    return {
        "vit_config": vit_config,
        "vit_model": vit_model,
        "convnext_config": convnext_config,
        "convnext_model": convnext_model,
    }


def _sync_num_channels(model: Any, num_input_channels: int) -> None:
    """Sync every captured num_channels copy (config/embeddings/stem)."""
    model.config.num_channels = num_input_channels
    embeddings = getattr(model, "embeddings", None)
    for owner in (embeddings, getattr(embeddings, "patch_embeddings", None)):
        if owner is not None and hasattr(owner, "num_channels"):
            owner.num_channels = num_input_channels


def _support_error(model_name: str) -> ImportError:
    return ImportError(
        f"DINOv3 model '{model_name}' requires a transformers version with "
        f"DINOv3 support (class Dinov3ViTModel/Dinov3ConvNextModel). "
        f"Upgrade: pip install -U 'autovision[transformers]'."
    )


class Dinov3Package(Package):
    name = "dinov3"

    @classmethod
    def list_model_names(cls) -> list[str]:
        if _dinov3_classes() is None:
            return []
        names: list[str] = []
        for model_name in _MODEL_INFO:
            names.append(f"{cls.name}/{model_name}")
            names.append(f"{cls.name}/{model_name}-notpretrained")
        return sorted(names)

    @classmethod
    def is_supported_model(cls, model: Any) -> bool:
        if isinstance(
            model, (TransformersDinov3ViTWrapper, TransformersDinov3ConvNextWrapper)
        ):
            return True
        classes = _dinov3_classes()
        if classes is None:
            return False
        return isinstance(model, (classes["vit_model"], classes["convnext_model"]))

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        classes = _dinov3_classes()
        if classes is None:
            raise _support_error(model_name)
        base_name = model_name.removesuffix("-notpretrained")
        pretrained = model_name == base_name
        if base_name not in _MODEL_INFO:
            supported = ", ".join(sorted(_MODEL_INFO))
            raise ValueError(
                f"Unknown dinov3 model: '{model_name}'. Supported models: {supported}."
            )
        info = _MODEL_INFO[base_name]
        config_kwargs = dict(model_args or {})
        if pretrained and load_weights:
            model_cls = (
                classes["vit_model"] if info["arch"] == "vit" else classes["convnext_model"]
            )
            model = load_pretrained_verified(model_cls, info["hf"], config_kwargs)
        elif info["arch"] == "vit":
            config = classes["vit_config"](
                hidden_size=info["hidden_size"],
                num_hidden_layers=info["num_hidden_layers"],
                num_attention_heads=info["num_attention_heads"],
                image_size=_IMAGE_SIZE,
                patch_size=_PATCH_SIZE,
                **config_kwargs,
            )
            model = classes["vit_model"](config)
        else:
            config = classes["convnext_config"](
                hidden_sizes=info["hidden_sizes"],
                image_size=_IMAGE_SIZE,
                patch_size=_PATCH_SIZE,
                **config_kwargs,
            )
            model = classes["convnext_model"](config)
        if num_input_channels != 3:
            cls._adapt_input_channels(model, num_input_channels)
        return model

    @classmethod
    def _adapt_input_channels(cls, model: Any, num_input_channels: int) -> None:
        from autovision._models._model_helpers import adapt_conv_input_channels

        if hasattr(model.config, "hidden_sizes"):  # ConvNeXt
            stem = model.embeddings.patch_embeddings
            new_stem = nn.Conv2d(
                num_input_channels,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=stem.bias is not None,
            )
            with torch.no_grad():
                new_stem.weight.copy_(
                    adapt_conv_input_channels(stem.weight.data, num_input_channels)
                )
                if stem.bias is not None:
                    new_stem.bias.copy_(stem.bias.data)
            model.embeddings.patch_embeddings = new_stem
            _sync_num_channels(model, num_input_channels)
        else:  # ViT
            projection = model.embeddings.patch_embeddings.projection
            new_projection = nn.Conv2d(
                num_input_channels,
                projection.out_channels,
                kernel_size=projection.kernel_size,
                stride=projection.stride,
                padding=projection.padding,
                bias=projection.bias is not None,
            )
            with torch.no_grad():
                new_projection.weight.copy_(
                    adapt_conv_input_channels(
                        projection.weight.data, num_input_channels
                    )
                )
                if projection.bias is not None:
                    new_projection.bias.copy_(projection.bias.data)
            model.embeddings.patch_embeddings.projection = new_projection
            _sync_num_channels(model, num_input_channels)

    @classmethod
    def get_model_wrapper(cls, model: Any) -> Any:
        if isinstance(model, TransformersDinov3ViTWrapper):
            return model
        if isinstance(model, TransformersDinov3ConvNextWrapper):
            return model
        if hasattr(getattr(model, "config", None), "hidden_sizes"):
            return TransformersDinov3ConvNextWrapper(model)
        return TransformersDinov3ViTWrapper(model)

    @classmethod
    def export_model(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapper = cls.get_model_wrapper(model)
        torch.save(wrapper.get_model().state_dict(), out)
        if log_example:
            print(
                f"Exported DINOv3 model to '{out}'. Rebuild with "
                f"autovision.get_wrapped_model('<dinov3 model name>') and "
                f"load_state_dict(torch.load('{out}'))."
            )


DINOV3_PACKAGE = Dinov3Package()

from autovision._models.package_helpers import register_package  # noqa: E402

register_package(DINOV3_PACKAGE)
