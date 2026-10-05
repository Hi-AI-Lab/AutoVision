# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv3 wrapper: ViT (CLS + registers) and ConvNeXt variants.

The ViT wrapper reuses the DINOv2 wrapping semantics (pooled feature = CLS
token, spatial features strip the prefix tokens); the ConvNeXt wrapper
exposes the last stage feature map directly. Both operate on
``transformers`` DINOv3 models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from autovision._models.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision._models.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)

__all__ = ["TransformersDinov3ConvNextWrapper", "TransformersDinov3ViTWrapper"]


class TransformersDinov3ViTWrapper(TransformersDinov2Wrapper):
    """DINOv3 ViT: identical semantics to the DINOv2 wrapper."""


class TransformersDinov3ConvNextWrapper(nn.Module):
    """DINOv3 ConvNeXt: the last stage feature map is the spatial feature."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    @property
    def hf_model(self) -> nn.Module:
        return self._model

    def get_model(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        outputs = self._model(pixel_values=x)
        return {"features": outputs.last_hidden_state}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}

    def feature_dim(self) -> int:
        return int(self.config.hidden_sizes[-1])

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "layernorm"}

    def make_teacher(self) -> "TransformersDinov3ConvNextWrapper":
        """Prepare for teacher duty: eval mode and no stochastic dropout.

        Also zeroes DropPath-style stochastic depth (HF ``Dinov3ConvnextDropPath``
        subclasses ``nn.Module``, not ``nn.Dropout``), so the guarantee survives
        a later ``train()`` call.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
        return self


def _adapt_convnext_stem_channels(model: nn.Module, num_input_channels: int) -> None:
    """Rebuild the ConvNeXt stem conv for a different channel count."""
    from autovision._models._model_helpers import adapt_conv_input_channels

    stem = model.embeddings.patch_embeddings  # Conv2d in transformers ConvNeXt
    if stem.in_channels == num_input_channels:
        return
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
