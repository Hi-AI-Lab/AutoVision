# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RT-DETR wrapper for raw ``nn.Module``\ s with a ``.backbone`` attribute.

This is an *internal* adapter (no package, no model list): it wraps a bare
detection module whose ``backbone`` returns a list of multi-scale feature
maps and exposes ``out_channels``. Used by task-model components and
available to users via ``autovision._model_wrappers.RTDETRModelWrapper``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autovision._models.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)

__all__ = ["RTDETRModelWrapper"]


class RTDETRModelWrapper(nn.Module):
    """Adapts a raw RT-DETR-like module (with ``.backbone``) to ``ModelWrapper``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError(
                f"RTDETRModelWrapper requires a model with a '.backbone' "
                f"attribute, got '{type(model).__name__}'."
            )
        out_channels = getattr(backbone, "out_channels", None)
        if out_channels is None:
            raise ValueError(
                "The model backbone must expose 'out_channels' "
                f"(list of per-scale channels), got '{type(backbone).__name__}'."
            )
        self._model = model

    def get_model(self) -> nn.Module:
        return self._model

    @property
    def backbone(self) -> nn.Module:
        return self._model.backbone

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}

    def feature_dim(self) -> int:
        return int(self.backbone.out_channels[-1])

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}
