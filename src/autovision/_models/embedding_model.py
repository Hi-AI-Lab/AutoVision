# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Embedding model: a backbone wrapper plus an optional linear embed head.

This is the pretraining-facing student model. Kept separate from
``ModelWrapper`` deliberately: the wrapper adapts a backbone, the embedding
model is what methods train and what gets exported.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from autovision._models.model_wrapper import ModelWrapper

__all__ = ["EmbeddingModel"]


class EmbeddingModel(nn.Module):
    def __init__(self, wrapped_model: ModelWrapper, embed_dim: int | None = None) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model
        feature_dim = wrapped_model.feature_dim()
        if embed_dim is None or embed_dim == feature_dim:
            self.embed_head: nn.Module = nn.Identity()
        else:
            self.embed_head = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)

    @property
    def embed_dim(self) -> int:
        if isinstance(self.embed_head, nn.Identity):
            return self.wrapped_model.feature_dim()
        return int(self.embed_head.out_channels)

    def forward(self, x: Tensor, pool: bool = True, **kwargs: Any) -> Tensor:
        features = self.wrapped_model.forward_features(x, **kwargs)
        if pool:
            out = self.wrapped_model.forward_pool(features)["pooled_features"]
        else:
            out = features["features"]
        return self.embed_head(out)

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Any
    ) -> list[dict[str, Tensor]]:
        """Intermediate-layer taps via the wrapped model.

        Requires a backbone wrapper implementing
        ``forward_multiscale_features`` (ViT backbones); each tap's
        ``features`` map passes through the embed head.
        """
        forward = getattr(self.wrapped_model, "forward_multiscale_features", None)
        if forward is None:
            raise ValueError(
                f"Backbone wrapper '{type(self.wrapped_model).__name__}' does "
                "not implement forward_multiscale_features; intermediate-layer "
                "methods require a ViT backbone (e.g. dinov2/*, dinov3/*)."
            )
        outputs = forward(x, layer_indices)
        return [
            {
                key: self.embed_head(value) if key == "features" else value
                for key, value in output.items()
            }
            for output in outputs
        ]
