# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RF-DETR wrapper on top of ``transformers``.

Per framework policy (see the DINOv2/DINOv3 packages), RF-DETR is provided
through HuggingFace ``transformers`` (``RfDetrModel``) instead of the
Roboflow ``rfdetr`` package. The HF model is a plain ``nn.Module`` — no
non-module bridging — and its DINOv2 backbone exposes multi-scale feature
maps (``BackboneOutput.feature_maps``, one per ``out_features`` stage).

Pretraining features come from the backbone only: the last feature map is
the spatial feature. The encoder/decoder stay in the module tree so state
dicts and exports remain full-model, but receive no gradients from feature
extraction (the engine's ``find_unused_parameters=True`` covers DDP).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from autovision._models.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    PatchSize,
)

__all__ = ["RFDETRModelWrapper"]


class RFDETRModelWrapper(nn.Module, PatchSize):
    """Adapts a ``transformers`` RF-DETR model to the ``ModelWrapper`` protocol."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError(
                f"RFDETRModelWrapper requires a transformers RF-DETR model "
                f"with a '.backbone' attribute, got '{type(model).__name__}'."
            )
        self._model = model  # transformers models are nn.Modules: register.

    @property
    def hf_model(self) -> nn.Module:
        return self._model

    def get_model(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    @property
    def backbone(self) -> nn.Module:
        return self._model.backbone

    # ── ModelWrapper protocol ───────────────────────────────────────────
    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        outputs = self.backbone(pixel_values=x)
        features = outputs.feature_maps[-1]
        if features.dim() != 4:
            raise ValueError(
                "Expected 4D (B, C, H, W) backbone feature maps (build the "
                f"backbone with reshape_hidden_states=True); got shape "
                f"{tuple(features.shape)}."
            )
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}

    def feature_dim(self) -> int:
        backbone_config = getattr(self.config, "backbone_config", None)
        hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Cannot determine the RF-DETR feature dim: the model config "
                "has no backbone_config.hidden_size."
            )
        return int(hidden_size)

    def patch_size(self) -> int:
        return int(self.config.backbone_config.patch_size)

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}

    def make_teacher(self) -> "RFDETRModelWrapper":
        """Prepare for teacher duty: eval mode and no stochastic dropout.

        Also zeroes DropPath-style stochastic depth: the HF DINOv2 drop-path
        subclasses ``nn.Module`` (attribute ``drop_prob``), not ``nn.Dropout``,
        so ``eval()`` alone would only keep it dormant and a later ``train()``
        call would silently re-enable it.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
        return self
