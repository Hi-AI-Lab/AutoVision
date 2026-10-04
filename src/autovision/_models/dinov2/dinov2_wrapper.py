# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv2 backbone wrapper on top of ``transformers``.

Per framework policy, DINOv2 is provided through HuggingFace
``transformers`` (``Dinov2Model``) instead of vendored code. Pretrained
weights load ``from_pretrained`` (network required); every model also has a
``-notpretrained`` variant built locally from the config (offline).

Wrapping semantics follow the baseline DINOv2 wrapper: the pooled feature is
the CLS token; spatial features strip the CLS + register tokens.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from torch import Tensor, nn

from autovision._models.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    PatchSize,
)

__all__ = ["TransformersDinov2Wrapper"]


class TransformersDinov2Wrapper(nn.Module, PatchSize):
    """Adapts a ``transformers.Dinov2Model`` to the ``ModelWrapper`` protocol."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model  # transformers models are nn.Modules: register.

    @property
    def hf_model(self) -> nn.Module:
        return self._model

    def get_model(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    # ── ModelWrapper protocol ───────────────────────────────────────────
    def forward_features(
        self, x: Tensor, bool_masked_pos: Tensor | None = None
    ) -> ForwardFeaturesOutput:
        if bool_masked_pos is None:
            outputs = self._model(pixel_values=x)
        else:
            if not getattr(self.config, "use_mask_token", False):
                raise ValueError(
                    "bool_masked_pos requires the HF mask token, which is "
                    "disabled by default; build the model with "
                    "model_args={'use_mask_token': True}."
                )
            # iBOT-style masking: masked patch positions are replaced by the
            # model's learned mask token (HF Dinov2Model embeddings).
            outputs = self._model(pixel_values=x, bool_masked_pos=bool_masked_pos)
        hidden = outputs.last_hidden_state  # (B, 1 + R + N, D)
        features, cls_token = self._reshape_patch_tokens(hidden)
        return {"features": features, "cls_token": cls_token}

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Features after the given 0-based block indices (baseline
        ``get_intermediate_layers`` semantics: patch tokens reshaped to
        (B, D, H, W), no final layernorm, CLS token per tap).

        Implemented via a full forward with ``output_hidden_states=True``;
        cost is identical to a plain forward whenever the last tap is the
        final block (true for all registry tap sets).
        """
        layer_indices = list(layer_indices)  # materialize: iterators would be
        # consumed by the validation loop below, silently returning no taps.
        num_blocks = int(self.config.num_hidden_layers)
        for layer_index in layer_indices:
            if not 0 <= layer_index < num_blocks:
                raise ValueError(
                    f"layer_index {layer_index} out of range for a model with "
                    f"{num_blocks} blocks (expected 0..{num_blocks - 1})."
                )
        outputs = self._model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # (embeddings, block_0, ...)
        results: list[dict[str, Any]] = []
        for layer_index in layer_indices:
            features, cls_token = self._reshape_patch_tokens(
                hidden_states[layer_index + 1]
            )
            results.append({"features": features, "cls_token": cls_token})
        return results

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        # Baseline semantics: DINOv2 pools via the CLS token, not avg pool.
        cls_token = x.get("cls_token")
        if cls_token is None:
            raise ValueError("forward_pool requires 'cls_token' from forward_features.")
        return {"pooled_features": cls_token[..., None, None]}

    def feature_dim(self) -> int:
        return int(self.config.hidden_size)

    def patch_size(self) -> int:
        return int(self.config.patch_size)

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}

    def make_teacher(self) -> "TransformersDinov2Wrapper":
        """Prepare for teacher duty: eval mode and no stochastic dropout.

        Also zeroes DropPath-style stochastic depth: HF ``Dinov2DropPath``
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

    # ── Internal helpers ────────────────────────────────────────────────
    def _reshape_patch_tokens(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Split the prefix tokens off a hidden state and reshape patches.

        Strips the CLS + register tokens and reshapes the patch tokens to a
        square (B, C, H, W) map. Returns ``(features, cls_token)``.
        """
        cls_token = hidden[:, 0]
        num_prefix = 1 + int(getattr(self.config, "num_register_tokens", 0) or 0)
        patches = hidden[:, num_prefix:]  # (B, N, D)
        num_patches = patches.shape[1]
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(
                f"Non-square DINOv2 feature map ({num_patches} patches) is "
                f"not supported."
            )
        channels = self.feature_dim()
        features = (
            patches.transpose(1, 2)
            .reshape(-1, channels, side, side)
            .contiguous()
        )
        return features, cls_token
