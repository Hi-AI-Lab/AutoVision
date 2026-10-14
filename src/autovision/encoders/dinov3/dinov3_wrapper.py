# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv3 wrapper: ViT (CLS + registers) and ConvNeXt variants.

The ViT wrapper reuses the DINOv2 wrapping semantics (pooled feature = CLS
token, spatial features strip the prefix tokens); the ConvNeXt wrapper
exposes the last stage feature map directly. Both operate on
``transformers`` DINOv3 models.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from autovision.encoders.dinov2.dinov2_wrapper import (
    TransformersDinov2Wrapper,
    _backfill_patch_tokens,
    _kept_patch_indices,
)
from autovision.encoders.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)

__all__ = ["TransformersDinov3ConvNextWrapper", "TransformersDinov3ViTWrapper"]


class TransformersDinov3ViTWrapper(TransformersDinov2Wrapper):
    """DINOv3 ViT: identical semantics to the DINOv2 wrapper.

    The HF layout differs (``embeddings`` / ``rope_embeddings`` / ``model``
    (= the encoder stack) / ``norm``; rope is part of the forward signature,
    not an embeddings internal — design doc 17 §4.3.1), so the masked
    multi-scale forward is orchestrated here instead of inherited.
    """

    def supports_masked_forward(self) -> bool:
        """True when the DINOv3 masked orchestration surface is complete.

        Requires the minimal attribute set the masked forward drives:
        ``embeddings``, the encoder stack (``model``, a
        ``DINOv3ViTEncoder`` with ``.layer``), and ``rope_embeddings``
        (design doc 20 §3.2). A transformers-side layout drift degrades to
        a config-time error instead of a mid-training failure.
        """
        hf_model = self._model
        return (
            hasattr(hf_model, "embeddings")
            and hasattr(hf_model, "model")
            and hasattr(hf_model, "rope_embeddings")
            and hasattr(getattr(hf_model, "model"), "layer")
        )

    def forward_multiscale_features_masked(
        self,
        x: Tensor,
        layer_indices: Sequence[int],
        keep_masks: Sequence[Tensor | None],
    ) -> list[dict[str, Any]]:
        """Multi-scale taps with per-image background patch-token dropping.

        DINOv3 recipe (design doc 20 §3.1): identical to the DINOv2 masked
        forward except that the rope table — computed per image from the
        pixel shape — is gathered with the SAME kept indices as the patch
        tokens. That keeps every kept token's original position embedding
        AND keeps ``apply_rotary_pos_emb``'s prefix inference
        (``num_tokens - sin.shape[-2]``) correct on the shortened sequence.
        Images whose mask is ``None`` run the plain dense per-image forward.

        Note (design doc 20 §4, difference V1): with the default
        ``pos_embed_rescale`` rope augmentation active (training mode), the
        per-image loop draws augmentation per image while the dense batch
        path draws once per batch — a declared training-time divergence;
        all bitwise invariants hold in eval mode.
        """
        if not self.supports_masked_forward():
            raise ValueError(
                "Masked multi-scale forward requires the DINOv3 internals "
                "layout (model.embeddings + model.model + "
                "model.rope_embeddings); this backbone exposes a different "
                "one — design doc 20 §3.2."
            )
        layer_indices = list(layer_indices)
        num_blocks = int(self.config.num_hidden_layers)
        for layer_index in layer_indices:
            if not 0 <= layer_index < num_blocks:
                raise ValueError(
                    f"layer_index {layer_index} out of range for a model with "
                    f"{num_blocks} blocks (expected 0..{num_blocks - 1})."
                )
        if len(keep_masks) != x.shape[0]:
            raise ValueError(
                f"Got {len(keep_masks)} keep masks for {x.shape[0]} images."
            )
        # Same prefix formula as _reshape_patch_tokens (CLS + registers).
        num_prefix = 1 + int(getattr(self.config, "num_register_tokens", 0) or 0)
        patch_size = self.patch_size()
        per_tap: list[list[dict[str, Any]]] = [[] for _ in layer_indices]
        for image_index in range(x.shape[0]):
            image = x[image_index : image_index + 1]
            mask = keep_masks[image_index]
            num_h_patches = image.shape[2] // patch_size
            num_w_patches = image.shape[3] // patch_size
            kept: Tensor | None = None
            if mask is not None:
                kept = _kept_patch_indices(
                    mask.to(device=image.device), num_h_patches, num_w_patches
                )
            hidden = self._model.embeddings(pixel_values=image)
            cos, sin = self._model.rope_embeddings(pixel_values=image)
            if kept is not None:
                # A kept sequence runs through gather/refill even when it
                # covers every patch: index_select of all indices reproduces
                # the dense values exactly, keeping the keep-all ≡ dense
                # invariant honest under test. The rope rows are gathered
                # with the same indices — per-position identity preserved,
                # prefix inference on the shortened sequence intact.
                prefix, patches = (
                    hidden[:, :num_prefix],
                    hidden[:, num_prefix:],
                )
                hidden = torch.cat(
                    (prefix, patches.index_select(1, kept)), dim=1
                )
                cos = cos.index_select(0, kept)
                sin = sin.index_select(0, kept)
            encoder_outputs = self._model.model(
                hidden,
                position_embeddings=(cos, sin),
                output_hidden_states=True,
            )
            hidden_states = encoder_outputs.hidden_states
            for tap_index, layer_index in enumerate(layer_indices):
                tap_hidden = hidden_states[layer_index + 1]
                if kept is not None:
                    # Zero-fill the dropped patch positions so the tap
                    # matches the dense (1, prefix + N, D) layout; gradients
                    # flow through index_copy_ into the kept tokens.
                    tap_hidden = _backfill_patch_tokens(
                        tap_hidden,
                        kept,
                        num_prefix,
                        num_h_patches,
                        num_w_patches,
                    )
                features, cls_token = self._reshape_patch_tokens(tap_hidden)
                per_tap[tap_index].append(
                    {"features": features, "cls_token": cls_token}
                )
        return [
            {
                "features": torch.cat(
                    [image_out["features"] for image_out in tap_images], dim=0
                ),
                "cls_token": torch.cat(
                    [image_out["cls_token"] for image_out in tap_images], dim=0
                ),
            }
            for tap_images in per_tap
        ]


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
    from autovision.encoders._model_helpers import adapt_conv_input_channels

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
