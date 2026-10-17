# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT model construction on top of ``transformers``.

Per framework policy the EOMT (Encoder-only Mask Transformer) is provided
through HuggingFace ``transformers`` (``EomtForUniversalSegmentation``)
instead of vendored code: the interaction blocks, Hungarian matcher,
mask/dice/CE criterion and scale blocks all live in the HF implementation.
This module only adds construction defaults and the class-head transfer
hook used when the target dataset has a different label/query layout.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import (
    EomtConfig,
    EomtDinov3Config,
    EomtDinov3ForUniversalSegmentation,
    EomtForUniversalSegmentation,
)

__all__ = ["build_eomt_model", "adapt_eomt_head", "ROPE_PERIODS_KEY"]

#: Reserved state-dict key for a checkpoint's RoPE periods. The DINOv3 task
#: model pops it before the strict HF load and installs ``1 / periods`` on the
#: HF rope module (``inv_freq`` is non-persistent and cannot be loaded through
#: ``load_state_dict``). Keeping the trained buffer authoritative avoids the
#: ~1e-3 inverse-frequency drift of recomputing from ``rope_theta``.
ROPE_PERIODS_KEY = "autovision.rope_periods"

# HF-hosted EOMT checkpoints used when ``load_weights`` is set without an
# explicit ``weight_source``. The DINOv3 id is the checkpoint family used by
# the official transformers EoMT-DINOv3 docs (config defaults mirror
# tue-mps/coco_panoptic_eomt_large_640_dinov3).
DEFAULT_DINOV2_WEIGHT_SOURCE = "tue-benchmark/eomt-dinov2-base"
DEFAULT_DINOV3_WEIGHT_SOURCE = "tue-mps/eomt-dinov3-coco-panoptic-base-640"


def build_eomt_model(
    num_labels: int,
    *,
    variant: Literal["dinov2", "dinov3"] = "dinov2",
    hidden_size: int = 384,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 6,
    num_queries: int = 100,
    num_blocks: int = 4,
    num_upscale_blocks: int = 2,
    image_size: int = 512,
    patch_size: int = 16,
    mlp_ratio: int = 4,
    use_swiglu_ffn: bool = False,
    use_gated_mlp: bool = False,
    drop_path_rate: float = 0.0,
    layer_norm_eps: float = 1e-6,
    load_weights: bool = False,
    weight_source: str | None = None,
) -> EomtForUniversalSegmentation | EomtDinov3ForUniversalSegmentation:
    """Build the EOMT universal segmentation model.

    With ``load_weights`` the model initializes from ``weight_source`` (a
    HF hub id or local path) and the class head transfers to ``num_labels``;
    otherwise the model is built locally from the config (offline,
    ``-notpretrained`` semantics).

    ``variant="dinov3"`` builds the HF ``EomtDinov3*`` classes (RoPE-based,
    storage tokens, ``use_gated_mlp`` instead of ``use_swiglu_ffn``); the
    FFN width is derived as ``mlp_ratio * hidden_size`` to keep the ratio
    knob shared across variants. ``use_swiglu_ffn`` is ignored for the
    DINOv3 variant.

    ``layer_norm_eps`` is applied to every ``nn.LayerNorm`` in the model,
    including the scale block's LayerNorm2d. HF's DINOv3 scale block
    hardcodes 1e-6 there instead of reading ``config.layer_norm_eps``;
    DINOv3 EoMT checkpoints trained with ``layernormbf16`` use 1e-5.
    """
    id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
    if variant not in ("dinov2", "dinov3"):
        raise ValueError(f"Unknown EOMT variant: '{variant}'.")
    if variant == "dinov3":
        config = EomtDinov3Config(
            id2label=id2label,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_queries=num_queries,
            num_blocks=num_blocks,
            num_upscale_blocks=num_upscale_blocks,
            image_size=image_size,
            patch_size=patch_size,
            intermediate_size=mlp_ratio * hidden_size,
            use_gated_mlp=use_gated_mlp,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
        )
        if load_weights:
            model = EomtDinov3ForUniversalSegmentation.from_pretrained(
                weight_source or DEFAULT_DINOV3_WEIGHT_SOURCE, config=config
            )
        else:
            model = EomtDinov3ForUniversalSegmentation(config)
        _set_layer_norm_eps(model, layer_norm_eps)
        return model
    config = EomtConfig(
        id2label=id2label,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_queries=num_queries,
        num_blocks=num_blocks,
        num_upscale_blocks=num_upscale_blocks,
        image_size=image_size,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        use_swiglu_ffn=use_swiglu_ffn,
        drop_path_rate=drop_path_rate,
        layer_norm_eps=layer_norm_eps,
    )
    if load_weights:
        model = EomtForUniversalSegmentation.from_pretrained(
            weight_source or DEFAULT_DINOV2_WEIGHT_SOURCE, config=config
        )
    else:
        model = EomtForUniversalSegmentation(config)
    _set_layer_norm_eps(model, layer_norm_eps)
    return model


def _set_layer_norm_eps(
    model: EomtForUniversalSegmentation | EomtDinov3ForUniversalSegmentation,
    eps: float,
) -> None:
    """Apply one LayerNorm epsilon to every norm in the model.

    ``EomtDinov3ScaleLayer`` constructs its ``LayerNorm2d`` with the class
    default instead of ``config.layer_norm_eps``, so setting the config
    alone would leave the upscale path on a different epsilon.
    """
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = float(eps)


@torch.no_grad()
def adapt_eomt_head(
    model: EomtForUniversalSegmentation, num_labels: int, num_queries: int
) -> None:
    """Class-head transfer pre-hook: rebuild the class predictor and the
    query embeddings (plus the criterion's empty-class weight) whenever
    the loaded checkpoint's head layout does not match the target dataset.

    Only the replaced modules are (re-)initialized; the backbone keeps its
    pretrained weights.
    """
    config = model.config
    if config.num_labels != num_labels:
        old = model.class_predictor
        model.class_predictor = nn.Linear(config.hidden_size, num_labels + 1)
        model._init_weights(model.class_predictor)
        config.num_labels = num_labels
        criterion = model.criterion
        criterion.num_labels = num_labels
        empty_weight = torch.ones(num_labels + 1)
        empty_weight[-1] = criterion.eos_coef
        criterion.empty_weight = empty_weight.to(old.weight.device)
    if config.num_queries != num_queries:
        model.query = nn.Embedding(num_queries, config.hidden_size)
        model._init_weights(model.query)
        config.num_queries = num_queries
