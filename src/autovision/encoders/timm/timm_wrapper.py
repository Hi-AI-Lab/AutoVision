# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""timm model wrapper: adapts any ``timm`` backbone to the EncoderProtocol protocol.

timm models are heterogeneous: ``forward_features`` may return NCHW feature
maps or NLC token sequences including class/prefix tokens depending on the
architecture family. The wrapper normalizes all of them to an NCHW feature
map (dropping prefix tokens and reshaping token sequences by their square
grid assumption — the same strategy as the baseline).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor, nn

from autovision.encoders.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)

if TYPE_CHECKING:
    pass

__all__ = ["TIMMModelWrapper", "architecture_info_for_name"]

logger = logging.getLogger("autovision")

# Architecture name prefix -> ArchitectureInfo (baseline table, verbatim
# semantics). Names come from ``model.pretrained_cfg["architecture"]``.
_TIMM_ARCH_NAME_PREFIXES: list[tuple[str, ArchitectureInfo]] = [
    # Transformers
    ("vit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("deit3_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("deit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("swinv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("swin_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beitv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beit3_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("eva02_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("eva_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("xcit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("flexivit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("tiny_vit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("cait_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("crossvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("pit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("tnt_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("twins_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("nest_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("sequencer2d_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("aimv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("convit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("gmixer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("gmlp_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("hieradet_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("hiera_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("mixer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("mvitv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("naflexvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("pvt_v2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("resmlp_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("sam2_hiera_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("samvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("visformer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("vitamin_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("volo_", {"model_type": "transformer", "norm_type": "layernorm"}),
    # Hybrids (LayerNorm)
    ("maxvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coat_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("cvt_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("levit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("cmt_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mix_transformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("caformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coatnet_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coatnext_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("davit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("edgenext_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientformerv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("fastvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("gcvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("maxxvitv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("maxxvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mobilevitv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mobilevit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("nextvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("repvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("shvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("swiftformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    # Hybrids (BatchNorm)
    ("bat_", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("botnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("eca_botnext", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("eca_halonext", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("halo", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("lambda_resnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("lamhalobotnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("sebotnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("sehalonet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    # Conv + LayerNorm
    ("convnextv2_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convnext_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convmixer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("poolformerv2_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("poolformer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convformer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("fasternet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("focalnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("inception_next_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("mambaout_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("rdnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("starnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    # Conv + BatchNorm
    ("resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("resnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("resnest", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("seresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("seresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetx_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnety_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetv_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetz_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("efficientnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("efficientnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_efficientnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv5_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv4_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv1_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenet_edgetpu", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobileone_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("shufflenet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("densenet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dpn", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dm_nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nf_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("vgg", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("squeezenet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tresnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hardcorenas_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cs3", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("csat", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspdarknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("darknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dla", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_resnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ecaresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ecaresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ese_vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("fbnetc_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("fbnetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gc_efficientnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gcresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gcresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gernet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ghostnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hgnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hgnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hrnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("inception_resnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("inception_v", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("lcnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("legacy_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mixnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nasnetalarge", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("pnasnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("repghostnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("repvgg_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("res2net", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("res2next", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("rexnetr_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("rexnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("sedarknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("selecsls", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("semnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("senet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("skresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("skresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("spnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_mixnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_mobilenetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tinynet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("wide_resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("xception", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    # timm-internal test models (mixed architectures).
    ("test_", {"model_type": "transformer", "norm_type": "layernorm"}),
]


def architecture_info_for_name(arch_name: str) -> ArchitectureInfo:
    """Resolve the ArchitectureInfo for a timm architecture name (lowercase)."""
    for prefix, info in _TIMM_ARCH_NAME_PREFIXES:
        if arch_name.startswith(prefix):
            return info
    logger.warning(
        "Could not infer architecture info for timm model '%s'; classifying "
        "as transformer (LayerNorm) — set weight_decay explicitly if this is "
        "wrong.", arch_name,
    )
    return {"model_type": "transformer", "norm_type": "layernorm"}


def _drop_prefix_tokens(model: nn.Module, x: Tensor) -> Tensor:
    """Remove class/prefix tokens from an NLC token sequence."""
    if x.ndim != 3:
        return x
    num_prefix = getattr(model, "num_prefix_tokens", None)
    if num_prefix is None:
        num_prefix = 1 if hasattr(model, "cls_token") else 0
    return x[:, num_prefix:]


def _to_nchw(x: Tensor) -> Tensor:
    """Reshape an NLC token sequence into NCHW (square grid assumption)."""
    if x.ndim != 3:
        return x
    batch, length, channels = x.shape
    side = int(length**0.5)
    return (
        x.reshape(batch, side, side, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _get_forward_features_fn(model: nn.Module) -> Callable[[nn.Module, Tensor], Tensor]:
    """Pick the most reliable NCHW feature extraction path for the model."""
    if hasattr(model, "forward_intermediates"):  # timm >= 1.0
        def via_forward_intermediates(module: nn.Module, x: Tensor) -> Tensor:
            intermediates = module.forward_intermediates(
                x, indices=1, output_fmt="NCHW", intermediates_only=True, norm=True,
            )
            return intermediates[0]

        return via_forward_intermediates
    if hasattr(model, "get_intermediate_layers"):  # timm 0.9.x - 1.0
        def via_get_intermediate_layers(module: nn.Module, x: Tensor) -> Tensor:
            intermediates = module.get_intermediate_layers(x, n=1, reshape=True, norm=True)
            return intermediates[0]

        return via_get_intermediate_layers

    def via_forward_features(module: nn.Module, x: Tensor) -> Tensor:
        features = module.forward_features(x)
        features = _drop_prefix_tokens(module, features)
        return _to_nchw(features)

    return via_forward_features


def _get_pool_layer(model: nn.Module) -> nn.Module:
    """Locate the model's global pooling module (timm stores it inconsistently)."""
    if hasattr(model, "global_pool") and callable(model.global_pool):
        return model.global_pool
    if (
        hasattr(model, "head")
        and hasattr(model.head, "global_pool")
        and callable(model.head.global_pool)
    ):
        return model.head.global_pool
    logger.warning(
        "Could not find a pooling layer on the timm model; defaulting to "
        "AdaptiveAvgPool2d."
    )
    return nn.AdaptiveAvgPool2d((1, 1))


class TIMMModelWrapper(nn.Module):
    """EncoderProtocol adapter for timm backbones."""

    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "forward_features"):
            raise ValueError("A timm model must expose 'forward_features'.")
        if not hasattr(model, "num_features"):
            raise ValueError("A timm model must expose 'num_features'.")
        super().__init__()
        self._model = model
        self._pool = _get_pool_layer(model)
        self._forward_features_fn = _get_forward_features_fn(model)

    def get_model(self) -> nn.Module:
        return self._model

    def feature_dim(self) -> int:
        return int(self._model.num_features)

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._forward_features_fn(self._model, x)}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        pooled = self._pool(x["features"])
        while pooled.ndim < 4:
            pooled = pooled.unsqueeze(-1)
        return {"pooled_features": pooled}

    def architecture_info(self) -> ArchitectureInfo:
        arch_name = ""
        pretrained_cfg = getattr(self._model, "pretrained_cfg", None)
        if isinstance(pretrained_cfg, dict):
            arch_name = str(pretrained_cfg.get("architecture", "")).lower()
        return architecture_info_for_name(arch_name)
