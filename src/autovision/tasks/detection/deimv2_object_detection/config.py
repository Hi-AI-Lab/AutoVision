# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DEIMv2 object detection: model configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from autovision.config.base import PydanticConfig
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "DEIMV2_MODEL_REGISTRY",
    "Deimv2ObjectDetectionTrainArgs",
    "resolve_deimv2_variant",
]

# Per-variant metadata from the official Intellindust DEIMv2 configs (COCO
# release): native input resolution, the stride divisor any custom
# resolution must be divisible by (max feat stride 32; the DINOv3 ViT adds
# patch 16, also covered), the official per-variant optimizer recipe, and
# the architecture kwargs for local (untrained) builds — mirroring the
# transformers-format checkpoint configs.
#
# Pretrained weights are the transformers-format conversions of the
# official Intellindust COCO checkpoints (harshaljanjani/*_Transformers on
# the Hub, apache-2.0 cards; the DINOv3-S conversion embeds DINOv3
# backbone weights, which carry the DINOv3 license terms). The official
# Intellindust/* repos ship the original-framework state dict and are NOT
# loadable by ``Deimv2ForObjectDetection`` directly.
DEIMV2_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "deimv2-dinov3-s": dict(
        hf="harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers",
        resolution=640,
        size_divisor=32,
        lr=5e-4,
        lr_backbone_ratio=0.05,  # official: dinov3 2.5e-5 vs 5e-4
        weight_decay=1e-4,
        # backbone out_indices [6, 9, 12] (stage positions over the 13-entry
        # embeddings+blocks sequence -> block outputs {5, 8, 11} 0-based)
        # match the RELEASED Intellindust checkpoint config verbatim
        # (DINOv3STAs.interaction_indexes [5, 8, 11]); the [3, 7, 11] in the
        # GitHub training yml is a stale pre-release value — do not "fix".
        config=dict(
            num_queries=300,
            decoder_layers=4,
            d_model=192,
            encoder_type="hybrid",
            encoder_hidden_dim=192,
            encoder_in_channels=[192, 192, 192],
            feat_strides=[8, 16, 32],
            encoder_ffn_dim=512,
            decoder_in_channels=[192, 192, 192],
            decoder_ffn_dim=512,
            num_feature_levels=3,
            decoder_n_points=[3, 6, 3],
            hidden_expansion=0.34,
            depth_mult=0.67,
            encode_proj_layers=[2],
            encoder_has_trailing_conv=False,
            backbone_config={
                "model_type": "dinov3_vit",
                "hidden_size": 192,
                "num_hidden_layers": 12,
                "num_attention_heads": 3,
                "patch_size": 16,
                "out_indices": [6, 9, 12],
                "out_features": ["stage6", "stage9", "stage12"],
            },
        ),
    ),
    "deimv2-hgnetv2-n": dict(
        hf="harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers",
        resolution=640,
        size_divisor=32,
        lr=8e-4,
        lr_backbone_ratio=0.5,  # official: backbone 4e-4 vs 8e-4
        weight_decay=1e-4,
        config=dict(
            num_queries=300,
            decoder_layers=3,
            d_model=128,
            encoder_type="hybrid",
            encoder_hidden_dim=128,
            encoder_in_channels=[512, 1024],
            feat_strides=[16, 32],
            encoder_ffn_dim=512,
            decoder_in_channels=[128, 128],
            decoder_ffn_dim=512,
            num_feature_levels=2,
            decoder_n_points=[6, 6],
            hidden_expansion=0.34,
            depth_mult=0.5,
            encode_proj_layers=[1],
            encoder_has_trailing_conv=True,
            backbone_config={
                "model_type": "hgnet_v2",
                "out_indices": [3, 4],
                "out_features": ["stage3", "stage4"],
                "depths": [3, 3, 3, 3],
            },
        ),
    ),
    # DEIMv2-L (design doc 22 PR-0 + v1.2 erratum): the original yml
    # (DEIMv2-main/configs/deimv2/deimv2_dinov3_l_coco.yml) loads the REAL
    # DINOv3 ViT-S/16 pretrained checkpoint (dinov3_vits16_pretrain_
    # lvd1689m — hidden 384, 6 heads, 12 layers, 4 register tokens) and
    # projects to hidden_dim 224 via the STA wrapper's 1x1 fusion convs
    # (the wrapper overwrites its embed_dim arg with the ViT's real width
    # — dinov3_adapter.py:103). The serialized config.json's
    # "embed_dim: 224" is the PROJECTED width, not the ViT width. Warm
    # start for the ViT comes from FsdcArgs.backbone_weights (facebook
    # conversion, manifest artifact facebook/dinov3-vits16-pretrain-
    # lvd1689m); the decoder trains fresh (no COCO conversion, PR-0).
    "deimv2-dinov3-l": dict(
        hf=None,
        resolution=640,
        size_divisor=32,
        lr=5e-4,  # official L yml: lr 0.0005, wd 0.000125
        lr_backbone_ratio=0.025,  # official L: dinov3 bucket 1.25e-5 / 5e-4
        weight_decay=1.25e-4,
        config=dict(
            num_queries=300,
            decoder_layers=4,
            d_model=224,
            encoder_type="hybrid",
            encoder_hidden_dim=224,
            encoder_in_channels=[224, 224, 224],
            feat_strides=[8, 16, 32],
            encoder_ffn_dim=896,
            decoder_in_channels=[224, 224, 224],
            decoder_ffn_dim=1792,
            num_feature_levels=3,
            decoder_n_points=[3, 6, 3],
            decoder_attention_heads=8,
            hidden_expansion=1.0,
            depth_mult=1.0,
            encode_proj_layers=[2],
            encoder_has_trailing_conv=False,
            spatial_tuning_adapter_inplanes=32,
            backbone_config={
                "model_type": "dinov3_vit",
                "hidden_size": 384,
                "num_hidden_layers": 12,
                "num_attention_heads": 6,
                "num_register_tokens": 4,
                "intermediate_size": 1536,
                "patch_size": 16,
                "out_indices": [6, 9, 12],
                "out_features": ["stage6", "stage9", "stage12"],
            },
        ),
    ),
    # Hermetic offline variant for the FSDC family's test suite (design
    # doc 22 §7): tiny DINOv3 ViT so ``tests/test_fsdc_native.py`` can
    # build the real HF stack without weights or network. Not trainable
    # in practice; mirrors the vittest pattern of the AV-DETR family.
    "fsdc-vittest": dict(
        hf=None,
        resolution=256,
        size_divisor=32,
        lr=5e-4,
        lr_backbone_ratio=0.05,
        weight_decay=1e-4,
        config=dict(
            num_queries=50,
            decoder_layers=2,
            d_model=64,
            encoder_type="hybrid",
            encoder_hidden_dim=64,
            encoder_in_channels=[64, 64, 64],
            feat_strides=[8, 16, 32],
            encoder_ffn_dim=128,
            decoder_in_channels=[64, 64, 64],
            decoder_ffn_dim=128,
            num_feature_levels=3,
            decoder_n_points=[3, 6, 3],
            decoder_attention_heads=4,
            hidden_expansion=0.34,
            depth_mult=0.34,
            encode_proj_layers=[2],
            encoder_has_trailing_conv=False,
            spatial_tuning_adapter_inplanes=16,
            backbone_config={
                "model_type": "dinov3_vit",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "patch_size": 16,
                "out_indices": [1, 2, 3],
                "out_features": ["stage1", "stage2", "stage3"],
            },
        ),
    ),
    "deimv2-hgnetv2-pico": dict(
        hf="harshaljanjani/DEIMv2_HGNetv2_PICO_COCO_Transformers",
        resolution=640,
        size_divisor=32,
        lr=2.5e-4,
        lr_backbone_ratio=0.05,  # official base recipe: 1.25e-5 vs 2.5e-4
        weight_decay=1.25e-4,
        config=dict(
            num_queries=200,
            decoder_layers=3,
            d_model=112,
            encoder_type="lite",
            encoder_layers=0,  # LiteEncoder has no AIFI
            encoder_hidden_dim=112,
            encoder_in_channels=[512],
            feat_strides=[16],
            encoder_ffn_dim=1024,
            decoder_in_channels=[112, 112],
            decoder_ffn_dim=320,
            num_feature_levels=2,
            decoder_n_points=[4, 2],
            hidden_expansion=0.34,
            depth_mult=0.5,
            use_gateway=False,
            share_bbox_head=True,
            encoder_has_trailing_conv=True,
            backbone_config={
                "model_type": "hgnet_v2",
                "out_indices": [3],
                "out_features": ["stage3"],
                "depths": [3, 3, 3],
            },
        ),
    ),
}


def resolve_deimv2_variant(model: str) -> dict[str, Any]:
    """Registry entry for a variant name (``deimv2-dinov3-s`` etc.)."""
    if model not in DEIMV2_MODEL_REGISTRY:
        supported = ", ".join(sorted(DEIMV2_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown deimv2 detection model '{model}'. Supported: {supported}."
        )
    return DEIMV2_MODEL_REGISTRY[model]


class Deimv2ObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training args for the DEIMv2 detection family (HF transformers).

    The loss (Hungarian matcher + VFL/MAL + L1 + GIoU + D-FINE-style
    FGL/DDF behind DEIMv2's dense one-to-one auxiliary matching) is the one
    inside HF ``Deimv2ForObjectDetection`` — a faithful port of the
    official ``DEIMCriterion`` — so it is consumed, not re-implemented.

    The optimizer defaults follow the official per-variant recipes (AdamW,
    betas 0.9/0.999, clip 0.1, EMA 0.9999 with 1000-update warmup over a
    total batch of 32): the backbone gets a variant-specific LR multiplier
    (HGNetv2 0.5x, DINOv3 0.05x) and norm/bias parameters are exempt from
    weight decay.
    """

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 32  # official total_batch_size
    default_steps: ClassVar[int] = 12_000

    model: str = "deimv2-hgnetv2-n"
    num_classes: int = 80
    # "auto" resolves to the variant's native resolution (640 for every
    # released DEIMv2 checkpoint); custom sizes must be divisible by the
    # max feature stride (32).
    image_size: Literal["auto"] | ImageSizeTuple = "auto"
    load_weights: bool = True  # pretrained COCO checkpoint via the verified loader
    num_select: int = 300  # eval decode width (official top-k)

    # optimizer / schedule — None resolves to the official per-variant
    # recipe in resolve_auto (see DEIMV2_MODEL_REGISTRY).
    lr: float | None = None
    lr_backbone: float | None = None
    weight_decay: float | None = None
    gradient_clip_val: float = 0.1
    #: torch.optim.AdamW's fused single-kernel step on CUDA (design doc 26
    #: W1). Off by default for run comparability — same contract as the
    #: AV-DETR family's flag.
    fused_optimizer: bool = False
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: Literal["auto", "step", "cosine"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    lr_milestones: tuple[float, ...] = (0.75,)  # fractions of total steps
    lr_gamma: float = 0.1

    # EMA — official ModelEMA: decay 0.9999 with a 1000-update warmup.
    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 1000

    eval_max_dets: int = 100

    def resolve_variant(self) -> dict[str, Any]:
        return resolve_deimv2_variant(self.model)

    def resolve_image_size(self) -> None:
        """Pin ``image_size`` to a concrete (h, w) tuple ("auto" -> variant
        native resolution) and validate the stride divisibility."""
        info = self.resolve_variant()
        if self.image_size == "auto":
            resolution = int(info["resolution"])
            self.image_size = (resolution, resolution)
            return
        height, width = self.image_size
        divisor = int(info["size_divisor"])
        for name, value in (("height", height), ("width", width)):
            if value % divisor:
                suggestion = max(divisor, round(value / divisor) * divisor)
                raise ValueError(
                    f"deimv2 image_size {self.image_size} is invalid: {name} "
                    f"{value} is not divisible by the max feature stride "
                    f"({divisor} for '{self.model}'). Try {suggestion}."
                )

    def resolve_auto(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        self.resolve_image_size()
        info = self.resolve_variant()
        if self.lr is None:
            self.lr = float(info["lr"])
        if self.lr_backbone is None:
            self.lr_backbone = self.lr * float(info["lr_backbone_ratio"])
        if self.weight_decay is None:
            self.weight_decay = float(info["weight_decay"])
        if self.lr_scheduler == "auto":
            self.lr_scheduler = "cosine"
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = max(100, min(500, total_steps // 10))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
