# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""LT-DETR object detection: model configuration and variant registry.

Model names follow ``ltdetr/<backbone>`` where ``<backbone>`` resolves
against the backbone packages (``dinov2``, ``dinov3``, ``edgecrafter``),
e.g. ``ltdetr/dinov2-vits14`` or ``ltdetr/dinov3-convnext-tiny``. All
DINOv2 sizes accept the package suffixes (``-noreg``, ``-notpretrained``).
``ltdetr/dinov2-vittest-notpretrained`` is a tiny offline test variant
(mirrors the baseline ``ViTTest``).

Variants mirror the baseline registry (lightly-train 0.17.0):

- **DINOv2 ViT** (patch 14): NoSTA wrapper — no spatial-prior fusion, the
  semantic taps are projected straight to ``hidden_dim``.
- **DINOv3 ViT** (patch 16, incl. the tiny ``vitt16``/``vitt16plus``
  distillations): STA wrapper with ``conv_inplane = factor × patch_size``.
- **DINOv3 ConvNeXt**: last-three-stage taps, whole backbone at backbone LR.
- **EdgeCrafter ECViT** (patch 16): pass-through wrapper; these variants
  default to the D-FINE decoder (baseline "v2" family). The published
  ``-eupe`` names are aliases of the same architectures.

Encoder/decoder recipes (hidden dims, FFN widths, expansion/depth
multipliers, sampling points, layer counts) are carried per variant in
:class:`LTDETRVariantSpec` so the assembled network matches the baseline
configs instead of a single global default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from autovision._configs.base import PydanticConfig
from autovision._task_models.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "LTDETR_MODEL_REGISTRY",
    "LTDETRVariantSpec",
    "LTDETRObjectDetectionTrainArgs",
    "list_ltdetr_model_names",
    "resolve_ltdetr_variant",
]


@dataclass(frozen=True)
class LTDETRVariantSpec:
    """Static per-variant architecture recipe (baseline-faithful)."""

    backbone_package: str  # "dinov2" | "dinov3" | "edgecrafter"
    backbone_name: str  # name inside the backbone package (suffixes allowed)
    family: Literal["vit", "convnext", "ecvit"]
    patch_size: int
    #: 0-based ViT block taps; unused for convnext (stage taps). ECViT
    #: averages two taps, so the tuple length is family-dependent.
    interaction_indexes: tuple[int, ...]
    #: CNN spatial-prior fusion (baseline: on for DINOv3 ViT, off for DINOv2).
    use_sta: bool
    #: STA channel budget; baseline rule ``factor × patch_size``.
    conv_inplane: int
    #: Shared projection target: encoder hidden, decoder hidden, and the
    #: backbone wrapper's 1x1 projection all run at this width.
    hidden_dim: int
    #: HybridEncoder transformer FFN width.
    enc_dim_feedforward: int
    enc_expansion: float = 1.0
    enc_depth_mult: float = 1.0
    #: Decoder depth (per variant; baseline uses 4 for ViT, 3/4/6 for CNN).
    dec_layers: int = 4
    #: Decoder FFN width (baseline rtdetrv2 decoder default is 1024).
    dec_dim_feedforward: int = 1024
    #: Deformable-attention sampling points per pyramid level.
    dec_num_points: tuple[int, ...] = (3, 6, 3)
    dec_num_heads: int = 8
    num_queries: int = 300
    project_features: bool = True
    #: convnext/vit dinov3 variants need transformers to ship DINOv3.
    requires_dinov3: bool = False
    #: Default decoder; baseline "v2" (ECViT) variants ship D-FINE.
    decoder_default: Literal["rtdetrv2", "dfine"] = "rtdetrv2"
    #: Per-variant default for ``backbone_lr_factor`` ("auto" resolution).
    backbone_lr_factor: float = 0.05


# Baseline ViT recipes. conv_inplane = factor × patch_size, per the
# LTDETRRTDETRBackboneWrapperConfig table (ViTSmall factor 2, Base/Large/
# Giant factor 4, ViTTiny(+Plus) factor 1).
_VIT_SIZES: dict[str, dict[str, Any]] = {
    "vits14": dict(
        interaction_indexes=(5, 8, 11),
        conv_inplane_factor=2,
        hidden_dim=224,
        enc_dim_feedforward=896,
        dec_dim_feedforward=1792,
    ),
    "vitb14": dict(
        interaction_indexes=(5, 8, 11),
        conv_inplane_factor=4,
        hidden_dim=768,
        enc_dim_feedforward=3072,
        dec_dim_feedforward=6144,
    ),
    "vitl14": dict(
        interaction_indexes=(11, 17, 23),
        conv_inplane_factor=4,
        hidden_dim=1024,
        enc_dim_feedforward=4096,
        dec_dim_feedforward=8192,
    ),
    "vitg14": dict(
        interaction_indexes=(19, 29, 39),
        conv_inplane_factor=4,
        hidden_dim=1536,
        enc_dim_feedforward=6144,
        dec_dim_feedforward=12288,
    ),
}

# DINOv3 ViT recipes (STA wrapper, patch 16). vitt16/vitt16plus are the
# tiny distillations: expansion/depth_mult shrink the HybridEncoder CNN.
_DINOV3_VIT_SIZES: dict[str, dict[str, Any]] = {
    "vitt16": dict(
        interaction_indexes=(3, 7, 11),
        conv_inplane_factor=1,
        hidden_dim=192,
        enc_dim_feedforward=512,
        enc_expansion=0.34,
        enc_depth_mult=0.67,
        dec_dim_feedforward=512,
    ),
    "vitt16plus": dict(
        interaction_indexes=(3, 7, 11),
        conv_inplane_factor=1,
        hidden_dim=256,
        enc_dim_feedforward=512,
        enc_expansion=0.67,
        enc_depth_mult=1.0,
        dec_dim_feedforward=512,
    ),
    "vits16": dict(
        interaction_indexes=(5, 8, 11),
        conv_inplane_factor=2,
        hidden_dim=224,
        enc_dim_feedforward=896,
        dec_dim_feedforward=1792,
    ),
    "vitb16": dict(
        interaction_indexes=(5, 8, 11),
        conv_inplane_factor=4,
        hidden_dim=768,
        enc_dim_feedforward=3072,
        dec_dim_feedforward=6144,
    ),
    "vitl16": dict(
        interaction_indexes=(11, 17, 23),
        conv_inplane_factor=4,
        hidden_dim=1024,
        enc_dim_feedforward=4096,
        dec_dim_feedforward=8192,
    ),
}

_CONVNEXT_SIZES: dict[str, dict[str, Any]] = {
    # Baseline CNN recipes: hidden 384 across sizes; decoder depth grows
    # tiny/small 3 -> base 4 -> large 6. Decoder FFN keeps the rtdetrv2
    # default (1024); sampling points stay (4, 4, 4).
    "convnext-tiny": dict(hidden_dim=384, dec_layers=3),
    "convnext-small": dict(hidden_dim=384, dec_layers=3),
    "convnext-base": dict(hidden_dim=384, dec_layers=4),
    "convnext-large": dict(hidden_dim=384, dec_layers=6),
}

_ECVIT_SIZES: dict[str, dict[str, Any]] = {
    "ecvitt": dict(
        hidden_dim=192,
        enc_dim_feedforward=512,
        enc_expansion=0.34,
        enc_depth_mult=0.67,
        dec_dim_feedforward=512,
        backbone_lr_factor=0.05,
    ),
    "ecvittplus": dict(
        hidden_dim=256,
        enc_dim_feedforward=512,
        enc_expansion=0.67,
        enc_depth_mult=1.0,
        dec_dim_feedforward=512,
        backbone_lr_factor=0.0025,
    ),
    "ecvits": dict(
        hidden_dim=256,
        enc_dim_feedforward=512,
        enc_expansion=0.67,
        enc_depth_mult=1.0,
        dec_dim_feedforward=512,
        backbone_lr_factor=0.0025,
    ),
    "ecvitsplus": dict(
        hidden_dim=256,
        enc_dim_feedforward=512,
        enc_expansion=0.67,
        enc_depth_mult=1.0,
        dec_dim_feedforward=512,
        backbone_lr_factor=0.0025,
    ),
}


def _build_registry() -> dict[str, LTDETRVariantSpec]:
    registry: dict[str, LTDETRVariantSpec] = {}

    # DINOv2 ViT variants (NoSTA per baseline): base + package suffixes
    # (-noreg, -notpretrained).
    for size, info in _VIT_SIZES.items():
        for suffix in ("", "-noreg", "-notpretrained"):
            registry[f"dinov2-{size}{suffix}"] = LTDETRVariantSpec(
                backbone_package="dinov2",
                backbone_name=f"{size}{suffix}",
                family="vit",
                patch_size=14,
                interaction_indexes=info["interaction_indexes"],
                use_sta=False,
                conv_inplane=info["conv_inplane_factor"] * 14,
                hidden_dim=info["hidden_dim"],
                enc_dim_feedforward=info["enc_dim_feedforward"],
                enc_expansion=info.get("enc_expansion", 1.0),
                enc_depth_mult=info.get("enc_depth_mult", 1.0),
                dec_dim_feedforward=info["dec_dim_feedforward"],
            )
    # Tiny offline test variant (baseline ViTTest analogue): built directly
    # from a local HF config, random init. Intentionally keeps the STA
    # fusion ACTIVE (unlike every real DINOv2 variant) so the fusion code
    # path stays covered by the offline test suite.
    registry["dinov2-vittest-notpretrained"] = LTDETRVariantSpec(
        backbone_package="dinov2",
        backbone_name="vittest-notpretrained",
        family="vit",
        patch_size=14,
        interaction_indexes=(1, 2, 3),
        use_sta=True,
        conv_inplane=16,
        hidden_dim=64,
        enc_dim_feedforward=512,
        dec_layers=2,
        dec_dim_feedforward=512,
    )
    # DINOv3 ViT variants (STA wrapper; usable once transformers ships
    # DINOv3). vitt16/vitt16plus have no public HF checkpoint: only their
    # -notpretrained builds resolve (see the dinov3 package).
    for size, info in _DINOV3_VIT_SIZES.items():
        for suffix in ("", "-notpretrained"):
            name = f"dinov3-{size}{suffix}"
            registry[name] = LTDETRVariantSpec(
                backbone_package="dinov3",
                backbone_name=f"{size}{suffix}",
                family="vit",
                patch_size=16,
                interaction_indexes=info["interaction_indexes"],
                use_sta=True,
                conv_inplane=info["conv_inplane_factor"] * 16,
                hidden_dim=info["hidden_dim"],
                enc_dim_feedforward=info["enc_dim_feedforward"],
                enc_expansion=info.get("enc_expansion", 1.0),
                enc_depth_mult=info.get("enc_depth_mult", 1.0),
                dec_dim_feedforward=info["dec_dim_feedforward"],
                requires_dinov3=True,
            )
    # DINOv3 ConvNeXt variants (usable once transformers ships DINOv3).
    for size, info in _CONVNEXT_SIZES.items():
        registry[f"dinov3-{size}"] = LTDETRVariantSpec(
            backbone_package="dinov3",
            backbone_name=size,
            family="convnext",
            patch_size=16,
            interaction_indexes=(1, 2, 3),  # unused for convnext (stage taps)
            use_sta=False,
            conv_inplane=0,
            hidden_dim=info["hidden_dim"],
            enc_dim_feedforward=2048,
            dec_layers=info["dec_layers"],
            dec_num_points=(4, 4, 4),
            requires_dinov3=True,
        )
    # EdgeCrafter ECViT variants (baseline "v2"): pass-through backbone,
    # D-FINE decoder by default.
    for size, info in _ECVIT_SIZES.items():
        registry[f"edgecrafter-{size}"] = LTDETRVariantSpec(
            backbone_package="edgecrafter",
            backbone_name=size,
            family="ecvit",
            patch_size=16,
            interaction_indexes=(10, 11),  # averaged by the wrapper itself
            use_sta=False,
            conv_inplane=0,
            hidden_dim=info["hidden_dim"],
            enc_dim_feedforward=info["enc_dim_feedforward"],
            enc_expansion=info["enc_expansion"],
            enc_depth_mult=info["enc_depth_mult"],
            dec_dim_feedforward=info["dec_dim_feedforward"],
            decoder_default="dfine",
            backbone_lr_factor=info["backbone_lr_factor"],
        )

    # Published "-eupe" names are architecture aliases (baseline registers
    # them against the same configs).
    aliases = {
        "dinov3-convnext-tiny-eupe": "dinov3-convnext-tiny",
        "dinov3-convnext-small-eupe": "dinov3-convnext-small",
        "dinov3-convnext-base-eupe": "dinov3-convnext-base",
        "dinov3-vitt16-eupe": "dinov3-vitt16",
        "dinov3-vits16-eupe": "dinov3-vits16",
        "dinov3-vitb16-eupe": "dinov3-vitb16",
    }
    for alias, target in aliases.items():
        registry[alias] = registry[target]
    return registry


LTDETR_MODEL_REGISTRY: dict[str, LTDETRVariantSpec] = _build_registry()


def resolve_ltdetr_variant(name: str) -> tuple[str, LTDETRVariantSpec]:
    """Split ``ltdetr/<backbone>`` into (backbone key, spec).

    Idempotent: accepts both ``ltdetr/<backbone>`` (public model name) and
    ``<backbone>`` (init-args form, e.g. from exported checkpoints).
    """
    prefix = "ltdetr/"
    key = name[len(prefix) :] if name.startswith(prefix) else name
    if not key or key.startswith(prefix) or "/" in key:
        raise ValueError(
            f"Unknown LT-DETR model: '{name}' (expected '{prefix}<backbone>')."
        )
    spec = LTDETR_MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(sorted(LTDETR_MODEL_REGISTRY))
        raise ValueError(f"Unknown LT-DETR backbone: '{key}'. Supported: {supported}.")
    return key, spec


def list_ltdetr_model_names() -> list[str]:
    return sorted(f"ltdetr/{key}" for key in LTDETR_MODEL_REGISTRY)


def ltetr_image_size_divisor(spec: LTDETRVariantSpec) -> int:
    """Image sizes must divide by ``2 × patch_size`` (baseline rule)."""
    return 2 * spec.patch_size


def check_image_size(image_size: tuple[int, int], spec: LTDETRVariantSpec) -> None:
    """Raise unless ``image_size`` is square and divides by ``2 × patch``.

    The detection pipeline is square-only end to end (the scale-jitter
    ladder and the batch resize are square), and the ViT patch grid plus
    the STA/encoder pyramids assume the 2 × patch divisibility.
    """
    divisor = ltetr_image_size_divisor(spec)
    height, width = image_size
    if height != width:
        raise ValueError(
            f"LT-DETR image_size {(height, width)} must be square (the "
            "detection pipeline resizes to square batches)."
        )
    if height % divisor or width % divisor:
        raise ValueError(
            f"LT-DETR image_size {(height, width)} must be divisible by "
            f"{divisor} (= 2 x patch_size for the '{spec.backbone_name}' "
            f"backbone); e.g. use ({-(-height // divisor) * divisor}, "
            f"{-(-width // divisor) * divisor})."
        )


class LTDETRObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training/architecture args for the LT-DETR detection family."""

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 12_000

    backbone: str = "dinov2-vits14-notpretrained"
    num_classes: int = 80
    image_size: ImageSizeTuple = (640, 640)
    #: Load the backbone's published pretrained weights (HF for
    #: dinov2/dinov3). Offline builds use ``-notpretrained`` backbone names.
    backbone_pretrained: bool = False
    #: Optional path to backbone-only weights loaded into the HF model.
    backbone_weights: Path | None = None

    #: Detection head family. "auto" picks the per-variant default
    #: ("dfine" for the ECViT "v2" variants, else "rtdetrv2"). "dfine"
    #: swaps in the distributional box refinement head (FGL/DDF).
    decoder: Literal["rtdetrv2", "dfine", "auto"] = "auto"
    #: D-FINE per-side bin resolution (``2 × reg_max + 1`` bins).
    reg_max: int = 32

    hidden_dim: int | Literal["auto"] = "auto"
    num_queries: int = 300
    dec_layers: int | Literal["auto"] = "auto"
    num_heads: int | Literal["auto"] = "auto"
    num_top_queries: int = 300

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0

    lr: float = 5e-4
    #: "auto" resolves to the per-variant recipe default (baseline: 0.05
    #: for small/tiny models, 0.0025 for the larger ECViT models).
    backbone_lr_factor: float | Literal["auto"] = "auto"
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_warmup_steps: int | Literal["auto"] = "auto"
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"

    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 2000

    def resolve_auto(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision._task_models.object_detection_components.flat_cosine import (
            resolve_flat_cosine_windows,
        )

        key, spec = resolve_ltdetr_variant(f"ltdetr/{self.backbone}")
        if self.hidden_dim == "auto":
            self.hidden_dim = spec.hidden_dim
        if self.dec_layers == "auto":
            self.dec_layers = spec.dec_layers
        if self.num_heads == "auto":
            self.num_heads = spec.dec_num_heads
        if self.backbone_lr_factor == "auto":
            self.backbone_lr_factor = spec.backbone_lr_factor
        warmup, flat, no_aug = resolve_flat_cosine_windows(
            total_steps, train_num_batches, gradient_accumulation_steps
        )
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = warmup
        if self.scheduler_flat_steps == "auto":
            self.scheduler_flat_steps = flat
        if self.scheduler_no_aug_steps == "auto":
            self.scheduler_no_aug_steps = no_aug

    def resolve_image_size(self) -> None:
        """Snap ``image_size`` to the backbone's patch grid (in place).

        The default is rounded UP to the next multiple of ``2 × patch_size``
        (baseline rule: 640 -> 644 for patch-14 backbones, 640 unchanged for
        patch 16). An explicitly set size must already be divisible — the
        ViT patch grid and the STA/encoder pyramids both assume it.
        """
        _, spec = resolve_ltdetr_variant(f"ltdetr/{self.backbone}")
        divisor = ltetr_image_size_divisor(spec)
        height, width = self.image_size
        if (height, width) == (640, 640):
            self.image_size = (
                -(-height // divisor) * divisor,
                -(-width // divisor) * divisor,
            )
            return
        check_image_size(self.image_size, spec)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
