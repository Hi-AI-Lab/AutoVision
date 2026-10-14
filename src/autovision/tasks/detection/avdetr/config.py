# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""AV-DETR object detection: model configuration and variant registry.

Model names follow ``avdetr/<backbone>`` where ``<backbone>`` resolves
against the backbone packages (``dinov2``, ``dinov3``, ``edgecrafter``),
e.g. ``avdetr/dinov2-vits14`` or ``avdetr/dinov3-convnext-tiny``. All
DINOv2 sizes accept the package suffixes (``-noreg``, ``-notpretrained``).
``avdetr/dinov2-vittest-notpretrained`` is a tiny offline test variant
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
:class:`AVDETRVariantSpec` so the assembled network matches the baseline
configs instead of a single global default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from autovision.config.base import PydanticConfig
from autovision.tasks.detection.components.distillation import (
    DistillParams,
)
from autovision.tasks.detection.components.ssl_contrastive import (
    SSLParams,
)
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "AVDETR_MODEL_REGISTRY",
    "AVDETRDensityArgs",
    "AVDETRDistillArgs",
    "AVDETROpenVocabArgs",
    "AVDETRSSLArgs",
    "AVDETRVariantSpec",
    "AVDETRObjectDetectionTrainArgs",
    "list_avdetr_model_names",
    "avdetr_default_batch_size",
    "resolve_avdetr_variant",
]


@dataclass(frozen=True)
class AVDETRVariantSpec:
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
    #: Default decoder; baseline "v2" (ECViT) variants ship the official
    #: D-FINE decoder ("dfine_official", weight-compatible with the
    #: published lightly-train D-FINE checkpoints), the fork's Dome-line
    #: variants "dfine" (RT-DETRv2-based port).
    decoder_default: Literal["rtdetrv2", "dfine", "dfine_official"] = "rtdetrv2"
    #: Per-variant default for ``backbone_lr_factor`` ("auto" resolution).
    backbone_lr_factor: float = 0.05


# Baseline ViT recipes. conv_inplane = factor × patch_size, per the
# AVDETRRTDETRBackboneWrapperConfig table (ViTSmall factor 2, Base/Large/
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


def _build_registry() -> dict[str, AVDETRVariantSpec]:
    registry: dict[str, AVDETRVariantSpec] = {}

    # DINOv2 ViT variants (NoSTA per baseline): base + package suffixes
    # (-noreg, -notpretrained).
    for size, info in _VIT_SIZES.items():
        for suffix in ("", "-noreg", "-notpretrained"):
            registry[f"dinov2-{size}{suffix}"] = AVDETRVariantSpec(
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
    registry["dinov2-vittest-notpretrained"] = AVDETRVariantSpec(
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
            registry[name] = AVDETRVariantSpec(
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
        registry[f"dinov3-{size}"] = AVDETRVariantSpec(
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
        registry[f"edgecrafter-{size}"] = AVDETRVariantSpec(
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
            # Baseline "v2" parity: the official gateway D-FINE decoder —
            # the architecture the published EdgeCrafter "-ltdetr-coco"
            # checkpoints were trained with (weight-compatible warm start).
            decoder_default="dfine_official",
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


AVDETR_MODEL_REGISTRY: dict[str, AVDETRVariantSpec] = _build_registry()


#: Legacy family key (R16 rename): "ltdetr/<backbone>" keeps working for
#: one release and emits a DeprecationWarning.
_LEGACY_FAMILY_PREFIX = "ltdetr/"
FAMILY_PREFIX = "avdetr/"


def resolve_avdetr_variant(name: str) -> tuple[str, AVDETRVariantSpec]:
    """Split ``avdetr/<backbone>`` into (backbone key, spec).

    Idempotent: accepts ``avdetr/<backbone>`` (public model name) and
    ``<backbone>`` (init-args form, e.g. from exported checkpoints). The
    pre-rename family key ``ltdetr/<backbone>`` is accepted as a
    deprecated alias.
    """
    if name.startswith(_LEGACY_FAMILY_PREFIX):
        warnings.warn(
            f'Model family key "{_LEGACY_FAMILY_PREFIX}..." is deprecated; '
            f'use "{FAMILY_PREFIX}..." instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        name = FAMILY_PREFIX + name[len(_LEGACY_FAMILY_PREFIX) :]
    key = name[len(FAMILY_PREFIX) :] if name.startswith(FAMILY_PREFIX) else name
    if not key or key.startswith(FAMILY_PREFIX) or "/" in key:
        raise ValueError(
            f"Unknown AV-DETR model: '{name}' (expected '{FAMILY_PREFIX}<backbone>')."
        )
    spec = AVDETR_MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(sorted(AVDETR_MODEL_REGISTRY))
        raise ValueError(f"Unknown AV-DETR backbone: '{key}'. Supported: {supported}.")
    return key, spec


def list_avdetr_model_names() -> list[str]:
    return sorted(f"{FAMILY_PREFIX}{key}" for key in AVDETR_MODEL_REGISTRY)


def avdetr_default_batch_size(backbone: str) -> int:
    """Baseline reference batch: 32 across the family, 16 for dinov2.

    Early resolution for ``batch_size="auto"`` — the train command needs
    the value before the step-based ``resolve_auto`` runs.
    """
    _, spec = resolve_avdetr_variant(f"avdetr/{backbone}")
    return 16 if spec.backbone_package == "dinov2" else 32


def avdetr_image_size_divisor(spec: AVDETRVariantSpec) -> int:
    """Image sizes must divide by ``2 × patch_size`` (baseline rule)."""
    return 2 * spec.patch_size


def check_image_size(image_size: tuple[int, int], spec: AVDETRVariantSpec) -> None:
    """Raise unless ``image_size`` is square and divides by ``2 × patch``.

    The detection pipeline is square-only end to end (the scale-jitter
    ladder and the batch resize are square), and the ViT patch grid plus
    the STA/encoder pyramids assume the 2 × patch divisibility.
    """
    divisor = avdetr_image_size_divisor(spec)
    height, width = image_size
    if height != width:
        raise ValueError(
            f"AV-DETR image_size {(height, width)} must be square (the "
            "detection pipeline resizes to square batches)."
        )
    if height % divisor or width % divisor:
        raise ValueError(
            f"AV-DETR image_size {(height, width)} must be divisible by "
            f"{divisor} (= 2 x patch_size for the '{spec.backbone_name}' "
            f"backbone); e.g. use ({-(-height // divisor) * divisor}, "
            f"{-(-width // divisor) * divisor})."
        )


class AVDETRDensityArgs(PydanticConfig):
    """Density-guided query/feature branch (Dome-DETR port, doc 13 §4.7).

    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit. Baseline Dome VisDrone recipe: min 250 / max 500,
    window 10, IoU threshold ``0.4 + 0.5 * density``.
    """

    model_config = {"extra": "forbid"}

    #: Always-kept proposal count ("auto" inherits ``num_queries``).
    min_num_select: int | Literal["auto"] = "auto"
    #: Upper bound on the per-image dynamic query count.
    max_num_select: int = 500
    #: Masked window attention on the stride-16 encoder level.
    use_mwas: bool = True
    #: Window side in stride-16 feature cells.
    mwas_window_size: int = 10
    #: Weight of the density-map MSE (Dome ``defe_density_map_weight``).
    density_map_weight: float = 1.0
    #: Extra penalty for underestimating high-density regions.
    density_underestimation_penalty: float = 0.3
    #: Weight of the object-count regression loss.
    count_loss_weight: float = 1.0
    #: Extra penalty for underestimating the object count.
    count_underestimation_weight: float = 2.0
    #: Window-gate threshold and descent step (baseline adaptive filter).
    window_threshold: float = 0.05
    window_threshold_step: float = 0.01
    #: Per-candidate NMS IoU threshold = base + scale * density.
    nms_iou_base: float = 0.4
    nms_iou_density_scale: float = 0.5
    #: GT heatmap sigma = ratio * box (w, h).
    gaussian_sigma_ratio: float = 1.2

    def resolved_min_num_select(self, num_queries: int) -> int:
        if self.min_num_select == "auto":
            return num_queries
        return int(self.min_num_select)


class AVDETRSSLArgs(PydanticConfig):
    """Per-object contrastive SSL branch (RF-DETR-SSL port, doc 14 §4.7).

    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit. ``mode="off"`` + ``view="two_crop"`` is the batch-matched
    no-SSL baseline: the doubled two-view data pipeline runs but neither
    the projection head nor the contrastive loss is built. Recommended
    start point (upstream M4 ablation): ``unsup`` / ``two_crop`` /
    ``loss_coef=0.1`` / ``temp=0.05`` / ``iou_match=0.55``.
    """

    model_config = {"extra": "forbid"}

    #: Contrastive objective: SimCSE InfoNCE ("unsup"), the SupCon-style
    #: same-label extension ("sup"), or their sum ("both"). "off" disables
    #: the loss while keeping the view's data pipeline (baseline runs).
    mode: Literal["off", "unsup", "sup", "both"] = "unsup"
    #: Weight λ of ``loss_ssl`` in the total loss. Upstream M4: 0.1–0.25 is
    #: the robust range; 1.0 was the only harmful setting. Constrained
    #: ``ge=0`` so the "off" baseline may carry 0; the validator requires
    #: ``> 0`` whenever the mode is active.
    loss_coef: float = Field(default=0.1, ge=0.0)
    #: Temperature of the cosine logits.
    temp: float = Field(default=0.05, gt=0.0)
    #: "two_crop" pairs every image with a second independent augmentation
    #: (doubled batch); "dropout" draws two stochastic samples through the
    #: projection head with zero data-pipeline changes (ablation control).
    view: Literal["two_crop", "dropout"] = "two_crop"
    #: Projection head architecture: SimCSE's Linear+Tanh or the paper's
    #: two-layer MLP ablation.
    projector: Literal["simcse", "mlp2"] = "simcse"
    #: Contrastive embedding width.
    projector_dim: int = Field(default=256, ge=1)
    #: Dropout applied to the head input (the dropout view's noise source).
    projector_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    #: Cross-view IoU threshold for instance correspondence (two_crop).
    iou_match: float = Field(default=0.55, gt=0.0, le=1.0)
    #: Score cross-view candidate pairs under the horizontal-mirror
    #: alignment too when pairing: the two views flip independently, so a
    #: mirrored object is the same object and its raw normalized-box IoU is
    #: ~0. Disable to restore the strict identity-alignment pairing.
    flip_invariant_pairing: bool = True
    #: Feed matched instances whose cross-view counterpart failed validation
    #: back into the pool as extra candidates (negatives under "unsup",
    #: soft positives via label equality under "sup") instead of dropping
    #: them — reclaims hard instances (design doc 14, D14).
    use_unpaired_as_negatives: bool = True
    #: Weight λ_w of the IoU soft-alignment logit bias (positives only).
    iou_bias: float = Field(default=0.5, ge=0.0)
    #: Exponent on the alignment IoU before the log bias.
    iou_gamma: float = Field(default=1.0, gt=0.0)
    #: Restrict same-label soft positives to same-image pairs (sup modes).
    same_image_positive_only: bool = False
    #: Gather the contrastive pool across DDP ranks (padded gather).
    gather_distributed: bool = True
    #: Apply the SSL objective only every k-th step. Controls the SSL
    #: gradient frequency only — it does NOT reduce the doubled two_crop
    #: pipeline compute (design doc 14 §4.4).
    every_n_steps: int = Field(default=1, ge=1)
    #: SIGReg distribution regularizer (design doc 16, LeVJEPA R1-A1):
    #: a third loss leg on the RAW projector outputs of the matched pool.
    #: 0 disables it. Requires mode != "off", view == "two_crop" and
    #: instance_source == "matched_gt". Reference finding: SIGReg mainly
    #: buys λ_ssl robustness (E2: λ=1.0 no longer collapses) rather than
    #: raw mAP.
    sigreg_coef: float = Field(default=0.0, ge=0.0)
    #: Random unit directions per SIGReg evaluation (resampled each step).
    sigreg_slices: int = Field(default=1024, ge=1)
    #: Trapezoidal quadrature knots for the Epps–Pulley integral.
    sigreg_knots: int = Field(default=17, ge=5)
    #: Instance pool source. "matched_gt" = the final Hungarian match
    #: (auxiliary objective alongside the detection losses).
    #: "encoder_candidates" = the decoder's top-k candidate features —
    #: label-free, requires ``mode="unsup"`` + ``view="dropout"``, and the
    #: contrastive loss REPLACES the detection losses entirely (SSL-only
    #: stage-1 pretraining; pair with a supervised stage-2 finetune).
    instance_source: Literal["matched_gt", "encoder_candidates"] = "matched_gt"

    @model_validator(mode="after")
    def _validate_ssl_configuration(self) -> "AVDETRSSLArgs":
        if self.mode != "off" and self.loss_coef <= 0.0:
            raise ValueError(
                f"ssl mode={self.mode!r} requires loss_coef > 0, got {self.loss_coef}."
            )
        if self.same_image_positive_only and self.view != "two_crop":
            # Under the dropout view every instance shares one image, so
            # the same-image restriction carries no information.
            raise ValueError(
                "ssl same_image_positive_only=True requires view='two_crop'."
            )
        if self.instance_source == "encoder_candidates":
            # Route B (design doc 14 §4.10): label-free SSL-only pretraining.
            if self.mode != "unsup":
                raise ValueError(
                    "ssl instance_source='encoder_candidates' is label-free "
                    "and requires mode='unsup'."
                )
            if self.view != "dropout":
                raise ValueError(
                    "ssl instance_source='encoder_candidates' requires "
                    "view='dropout' (candidate proposals are not comparable "
                    "across independent crops)."
                )
            if self.every_n_steps > 1:
                # The stride gate exists to interleave the SSL gradient with
                # the detection losses; here there is nothing to interleave
                # with, so a >1 value would be silently meaningless.
                raise ValueError(
                    "ssl every_n_steps has no effect with "
                    "instance_source='encoder_candidates'; use 1."
                )
            if self.sigreg_coef > 0:
                # SIGReg (design doc 16 R1-A1) rides the matched-pool taps,
                # which only exist for the auxiliary objective.
                raise ValueError(
                    "ssl sigreg_coef>0 requires "
                    "instance_source='matched_gt' (encoder_candidates has "
                    "no Hungarian match to pool from)."
                )
        if self.sigreg_coef > 0:
            if self.mode == "off":
                raise ValueError(
                    "ssl sigreg_coef>0 requires an active mode "
                    "(mode='off' builds the head for nobody)."
                )
            if self.view != "two_crop":
                # v1 scope (reference risk table): the SIGReg pool is the
                # two_crop matched pool; the dropout view shares one query
                # set, where the statistic adds nothing over dropout itself.
                raise ValueError(
                    "ssl sigreg_coef>0 requires view='two_crop' (v1 scope)."
                )
            if self.projector == "simcse":
                # The raw SIGReg tap is the post-Tanh output, bounded to
                # (-1, 1)^K, so the N(0, I) target is unreachable in
                # principle: the statistic floors away from zero and the
                # gradient pushes the Linear pre-activations into the Tanh
                # saturation regime. The mlp2 head ends in an unbounded
                # Linear — the space the LeJEPA regularizer assumes.
                raise ValueError(
                    "ssl sigreg_coef>0 requires projector='mlp2' (the "
                    "'simcse' head's raw tap is post-Tanh, bounded to "
                    "(-1, 1)^K, so the N(0, I) target is unreachable)."
                )
        return self

    def to_ssl_params(self) -> SSLParams:
        """Map onto the shared-component :class:`SSLParams` contract."""
        return SSLParams(
            mode=self.mode,
            loss_coef=self.loss_coef,
            temp=self.temp,
            view=self.view,
            iou_match=self.iou_match,
            iou_bias=self.iou_bias,
            iou_gamma=self.iou_gamma,
            flip_invariant_pairing=self.flip_invariant_pairing,
            use_unpaired_as_negatives=self.use_unpaired_as_negatives,
            same_image_positive_only=self.same_image_positive_only,
            gather_distributed=self.gather_distributed,
            every_n_steps=self.every_n_steps,
            instance_source=self.instance_source,
            sigreg_coef=self.sigreg_coef,
            sigreg_slices=self.sigreg_slices,
            sigreg_knots=self.sigreg_knots,
        )


class AVDETRTokenDropArgs(PydanticConfig):
    """Window-granularity in-backbone background token dropping (doc 17).

    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit. Per training sample, a window-grid keep mask is derived
    from the GT boxes (windows overlapping a margin-dilated box are always
    kept; a uniform random ``keep_ratio`` subset of the background windows
    survives) and the ViT backbone skips every dropped window's patch
    tokens for the whole encoder stack. Train-only: validation and export
    never see masks and run the dense backbone.

    Selection reuses the box-aware eraser's protection semantics (doc 16)
    at window granularity; the mask is built in ``training_step`` on the
    final canvas, so transform/collate are untouched.
    """

    model_config = {"extra": "forbid"}

    #: Window edge in pixels. Must be a multiple of the backbone patch
    #: size, and the canvas must be divisible by it (samples that are not
    #: silently run dense). 32px on an 800px canvas = a 25x25 window grid.
    window_size: int = Field(default=32, ge=1)
    #: Fraction of background windows kept when the mask applies, in (0, 1].
    keep_ratio: float = Field(default=0.4, gt=0.0, le=1.0)
    #: Per-sample application probability; unapplied samples run dense.
    prob: float = Field(default=1.0, ge=0.0, le=1.0)
    #: GT box dilation fraction forwarded to the protection mask (doc 16
    #: semantics: floored at 2px; 0 disables dilation).
    padding: float = Field(default=0.15, ge=0.0)
    #: Step at which token dropping turns OFF (Dome aug-stop cadence —
    #: training ends on the dense distribution it deploys with). ``"auto"``
    #: resolves to the shared step schedule; ``None`` keeps it on for the
    #: whole run. An unresolved ``"auto"`` stays active (transform precedent).
    step_stop: int | None | Literal["auto"] = "auto"


class AVDETRDistillArgs(PydanticConfig):
    """Cross-model distillation branch (design doc 23).

    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit: no teacher is built, no distillation losses join the
    criterion. The teacher produces soft targets for the student's
    union-matched queries via a GT-anchored join — both models run their
    own Hungarian match against the same targets, and every ground truth
    present in both matches becomes a (student query, teacher query)
    supervision pair. Three legs ride the final union assignment:
    classification soft targets, box soft targets, and a matched-only
    cross-model variant of the D-FINE distribution distillation (DDF).

    Teacher sources (``teacher_model`` unless ``teacher_checkpoint`` is
    set): ``"self:ema"`` (the run's own EMA copy — the mean-teacher
    control arm) or a public task-model name (``"deimv2/deimv2-dinov3-s"``).
    ``teacher_checkpoint`` points at an AutoVision exported ``.pt``
    (``exported_best.pt``), which is the recommended same-domain teacher
    (a deimv2 VisDrone arm's best checkpoint).
    """

    model_config = {"extra": "forbid"}

    #: Teacher source when ``teacher_checkpoint`` is None: ``"self:ema"``
    #: or a public task-model name (``"<family>/<model>"``). Ignored when
    #: ``teacher_checkpoint`` is set.
    teacher_model: str = "self:ema"
    #: Path to an AutoVision exported model (``exported_*.pt``) used as
    #: the teacher. Takes precedence over ``teacher_model``.
    teacher_checkpoint: Path | None = None
    #: Teacher input canvas; ``None`` runs the teacher on the student's
    #: canvas. Normalized cxcywh targets and predictions are
    #: resolution-independent, so a smaller teacher canvas (its native
    #: training resolution) is valid and cheaper (design doc 23 D6).
    teacher_image_size: tuple[int, int] | None = None
    #: Weight of the classification soft-target leg (teacher confidence
    #: for the assigned GT label as the BCE target of the student logit).
    cls_coef: float = Field(default=1.0, ge=0.0)
    #: Weight of the box soft-target leg (L1 + ``box_giou_coef`` ×
    #: (1 − GIoU) against the teacher's refined box, on top of — never
    #: instead of — the GT box losses).
    box_coef: float = Field(default=1.0, ge=0.0)
    #: GIoU weight inside the box soft-target leg.
    box_giou_coef: float = Field(default=2.0, ge=0.0)
    #: Weight of the matched-only cross-model DDF leg. Requires a teacher
    #: whose side-distribution space matches the student's
    #: (``reg_max``/``reg_scale``); otherwise the leg is dropped with a
    #: one-time warning (the cls/box legs are bin-space free).
    ddf_coef: float = Field(default=1.5, ge=0.0)
    #: Temperature of the DDF KL (official self-distillation value).
    ddf_temperature: float = Field(default=5.0, gt=0.0)
    #: First optimizer step that computes distillation losses: matches
    #: drift badly early in training, so teacher soft targets wait out
    #: the warmup (design doc 23 D7).
    start_step: int = Field(default=1000, ge=0)
    #: Compute the teacher pass and distillation losses only every k-th
    #: step (the teacher forward is pure overhead for the skipped steps).
    every_n_steps: int = Field(default=1, ge=1)
    #: Reject teachers whose class count differs from the student's. With
    #: ``False``, a class-count mismatch forces ``cls_coef = 0`` (box and
    #: DDF legs live in normalized coordinates / bin space and survive a
    #: class-space change; the classification leg does not).
    require_same_class_space: bool = True

    @model_validator(mode="after")
    def _validate_distill_configuration(self) -> "AVDETRDistillArgs":
        if self.teacher_checkpoint is None and not self.teacher_model:
            raise ValueError(
                "distill requires teacher_checkpoint or a non-empty "
                "teacher_model ('self:ema' or '<family>/<model>')."
            )
        if self.ddf_coef > 0 and self.ddf_temperature <= 0:
            raise ValueError("distill ddf_temperature must be positive.")
        return self

    def resolved_cls_coef(self, teacher_num_classes: int, student_num_classes: int) -> float:
        """Effective classification-leg weight after the class-space rule."""
        if teacher_num_classes == student_num_classes:
            return self.cls_coef
        if self.require_same_class_space:
            raise ValueError(
                f"distill teacher has {teacher_num_classes} classes but the "
                f"student has {student_num_classes}; pass "
                "require_same_class_space=False to keep the box/DDF legs "
                "(the classification leg is dropped)."
            )
        return 0.0

    def to_distill_params(self) -> DistillParams:
        """Map onto the shared-component :class:`DistillParams` contract."""
        return DistillParams(
            cls_coef=self.cls_coef,
            box_coef=self.box_coef,
            box_giou_coef=self.box_giou_coef,
            ddf_coef=self.ddf_coef,
            ddf_temperature=self.ddf_temperature,
        )


class AVDETROpenVocabArgs(PydanticConfig):
    """Text-contrastive open-vocabulary branch (design doc 27).

    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit: no text module is built and the per-class Linear score
    heads stay in place. When enabled, the per-layer decoder score heads
    become :class:`~...open_vocab.ContrastiveClassHead` instances whose
    logits are ``logit_scale * cos(query_proj, text_embed) + class_bias``
    — emitted as the SAME ``(B, Q, num_classes)`` reduced view, so the
    criterion, matcher, decode path, and the Dome dynamic selection are
    untouched (the "reduce-then-reuse" red line, doc 27 §3.2).

    The class names are prompt-encoded ONCE through a frozen text tower
    (cached under the framework cache dir) into the decoder's persistent
    ``ov_text_embeds`` buffer — the tower itself never enters the module
    tree / state_dict / optimizer / export surface.
    """

    model_config = {"extra": "forbid"}

    #: Frozen text tower. The real encoders download through the verified
    #: weight path on first use (sha256 pinning is gate G1, doc 27 §6);
    #: ``identity`` is a deterministic hash embedder for hermetic tests
    #: and offline smoke runs — not a research option.
    text_encoder: Literal["siglip-base", "clip-b32", "clip-b16", "identity"] = (
        "siglip-base"
    )
    #: Vocabulary, canonical dataset order. ``len(class_names)`` must equal
    #: ``num_classes`` (checked on the train args); the task model
    #: cross-checks the dataset categories at build time.
    class_names: tuple[str, ...]
    #: Single prompt template (the class-name slot is ``{}``).
    prompt_template: str = "a photo of a {}."
    #: Template ensemble (per-class mean over templates, CLIP convention).
    #: ``None`` runs the single ``prompt_template``.
    prompt_templates: tuple[str, ...] | None = None
    #: Per-class bias restoring the baseline score-head calibration
    #: (init = ``bias_init_with_prob(0.01)``); fresh classes extrapolate
    #: with the same prior (doc 27 §3.4).
    use_class_bias: bool = True
    #: Learnable similarity scale: init ``ln(1/0.07)`` (CLIP convention),
    #: clamped at ``logit_scale_max`` = ``ln(100)``.
    logit_scale_init: float = Field(default=2.659, gt=0.0)
    logit_scale_max: float = Field(default=4.605, gt=0.0)
    #: Cache the encoded vocabulary under the framework cache dir
    #: (recommended; the cache key covers encoder + vocabulary + templates).
    cache_embeddings: bool = True
    # ── v2 (design doc 27 §3.5/§3.6) ────────────────────────────────────
    #: Language-guided query selection: the ENCODER score head becomes a
    #: contrastive head too, so the two-stage top-k (and the Dome dynamic
    #: selection's ranking signal) rank cells by text alignment instead of
    #: learned class logits. The selection code itself is untouched — the
    #: reduced view keeps ``logits.max(-1)`` semantics.
    language_guided_selection: bool = False
    #: Zero-init gated text cross-attention in every decoder layer (after
    #: the image cross-attention). A bitwise identity at init.
    decoder_text_attention: bool = False
    #: Heads of the text cross-attention sublayer.
    text_attn_heads: int = Field(default=8, ge=1)
    # ── zero-shot protocol (doc 27 §3.7) ────────────────────────────────
    #: Novel-class subset for the leave-out protocol: ``compute_metrics``
    #: emits ``base_ap`` / ``novel_ap`` as per-class-AP aggregates over
    #: this subset and its complement. ``None`` keeps the plain metrics.
    novel_eval_classes: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_open_vocab_configuration(self) -> "AVDETROpenVocabArgs":
        if not self.class_names:
            raise ValueError("open_vocab requires a non-empty class_names tuple.")
        if len(set(self.class_names)) != len(self.class_names):
            duplicates = sorted(
                {name for name in self.class_names if self.class_names.count(name) > 1}
            )
            raise ValueError(
                f"open_vocab class_names must be unique; duplicated: {duplicates}."
            )
        templates = (
            self.prompt_templates
            if self.prompt_templates is not None
            else (self.prompt_template,)
        )
        for template in templates:
            if "{}" not in template:
                raise ValueError(
                    f"open-vocab prompt template {template!r} lacks the "
                    f"'{{}}' class-name slot."
                )
        if self.logit_scale_max < self.logit_scale_init:
            raise ValueError(
                f"open_vocab logit_scale_max ({self.logit_scale_max}) must be "
                f">= logit_scale_init ({self.logit_scale_init})."
            )
        if self.novel_eval_classes is not None:
            unknown = [
                name for name in self.novel_eval_classes
                if name not in self.class_names
            ]
            if not self.novel_eval_classes:
                raise ValueError(
                    "open_vocab novel_eval_classes must be a non-empty subset "
                    "of class_names (or None)."
                )
            if unknown:
                raise ValueError(
                    f"open_vocab novel_eval_classes are not all in class_names: "
                    f"{unknown}."
                )
            # The FULL set is legal (doc 27 §3.7/G4): novel_ap then equals
            # the plain mAP and base_ap degrades to the -1.0 "empty subset"
            # sentinel — the protocol's self-check configuration.
        return self


class AVDETRObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training/architecture args for the AV-DETR detection family."""

    model_config = {"extra": "forbid"}

    #: Reference batch for the ``lr × sqrt(global/default)`` LR scaling.
    #: Baseline: 32 across the AVDETR family, 16 only for the dinov2 V2
    #: args — "auto" resolves by backbone package in :meth:`resolve_auto`
    #: (``batch_size="auto"`` resolves through
    #: :func:`avdetr_default_batch_size` with the same rule).
    lr_reference_batch: int | Literal["auto"] = "auto"
    default_steps: ClassVar[int] = 12_000
    backbone: str = "dinov2-vits14-notpretrained"
    num_classes: int = 80
    image_size: ImageSizeTuple = (640, 640)
    #: Load the backbone's published pretrained weights (HF for
    #: dinov2/dinov3). Offline builds use ``-notpretrained`` backbone names.
    backbone_pretrained: bool = False
    #: Optional path to backbone-only weights loaded into the HF model.
    backbone_weights: Path | None = None
    #: Decoder body width, decoupled from the encoder ``hidden_dim``.
    #: ``None`` keeps the shared width (fork checkpoint compatibility);
    #: 256 matches the baseline ConvNeXt detector layout required to load
    #: the published COCO checkpoints (encoder 384 + decoder 256). ViT
    #: baseline checkpoints share the spec width, so ``None`` is correct
    #: there (vitt16 192 / vits16 224 / vitt16plus 256).
    decoder_hidden_dim: int | None = None
    #: Optional path to a published baseline AV-DETR detector checkpoint
    #: (lightly-train "…-ltdetr-coco" .pt): warm-starts backbone + encoder
    #: + decoder for both families (ConvNeXt via the old-ConvNeXt remap,
    #: ViT via the Meta-ViT remap including the STA wrapper extras).
    #: Requires the baseline-faithful shapes (ConvNeXt: decoder_hidden_
    #: dim=256 at the checkpoint's encoder width; ViT: the spec recipe);
    #: class-count-dependent decoder heads are skipped when the class
    #: count differs. See coco_checkpoint.load_baseline_coco_checkpoint.
    detector_checkpoint: Path | None = None

    #: Detection head family. "auto" picks the per-variant default
    #: ("dfine_official" for the ECViT "v2" variants, else "rtdetrv2").
    #: "dfine" is the fork's RT-DETRv2-based D-FINE port (Dome line);
    #: "dfine_official" is the official gateway-based D-FINE decoder,
    #: weight-compatible with the published lightly-train D-FINE
    #: checkpoints.
    decoder: Literal["rtdetrv2", "dfine", "dfine_official", "auto"] = "auto"
    #: D-FINE per-side bin resolution (``reg_max + 1`` bins).
    reg_max: int = 32
    #: D-FINE weighting-function curvature (official ``reg_scale``): bins
    #: encode distances in ``w / reg_scale`` units, geometric spacing spans
    #: ``±2 · 0.5 · reg_scale`` units (``±reg_scale × w`` of reach).
    reg_scale: float = 4.0
    #: HybridEncoder width knobs (official Dome-S "S" recipe: 1024 / 0.5 /
    #: 0.34). ``"auto"`` follows the variant spec — the ConvNeXt specs
    #: carry the heavy baseline values (2048 / 1.0 / 1.0), a ~4-6x larger
    #: randomly-initialized encoder that measurably slows early
    #: convergence on dense tiny-object benchmarks.
    enc_dim_feedforward: int | Literal["auto"] = "auto"
    enc_expansion: float | Literal["auto"] = "auto"
    enc_depth_mult: float | Literal["auto"] = "auto"

    hidden_dim: int | Literal["auto"] = "auto"
    num_queries: int = 300
    dec_layers: int | Literal["auto"] = "auto"
    num_heads: int | Literal["auto"] = "auto"
    num_top_queries: int = 300

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    #: Crash guard for extreme-dense batches: cap the per-image DN slot
    #: count (DN queries = 2 × min(max-GT, cap) per group). Unbounded, the
    #: decoder's self-attention over (DN + learned) queries OOMs a 24 GB
    #: card on batches containing VisDrone's 500+ GT images. ``None``
    #: keeps the official unbounded behavior; a cap only affects batches
    #: above it, where ``num_group`` is already floored to 1 — DN
    #: supervision then covers the first ``cap`` GTs of those images.
    dn_max_gt_cap: int | None = None
    #: D-FINE only: FGL weight on the denoising (``_dn_*``) branches.
    #: ``None`` follows the main FGL weight (baseline parity). Dense
    #: tiny-object batches (VisDrone) push CDN to a single group of
    #: 2 × max-GT noised queries whose FGL residuals saturate the reg_max
    #: bins — the dn FGL term then dominates the total loss (~40 % in the
    #: 10-epoch pilot) without converging; lower it there.
    dn_fgl_loss_weight: float | None = None

    #: IoU term of the supervised box loss (design doc 15, D-FINE decoder
    #: only). "giou" keeps the upstream 1−GIoU emission; "siou" emits
    #: loss_siou instead (Scylla-IoU, arXiv:2205.12740) — mutually
    #: exclusive emissions, matcher cost untouched. Single-variable
    #: ablation switch, NOT a default improvement (expected gain is
    #: noise-level; design doc 15 §1.3).
    box_regression_iou: Literal["giou", "siou"] = "giou"
    #: Weight of loss_siou in siou mode; None inherits the loss_giou weight
    #: (E0 probe: loss-scale ratio 0.953 on VisDrone — no down-scaling needed).
    siou_loss_coef: float | None = Field(default=None, ge=0.0)
    #: SIoU shape-cost exponent (paper experience value 4; no per-dataset search).
    siou_theta: float = Field(default=4.0, gt=0.0)

    #: "auto" follows the backbone-family recipe (baseline lightly-train
    #: 0.17.0): DINOv2 ViT 1e-4, CNN 5e-4.
    lr: float | Literal["auto"] = "auto"
    #: "auto" resolves to the per-variant recipe default (baseline: 0.05
    #: for small/tiny models, 0.0025 for the larger ECViT models).
    backbone_lr_factor: float | Literal["auto"] = "auto"
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
    #: torch.optim.AdamW's fused single-kernel step on CUDA (design doc 26
    #: W1; measured 8.70 → 0.71 ms/step on a 21M/140-tensor shape). Off by
    #: default: the fused update reorders float ops, so in-flight ablation
    #: runs stay comparable — flip for fresh runs.
    fused_optimizer: bool = False
    #: Vectorized criterion match-index paths (design doc 26 W2): the FGL
    #: target/apply and DDF weight per-image loops flatten into single
    #: batched calls with row-aligned IoU. Numerically equivalent up to
    #: float summation order (< 1e-6 fp32 in tests); off by default for
    #: the same run-comparability reason as ``fused_optimizer``.
    criterion_flat_losses: bool = False
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_warmup_steps: int | Literal["auto"] = "auto"
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"
    #: LR schedule family. ``"flat_cosine"``: the baseline warmup → plateau
    #: → cosine → restart-tail window. ``"multistep"``: the official
    #: Dome/D-FINE recipe — linear warmup, then constant LR with ``gamma``
    #: drops at ``lr_milestones`` fractions of the run (official: ×0.8 at
    #: epochs 80/120 of 160, i.e. the LR never drops below 64 % of peak).
    #: The multistep family is the measured-safe choice on dense
    #: tiny-object benchmarks: deep cosine annealing plus the restart tail
    #: degraded end-of-run mAP in the 10-epoch VisDrone pilots.
    lr_scheduler: Literal["flat_cosine", "multistep", "linear", "auto"] = "auto"
    lr_milestones: tuple[float, ...] | Literal["auto"] = "auto"
    lr_gamma: float = 0.8

    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 2000

    #: Detections-per-image cap in the COCO eval (pycocotools default 100).
    #: Dense benchmarks use the official higher cap — VisDrone: 500.
    eval_max_dets: int = 100

    #: Detection pyramid depth: 3 (strides 8/16/32, historical default) or
    #: 4 (strides 4/8/16/32 — the Dome-faithful tiny-object pyramid,
    #: design doc 13 §5/D1). 4 requires a ConvNeXt-family backbone (the
    #: only wrapper exposing a stride-4 stage); ViT/ECViT are fixed
    #: 3-level and reject the value.
    pyramid_levels: int = 3

    #: Density-guided dynamic queries / DeFE / MWAS branch (Dome-DETR).
    #: ``None`` keeps the baseline contract bit-for-bit.
    density: AVDETRDensityArgs | None = None

    #: Per-object contrastive SSL branch (design doc 14). ``None`` keeps
    #: the baseline contract bit-for-bit; ``mode="off"`` + ``view=
    #: "two_crop"`` is the batch-matched no-SSL baseline.
    ssl: AVDETRSSLArgs | None = None

    #: In-backbone background token dropping (design doc 17, ViT family
    #: only). ``None`` keeps the baseline contract bit-for-bit; the
    #: validation/export paths are always dense regardless.
    token_drop: AVDETRTokenDropArgs | None = None

    #: Cross-model distillation branch (design doc 23). ``None`` keeps the
    #: baseline contract bit-for-bit; the teacher never enters the
    #: state_dict / optimizer / EMA / export surfaces.
    distill: AVDETRDistillArgs | None = None

    #: Text-contrastive open-vocabulary branch (design doc 27). ``None``
    #: keeps the baseline contract bit-for-bit: the per-class Linear score
    #: heads stay, no text asset is built.
    open_vocab: AVDETROpenVocabArgs | None = None

    @model_validator(mode="after")
    def _validate_avdetr_train_args(self) -> "AVDETRObjectDetectionTrainArgs":
        if self.open_vocab is not None:
            expected = len(self.open_vocab.class_names)
            if expected != self.num_classes:
                raise ValueError(
                    f"open_vocab carries {expected} class names but the model "
                    f"trains num_classes={self.num_classes}; the vocabulary "
                    f"order defines the label space."
                )
        return self

    def resolve_auto(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision.tasks.shared.flat_cosine import (
            resolve_flat_cosine_windows,
        )

        key, spec = resolve_avdetr_variant(f"avdetr/{self.backbone}")
        # Baseline recipe split is by MODEL-NAME PREFIX, not by network
        # family: only `dinov2/*` walks the "V2 args" (lr 1e-4, backbone
        # factor 1e-2, linear schedule, warmup 2000, reference batch 16).
        # dinov3 ViT and ConvNeXt variants share the AVDETR recipe (5e-4 /
        # 0.05 spec default / flat-cosine / schedule warmup / batch 32);
        # the large ECViT variants carry 0.0025 in the spec table.
        package_is_dinov2 = spec.backbone_package == "dinov2"
        if self.hidden_dim == "auto":
            self.hidden_dim = spec.hidden_dim
        if self.dec_layers == "auto":
            self.dec_layers = spec.dec_layers
        if self.num_heads == "auto":
            self.num_heads = spec.dec_num_heads
        if package_is_dinov2:
            if self.backbone_lr_factor == "auto":
                self.backbone_lr_factor = 1e-2
            if self.lr_warmup_steps == "auto":
                self.lr_warmup_steps = 2000
        elif self.backbone_lr_factor == "auto":
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
        if self.lr_milestones == "auto":
            # Official Dome-S ratio: ×0.8 at epochs 80/120 of 160.
            self.lr_milestones = (0.5, 0.75)
        if self.lr == "auto":
            self.lr = 1e-4 if package_is_dinov2 else 5e-4
        if self.lr_scheduler == "auto":
            # dinov2 V2 args train with the linear recipe (warmup →
            # constant); everything else with flat-cosine (baseline
            # defaults).
            self.lr_scheduler = "linear" if package_is_dinov2 else "flat_cosine"
        if self.enc_dim_feedforward == "auto":
            self.enc_dim_feedforward = spec.enc_dim_feedforward
        if self.enc_expansion == "auto":
            self.enc_expansion = spec.enc_expansion
        if self.enc_depth_mult == "auto":
            self.enc_depth_mult = spec.enc_depth_mult
        if self.lr_reference_batch == "auto":
            self.lr_reference_batch = 16 if package_is_dinov2 else 32

    def resolve_image_size(self) -> None:
        """Snap ``image_size`` to the backbone's patch grid (in place).

        The default is rounded UP to the next multiple of ``2 × patch_size``
        (baseline rule: 640 -> 644 for patch-14 backbones, 640 unchanged for
        patch 16). An explicitly set size must already be divisible — the
        ViT patch grid and the STA/encoder pyramids both assume it.
        """
        _, spec = resolve_avdetr_variant(f"avdetr/{self.backbone}")
        divisor = avdetr_image_size_divisor(spec)
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
