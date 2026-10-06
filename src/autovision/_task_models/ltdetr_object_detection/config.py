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

from pydantic import Field, model_validator

from autovision._configs.base import PydanticConfig
from autovision._task_models.object_detection_components.ssl_contrastive import (
    SSLParams,
)
from autovision._task_models.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "LTDETR_MODEL_REGISTRY",
    "LTDETRDensityArgs",
    "LTDETRSSLArgs",
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


class LTDETRDensityArgs(PydanticConfig):
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


class LTDETRSSLArgs(PydanticConfig):
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
    #: Instance pool source. "matched_gt" = the final Hungarian match
    #: (auxiliary objective alongside the detection losses).
    #: "encoder_candidates" = the decoder's top-k candidate features —
    #: label-free, requires ``mode="unsup"`` + ``view="dropout"``, and the
    #: contrastive loss REPLACES the detection losses entirely (SSL-only
    #: stage-1 pretraining; pair with a supervised stage-2 finetune).
    instance_source: Literal["matched_gt", "encoder_candidates"] = "matched_gt"

    @model_validator(mode="after")
    def _validate_ssl_configuration(self) -> "LTDETRSSLArgs":
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
    #: D-FINE only: FGL weight on the denoising (``_dn_*``) branches.
    #: ``None`` follows the main FGL weight (baseline parity). Dense
    #: tiny-object batches (VisDrone) push CDN to a single group of
    #: 2 × max-GT noised queries whose FGL residuals saturate the reg_max
    #: bins — the dn FGL term then dominates the total loss (~40 % in the
    #: 10-epoch pilot) without converging; lower it there.
    dn_fgl_loss_weight: float | None = None

    #: "auto" follows the backbone-family recipe (baseline lightly-train
    #: 0.17.0): DINOv2 ViT 1e-4, CNN 5e-4.
    lr: float | Literal["auto"] = "auto"
    #: "auto" resolves to the per-variant recipe default (baseline: 0.05
    #: for small/tiny models, 0.0025 for the larger ECViT models).
    backbone_lr_factor: float | Literal["auto"] = "auto"
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
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
    density: LTDETRDensityArgs | None = None

    #: Per-object contrastive SSL branch (design doc 14). ``None`` keeps
    #: the baseline contract bit-for-bit; ``mode="off"`` + ``view=
    #: "two_crop"`` is the batch-matched no-SSL baseline.
    ssl: LTDETRSSLArgs | None = None

    def resolve_auto(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision._task_models.object_detection_components.flat_cosine import (
            resolve_flat_cosine_windows,
        )

        key, spec = resolve_ltdetr_variant(f"ltdetr/{self.backbone}")
        family_is_vit = spec.family == "vit"
        if self.hidden_dim == "auto":
            self.hidden_dim = spec.hidden_dim
        if self.dec_layers == "auto":
            self.dec_layers = spec.dec_layers
        if self.num_heads == "auto":
            self.num_heads = spec.dec_num_heads
        if family_is_vit:
            # DINOv2 ViT family recipe (baseline V2 args: lr 1e-4, factor
            # 1e-2, linear schedule, warmup 2000). The spec table carries
            # the CNN 0.05 factor — overridden here.
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
            self.lr = 1e-4 if spec.family == "vit" else 5e-4
        if self.lr_scheduler == "auto":
            # DINOv2 ViT family trains with the linear recipe (warmup →
            # constant); CNN families with flat-cosine (baseline defaults).
            self.lr_scheduler = "linear" if spec.family == "vit" else "flat_cosine"
        if self.enc_dim_feedforward == "auto":
            self.enc_dim_feedforward = spec.enc_dim_feedforward
        if self.enc_expansion == "auto":
            self.enc_expansion = spec.enc_expansion
        if self.enc_depth_mult == "auto":
            self.enc_depth_mult = spec.enc_depth_mult

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
