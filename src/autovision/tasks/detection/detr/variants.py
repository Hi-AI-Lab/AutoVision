# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR variant registry: ``detr/<backbone>`` architecture specs.

Model names follow ``detr/<backbone>`` where ``<backbone>`` resolves
against the backbone packages (``dinov2``, ``dinov3``),
e.g. ``detr/dinov2-vits14`` or ``detr/dinov3-convnext-tiny``. All
DINOv2 sizes accept the package suffixes (``-noreg``, ``-notpretrained``).
``detr/dinov2-vittest-notpretrained`` is a tiny offline test variant
(mirrors the baseline ``ViTTest``).

Variants mirror the published baseline registry (lineage:
``docs/provenance/LEDGER.md``):

- **DINOv2 ViT** (patch 14): NoSTA wrapper — no spatial-prior fusion, the
  semantic taps are projected straight to ``hidden_dim``.
- **DINOv3 ViT** (patch 16, incl. the tiny ``vitt16``/``vitt16plus``
  distillations): STA wrapper with ``conv_inplane = factor x patch_size``.
- **DINOv3 ConvNeXt**: last-three-stage taps, whole backbone at backbone LR.
  The published ``-eupe`` names are architecture aliases of these configs.

Encoder/decoder recipes (hidden dims, FFN widths, expansion/depth
multipliers, sampling points, layer counts) are carried per variant in
:class:`DETRVariantSpec` so the assembled network matches the baseline
configs instead of a single global default.

Train-argument dataclasses live in :mod:`.config`, which re-exports the
registry helpers so ``config`` stays the single import surface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DETR_MODEL_REGISTRY",
    "DETRVariantSpec",
    "FAMILY_PREFIX",
    "SPECTRUM",
    "SpectrumTier",
    "check_image_size",
    "detr_default_batch_size",
    "detr_image_size_divisor",
    "list_detr_model_names",
    "list_spectrum_tiers",
    "resolve_detr_variant",
    "resolve_spectrum_tier",
]


@dataclass(frozen=True)
class DETRVariantSpec:
    """Static per-variant architecture recipe (baseline-faithful)."""

    backbone_package: str  # "dinov2" | "dinov3" | "timm" | "gtr"
    backbone_name: str  # name inside the backbone package (suffixes allowed)
    family: Literal["vit", "convnext", "timm_cnn", "gtr"]
    patch_size: int
    #: 0-based ViT block taps; unused for convnext (stage taps).
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
    #: Default decoder; "dfine_official" ships the official D-FINE decoder
    #: (weight-compatible with the published baseline D-FINE checkpoints),
    #: the fork's Dome-line variants "dfine" (RT-DETRv2-based port).
    decoder_default: Literal["rtdetrv2", "dfine", "dfine_official"] = "rtdetrv2"
    #: Per-variant default for ``backbone_lr_factor`` ("auto" resolution).
    backbone_lr_factor: float = 0.05


# Baseline ViT recipes. conv_inplane = factor × patch_size, per the
# DETRRTDETRBackboneWrapperConfig table (ViTSmall factor 2, Base/Large/
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

# Spectrum width recipes for the timm CNN line (HGNetV2), P1-frozen per
# roadmap design doc 29 §3.1 (D1/D1b) — the widths are NOT guesses: they
# were read off the cached official D-FINE-S COCO checkpoint
# (hidden 256, encoder FFN 1024, decoder 3 layers, sampling points
# (4, 4, 4), 8 heads), so the ``dfine_official`` decoder assembles
# weight-compatible with the Apache-2.0 official assets. backbone_lr_factor
# 0.1 = pretrained-CNN fine-tune convention (RT-DETR trains HGNetV2 from
# scratch at 1.0; our spectrum starts from timm ImageNet weights).
_HGNETV2_SIZES: dict[str, dict[str, Any]] = {
    "hgnetv2-b0": dict(backbone_name="hgnetv2_b0", dec_layers=3),
    "hgnetv2-b1": dict(backbone_name="hgnetv2_b1", dec_layers=3),
}
_HGNETV2_RECIPE: dict[str, Any] = dict(
    hidden_dim=256,
    enc_dim_feedforward=1024,
    dec_layers=3,
    dec_dim_feedforward=1024,
    dec_num_points=(4, 4, 4),
    backbone_lr_factor=0.1,
)



def _build_registry() -> dict[str, DETRVariantSpec]:
    registry: dict[str, DETRVariantSpec] = {}

    # DINOv2 ViT variants (NoSTA per baseline): base + package suffixes
    # (-noreg, -notpretrained).
    for size, info in _VIT_SIZES.items():
        for suffix in ("", "-noreg", "-notpretrained"):
            registry[f"dinov2-{size}{suffix}"] = DETRVariantSpec(
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
    registry["dinov2-vittest-notpretrained"] = DETRVariantSpec(
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
            registry[name] = DETRVariantSpec(
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
        registry[f"dinov3-{size}"] = DETRVariantSpec(
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
    # Spectrum CNN line: timm HGNetV2 (Apache-2.0 upstream) — the P1
    # backbone for the N/S tiers (roadmap doc 29 §3.1). Widths are
    # dfine_s-compatible (see _HGNETV2_RECIPE) so the official decoder
    # assembles weight-compatible from the Apache-2.0 D-FINE assets.
    for size, info in _HGNETV2_SIZES.items():
        registry[size] = DETRVariantSpec(
            backbone_package="timm",
            backbone_name=info["backbone_name"],
            family="timm_cnn",
            patch_size=32,
            interaction_indexes=(),  # unused for CNN stage taps
            use_sta=False,
            conv_inplane=0,
            hidden_dim=_HGNETV2_RECIPE["hidden_dim"],
            enc_dim_feedforward=_HGNETV2_RECIPE["enc_dim_feedforward"],
            dec_layers=info["dec_layers"],
            dec_dim_feedforward=_HGNETV2_RECIPE["dec_dim_feedforward"],
            dec_num_points=_HGNETV2_RECIPE["dec_num_points"],
            decoder_default="dfine_official",
            backbone_lr_factor=_HGNETV2_RECIPE["backbone_lr_factor"],
        )
    # GTR assembly-compatible recipe (roadmap design doc 31, G1): widths
    # read off the gtr_s_coco EMA checkpoint (hidden 192, decoder FFN 512,
    # 4 layers, points (4,4,4) — 104 direct key matches, 12 residual
    # mismatches ALL FFN width, resolved by this spec). Decoder assembles
    # from the MIT GTR asset via dfine_official_warm_start(
    # source_format="gtr"); the backbone stays on our timm HGNetV2 line
    # (GTR's own backbone is a different GLA family — doc 31 G2).
    registry["gtr-s"] = DETRVariantSpec(
        backbone_package="timm",
        backbone_name="hgnetv2_b1",
        family="timm_cnn",
        patch_size=32,
        interaction_indexes=(),
        use_sta=False,
        conv_inplane=0,
        hidden_dim=192,
        enc_dim_feedforward=1024,
        dec_layers=4,
        dec_dim_feedforward=512,
        dec_num_points=(4, 4, 4),
        decoder_default="dfine_official",
        backbone_lr_factor=0.1,
    )
    # GTR GLA-backbone line (roadmap design doc 31 G2, MIT re-port under
    # tasks/detection/shared/gtr/): the REAL GTR backbone (Gated Linear
    # Attention + Spatial SwiGLU + bidirectional scan, DEIMv2-vit-tiny
    # lineage) + GTREncoder pyramid. Requires the "gtr" extra
    # (flash-linear-attention) at construction; Triton kernels need CUDA.
    # Widths from the official coco_finetune configs (S: 192/3 heads).
    # NOT a frozen spectrum tier — G2 acceptance (COCO trend + export)
    # pending; assembly variant "gtr-s" (HGNetV2 backbone) stays the
    # decoder-only path.
    registry["gtrvit-s"] = DETRVariantSpec(
        backbone_package="gtr",
        backbone_name="gtr_tiny_spatial_swiglu",
        family="gtr",
        patch_size=16,
        interaction_indexes=(3, 7, 11),
        use_sta=False,
        conv_inplane=0,
        hidden_dim=192,
        enc_dim_feedforward=1024,  # unused: GTREncoder has no AIFI FFN
        dec_layers=4,
        dec_dim_feedforward=512,
        dec_num_points=(4, 4, 4),
        decoder_default="dfine_official",
        backbone_lr_factor=0.1,
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


DETR_MODEL_REGISTRY: dict[str, DETRVariantSpec] = _build_registry()


@dataclass(frozen=True)
class SpectrumTier:
    """One product tier of the DETR spectrum (roadmap doc 29 §3.1).

    ``status`` carries the D10 freeze discipline: "frozen" tiers are
    product commitments (components only change via a minor version plus
    new weight entries); "planned" tiers are declared targets still
    awaiting their family implementation.
    """

    tier: str
    variant_key: str  # key into DETR_MODEL_REGISTRY ("" when planned)
    route: Literal["distill", "assembly-finetune"]
    benchmark_anchor: str
    status: Literal["frozen", "planned"] = "planned"
    note: str = ""


#: Product spectrum, P1 state (design doc 29 §3.1, frozen 2026-10-15).
SPECTRUM: dict[str, SpectrumTier] = {
    "n": SpectrumTier(
        tier="n",
        variant_key="hgnetv2-b0",
        route="distill",
        benchmark_anchor="YOLOv8n / YOLO11n",
        status="frozen",
        note=(
            "D1b: dfine_n official weights do not exist — N is the "
            "distillation-first tier (12ep). Teacher candidates: RF-DETR-n "
            "(Apache, HF), dfine_s decoder knowledge, or GTR-S COCO (MIT, "
            "local: Dino-Dome-Research/weight/gtr/det — Intellindust "
            "Gated-Token-Recurrence, decoder shape-compatible with "
            "dfine_official at hidden 192)."
        ),
    ),
    "s": SpectrumTier(
        tier="s",
        variant_key="hgnetv2-b1",
        route="assembly-finetune",
        benchmark_anchor="YOLOv8s / YOLO11s",
        status="frozen",
        note=(
            "Widths are dfine_s-assembly-compatible; warm-start from the "
            "Apache-2.0 d-fine/dfine_s_coco asset."
        ),
    ),
    "m": SpectrumTier(
        tier="m",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="YOLOv8m / RT-DETR-R18",
        note=(
            "Backbone line pending a timm/torchvision ConvNeXt-T family "
            "implementation (current convnext-* entries are DINOv3-ConvNeXt)."
        ),
    ),
    "l": SpectrumTier(
        tier="l",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="RT-DETR-R34 / YOLOv8l",
        note="Backbone line pending (see m).",
    ),
    "x": SpectrumTier(
        tier="x",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="RT-DETR-R50 / YOLOv8x",
        note="Optional tier; ConvNeXt-B / DINOv2-S backbone candidates.",
    ),
    # Planned tier WITH a component key (unlike m/l/x): the gtr-s width
    # recipe exists and assembles (doc 31 G1), but the tier freeze awaits
    # the G1 acceptance review (D10: planned ≠ product commitment).
    "gtr-s": SpectrumTier(
        tier="gtr-s",
        variant_key="gtr-s",
        route="assembly-finetune",
        benchmark_anchor="GTR-S COCO AP 53.6（MIT，官方口径）",
        status="planned",
        note=(
            "Doc 31 G1: decoder assembles from gtr_s_coco EMA "
            "(hidden 192/FFN 512); frozen pending assembly-coverage and "
            "COCO trend review."
        ),
    ),
}


def list_spectrum_tiers() -> list[str]:
    return sorted(SPECTRUM)


#: D10 public model strings: frozen tiers become first-class registry keys
#: so ``detr/{n,s}`` flow through the standard resolution/training/export
#: plumbing unchanged. Planned tiers deliberately get NO registry entry.
for _tier_key, _tier_entry in SPECTRUM.items():
    if _tier_entry.status == "frozen" and _tier_entry.variant_key:
        DETR_MODEL_REGISTRY[_tier_key] = DETR_MODEL_REGISTRY[_tier_entry.variant_key]


def resolve_spectrum_tier(tier: str) -> SpectrumTier:
    """Resolve a spectrum tier name (``"n"`` or ``"detr/n"``)."""
    key = tier[len(FAMILY_PREFIX) :] if tier.startswith(FAMILY_PREFIX) else tier
    entry = SPECTRUM.get(key)
    if entry is None:
        supported = ", ".join(sorted(SPECTRUM))
        raise ValueError(f"Unknown spectrum tier: '{tier}'. Supported: {supported}.")
    return entry


#: Deprecated family keys, each accepted with a DeprecationWarning for
#: one release after its rename: "avdetr/" (2026-10-15 detr rename),
#: "ltdetr/" (R16 rename).
_LEGACY_FAMILY_PREFIXES = ("avdetr/", "ltdetr/")
FAMILY_PREFIX = "detr/"


def resolve_detr_variant(name: str) -> tuple[str, DETRVariantSpec]:
    """Split ``detr/<backbone>`` into (backbone key, spec).

    Idempotent: accepts ``detr/<backbone>`` (public model name) and
    ``<backbone>`` (init-args form, e.g. from exported checkpoints). The
    pre-rename family keys ``avdetr/<backbone>`` and ``ltdetr/<backbone>``
    are accepted as deprecated aliases.
    """
    for legacy in _LEGACY_FAMILY_PREFIXES:
        if name.startswith(legacy):
            warnings.warn(
                f'Model family key "{legacy}..." is deprecated; '
                f'use "{FAMILY_PREFIX}..." instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            name = FAMILY_PREFIX + name[len(legacy) :]
            break
    key = name[len(FAMILY_PREFIX) :] if name.startswith(FAMILY_PREFIX) else name
    if not key or key.startswith(FAMILY_PREFIX) or "/" in key:
        raise ValueError(
            f"Unknown DETR model: '{name}' (expected '{FAMILY_PREFIX}<backbone>')."
        )
    spec = DETR_MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(sorted(DETR_MODEL_REGISTRY))
        raise ValueError(f"Unknown DETR backbone: '{key}'. Supported: {supported}.")
    return key, spec


def list_detr_model_names() -> list[str]:
    return sorted(f"{FAMILY_PREFIX}{key}" for key in DETR_MODEL_REGISTRY)


def detr_default_batch_size(backbone: str) -> int:
    """Baseline reference batch: 32 across the family, 16 for dinov2.

    Early resolution for ``batch_size="auto"`` — the train command needs
    the value before the step-based ``resolve_auto_fields`` runs.
    """
    _, spec = resolve_detr_variant(f"detr/{backbone}")
    return 16 if spec.backbone_package == "dinov2" else 32


def detr_image_size_divisor(spec: DETRVariantSpec) -> int:
    """Image sizes must divide by ``2 × patch_size`` (baseline rule)."""
    return 2 * spec.patch_size


def check_image_size(image_size: tuple[int, int], spec: DETRVariantSpec) -> None:
    """Raise unless ``image_size`` is square and divides by ``2 × patch``.

    The detection pipeline is square-only end to end (the scale-jitter
    ladder and the batch resize are square), and the ViT patch grid plus
    the STA/encoder pyramids assume the 2 × patch divisibility.
    """
    divisor = detr_image_size_divisor(spec)
    height, width = image_size
    if height != width:
        raise ValueError(
            f"DETR image_size {(height, width)} must be square (the "
            "detection pipeline resizes to square batches)."
        )
    if height % divisor or width % divisor:
        raise ValueError(
            f"DETR image_size {(height, width)} must be divisible by "
            f"{divisor} (= 2 x patch_size for the '{spec.backbone_name}' "
            f"backbone); e.g. use ({-(-height // divisor) * divisor}, "
            f"{-(-width // divisor) * divisor})."
        )
