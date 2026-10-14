# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DETR variant registry: ``detr/<backbone>`` architecture specs.

Model names follow ``detr/<backbone>`` where ``<backbone>`` resolves
against the backbone packages (``dinov2``, ``dinov3``, ``edgecrafter``),
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
- **EdgeCrafter ECViT** (patch 16): pass-through wrapper; these variants
  default to the D-FINE decoder (baseline "v2" family). The published
  ``-eupe`` names are aliases of the same architectures.

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
    "detr_default_batch_size",
    "detr_image_size_divisor",
    "check_image_size",
    "list_detr_model_names",
    "resolve_detr_variant",
]


@dataclass(frozen=True)
class DETRVariantSpec:
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
    #: published baseline D-FINE checkpoints), the fork's Dome-line
    #: variants "dfine" (RT-DETRv2-based port).
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
    # EdgeCrafter ECViT variants (baseline "v2"): pass-through backbone,
    # D-FINE decoder by default.
    for size, info in _ECVIT_SIZES.items():
        registry[f"edgecrafter-{size}"] = DETRVariantSpec(
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


DETR_MODEL_REGISTRY: dict[str, DETRVariantSpec] = _build_registry()


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
    the value before the step-based ``resolve_auto`` runs.
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
