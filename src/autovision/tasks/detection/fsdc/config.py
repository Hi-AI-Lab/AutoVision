# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""FSDC-DETR object detection: model configuration (design doc 22 §4.5).

The FSDC family reuses the DEIMv2 variant registry for its base
architecture: every entry maps an ``fsdc-*`` name onto a
``deimv2-*`` base variant whose HF config drives the shared decoder /
criterion stack, while the native dual-branch encoder and FSC hybrid
encoder are built from :class:`FsdcArgs`. ``fsdc-vittest`` is a hermetic
offline build for the test suite (registered in the DEIMv2 registry,
design doc 22 §7).
"""

from __future__ import annotations

from typing import Any

from autovision.config.base import PydanticConfig
from autovision.tasks.detection.deimv2.config import (
    Deimv2ObjectDetectionTrainArgs,
    resolve_deimv2_variant,
)

__all__ = [
    "FSDC_MODEL_REGISTRY",
    "FsdcArgs",
    "FsdcObjectDetectionTrainArgs",
    "resolve_fsdc_variant",
]

FSDC_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Paper anchor (FSDC-DETR Table 1 baseline DEIMv2-L): DINOv3 ViT-S
    # dual branch + STA, hidden 224 — base architecture pinned from the
    # official Intellindust config.json (PR-0). Warm-start: unavailable
    # (no transformers conversion on the Hub), decoder trains fresh.
    "fsdc-l": dict(base_variant="deimv2-dinov3-l"),
    # Cheap sanity arm: identical to fsdc-l but hidden 192 and the decoder
    # warm-starts from the in-repo DEIMv2-S COCO conversion (the only
    # DINOv3 conversion, PR-0) whenever load_weights=True.
    "fsdc-s": dict(base_variant="deimv2-dinov3-s"),
    # Hermetic test build (see the DEIMv2 registry entry).
    "fsdc-vittest": dict(base_variant="fsdc-vittest"),
}


def resolve_fsdc_variant(model: str) -> dict[str, Any]:
    """Registry entry for an fsdc variant name (``fsdc-l`` etc.)."""
    if model not in FSDC_MODEL_REGISTRY:
        supported = ", ".join(sorted(FSDC_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown fsdc detection model '{model}'. Supported: {supported}."
        )
    return FSDC_MODEL_REGISTRY[model]


class FsdcArgs(PydanticConfig):
    """Frequency-branch switches for the native FSDC modules (doc 22 §4.5).

    ``None`` (the train-arg default) keeps the stock DEIMv2 wrapper
    bit-for-bit; an explicit config swaps the conv encoder and hybrid
    encoder for the native implementations.
    """

    model_config = {"extra": "forbid"}

    # 3 = DEIMv2-faithful pyramid (strides 8/16/32); 4 = paper Fig.1
    # reading (P2-P5, strides 4/8/16/32 — capacity confound probe, G2).
    sta_levels: int = 3
    use_dbfsaf: bool = True
    dbfsaf_gamma: float = 0.5
    use_sfs_ff: bool = True
    # FSD-Down exists ONLY inside SFS-FF's Eq.10 in the native route;
    # enabling it without SFS-FF has no landing spot.
    use_fsd_down: bool = True
    freeze_wavelet: bool = False  # G5: False = learnable (Haar-initialized)
    fdconv_use_ksm: bool = False
    fdconv_lite: bool = False
    fft_export_fallback: bool = False
    # DINOv3 ViT pretrained checkpoint (safetensors, facebook conversion)
    # loaded into the native backbone after the swap — REQUIRED for
    # faithful fsdc-l: the official DEIMv2-L finetunes the LVD-1689M
    # pretrained ViT-S (deimv2_dinov3_l_coco.yml weights_path), and the
    # 0.025x backbone LR bucket assumes a pretrained ViT. None = random
    # init (architecture-only ablations).
    backbone_weights: str | None = None

    def validate_args(self) -> None:
        if self.sta_levels not in (3, 4):
            raise ValueError(
                f"fsdc.sta_levels must be 3 or 4, got {self.sta_levels}."
            )
        if self.use_fsd_down and not self.use_sfs_ff:
            raise ValueError(
                "fsdc.use_fsd_down has no landing spot without SFS-FF "
                "(FSD-Down is embedded in SFS-FF's Eq.10 alignment); "
                "enable use_sfs_ff or drop use_fsd_down."
            )
        if not 0 < self.dbfsaf_gamma < 1:
            raise ValueError(
                f"fsdc.dbfsaf_gamma must be in (0, 1), got {self.dbfsaf_gamma}."
            )


class FsdcObjectDetectionTrainArgs(Deimv2ObjectDetectionTrainArgs):
    """Training args for the FSDC detection family.

    Everything except ``model``/``fsdc`` is inherited from the DEIMv2
    args: the official optimizer recipe (AdamW + backbone LR buckets +
    wd exemptions + EMA) resolves against the BASE variant's registry
    entry via :meth:`resolve_variant`, and the HF criterion is consumed
    unchanged (design doc 22 §3.2/§4.6).
    """

    model_config = {"extra": "forbid"}

    model: str = "fsdc-l"
    fsdc: FsdcArgs | dict | None = None
    # Family default OFF: the paper-anchor variant (fsdc-l) has no
    # transformers-format COCO conversion (PR-0), so an inherited
    # load_weights=True would fail at construction; fsdc-s warm-start is
    # an explicit opt-in (scripts/visdrone_fsdc_800.py).
    load_weights: bool = False

    def resolve_variant(self) -> dict[str, Any]:
        # The image-size divisor and the optimizer recipe both key off the
        # BASE deimv2 variant, not the fsdc wrapper name.
        base = resolve_fsdc_variant(self.model)["base_variant"]
        return resolve_deimv2_variant(base)
