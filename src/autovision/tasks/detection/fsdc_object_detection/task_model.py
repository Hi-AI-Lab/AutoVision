# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""FSDC-DETR task model: DEIMv2 trunk + native frequency modules (doc 22 §3.2).

Assembly strategy — "HF trunk + sub-module replacement" (the established
pattern in this repo's DEIMv2 wrapper, cf. the enc_score_head surgery):
the parent class builds the complete HF ``Deimv2ForObjectDetection``
(optionally loading the COCO conversion for warm-start, e.g. fsdc-s),
then this class swaps ``conv_encoder`` for the native dual-branch encoder
and ``encoder`` for the FSC hybrid encoder. The decoder, the criterion
(dense O2O + MAL/FGL/DDF/L1/GIoU), and every head stay the official HF
objects — nothing is re-implemented and no loss code changes.

``fsdc=None`` performs no replacement: the wrapper IS the stock
DEIMv2 task model (bit-for-bit, including the init-args snapshot).

Checkpoint round trip: the init-args snapshot taken by the parent
records the BASE deimv2 variant; this class rewrites it to the fsdc
model name and the fsdc config dict so the pickle-free safe loader
reconstructs the same native stack (AV-DETR density/ssl snapshot
pattern).
"""

from __future__ import annotations

from typing import Any

from autovision.tasks.detection.deimv2_object_detection.task_model import (
    Deimv2ObjectDetection,
)
from autovision.tasks.detection.fsdc_object_detection.config import (
    FsdcArgs,
    resolve_fsdc_variant,
)
from autovision.tasks.detection.fsdc_object_detection.dual_branch_backbone import (
    FsdcDinoDualBranchEncoder,
)
from autovision.tasks.detection.fsdc_object_detection.fsc_encoder import (
    FsdcHybridEncoder,
)

__all__ = ["FsdcObjectDetection"]


def _coerce_fsdc(fsdc: FsdcArgs | dict | None) -> FsdcArgs | None:
    if fsdc is None or isinstance(fsdc, FsdcArgs):
        return fsdc
    if isinstance(fsdc, dict):
        return FsdcArgs(**fsdc)
    raise TypeError(
        f"fsdc must be FsdcArgs, dict, or None; got {type(fsdc).__name__}."
    )


class FsdcObjectDetection(Deimv2ObjectDetection):
    """DEIMv2 trunk + native FSDC dual-branch encoder and FSC encoder."""

    def __init__(
        self,
        num_classes: int,
        model: str = "fsdc-l",
        image_size: tuple[int, int] = (640, 640),
        num_select: int = 300,
        load_weights: bool = False,
        fsdc: FsdcArgs | dict | None = None,
        **kwargs: Any,
    ) -> None:
        spec = resolve_fsdc_variant(model)
        fsdc_args = _coerce_fsdc(fsdc)
        if fsdc_args is not None:
            fsdc_args.validate_args()
        # 4-level pyramid (G2) must be expressed BEFORE the HF build: the
        # deformable attention bakes num_points_scale buffers from the
        # config, so a post-hoc mutation cannot reach the decoder. The
        # overrides follow the registry dict's own construction rules
        # (decoder_n_points extended by prepending the finest level's
        # value: [3, 6, 3] -> [3, 3, 6, 3]).
        config_overrides = None
        if fsdc_args is not None and fsdc_args.sta_levels == 4:
            from autovision.tasks.detection.deimv2_object_detection.config import (
                resolve_deimv2_variant,
            )

            base_config = resolve_deimv2_variant(spec["base_variant"])["config"]
            width = base_config["decoder_in_channels"][-1]
            points = list(base_config["decoder_n_points"])
            config_overrides = dict(
                num_feature_levels=4,
                encoder_in_channels=[width] * 4,
                feat_strides=[4, 8, 16, 32],
                decoder_in_channels=[width] * 4,
                decoder_n_points=[points[0]] + points,
            )
        # The parent builds the full HF stack (config, optional COCO
        # conversion, class-head retile) against the BASE variant.
        super().__init__(
            num_classes=num_classes,
            model=spec["base_variant"],
            image_size=image_size,
            num_select=num_select,
            load_weights=load_weights,
            config_overrides=config_overrides,
            **kwargs,
        )
        # Snapshot correction: record the fsdc name + config so the safe
        # loader rebuilds this exact stack (see module docstring).
        self._init_args["model"] = model
        self._init_args["fsdc"] = (
            fsdc_args.model_dump() if fsdc_args is not None else None
        )
        self.fsdc_model_name = model
        self.fsdc_args = fsdc_args
        if fsdc_args is not None:
            self._install_native_modules(fsdc_args)

    # ── native module installation ──────────────────────────────────────
    def _install_native_modules(self, fsdc: FsdcArgs) -> None:
        inner = self._model.model  # Deimv2Model
        config = inner.config
        source_backbone = getattr(inner.conv_encoder, "backbone", None)
        if source_backbone is None:
            raise ValueError(
                "fsdc requires a DINOv3 dual-branch base variant (the "
                "native encoder transfers the pretrained ViT from the "
                "replaced HF conv encoder); the resolved base has no "
                "conv_encoder.backbone."
            )
        inner.conv_encoder = FsdcDinoDualBranchEncoder(
            config, fsdc, source_backbone=source_backbone
        )
        inner.encoder = FsdcHybridEncoder(config, fsdc)
