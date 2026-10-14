# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Warm-start assembly from Apache upstream components (design doc 28, §6.2).

:func:`load_dfine_official_warm_start` initializes an DETR task model
from two Apache-2.0 sources:

- **encoder + decoder** ← an official D-FINE COCO checkpoint
  (``Peterande/D-FINE``, e.g. asset ``d-fine/dfine_s_coco``): the DETR
  hybrid encoder and official D-FINE decoder are re-ports of the same
  architecture, so tensors transfer through a small key-normalization
  (``cv1/cv2`` → ``conv1/conv2`` block renames and the upstream
  ``downsample_convs.N.0.cv1.*`` Sequential nesting) with **shape-driven**
  acceptance — any tensor whose normalized name has no same-shape target
  (different width recipe, class-count head, deploy-time buffers such as
  ``anchors``/``valid_mask``) is skipped and reported.
- **backbone** ← the EdgeCrafter ECViT backbone weights via the wrapper's
  own strict loader (``encoders/ecvit/ecvit.py``).

Widths are a property of the DETR variant, not of this loader: variants
whose encoder/decoder hidden width differs from the D-FINE checkpoint (256)
simply keep those parts at their fresh initialization and the coverage
report says so. The returned summary dict is what
``scripts/build_warm_start_coco.py`` records in the AVWarmStart ``spec``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

from autovision.registry.downloadable import resolve_downloadable
from autovision.registry.warm_start import WarmStartError

__all__ = ["load_dfine_official_warm_start"]

logger = logging.getLogger("autovision")

#: Submodule renames between the upstream D-FINE block naming and the
#: DETR re-port (``CSPRepLayer.cv1/cv2`` → ``conv1/conv2``).
_BLOCK_RENAMES = ((".cv1.", ".conv1."), (".cv2.", ".conv2."))

#: D-FINE wraps ``downsample_convs`` entries in ``Sequential(...)[0]`` with
#: an inner ``cv1``; the DETR port keeps the bare conv/norm pair.
_DOWNSAMPLE_NESTING = re.compile(r"^(encoder\.downsample_convs\.\d+)\.0\.cv1\.")

#: Class-count-dependent decoder heads — shape mismatches against the COCO
#: (80-class) checkpoint are expected and skipped for other class counts.
_CLASS_COUNT_MARKERS = (
    "dec_score_head",
    "enc_score_head",
    "denoising_class_embed",
    "pre_bbox_head",
)


def _normalize_dfine_key(key: str) -> str:
    """Map an official-checkpoint key onto the DETR layout."""
    key = _DOWNSAMPLE_NESTING.sub(r"\1.", key)
    for dfine_name, av_name in _BLOCK_RENAMES:
        if ".fpn_blocks." in key or ".pan_blocks." in key:
            key = key.replace(dfine_name, av_name)
    return key


def _resolve_checkpoint(source: str | Path) -> Path:
    """Accept an asset manifest name or a direct path."""
    path = Path(source)
    if path.is_file():
        return path
    return resolve_downloadable(str(source))


def load_dfine_official_warm_start(
    task_model: torch.nn.Module,
    dfine_checkpoint: str | Path,
    backbone_weights: str | Path | None = None,
    min_decoder_coverage: float = 0.5,
) -> dict[str, Any]:
    """Initialize ``task_model`` from Apache upstream components, in place.

    Args:
        task_model: an ``DETRObjectDetection`` instance whose decoder is
            the official D-FINE one (``decoder="dfine_official"``).
        dfine_checkpoint: official D-FINE COCO checkpoint — an asset name
            (``d-fine/dfine_s_coco``, resolved through the licensed asset
            manifest with mirror fallback) or a direct path.
        backbone_weights: optional EdgeCrafter backbone weights (e.g. asset
            ``edgecrafter/ecvitt``); loaded by the wrapper's strict loader.
        min_decoder_coverage: refuse silently-poor decoder transfers — the
            call raises when fewer than this fraction of the checkpoint's
            decoder tensors find a same-shape target.

    Returns a summary dict (``loaded/skipped/coverage`` breakdowns) for the
    AVWarmStart ``spec``.
    """
    path = _resolve_checkpoint(dfine_checkpoint)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(state, dict):
        raise WarmStartError(f"'{path}' is not a D-FINE checkpoint state dict.")

    target = task_model.state_dict()

    loaded: dict[str, str] = {}
    skipped_shape: list[tuple[str, str]] = []
    skipped_class_count: list[str] = []
    dropped_buffers: list[str] = []
    discarded_backbone = 0
    unmatched_structure: list[str] = []
    transfer: dict[str, torch.Tensor] = {}

    for dfine_key, tensor in state.items():
        av_key = _normalize_dfine_key(dfine_key)
        if av_key.startswith("backbone."):
            discarded_backbone += 1  # cspdarknet: the target brings its own
            continue
        if av_key not in target:
            if dfine_key.endswith(("anchors", "valid_mask")):
                dropped_buffers.append(dfine_key)
            else:
                unmatched_structure.append(dfine_key)
            continue
        if tuple(target[av_key].shape) == tuple(tensor.shape):
            transfer[av_key] = tensor
            loaded[dfine_key] = av_key
        elif any(marker in av_key for marker in _CLASS_COUNT_MARKERS):
            skipped_class_count.append(f"{dfine_key} -> {av_key}")
        else:
            skipped_shape.append((f"{dfine_key} -> {av_key}",
                                  f"{tuple(tensor.shape)} vs {tuple(target[av_key].shape)}"))

    dfine_decoder_total = sum(1 for k in state if k.startswith("decoder."))
    decoder_loaded = sum(1 for k in loaded if k.startswith("decoder."))
    coverage = decoder_loaded / dfine_decoder_total if dfine_decoder_total else 0.0
    if coverage < min_decoder_coverage:
        raise WarmStartError(
            f"D-FINE decoder coverage {coverage:.0%} ({decoder_loaded}/"
            f"{dfine_decoder_total}) is below the required "
            f"{min_decoder_coverage:.0%}; the checkpoint and this DETR "
            f"variant are too far apart for a warm start. Skipped by shape: "
            f"{skipped_shape[:6]}"
        )

    missing, unexpected = task_model.load_state_dict(transfer, strict=False)
    if unexpected:
        raise WarmStartError(f"Unexpected keys after transfer: {unexpected[:6]}")
    del missing  # expected: every target tensor outside the transfer set
    logger.info(
        "D-FINE warm start: %d tensors transferred (decoder %.0f%%); "
        "%d class-count skips, %d width skips, %d dropped buffers/backbone.",
        len(transfer), coverage * 100, len(skipped_class_count),
        len(skipped_shape), len(dropped_buffers),
    )

    if backbone_weights is not None:
        backbone_path = Path(backbone_weights)
        if not backbone_path.is_file():
            backbone_path = resolve_downloadable(str(backbone_weights))
        # The detection backbone wrapper delegates to the model wrapper that
        # owns the strict EdgeCrafter loader.
        candidates = [task_model.backbone,
                      getattr(task_model.backbone, "_model_wrapper", None)]
        loader = None
        for candidate in candidates:
            loader = getattr(candidate, "_load_backbone_weights", None)
            if loader is not None:
                break
        if loader is None:
            raise WarmStartError(
                f"{type(task_model.backbone).__name__} does not expose "
                f"backbone weight loading; pass a variant with an "
                f"ECViT/EdgeCrafter backbone."
            )
        loader(backbone_path)

    return {
        "dfine_checkpoint": str(path),
        "tensors_transferred": len(transfer),
        "decoder_coverage": coverage,
        "class_count_skips": skipped_class_count,
        "width_skips": skipped_shape,
        "dropped_buffers": dropped_buffers,
        "unmatched_structure": unmatched_structure,
        "discarded_backbone_tensors": discarded_backbone,
        "backbone_weights": str(backbone_weights) if backbone_weights else None,
    }
