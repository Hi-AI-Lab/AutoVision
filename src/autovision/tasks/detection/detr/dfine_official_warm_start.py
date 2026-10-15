# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Assemble a DETR spectrum network from the official D-FINE COCO assets.

Rebuild of the assembly loader (the ECViT-era module was removed together
with its backbone line, commit ``588530d``; roadmap design doc 29 P1 /
LEDGER P3-1). The transfer is **shape-driven**: source keys are matched
against the target network's state dict by name and exact tensor shape,
and only tensors matching both are copied — a width or class-count
mismatch becomes a skip, never a silent reshape. The decoder coverage
gate refuses assemblies below ``min_decoder_coverage`` (honest rejection
over partial warm-starts).

Backbone weights do NOT flow through here: the timm HGNetV2 line loads
its own ImageNet weights at construction (``load_weights=True``); this
module transfers encoder/decoder knowledge from the Apache-2.0 D-FINE
official checkpoints only.

Empirical fit (P1 freeze, 2026-10-15): ``detr/s`` (hgnetv2-b1, hidden
256, COCO 80 classes) vs ``d-fine/dfine_s_coco`` — 168 exact name+shape
matches, zero shape conflicts, decoder coverage 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from autovision.registry.assets import LicenseRef
from autovision.registry.warm_start import save_warm_start
from autovision.tasks.detection.detr.task_model import DETRObjectDetection

__all__ = [
    "WarmStartAssemblyError",
    "build_dfine_official_warm_start",
]

logger = logging.getLogger("autovision")

#: D-FINE official checkpoints keep the pre-rename family module names;
#: the historical normalization rules from the ECViT-era loader (kept for
#: any residual divergence beyond the direct matches).
_KEY_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("cv1", "conv1"),
    ("cv2", "conv2"),
)


class WarmStartAssemblyError(RuntimeError):
    pass


def _normalize_key(key: str) -> str:
    for source, target in _KEY_NORMALIZATIONS:
        key = key.replace(source, target)
    return key


def build_dfine_official_warm_start(
    model_name: str,
    out: Path | str,
    checkpoint: Path | str,
    *,
    num_classes: int = 80,
    image_size: tuple[int, int] = (640, 640),
    backbone_pretrained: bool = True,
    license_id: str = "Apache-2.0",
    origin: str = "https://huggingface.co/Peterande/D-FINE",
    min_decoder_coverage: float = 0.5,
    extra_spec: dict[str, Any] | None = None,
    source_format: str = "dfine",
) -> dict[str, Any]:
    """Assemble ``model_name`` encoder/decoder from a D-FINE checkpoint.

    Builds the target network (backbone from timm ImageNet weights when
    ``backbone_pretrained``), transfers every source tensor whose
    normalized name and shape match a target tensor exactly, enforces the
    decoder coverage gate, and writes an AVWarmStart v1 payload. Returns
    the assembly report (counts + coverage) for model-card provenance.

    ``source_format``: ``"dfine"`` (official D-FINE ``{"model": ...}``
    payloads) or ``"gtr"`` (Intellindust GTR checkpoints — ``{model, ema,
    last_epoch}``, EMA weights used per the GTR README基准口径; roadmap
    doc 31 G1). ``license_id``/``origin`` should then reflect the GTR
    MIT provenance.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise WarmStartAssemblyError(
            f"D-FINE checkpoint not found: '{checkpoint}'. Resolve the "
            "registered asset (d-fine/dfine_{s,m,l}_coco) first."
        )
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(source, dict) and "model" in source:
        if source_format == "gtr":
            ema_module = (source.get("ema") or {}).get("module")
            source = ema_module if ema_module is not None else source["model"]
        else:
            source = source["model"]

    network = DETRObjectDetection(
        num_classes=num_classes,
        model_name=model_name,
        image_size=image_size,
        decoder="dfine_official",
        backbone_pretrained=backbone_pretrained,
        load_weights=backbone_pretrained,
    )
    target = network.state_dict()

    normalized_source = {_normalize_key(k): v for k, v in source.items()}
    transferred: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    matched_by: dict[str, str] = {}
    for key, tensor in target.items():
        candidate = normalized_source.get(key)
        if candidate is None:
            continue
        if candidate.shape != tensor.shape:
            skipped_shape.append(key)
            continue
        transferred[key] = candidate.clone()
        matched_by[key] = "direct"

    missing, _ = network.load_state_dict(transferred, strict=False)
    decoder_target_keys = [k for k in target if k.startswith("decoder.")]
    decoder_target_numel = sum(target[k].numel() for k in decoder_target_keys)
    decoder_transferred_numel = sum(
        target[k].numel() for k in transferred if k in decoder_target_keys
    )
    coverage = (
        decoder_transferred_numel / decoder_target_numel
        if decoder_target_numel
        else 0.0
    )
    report = {
        "source_checkpoint": str(checkpoint),
        "source_format": source_format,
        "model_name": model_name,
        "source_keys": len(source),
        "target_keys": len(target),
        "transferred": len(transferred),
        "transferred_numel": sum(v.numel() for v in transferred.values()),
        "skipped_shape": len(skipped_shape),
        "decoder_coverage": round(coverage, 4),
        "load_state_dict_missing": len(missing),
    }
    logger.info("D-FINE warm-start assembly: %s", report)
    if coverage < min_decoder_coverage:
        raise WarmStartAssemblyError(
            f"Decoder transfer coverage {coverage:.3f} < "
            f"{min_decoder_coverage:.2f} for '{model_name}' from "
            f"'{checkpoint.name}' — refusing to write a partial "
            f"warm-start (tally: {report})."
        )

    license_ref = LicenseRef(id=license_id, origin=origin)
    spec = {
        "family": "detr",
        "backbone": model_name,
        "classes": num_classes,
        "decoder_coverage": report["decoder_coverage"],
        "assembly": "dfine_official_warm_start v2 (shape-driven)",
        "source_checkpoint": checkpoint.name,
        **(extra_spec or {}),
    }
    save_warm_start(
        out,
        model={k: v.to(torch.float32) for k, v in transferred.items()},
        spec=spec,
        license=license_ref,
    )
    return report
