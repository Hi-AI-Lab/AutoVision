# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Depth dataset over paired image/depth directories.

Layout: an images directory and a depths directory with matching stems
(``abc.jpg`` <-> ``abc.npy`` or ``abc.png``). ``.npy`` maps are float
meters verbatim; 16-bit PNG maps are divided by ``png_depth_scale``
(default 1000 — the NYU / Depth-Anything convention of millimetres).
Invalid pixels (0 / NaN) stay in the map — the loss and metrics mask them
by depth range.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import (
    build_relative_lookup,
    match_with_stem_fallback,
    open_image,
)

__all__ = ["DepthEstimationDataset", "list_depth_pairs"]

logger = logging.getLogger("autovision")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _prefer_npy(candidate: Path, current: Path) -> Path:
    """Depth maps prefer ``.npy`` over ``.png`` at the same key."""
    return candidate if candidate.suffix.lower() == ".npy" else current


def list_depth_pairs(
    images_dir: Path, depths_dir: Path
) -> list[tuple[Path, Path]]:
    """Match image/depth files, preferring the suffix-stripped relative path.

    H3-2i: matching is keyed by the path relative to each root first (so
    nested trees never cross-match same-stem files); the bare stem remains
    a fallback (warned once when shared across depth directories). At one
    key ``.npy`` wins over ``.png``.
    """
    images_dir, depths_dir = Path(images_dir), Path(depths_dir)
    depth_paths = [
        path
        for path in sorted(depths_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".npy", ".png"}
    ]
    by_relative, by_stem, ambiguous = build_relative_lookup(
        depth_paths, depths_dir, prefer=_prefer_npy
    )
    pairs: list[tuple[Path, Path]] = []
    stem_fallbacks: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        depth_path, fell_back = match_with_stem_fallback(
            image_path, images_dir, by_relative, by_stem
        )
        if depth_path is None:
            continue
        if fell_back:
            if image_path.stem in ambiguous:
                stem_fallbacks.append((image_path, depth_path))
        pairs.append((image_path, depth_path))
    if stem_fallbacks:
        image_path, depth_path = stem_fallbacks[0]
        logger.warning(
            "%d image(s) matched depth maps by bare stem because the "
            "relative path missed, and that stem is shared across depth "
            "directories — the pairing may be wrong (H3-2i); first example: "
            "'%s' -> '%s'.",
            len(stem_fallbacks), image_path, depth_path,
        )
    return pairs


def _read_depth(path: Path, png_depth_scale: float) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            array = array.squeeze()
        return torch.from_numpy(array)
    from torchvision.io import read_image
    from torchvision.io.image import ImageReadMode

    raw = read_image(str(path), mode=ImageReadMode.UNCHANGED)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.dtype == torch.uint8:
        # 8-bit depth maps are rare; treat as meters unless scaled.
        return raw.float()
    return raw.float() / png_depth_scale


class DepthEstimationDataset(Dataset):
    """Yields ``{"image": (C,H,W) uint8, "depth": (H,W) float meters,
    "original_size": (w, h)}`` per pair; the transform owns resize /
    normalize / flips (joint image+depth)."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        transform: Any = None,
        png_depth_scale: float = 1000.0,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform
        self.png_depth_scale = float(png_depth_scale)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, depth_path = self.pairs[index]
        image = open_image(image_path)
        height, width = image.shape[-2:]
        depth = _read_depth(depth_path, self.png_depth_scale)
        if tuple(depth.shape) != (height, width):
            depth = torch.nn.functional.interpolate(
                depth.view(1, 1, *depth.shape),
                size=(height, width),
                mode="nearest",
            ).view(height, width)
        item = {
            "image": image,
            "depth": depth,
            "original_size": (width, height),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
