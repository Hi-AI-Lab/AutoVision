# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation dataset over paired image/mask directories.

Layout: an images directory and a masks directory containing files with
matching stems (``abc.jpg`` <-> ``abc.png``). Masks are single-channel
class-index images; label 255 marks ignored pixels.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import (
    build_relative_lookup,
    match_with_stem_fallback,
    open_image,
)
from autovision.errors import AutoVisionError

__all__ = ["SemanticSegmentationDataset", "list_segmentation_pairs"]

logger = logging.getLogger("autovision")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_mask(path: Path) -> torch.Tensor:
    from torchvision.io import read_image
    from torchvision.io.image import ImageReadMode

    mask = read_image(str(path), mode=ImageReadMode.UNCHANGED)
    if mask.ndim == 3:
        # RGB(A) palette mask: any color-collapsed mask works as long as all
        # channels agree; otherwise the R channel is the class index (the
        # convention for 8-bit grayscale masks saved as PNG).
        mask = mask[0]
    return mask.long()


def list_segmentation_pairs(
    images_dir: Path, masks_dir: Path, ignore_index: int = 255
) -> tuple[list[tuple[Path, Path]], int]:
    """Match image/mask files, preferring the suffix-stripped relative path.

    H3-2i: bare-stem matching cross-matched masks in nested trees where two
    directories both contain e.g. ``a.jpg``; the relative path is now the
    primary key and the bare stem only a fallback (warned once when the
    stem is shared across mask directories — the old silent mis-pairing).

    Returns ``(pairs, max_label)`` where ``max_label`` is the highest class
    index found in any sampled mask (0 for an empty list). Pixels labeled
    ``ignore_index`` (255, the ignore convention) do not count towards the
    label range.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    mask_paths = [path for path in sorted(masks_dir.rglob("*")) if path.is_file()]
    by_relative, by_stem, ambiguous = build_relative_lookup(mask_paths, masks_dir)
    pairs: list[tuple[Path, Path]] = []
    stem_fallbacks: list[tuple[Path, Path]] = []
    max_label = 0
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        mask_path, fell_back = match_with_stem_fallback(
            image_path, images_dir, by_relative, by_stem
        )
        if mask_path is None:
            continue
        if fell_back:
            if image_path.stem in ambiguous:
                stem_fallbacks.append((image_path, mask_path))
        pairs.append((image_path, mask_path))
        mask = _read_mask(mask_path)
        valid = mask[mask != ignore_index]
        if valid.numel():
            max_label = max(max_label, int(valid.max().item()))
    if stem_fallbacks:
        image_path, mask_path = stem_fallbacks[0]
        logger.warning(
            "%d image(s) matched segmentation masks by bare stem because the "
            "relative path missed, and that stem is shared across mask "
            "directories — the pairing may be wrong (H3-2i); first example: "
            "'%s' -> '%s'.",
            len(stem_fallbacks), image_path, mask_path,
        )
    if not pairs:
        raise ValueError(
            f"No image/mask pairs found ('{images_dir}' x '{masks_dir}'); "
            f"masks must share the image file stem."
        )
    return pairs, max_label + 1


class SemanticSegmentationDataset(Dataset):
    """Yields ``{"image", "mask"}`` after the joint transform."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        transform: Any = None,
        num_channels: int = 3,
    ) -> None:
        self.pairs = [(Path(image), Path(mask)) for image, mask in pairs]
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        image = open_image(image_path, num_channels=self.num_channels)
        mask = _read_mask(mask_path)
        if mask.shape != image.shape[-2:]:
            # Segmentation labels must stay pixel-aligned — resampling a
            # class map is lossy and semantically wrong, so a mismatched
            # pair is annotation corruption. It used to surface only as a
            # confusing crop/resize failure deep in the transform (DATA-11).
            raise AutoVisionError(
                f"Segmentation pair '{image_path.name}' / '{mask_path.name}' "
                f"is misaligned: image spatial size {tuple(image.shape[-2:])} "
                f"vs mask {tuple(mask.shape)} (image '{image_path}', mask "
                f"'{mask_path}')."
            )
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name,
        }
