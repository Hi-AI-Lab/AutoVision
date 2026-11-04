# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining dataset over plain images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset

from autovision.augment.method_transform import MethodTransform
from autovision.data.file_helpers import open_mask
from autovision.data.image_cache import SharedImagePool, open_image_cached
from autovision.types import DatasetItem

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Multi-view pretraining dataset.

    ``image_filenames`` is any sequence of paths (a plain list, or a
    ``MemoryMappedSequence`` for zero-copy sharing across workers).

    When ``mask_dir`` is set, a region-id mask is loaded for every image
    (``<image stem>.png`` inside ``mask_dir``) and passed through the
    transform, which must emit a ``mask`` entry per view (e.g. the DetCon
    transforms). Masks enable region-contrastive pretraining objectives.

    ``image_cache`` opts into the read cache of
    :mod:`autovision.data.image_cache` (default ``"none"`` keeps the
    plain ``open_image`` decode): ``"disk"`` stores ``<stem>.npy``
    sidecars next to the sources, ``"ram"`` reads from a
    :class:`SharedImagePool` (see :meth:`build_image_pool`); misses fall
    back to the source decode either way.
    """

    def __init__(
        self,
        image_dir: Path | None,
        image_filenames: Sequence[str | Path],
        transform: MethodTransform,
        num_channels: int = 3,
        mask_dir: Path | None = None,
        image_cache: Literal["none", "disk", "ram"] = "none",
        image_pool: SharedImagePool | None = None,
    ) -> None:
        if image_cache not in ("none", "disk", "ram"):
            raise ValueError(
                f"image_cache must be 'none', 'disk' or 'ram', got {image_cache!r}"
            )
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.transform = transform
        self.num_channels = num_channels
        self.mask_dir = mask_dir
        self.image_cache: Literal["none", "disk", "ram"] = image_cache
        self.image_pool = image_pool

    def __len__(self) -> int:
        return len(self.image_filenames)

    def build_image_pool(
        self, *, sample: int = 30, margin: float = 1.2
    ) -> SharedImagePool | None:
        """Pack this dataset's images into a :class:`SharedImagePool`.

        Builds the pool in the caller's process (do this before creating
        the dataloader so forked workers share it copy-on-write) and
        stores it on ``self.image_pool``. On a ``/dev/shm`` shortfall the
        pool is ``None`` and the dataset degrades to ``disk`` cache mode.

        Args:
            sample: Number of images decoded for the capacity estimate.
            margin: Capacity headroom multiplier over the sampled average.

        Returns:
            The pool, or ``None`` when RAM caching is unavailable.
        """
        paths = [
            self._resolve_image_path(i) for i in range(len(self.image_filenames))
        ]
        pool = SharedImagePool.build(
            paths, num_channels=self.num_channels, sample=sample, margin=margin
        )
        if pool is None:
            self.image_cache = "disk"
        else:
            self.image_pool = pool
        return pool

    def _resolve_image_path(self, index: int) -> Path:
        """Resolve ``image_filenames[index]`` against ``image_dir``."""
        filename = self.image_filenames[index]
        filename_str = (
            filename["filenames"] if isinstance(filename, dict) else str(filename)
        )
        path = Path(filename_str)
        if not path.is_absolute() and self.image_dir is not None:
            path = self.image_dir / path
        return path

    def __getitem__(self, index: int) -> DatasetItem:
        path = self._resolve_image_path(index)
        image = open_image_cached(
            path,
            num_channels=self.num_channels,
            mode=self.image_cache,
            pool=self.image_pool,
        )
        transform_input: dict = {"image": image}
        if self.mask_dir is not None:
            transform_input["mask"] = open_mask(
                (self.mask_dir / path.name).with_suffix(".png")
            )
        transformed: list[dict[str, Any]] = self.transform(transform_input)
        item: DatasetItem = {
            "filename": path.name,
            "views": [view["image"] for view in transformed],
        }
        if transformed and all("mask" in view for view in transformed):
            item["masks"] = [view["mask"] for view in transformed]
        if transformed and all("geometry" in view for view in transformed):
            item["geometries"] = [view["geometry"] for view in transformed]
        return item
