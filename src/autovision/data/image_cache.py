# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image read caching: ``.npy`` disk sidecars and a shared RAM pool.

Opt-in layers over :func:`autovision.data.file_helpers.open_image`
(design ``docs/design_ultralytics_adoption.md`` §4.10 D10). ``open_image``
stays the single decode contract; this module only decides whether a
decoded image is stored and reused. Every ``mode`` returns the same object
as ``open_image``: a uint8 ``(C, H, W)`` tensor.

- ``mode="none"``: direct passthrough, byte-identical to the uncached path.
- ``mode="disk"``: ``<stem>.npy`` sidecar next to the source image. The
  sidecar is an uncompressed numpy zip container (``np.savez`` written
  into a file handle, so the name stays ``.npy``) holding the decoded
  array plus a ``meta`` int64 array ``[schema_version, source_mtime_ns]``.
  Storing the metadata inside the sidecar instead of a second JSON file
  keeps exactly one extra file per image and commits pixels + mtime in a
  single ``os.replace`` — two sidecar files could go stale independently.
- ``mode="ram"``: :class:`SharedImagePool` packs all images into one
  contiguous uint8 tensor built once in the dataloader-builder process;
  forked workers share it copy-on-write.

The cache is strictly read-through and best-effort: misses, staleness,
corruption and unwritable sidecars all fall back to the source decode and
never fail the read.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from autovision.data.file_helpers import open_image

__all__ = ["CacheMode", "SharedImagePool", "open_image_cached"]

logger = logging.getLogger(__name__)

CacheMode = Literal["none", "disk", "ram"]

# Sidecar container schema. Bump when the meta layout changes.
_META_VERSION = 1
# Index of the source mtime_ns entry inside the meta array.
_META_MTIME_NS = 1
_SHM_DIR = Path("/dev/shm")


def open_image_cached(
    path: Path | str,
    *,
    num_channels: int,
    mode: CacheMode,
    pool: SharedImagePool | None = None,
) -> torch.Tensor:
    """Decode an image through the optional cache layers.

    Args:
        path: Image path (resolved; same contract as ``open_image``).
        num_channels: Requested channel count, forwarded to ``open_image``.
        mode: ``"none"`` passthrough, ``"disk"`` npy sidecar, ``"ram"``
            shared pool lookup.
        pool: Shared pool for ``mode="ram"``; ``None`` or a pool miss
            falls back to the source decode.

    Returns:
        uint8 ``(C, H, W)`` tensor, identical to ``open_image`` output.
    """
    if mode == "none":
        return open_image(path, num_channels=num_channels)
    if mode not in ("disk", "ram"):
        raise ValueError(f"Unknown image cache mode: {mode!r}")
    resolved = Path(path)
    if mode == "ram":
        if pool is not None:
            cached = pool.get(resolved)
            if cached is not None:
                return cached
        return open_image(resolved, num_channels=num_channels)
    return _open_image_disk(resolved, num_channels=num_channels)


def _open_image_disk(path: Path, *, num_channels: int) -> torch.Tensor:
    """Disk-cache read path: sidecar hit, else decode + rebuild sidecar."""
    sidecar = _sidecar_path(path)
    # Stat before decoding: if the source changes mid-decode the stored
    # mtime is older than the file, so the next read rebuilds (conservative).
    source_mtime_ns = _source_mtime_ns(path)
    if source_mtime_ns is not None:
        cached = _read_sidecar(
            sidecar, num_channels=num_channels, source_mtime_ns=source_mtime_ns
        )
        if cached is not None:
            return torch.from_numpy(cached)
    tensor = open_image(path, num_channels=num_channels)
    if source_mtime_ns is not None:
        _write_sidecar(sidecar, tensor.numpy(), source_mtime_ns)
    return tensor


def _sidecar_path(path: Path) -> Path:
    """Sidecar lives beside the source as ``<stem>.npy``."""
    return path.with_suffix(".npy")


def _source_mtime_ns(path: Path) -> int | None:
    """Source mtime in ns, or ``None`` when the file cannot be stat'd."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _read_sidecar(
    sidecar: Path, *, num_channels: int, source_mtime_ns: int
) -> np.ndarray | None:
    """Return the cached array if fresh and well-formed, else ``None``.

    A corrupt container (bad zip, truncated member) is unlinked so the
    rebuild replaces it; a merely stale or mismatched sidecar is left in
    place for the atomic rebuild to overwrite.
    """
    try:
        with np.load(sidecar) as data:
            meta = np.asarray(data["meta"])
            image = np.asarray(data["image"])
    except Exception:  # noqa: BLE001 - corrupt sidecars may fail in many ways
        _unlink_quiet(sidecar)
        return None
    if meta.size < 2 or int(meta[0]) != _META_VERSION:
        return None
    if int(meta[_META_MTIME_NS]) != source_mtime_ns:
        return None
    if image.dtype != np.uint8 or image.ndim != 3:
        return None
    if int(image.shape[0]) != num_channels:
        return None
    return image


def _write_sidecar(sidecar: Path, image: np.ndarray, source_mtime_ns: int) -> None:
    """Atomically publish the sidecar; concurrent rebuilds: last one wins.

    The unique tmp name lets parallel first-readers coexist; ``os.replace``
    is atomic, so concurrent readers only ever observe a complete file.
    """
    tmp = sidecar.with_name(f"{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                image=np.ascontiguousarray(image),
                meta=np.array(
                    [_META_VERSION, source_mtime_ns], dtype=np.int64
                ),
            )
        os.replace(tmp, sidecar)
    except OSError:
        # Cache writes are best-effort (read-only dataset dirs, disk full).
        _unlink_quiet(tmp)


def _unlink_quiet(path: Path) -> None:
    """Remove ``path`` ignoring absence or permission errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _shm_free_bytes() -> int | None:
    """Free bytes on the shm device, or ``None`` when unconstrained.

    Without ``/dev/shm`` (e.g. macOS) there is no shared-memory device to
    account for: forked workers share plain anonymous pages copy-on-write.
    """
    if not _SHM_DIR.is_dir():
        return None
    return int(shutil.disk_usage(_SHM_DIR).free)


class SharedImagePool:
    """Read-only flat uint8 tensor holding whole decoded images.

    Built once in the process that constructs the dataloader; forked
    workers inherit the storage copy-on-write, so pooled images cost no
    per-worker memory. Lookups are keyed by resolved path string.
    """

    def __init__(
        self,
        storage: torch.Tensor,
        starts: list[int],
        shapes: list[tuple[int, int, int]],
        index: dict[str, int],
    ) -> None:
        self._storage = storage
        self._starts = starts
        self._shapes = shapes
        self._index = index

    @property
    def num_pooled(self) -> int:
        """Number of images actually held in the pool."""
        return len(self._index)

    @classmethod
    def build(
        cls,
        paths: Sequence[str | Path],
        *,
        num_channels: int,
        sample: int = 30,
        margin: float = 1.2,
    ) -> SharedImagePool | None:
        """Decode and pack ``paths`` into one contiguous uint8 tensor.

        Capacity is extrapolated from up to ``sample`` evenly-spaced
        decodes times ``margin``. When ``/dev/shm`` exists but cannot hold
        the estimate, returns ``None`` so the caller can fall back to
        ``disk`` mode. Images that overflow the estimate are not pooled;
        reads of those paths fall back to the source decode.

        Args:
            paths: Resolved image paths, in dataset order.
            num_channels: Requested channel count for decoding.
            sample: Number of images decoded for the size estimate.
            margin: Capacity headroom multiplier over the sampled average.

        Returns:
            The filled pool, or ``None`` when the shm device is too small.
        """
        resolved = [Path(p) for p in paths]
        total = len(resolved)
        if total == 0:
            return cls(torch.empty(0, dtype=torch.uint8), [], [], {})
        sample_indices = np.unique(
            np.linspace(0, total - 1, num=min(sample, total)).astype(np.int64)
        )
        sampled_bytes = [
            open_image(resolved[i], num_channels=num_channels).numel()
            for i in sample_indices
        ]
        average = max(int(sum(sampled_bytes) / len(sampled_bytes)), 1)
        capacity = int(average * total * margin)
        free = _shm_free_bytes()
        if free is not None and free < capacity:
            logger.warning(
                "SharedImagePool needs ~%.1f MiB but /dev/shm has only "
                "%.1f MiB free; falling back to disk image cache.",
                capacity / (1 << 20),
                free / (1 << 20),
            )
            return None
        storage = torch.empty(capacity, dtype=torch.uint8)
        starts: list[int] = []
        shapes: list[tuple[int, int, int]] = []
        index: dict[str, int] = {}
        cursor = 0
        for position, path in enumerate(resolved):
            tensor = open_image(path, num_channels=num_channels)
            needed = tensor.numel()
            if cursor + needed > capacity:
                logger.warning(
                    "SharedImagePool capacity exhausted after %d/%d images; "
                    "the rest fall back to source reads.",
                    position,
                    total,
                )
                break
            storage[cursor : cursor + needed] = tensor.reshape(-1)
            starts.append(cursor)
            shapes.append(
                (int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2]))
            )
            index[str(path)] = position
            cursor += needed
        logger.info(
            "SharedImagePool packed %d/%d images (%.1f MiB).",
            len(index),
            total,
            cursor / (1 << 20),
        )
        return cls(storage=storage, starts=starts, shapes=shapes, index=index)

    def get(self, path: Path | str) -> torch.Tensor | None:
        """Return the pooled image for ``path`` as a private copy.

        A copy (not a view) is returned on purpose: in-place transforms
        must never touch the storage shared across dataloader workers.
        ``None`` means the path is not pooled; callers fall back to the
        source decode.
        """
        position = self._index.get(str(path))
        if position is None:
            return None
        channels, height, width = self._shapes[position]
        start = self._starts[position]
        count = channels * height * width
        return (
            self._storage.narrow(0, start, count)
            .view(channels, height, width)
            .clone()
        )
