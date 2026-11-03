# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image read cache: npy sidecars, shared RAM pool, and dataset wiring."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from autovision.augment.transform_args import (
    MethodTransformArgs,
    NormalizeArgs,
    RandomResizeArgs,
)
from autovision.data import ImageDataset
from autovision.data import image_cache as image_cache_module
from autovision.data.file_helpers import open_image
from autovision.data.image_cache import SharedImagePool, open_image_cached


def _write_image(
    path: Path, seed: int = 0, size: tuple[int, int] = (16, 16)
) -> Path:
    """Write a deterministic random RGB image at ``path`` and return it."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _pin_shm(monkeypatch: pytest.MonkeyPatch, free: int = 1 << 30) -> None:
    """Point the shm check at a real dir with a deterministic free value."""
    monkeypatch.setattr(image_cache_module, "_SHM_DIR", Path("/tmp"))
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: SimpleNamespace(free=free)
    )


class _PassthroughTransform:
    """Deterministic MethodTransform stub: one view, the input unchanged."""

    transform_args = MethodTransformArgs(
        image_size=(16, 16),
        random_resize=RandomResizeArgs(min_scale=1.0, max_scale=1.0),
        normalize=NormalizeArgs(mean=(0.5,), std=(0.25,)),
    )

    @staticmethod
    def transform_args_cls() -> type[MethodTransformArgs]:
        return MethodTransformArgs

    def __call__(self, input: dict) -> list[dict]:
        return [{"image": input["image"]}]


class TestDiskCache:
    def test_unknown_mode_rejected(self, tmp_path):
        """A bogus cache mode raises instead of silently decoding."""
        path = _write_image(tmp_path / "img.png")
        with pytest.raises(ValueError, match="cache mode"):
            open_image_cached(path, num_channels=3, mode="bogus")

    def test_first_read_populates_sidecar(self, tmp_path):
        """Disk mode decodes the source once and writes a fresh sidecar."""
        path = _write_image(tmp_path / "img.png", seed=1)
        cached = open_image_cached(path, num_channels=3, mode="disk")
        sidecar = tmp_path / "img.npy"
        assert sidecar.exists()
        expected = open_image(path, num_channels=3)
        assert torch.equal(cached, expected)
        with np.load(sidecar) as data:
            assert np.array_equal(data["image"], expected.numpy())
            meta = np.asarray(data["meta"])
        assert int(meta[0]) == image_cache_module._META_VERSION
        assert int(meta[1]) == path.stat().st_mtime_ns

    def test_none_mode_writes_no_sidecar(self, tmp_path):
        """mode="none" is the uncached path: same pixels, no sidecar files."""
        path = _write_image(tmp_path / "img.png", seed=2)
        uncached = open_image_cached(path, num_channels=3, mode="none")
        assert torch.equal(uncached, open_image(path, num_channels=3))
        assert list(tmp_path.glob("*.npy")) == []

    def test_mtime_change_triggers_rebuild(self, tmp_path):
        """A changed source mtime invalidates the sidecar and rebuilds it."""
        path = _write_image(tmp_path / "img.png", seed=3)
        stale = open_image_cached(path, num_channels=3, mode="disk")
        _write_image(path, seed=4)  # different pixels, fresh mtime
        new_mtime = path.stat().st_mtime_ns + 10**9
        os.utime(path, ns=(new_mtime, new_mtime))
        rebuilt = open_image_cached(path, num_channels=3, mode="disk")
        assert not torch.equal(stale, rebuilt)
        assert torch.equal(rebuilt, open_image(path, num_channels=3))
        with np.load(tmp_path / "img.npy") as data:
            assert int(data["meta"][1]) == new_mtime

    def test_shape_mismatch_triggers_rebuild(self, tmp_path):
        """A sidecar with an unexpected channel count is rebuilt from source."""
        path = _write_image(tmp_path / "img.png", seed=5)
        assert open_image_cached(path, num_channels=3, mode="disk") is not None
        mtime = path.stat().st_mtime_ns
        sidecar = tmp_path / "img.npy"
        with sidecar.open("wb") as handle:
            np.savez(
                handle,
                image=np.zeros((1, 4, 4), dtype=np.uint8),
                meta=np.array(
                    [image_cache_module._META_VERSION, mtime], dtype=np.int64
                ),
            )
        restored = open_image_cached(path, num_channels=3, mode="disk")
        assert torch.equal(restored, open_image(path, num_channels=3))
        with np.load(sidecar) as data:
            assert data["image"].shape == (3, 16, 16)

    def test_corrupt_sidecar_is_unlinked_and_rebuilt(self, tmp_path):
        """A corrupt sidecar is discarded and the next read restores it."""
        path = _write_image(tmp_path / "img.png", seed=6)
        assert open_image_cached(path, num_channels=3, mode="disk") is not None
        sidecar = tmp_path / "img.npy"
        sidecar.write_bytes(b"definitely not an npy container")
        recovered = open_image_cached(path, num_channels=3, mode="disk")
        assert torch.equal(recovered, open_image(path, num_channels=3))
        with np.load(sidecar) as data:
            assert data["image"].shape == (3, 16, 16)

    def test_concurrent_first_read_leaves_valid_sidecar(self, tmp_path):
        """Parallel first-readers of one image all succeed; sidecar loads."""
        path = _write_image(tmp_path / "img.png", seed=7)
        expected = open_image(path, num_channels=3)

        def read(_index: int) -> torch.Tensor:
            return open_image_cached(path, num_channels=3, mode="disk")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(read, range(16)))
        assert all(torch.equal(result, expected) for result in results)
        with np.load(tmp_path / "img.npy") as data:
            assert np.array_equal(data["image"], expected.numpy())


class TestRamPool:
    def test_pool_get_matches_direct_read(self, tmp_path, monkeypatch):
        """Every pooled image decodes identically to the source file."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(5)]
        _pin_shm(monkeypatch)
        pool = SharedImagePool.build(paths, num_channels=3)
        assert pool is not None
        assert pool.num_pooled == len(paths)
        for path in paths:
            cached = open_image_cached(path, num_channels=3, mode="ram", pool=pool)
            assert torch.equal(cached, open_image(path, num_channels=3))

    def test_ram_mode_falls_back_on_miss(self, tmp_path, monkeypatch):
        """ram mode without a pool, or on a pooled-out path, reads the source."""
        path = _write_image(tmp_path / "img.png", seed=8)
        expected = open_image(path, num_channels=3)
        assert torch.equal(
            open_image_cached(path, num_channels=3, mode="ram"), expected
        )
        _pin_shm(monkeypatch)
        pool = SharedImagePool.build([path], num_channels=3)
        assert pool is not None
        other = _write_image(tmp_path / "other.png", seed=9)
        assert torch.equal(
            open_image_cached(other, num_channels=3, mode="ram", pool=pool),
            open_image(other, num_channels=3),
        )

    def test_capacity_overflow_falls_back_to_source(self, tmp_path, monkeypatch):
        """Images that overflow the estimate stay unpooled; reads still work."""
        small = _write_image(tmp_path / "small.png", seed=0, size=(16, 16))
        huge = _write_image(tmp_path / "huge.png", seed=1, size=(64, 64))
        _pin_shm(monkeypatch)
        # sample=1 estimates from the small image, so the huge one overflows.
        pool = SharedImagePool.build(
            [small, huge], num_channels=3, sample=1, margin=1.0
        )
        assert pool is not None
        assert pool.num_pooled == 1
        assert pool.get(small) is not None
        assert torch.equal(
            open_image_cached(huge, num_channels=3, mode="ram", pool=pool),
            open_image(huge, num_channels=3),
        )

    def test_build_degrades_when_shm_is_short(self, tmp_path, monkeypatch):
        """A full /dev/shm makes build return None; room makes it build."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(2)]
        monkeypatch.setattr(image_cache_module, "_SHM_DIR", tmp_path)
        monkeypatch.setattr(
            shutil, "disk_usage", lambda _path: SimpleNamespace(free=1)
        )
        assert SharedImagePool.build(paths, num_channels=3) is None
        monkeypatch.setattr(
            shutil, "disk_usage", lambda _path: SimpleNamespace(free=1 << 30)
        )
        assert SharedImagePool.build(paths, num_channels=3) is not None

    def test_dataloader_workers_match_direct_read(self, tmp_path, monkeypatch):
        """num_workers>0 reads from the shared pool equal uncached reads."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(6)]
        direct = ImageDataset(None, paths, _PassthroughTransform())
        ram = ImageDataset(None, paths, _PassthroughTransform(), image_cache="ram")
        _pin_shm(monkeypatch)
        assert ram.build_image_pool() is not None
        direct_loader = DataLoader(direct, batch_size=2, shuffle=False)
        ram_loader = DataLoader(ram, batch_size=2, shuffle=False, num_workers=2)
        for direct_batch, ram_batch in zip(direct_loader, ram_loader):
            assert ram_batch["filename"] == direct_batch["filename"]
            for direct_view, ram_view in zip(
                direct_batch["views"], ram_batch["views"]
            ):
                assert torch.equal(direct_view, ram_view)


class TestImageDatasetWiring:
    def test_default_construction_matches_uncached_path(self, tmp_path):
        """Default image_cache="none" keeps today's behavior exactly."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(3)]
        dataset = ImageDataset(None, paths, _PassthroughTransform())
        assert dataset.image_cache == "none"
        assert dataset.image_pool is None
        for index, path in enumerate(paths):
            assert torch.equal(
                dataset[index]["views"][0], open_image(path, num_channels=3)
            )
        assert list(tmp_path.glob("*.npy")) == []

    def test_disk_cache_dataset_roundtrip_and_reuse(self, tmp_path):
        """A disk-cache dataset writes one sidecar per image and stays exact."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(3)]
        dataset = ImageDataset(
            None, paths, _PassthroughTransform(), image_cache="disk"
        )
        expected = [open_image(path, num_channels=3) for path in paths]
        for index in range(len(paths)):
            assert torch.equal(dataset[index]["views"][0], expected[index])
        assert len(list(tmp_path.glob("*.npy"))) == len(paths)
        for index in range(len(paths)):  # second pass serves from the sidecars
            assert torch.equal(dataset[index]["views"][0], expected[index])

    def test_build_image_pool_stores_pool(self, tmp_path, monkeypatch):
        """build_image_pool stores the pool on the dataset in ram mode."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(2)]
        dataset = ImageDataset(
            None, paths, _PassthroughTransform(), image_cache="ram"
        )
        _pin_shm(monkeypatch)
        pool = dataset.build_image_pool()
        assert pool is not None
        assert dataset.image_pool is pool
        assert torch.equal(
            open_image_cached(paths[0], num_channels=3, mode="ram", pool=pool),
            open_image(paths[0], num_channels=3),
        )

    def test_build_image_pool_degrades_to_disk(self, tmp_path, monkeypatch):
        """A shm shortfall degrades the dataset to disk cache, still exact."""
        paths = [_write_image(tmp_path / f"img_{i}.png", seed=i) for i in range(2)]
        dataset = ImageDataset(
            None, paths, _PassthroughTransform(), image_cache="ram"
        )
        monkeypatch.setattr(image_cache_module, "_SHM_DIR", tmp_path)
        monkeypatch.setattr(
            shutil, "disk_usage", lambda _path: SimpleNamespace(free=1)
        )
        assert dataset.build_image_pool() is None
        assert dataset.image_cache == "disk"
        assert dataset.image_pool is None
        assert torch.equal(
            dataset[0]["views"][0], open_image(paths[0], num_channels=3)
        )
