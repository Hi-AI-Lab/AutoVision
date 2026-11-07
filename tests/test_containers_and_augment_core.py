# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for D27 (metric containers) and D24 (augmentation protocol)."""

from __future__ import annotations

import numpy as np
import pytest

from autovision.augment.core import (
    SEMANTIC_IGNORE_VALUE,
    GeoParams,
    GeoTransform,
    nearest_resize_to,
)
from autovision.metrics.containers import TaskMetrics, empty_empty_score


class _Accuracy(TaskMetrics):
    """Minimal concrete container: mean of a per-batch correct-count ratio."""

    def process(self) -> dict[str, float]:
        correct = self._concatenate(self.stats["correct"])
        total = self._concatenate(self.stats["total"])
        if not total.size:
            return {"accuracy": 0.0}
        return {"accuracy": float(correct.sum() / max(total.sum(), 1))}


class TestTaskMetrics:
    """Lazy stats buffer, merge protocol, and column validation (D27)."""

    def test_update_and_process(self) -> None:
        """Per-batch rows concatenate at process time, not per batch."""
        metrics = _Accuracy(["correct", "total"])
        metrics.update(correct=np.array([3]), total=np.array([4]))
        metrics.update(correct=np.array([1]), total=np.array([4]))
        assert metrics.process() == {"accuracy": 0.5}

    def test_update_rejects_missing_and_unknown_columns(self) -> None:
        """Both silent failure directions raise with the column names."""
        metrics = _Accuracy(["correct", "total"])
        with pytest.raises(ValueError, match="missing column"):
            metrics.update(correct=np.array([1]))
        with pytest.raises(ValueError, match="undeclared column"):
            metrics.update(correct=np.array([1]), total=np.array([1]), extra=2)

    def test_export_merge_roundtrip(self) -> None:
        """Two ranks' raw buffers merge into one (DDP gather shape)."""
        rank0 = _Accuracy(["correct", "total"])
        rank1 = _Accuracy(["correct", "total"])
        rank0.update(correct=np.array([2]), total=np.array([4]))
        rank1.update(correct=np.array([3]), total=np.array([4]))
        rank0.merge_state(rank1.export_state())
        assert rank0.process() == {"accuracy": 0.625}

    def test_merge_rejects_unknown_columns(self) -> None:
        """Foreign schemas fail loudly at merge, not at process time."""
        rank0 = _Accuracy(["correct", "total"])
        with pytest.raises(ValueError, match="unknown column"):
            rank0.merge_state({"bogus": [1]})

    def test_reset_clears_buffer(self) -> None:
        """A fresh evaluation pass starts from an empty buffer."""
        metrics = _Accuracy(["correct", "total"])
        metrics.update(correct=np.array([1]), total=np.array([1]))
        metrics.reset()
        assert metrics.stats["correct"] == []

    def test_empty_empty_score_convention(self) -> None:
        """No GT + no predictions is a perfect image, not 0/0 nan (D27)."""
        assert empty_empty_score(0, 0) == 1.0
        assert empty_empty_score(3, 0) == 0.0
        assert empty_empty_score(0, 2) == 0.0
        assert empty_empty_score(3, 2) is None  # real work: caller scores


class _FakeFlip(GeoTransform):
    """Reference transform: horizontal flip driven by one sampled coin."""

    def sample_params(self, image: np.ndarray) -> GeoParams:
        return GeoParams(random_draws=(float(np.random.random()),))

    def apply_image(self, image: np.ndarray, params: GeoParams) -> np.ndarray:
        if params.random_draws[0] < 0.5:
            return image
        return np.flip(image, axis=1).copy()

    def apply_instances(self, instances: dict, params: GeoParams) -> dict:
        if params.random_draws[0] < 0.5:
            return instances
        width = instances["width"]
        boxes = instances["boxes"].copy()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return {**instances, "boxes": boxes}

    def apply_semantic_mask(
        self, semantic_mask: np.ndarray, params: GeoParams
    ) -> np.ndarray:
        if params.random_draws[0] < 0.5:
            return semantic_mask
        return np.flip(semantic_mask, axis=1).copy()

    def apply_depth(self, depth: np.ndarray, params: GeoParams) -> np.ndarray:
        if params.random_draws[0] < 0.5:
            return depth
        return np.flip(depth, axis=1).copy()


class TestGeoTransformProtocol:
    """One sampled params object drives all four targets (design D24)."""

    def test_all_targets_moved_by_the_same_draw(self) -> None:
        """Image/mask/depth flip together or not at all — shared identity."""
        image = np.arange(16, dtype=np.float32).reshape(2, 8)
        mask = np.zeros((2, 8), dtype=np.uint8)
        mask[0, 0] = 7  # asymmetric marker
        depth = np.full((2, 8), 3, dtype=np.float16)
        depth[0, 0] = 9
        transform = _FakeFlip()
        params = GeoParams(random_draws=(1.0,))  # force the flip branch
        out = transform(
            image=image,
            params=params,
            instances={"boxes": np.array([[1.0, 0.0, 3.0, 2.0]]), "width": 8.0},
            semantic_mask=mask,
            depth=depth,
        )
        assert out["image"][0, 0] == 7.0  # flipped column order
        assert out["semantic_mask"][0, 7] == 7
        assert out["depth"][0, 7] == 9
        assert out["instances"]["boxes"][0].tolist() == [5.0, 0.0, 7.0, 2.0]

    def test_undeclared_mask_hook_raises(self) -> None:
        """A geometry transform without a mask hook must not pass silently."""

        class _ImageOnly(GeoTransform):
            def sample_params(self, image: np.ndarray) -> GeoParams:
                return GeoParams()

            def apply_image(self, image: np.ndarray, params: GeoParams) -> np.ndarray:
                return image

        with pytest.raises(NotImplementedError, match="apply_semantic_mask"):
            _ImageOnly()(image=np.zeros((2, 2)), semantic_mask=np.zeros((2, 2)))

    def test_nearest_resize_keeps_dtype_and_values(self) -> None:
        """Label/depth maps resize by exact nearest neighbor, dtype intact."""
        mask = np.arange(25, dtype=np.uint8).reshape(5, 5)
        resized = nearest_resize_to(mask, (10, 10))
        assert resized.shape == (10, 10)
        assert resized.dtype == np.uint8
        # Top-left 2x2 block of the 10x10 all maps to source pixel (0, 0).
        assert (resized[:2, :2] == 0).all()

    def test_semantic_ignore_value_is_the_documented_constant(self) -> None:
        """The ignore fill rides on the module constant, not magic 255s."""
        assert SEMANTIC_IGNORE_VALUE == 255
