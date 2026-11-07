# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Data-layer hardening batch 2 (H3-2i).

H3-2i: pair matching between images and their masks/depth maps is keyed by
the suffix-stripped RELATIVE path first. The old bare-stem matching
cross-matched files in nested trees where two directories both contain
e.g. ``a.jpg`` — every image silently paired with the SAME (last-sorted)
``a.png``. The bare stem remains a fallback for differently-shaped trees,
with a one-time warning when the stem is shared across mask directories
(the old silent mis-pairing). Flat layouts must behave exactly as before.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autovision.data.depth_estimation_dataset import list_depth_pairs
from autovision.data.image_dataset import ImageDataset
from autovision.data.mask_panoptic_segmentation_dataset import list_panoptic_pairs
from autovision.data.semantic_segmentation_dataset import list_segmentation_pairs


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
    return path


def _write_mask(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").save(path)
    return path


# ── semantic segmentation ──────────────────────────────────────────────────


class TestSemanticPairMatching:
    def test_nested_same_stem_images_pair_with_their_own_masks(self, tmp_path: Path):
        """H3-2i: two ``a.jpg`` under different subdirectories each pair with
        the mask in THEIR subdirectory (previously both got the same,
        last-sorted mask)."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        left = _write_image(images / "left" / "a.jpg")
        right = _write_image(images / "right" / "a.jpg")
        _write_mask(masks / "left" / "a.png", value=1)
        _write_mask(masks / "right" / "a.png", value=2)

        pairs, _num_classes = list_segmentation_pairs(images, masks)
        assert len(pairs) == 2
        mapping = {str(image): mask for image, mask in pairs}
        assert mapping[str(left)] == masks / "left" / "a.png"
        assert mapping[str(right)] == masks / "right" / "a.png"

    def test_flat_layout_is_unchanged(self, tmp_path: Path):
        """H3-2i: a flat images/masks layout keeps the exact stem pairing
        (relative keys coincide with stems there)."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_image(images / "a.jpg")
        _write_image(images / "b.jpg")
        _write_mask(masks / "a.png", value=1)
        _write_mask(masks / "b.png", value=2)

        pairs, _num_classes = list_segmentation_pairs(images, masks)
        assert [(image.name, mask.name) for image, mask in pairs] == [
            ("a.jpg", "a.png"),
            ("b.jpg", "b.png"),
        ]

    def test_cross_directory_stem_fallback_warns_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """H3-2i: when the relative key misses and the bare stem matches
        masks in several directories, one warning names the risk."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_image(images / "a.jpg")
        _write_mask(masks / "left" / "a.png", value=1)
        _write_mask(masks / "right" / "a.png", value=2)

        with caplog.at_level(logging.WARNING, logger="autovision"):
            pairs, _ = list_segmentation_pairs(images, masks)
        assert len(pairs) == 1  # deterministic pick: first sorted mask
        assert pairs[0][1] == masks / "left" / "a.png"
        warnings = [r for r in caplog.records if "H3-2i" in r.getMessage()]
        assert len(warnings) == 1
        assert str(images / "a.jpg") in warnings[0].getMessage()


# ── panoptic segmentation ──────────────────────────────────────────────────


class TestPanopticPairMatching:
    def test_nested_same_stem_images_pair_with_their_own_masks(self, tmp_path: Path):
        """H3-2i: panoptic pair listing matches by relative path too."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        left = _write_image(images / "left" / "a.jpg")
        right = _write_image(images / "right" / "a.jpg")
        _write_mask(masks / "left" / "a.png", value=1)
        _write_mask(masks / "right" / "a.png", value=2)

        pairs = list_panoptic_pairs(images, masks)
        assert len(pairs) == 2
        mapping = {str(image): mask for image, mask in pairs}
        assert mapping[str(left)] == masks / "left" / "a.png"
        assert mapping[str(right)] == masks / "right" / "a.png"

    def test_flat_layout_is_unchanged(self, tmp_path: Path):
        """H3-2i: flat panoptic layouts keep the stem pairing and the
        no-pairs error."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        first = _write_image(images / "a.jpg")
        _write_mask(masks / "a.png", value=1)

        pairs = list_panoptic_pairs(images, masks)
        assert pairs == [(first, masks / "a.png")]


# ── depth estimation ───────────────────────────────────────────────────────


class TestDepthPairMatching:
    def test_nested_same_stem_images_pair_with_their_own_depths(self, tmp_path: Path):
        """H3-2i: nested ``a.png`` images each pair with their own ``a.npy``."""
        images = tmp_path / "images"
        depths = tmp_path / "depths"
        left = _write_image(images / "left" / "a.png")
        right = _write_image(images / "right" / "a.png")
        (depths / "left").mkdir(parents=True)
        (depths / "right").mkdir(parents=True)
        np.save(depths / "left" / "a.npy", np.full((16, 16), 1.5, dtype=np.float32))
        np.save(depths / "right" / "a.npy", np.full((16, 16), 3.5, dtype=np.float32))

        pairs = list_depth_pairs(images, depths)
        assert len(pairs) == 2
        mapping = {str(image): depth for image, depth in pairs}
        assert mapping[str(left)] == depths / "left" / "a.npy"
        assert mapping[str(right)] == depths / "right" / "a.npy"

    def test_npy_still_wins_over_png_in_flat_layouts(self, tmp_path: Path):
        """H3-2i: the flat ``.npy``-over-``.png`` preference is preserved."""
        images = tmp_path / "images"
        depths = tmp_path / "depths"
        image = _write_image(images / "a.png")
        _write_mask(depths / "a.png", value=7)
        np.save(depths / "a.npy", np.full((16, 16), 2.5, dtype=np.float32))

        pairs = list_depth_pairs(images, depths)
        assert pairs == [(image, depths / "a.npy")]


# ── pretraining ImageDataset ───────────────────────────────────────────────


class TestImageDatasetMaskMatching:
    def test_nested_same_stem_images_get_their_own_masks(self, tmp_path: Path):
        """H3-2i: ImageDataset resolves ``sub/a.jpg`` to ``sub/a.png`` under
        the mask dir (the old flat ``<stem>.png`` lookup cross-matched)."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        left = _write_image(images / "left" / "a.jpg")
        right = _write_image(images / "right" / "a.jpg")
        _write_mask(masks / "left" / "a.png", value=1)
        _write_mask(masks / "right" / "a.png", value=2)

        dataset = ImageDataset(
            image_dir=images,
            image_filenames=[left, right],
            transform=None,
            mask_dir=masks,
        )
        assert dataset._mask_path_for(left) == masks / "left" / "a.png"
        assert dataset._mask_path_for(right) == masks / "right" / "a.png"

    def test_flat_layout_resolves_to_the_same_file_as_before(self, tmp_path: Path):
        """H3-2i: a flat image/mask layout keeps the legacy ``<stem>.png``
        result (image_dir=None, absolute filenames like the CLI passes)."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        image = _write_image(images / "a.jpg")
        _write_mask(masks / "a.png", value=1)

        dataset = ImageDataset(
            image_dir=None,
            image_filenames=[image],
            transform=None,
            mask_dir=masks,
        )
        assert dataset._mask_path_for(image) == masks / "a.png"

    def test_ambiguous_stem_fallback_warns_once_per_process(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """H3-2i: a stem fallback onto a shared stem warns exactly once in
        this process, naming the image and the picked mask."""
        from autovision.data.image_dataset import _STEM_FALLBACK_WARNED

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        image = _write_image(images / "a.jpg")
        _write_mask(masks / "left" / "a.png", value=1)
        _write_mask(masks / "right" / "a.png", value=2)

        dataset = ImageDataset(
            image_dir=images,
            image_filenames=[image],
            transform=None,
            mask_dir=masks,
        )
        with caplog.at_level(logging.WARNING, logger="autovision"):
            mask_path = dataset._mask_path_for(image)
            dataset._mask_path_for(image)  # second call must stay silent
        assert mask_path == masks / "left" / "a.png"
        warnings = [r for r in caplog.records if "H3-2i" in r.getMessage()]
        assert len(warnings) == 1
        _STEM_FALLBACK_WARNED.discard((str(masks), "a"))

    def test_missing_mask_returns_the_legacy_flat_path(self, tmp_path: Path):
        """H3-2i: a stem with no mask anywhere still resolves to the legacy
        flat path, so the decode fails on the missing file as before."""
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        image = _write_image(images / "a.jpg")

        dataset = ImageDataset(
            image_dir=images,
            image_filenames=[image],
            transform=None,
            mask_dir=masks,
        )
        assert dataset._mask_path_for(image) == masks / "a.png"
