# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for COCO polygon sanitization on import (design doc D19).

Malformed segmentation polygons must degrade to their bounding boxes (or
drop the instance when no bbox exists) instead of aborting dataset
construction; warnings aggregate once per annotations file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autovision.data.coco_instance_segmentation_dataset import (
    COCOInstanceSegmentationDataset,
)
from autovision.geometry.polygons import is_valid_polygon, polygon_from_bbox

RECT = [10, 12, 20, 16]  # x, y, w, h — covers the valid polygon below
VALID_POLYGON = [
    10.0,
    12.0,
    30.0,
    12.0,
    30.0,
    28.0,
    10.0,
    28.0,
]


def _payload(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal COCO payload with one 64x64 image and one category."""
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": annotations,
    }


def _build_dataset(
    tmp_path: Path, annotations: list[dict[str, Any]]
) -> COCOInstanceSegmentationDataset:
    """Write the payload (and the referenced image) and build the dataset."""
    from PIL import Image

    Image.new("RGB", (64, 64)).save(tmp_path / "a.jpg")
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_payload(annotations)))
    return COCOInstanceSegmentationDataset(path, images_dir=tmp_path)


def _annotation(**overrides: Any) -> dict[str, Any]:
    """One annotation with sensible defaults, overridden per test."""
    base: dict[str, Any] = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": list(RECT),
        "segmentation": [list(VALID_POLYGON)],
    }
    base.update(overrides)
    return base


class TestPolygonPrimitives:
    """``is_valid_polygon`` whitelist and the bbox fallback shape."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(list(VALID_POLYGON), True, id="valid-4-points"),
            pytest.param([0.0, 1.0, 2.0], False, id="odd-length"),
            pytest.param([0.0, 1.0, 2.0, 3.0], False, id="single-point"),
            pytest.param(
                [0.0, 1.0, float("nan"), 3.0, 4.0, 5.0], False, id="nan"
            ),
            pytest.param(
                [0, 1, 2, 3, 4, "x"], False, id="non-numeric"
            ),
            pytest.param("not-a-polygon", False, id="wrong-type"),
            pytest.param(np.arange(8.0), True, id="numpy-array"),
            pytest.param(np.zeros((2, 2)), False, id="2d-array"),
        ],
    )
    def test_is_valid_polygon(self, values: Any, expected: bool) -> None:
        """The whitelist accepts exactly flat, even, finite >= 3-point polys."""
        assert is_valid_polygon(values) is expected

    def test_polygon_from_bbox_shape(self) -> None:
        """The fallback polygon is the bbox rectangle as 4 (x, y) points."""
        poly = polygon_from_bbox([10, 12, 20, 16])
        assert poly.shape == (4, 2)
        assert poly.tolist() == [
            [10.0, 12.0],
            [30.0, 12.0],
            [30.0, 28.0],
            [10.0, 28.0],
        ]


class TestDatasetSanitization:
    """Malformed-polygon handling in ``COCOInstanceSegmentationDataset``."""

    def test_valid_polygons_unchanged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Well-formed imports stay silent and keep every instance."""
        with caplog.at_level(logging.WARNING):
            dataset = _build_dataset(tmp_path, [_annotation()])
        row = dataset[0]
        assert row["bboxes"].shape == (1, 4)
        assert row["masks"].shape == (1, 64, 64)
        assert row["masks"].sum() > 0
        assert "malformed" not in caplog.text

    def test_malformed_polygon_falls_back_to_bbox(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An odd-length polygon degrades to the bbox rectangle, not a crash."""
        annotations = [
            _annotation(id=1),
            _annotation(id=2, segmentation=[[0.0, 1.0, 2.0]]),
        ]
        with caplog.at_level(logging.WARNING):
            dataset = _build_dataset(tmp_path, annotations)
        row = dataset[0]
        assert row["bboxes"].shape == (2, 4)  # instance survives
        mask = row["masks"][1]
        x, y, w, h = RECT
        expected_area = w * h
        assert int(mask.sum()) == expected_area  # exactly the bbox rectangle
        assert caplog.text.count("fell back") == 1  # aggregated once per file

    def test_flat_single_polygon_accepted(self, tmp_path: Path) -> None:
        """COCO's flat single-polygon form decodes like the nested form."""
        annotations = [_annotation(segmentation=list(VALID_POLYGON))]
        dataset = _build_dataset(tmp_path, annotations)
        row = dataset[0]
        assert row["masks"].shape == (1, 64, 64)
        assert row["masks"].sum() > 0

    def test_empty_segmentation_falls_back_to_bbox(self, tmp_path: Path) -> None:
        """An empty ``segmentation`` list is malformed, not silently empty."""
        annotations = [_annotation(segmentation=[])]
        dataset = _build_dataset(tmp_path, annotations)
        assert dataset[0]["bboxes"].shape == (1, 4)

    def test_unusable_geometry_without_bbox_is_dropped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed polygon AND no bbox: the instance is dropped with a note."""
        annotations = [
            _annotation(id=1),
            _annotation(id=2, segmentation=[[0.0]], bbox=None),
        ]
        with caplog.at_level(logging.WARNING):
            dataset = _build_dataset(tmp_path, annotations)
        row = dataset[0]
        assert row["bboxes"].shape == (1, 4)  # only the healthy instance
        assert "no usable geometry" in caplog.text

    def test_compressed_rle_string_counts_unchanged(self, tmp_path: Path) -> None:
        """Compressed RLE (string counts) decodes directly — crowd form."""
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(np.eye(64, dtype=np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations = [_annotation(segmentation=rle)]
        dataset = _build_dataset(tmp_path, annotations)
        mask = dataset[0]["masks"][0]
        assert int(mask.sum()) == 64  # the diagonal survived intact

    def test_uncompressed_rle_int_counts_unchanged(self, tmp_path: Path) -> None:
        """Uncompressed RLE (int counts) goes through frPyObjects as before."""
        counts = [0] + [1, 63] * 64  # the 64x64 diagonal, column-major
        annotations = [_annotation(segmentation={"size": [64, 64], "counts": counts})]
        dataset = _build_dataset(tmp_path, annotations)
        mask = dataset[0]["masks"][0]
        assert int(mask.sum()) == 64
