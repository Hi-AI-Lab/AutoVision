# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training-side large-image slicing: ``split_large_image_dataset`` (D9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from autovision.data.split_large_images import (
    _clip_polygon_to_canvas,
    split_large_image_dataset,
    tile_window_starts,
)
from autovision.errors import AutoVisionError
from autovision.tasks.shared.tiling_utils import tile_image

SOURCE_W, SOURCE_H = 2048, 1024
WINDOW = 1024  # crop used across the slicing tests


def _write_image(path: Path, width: int = SOURCE_W, height: int = SOURCE_H) -> Path:
    """Write a uniform synthetic image (pixel content is geometry-irrelevant)."""
    Image.new("RGB", (width, height), (40, 90, 160)).save(path)
    return path


def _yolo_hbb(cls: int, cx: float, cy: float, w: float, h: float) -> str:
    """YOLO HBB row normalized on the 2048x1024 source (exact digits)."""
    return (
        f"{cls} {cx / SOURCE_W:.9f} {cy / SOURCE_H:.9f} "
        f"{w / SOURCE_W:.9f} {h / SOURCE_H:.9f}"
    )


def _yolo_obb(cls: int, points: list[tuple[float, float]]) -> str:
    """YOLO OBB row for pixel-space corners on the 2048x1024 source."""
    values: list[float] = []
    for x, y in points:
        values.extend((x / SOURCE_W, y / SOURCE_H))
    return f"{cls} " + " ".join(f"{value:.9f}" for value in values)


def _read_rows(path: Path) -> list[list[float]]:
    """Parse a YOLO label file into float rows (field 0 is the class)."""
    return [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tile_xy(tile_name: str) -> tuple[int, int]:
    """Decode the ``{stem}__{x}___{y}{suffix}`` source-coordinate encoding."""
    encoded = tile_name.split("__", 1)[1]
    x_part, _, y_part = encoded.rpartition("___")
    return int(x_part), int(y_part.rsplit(".", 1)[0])


def _assert_rows_equal(actual: list[list[float]], expected: list[tuple]) -> None:
    """Order-insensitive label-row comparison (class exact, coords approx)."""
    assert len(actual) == len(expected)
    for row, want in zip(sorted(actual), sorted(expected)):
        assert row[0] == want[0]
        assert row[1:] == pytest.approx(list(want[1:]), abs=1e-6)


def test_hbb_windows_and_target_assignment(tmp_path: Path) -> None:
    """Windows tile 2048x1024 flush to the edges; targets land per their IoF."""
    # Arrange: targets with hand-checked per-window IoF (crop 1024, thr 0.7).
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "aerial.png")
    targets = [
        (0, 256, 512, 100, 100),  # inside window x=0 only
        (1, 1024, 512, 200, 100),  # IoF 0.5 at x=0/x=1024 -> only x=819
        (0, 1900, 256, 200, 200),  # inside window x=1024 only
        (1, 915, 800, 130, 120),  # fully inside x=0 AND x=819 (overlap band)
    ]
    (labels / "aerial.txt").write_text(
        "\n".join(_yolo_hbb(*target) for target in targets) + "\n", encoding="utf-8"
    )

    # Act: positional images/labels/out pins the documented signature.
    out = split_large_image_dataset(images, labels, tmp_path / "out", crop=WINDOW, gap=0.2)

    # Assert: three flush windows; each target only in its IoF-passing windows.
    assert out == tmp_path / "out"
    x_starts = [0, 819, 1024]
    assert sorted(path.name for path in (out / "images").iterdir()) == sorted(
        f"aerial__{x}___0.png" for x in x_starts
    )
    for name in (out / "images").iterdir():
        with Image.open(name) as tile:
            assert tile.size == (WINDOW, WINDOW)  # full crop, no out-of-bounds
    expected = {
        0: [(0, 256, 512, 100, 100), (1, 915, 800, 130, 120)],
        819: [(1, 1024, 512, 200, 100), (1, 915, 800, 130, 120)],
        1024: [(0, 1900, 256, 200, 200)],
    }
    for x in x_starts:
        rows = _read_rows(out / "labels" / f"aerial__{x}___0.txt")
        wants = [
            (float(cls), (cx - x) / WINDOW, cy / WINDOW, w / WINDOW, h / WINDOW)
            for cls, cx, cy, w, h in expected[x]
        ]
        _assert_rows_equal(rows, wants)


@pytest.mark.parametrize(
    ("gap", "expected_x_starts"),
    [
        pytest.param(0.0, [0, 1024], id="gap-0-abutting-windows"),
        pytest.param(0.2, [0, 819, 1024], id="gap-0.2-overlapping-windows"),
    ],
)
def test_gap_controls_window_count_and_edge_alignment(
    tmp_path: Path, gap: float, expected_x_starts: list[int]
) -> None:
    """Stride follows gap; the last column sits flush with the right edge."""
    # Arrange: one source image, no labels (window layout is under test).
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "aerial.png")

    # Act
    out = split_large_image_dataset(images, None, tmp_path / "out", crop=WINDOW, gap=gap)

    # Assert: encoded tile coordinates equal the expected window starts.
    # Sort by the DECODED coordinates: name order is lexicographic
    # ("1024" < "819"), which is not the raster-scan layout under test.
    tiles = [
        path.name
        for path in sorted((out / "images").iterdir(), key=lambda p: _tile_xy(p.name))
    ]
    assert [_tile_xy(name)[0] for name in tiles] == expected_x_starts
    assert all(_tile_xy(name)[1] == 0 for name in tiles)
    assert max(_tile_xy(name)[0] for name in tiles) == SOURCE_W - WINDOW


@pytest.mark.parametrize(
    ("min_window_rate", "scenario"),
    [
        pytest.param(0.6, "all-windows-dropped-fallback", id="fallback-batch"),
        pytest.param(0.4, "windows-pass-the-normal-gate", id="normal-gate"),
    ],
)
def test_low_fill_windows_fall_back_to_best_batch(
    tmp_path: Path, min_window_rate: float, scenario: str
) -> None:
    """A 2048x513 image (window rate 0.501) survives via the fallback batch."""
    del scenario  # both routes must yield the identical tile set
    # Arrange: every window has rate 513/1024 ~ 0.501 (dropped at 0.6, kept
    # at 0.4); one target only inside window x=0.
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "wide.png", width=SOURCE_W, height=513)
    (labels / "wide.txt").write_text(
        _yolo_hbb(0, 350, 256, 100, 100) + "\n", encoding="utf-8"
    )
    assert tile_window_starts(513, SOURCE_W, WINDOW, 0.2) == [
        (0, 0), (819, 0), (1024, 0)
    ]

    # Act
    out = split_large_image_dataset(
        images, labels, tmp_path / "out",
        crop=WINDOW, gap=0.2, min_window_rate=min_window_rate,
    )

    # Assert: nothing is lost, tiles keep the clipped (flush-bottom) height.
    names = sorted(path.name for path in (out / "images").iterdir())
    assert names == ["wide__0___0.png", "wide__1024___0.png", "wide__819___0.png"]
    for name in names:
        with Image.open(out / "images" / name) as tile:
            assert tile.size == (WINDOW, 513)
        rows = _read_rows(out / "labels" / f"{Path(name).stem}.txt")
        assert len(rows) == (1 if name.startswith("wide__0___") else 0)


def test_obb_truncation_across_windows(tmp_path: Path) -> None:
    """OBB targets at the IoF threshold are kept un-clipped; below it dropped."""
    # Arrange: square A straddles x=1024 with window IoFs of exactly
    # 0.7 / 1.0 / 0.3; square B sits fully inside windows x=819 and x=1024.
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "obb.png")
    quad_a = [(884, 412), (1084, 412), (1084, 612), (884, 612)]
    quad_b = [(1404, 200), (1544, 200), (1544, 340), (1404, 340)]
    (labels / "obb.txt").write_text(
        _yolo_obb(2, quad_a) + "\n" + _yolo_obb(3, quad_b) + "\n", encoding="utf-8"
    )

    # Act
    out = split_large_image_dataset(
        images, labels, tmp_path / "out", crop=WINDOW, gap=0.2, label_format="obb"
    )

    # Assert: A kept at x=0 (IoF == threshold) and x=819, dropped at x=1024;
    # B kept at x=819 and x=1024. Kept rows carry the ORIGINAL polygon.
    def expected_obb(cls: int, quad: list[tuple[float, float]], wx: int) -> tuple:
        return (
            float(cls),
            *(coord for point in quad for coord in (
                (point[0] - wx) / WINDOW, point[1] / WINDOW
            )),
        )

    expected = {
        0: [expected_obb(2, quad_a, 0)],
        819: [expected_obb(2, quad_a, 819), expected_obb(3, quad_b, 819)],
        1024: [expected_obb(3, quad_b, 1024)],
    }
    for x, wants in expected.items():
        rows = _read_rows(out / "labels" / f"obb__{x}___0.txt")
        _assert_rows_equal(rows, wants)


def test_clip_polygon_to_canvas_cases() -> None:
    """Sutherland–Hodgman: inside unchanged, outside empty, boundary exact."""
    # Arrange / Act / Assert: fully-inside polygon comes back unchanged.
    inside = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
    clipped_in = _clip_polygon_to_canvas(np.asarray(inside), (100, 100))
    assert clipped_in.tolist() == [list(p) for p in inside]

    # Assert: fully-outside polygon degenerates below three points.
    outside = [(200.0, 10.0), (300.0, 10.0), (300.0, 50.0), (200.0, 50.0)]
    clipped_out = _clip_polygon_to_canvas(np.asarray(outside), (100, 100))
    assert clipped_out.shape[0] < 3

    # Assert: straddling polygon is cut exactly at the canvas edge.
    straddle = [(80.0, 40.0), (120.0, 40.0), (120.0, 80.0), (80.0, 80.0)]
    clipped_edge = _clip_polygon_to_canvas(np.asarray(straddle), (100, 100))
    assert clipped_edge.tolist() == [
        [80.0, 40.0], [100.0, 40.0], [100.0, 80.0], [80.0, 80.0],
    ]


def test_tiles_jsonl_manifest_traces_source_and_window(tmp_path: Path) -> None:
    """Every manifest row parses, names its source, and matches a real tile."""
    # Arrange: two targets in window x=0, one in x=1024.
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "aerial.png")
    (labels / "aerial.txt").write_text(
        "\n".join([
            _yolo_hbb(0, 256, 512, 100, 100),
            _yolo_hbb(1, 915, 800, 130, 120),
        ])
        + "\n",
        encoding="utf-8",
    )

    # Act
    out = split_large_image_dataset(images, labels, tmp_path / "out", crop=WINDOW, gap=0.2)

    # Assert: parseable JSONL, exact keys, traceable source/window/files.
    records = [
        json.loads(line)
        for line in (out / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(list((out / "images").iterdir()))
    assert all(
        set(record) == {"tile", "source", "window", "targets"} for record in records
    )
    by_tile = {record["tile"]: record for record in records}
    assert by_tile["aerial__0___0.png"] == {
        "tile": "aerial__0___0.png",
        "source": "aerial.png",
        "window": [0, 0, WINDOW, WINDOW],
        "targets": 2,
    }
    assert by_tile["aerial__1024___0.png"]["window"] == [1024, 0, WINDOW, WINDOW]
    assert by_tile["aerial__1024___0.png"]["targets"] == 0
    for record in records:
        assert (out / "images" / str(record["tile"])).is_file()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"crop": 32}, "crop", id="crop-below-minimum"),
        pytest.param({"crop": 1024.5}, "crop", id="crop-not-an-integer"),
        pytest.param({"crop": "1024"}, "crop", id="crop-string"),
        pytest.param({"gap": 0.9}, "gap", id="gap-at-upper-bound"),
        pytest.param({"gap": -0.1}, "gap", id="gap-negative"),
        pytest.param({"iof_thr": 0.4}, "iof_thr", id="iof-below-range"),
        pytest.param({"iof_thr": 1.1}, "iof_thr", id="iof-above-range"),
        pytest.param({"min_window_rate": 0.0}, "min_window_rate", id="rate-zero"),
        pytest.param({"min_window_rate": 1.5}, "min_window_rate", id="rate-above-one"),
        pytest.param({"label_format": "polygon"}, "label_format", id="unknown-format"),
    ],
)
def test_rejects_invalid_parameters(tmp_path: Path, kwargs: dict, match: str) -> None:
    """Semantically invalid arguments raise AutoVisionError before any work."""
    # Arrange: a valid minimal dataset (errors must come from validation).
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "a.png")
    (labels / "a.txt").write_text("", encoding="utf-8")

    # Act / Assert
    with pytest.raises(AutoVisionError, match=match):
        split_large_image_dataset(images, labels, tmp_path / "out", **kwargs)


def test_images_only_when_labels_dir_is_none(tmp_path: Path) -> None:
    """Test-only slicing: images and manifest exist, no labels are written."""
    # Arrange
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "aerial.png")

    # Act
    out = split_large_image_dataset(images, None, tmp_path / "out", crop=WINDOW, gap=0.2)

    # Assert
    assert (out / "images" / "aerial__0___0.png").is_file()
    assert not (out / "labels").exists()
    records = [
        json.loads(line)
        for line in (out / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["targets"] == 0 for record in records)


def test_missing_label_file_is_background(tmp_path: Path) -> None:
    """An image without a label file slices fine with empty background labels."""
    # Arrange: labeled.png has a target; unlabeled.png has no label file.
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "labeled.png")
    _write_image(images / "unlabeled.png")
    (labels / "labeled.txt").write_text(
        _yolo_hbb(0, 256, 512, 100, 100) + "\n", encoding="utf-8"
    )

    # Act
    out = split_large_image_dataset(images, labels, tmp_path / "out", crop=WINDOW, gap=0.2)

    # Assert: the labeled tile keeps its target, the other is an empty file.
    assert len(_read_rows(out / "labels" / "labeled__0___0.txt")) == 1
    assert (out / "labels" / "unlabeled__0___0.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("height", "width", "tile", "overlap"),
    [
        pytest.param(64, 100, 32, 0.25, id="wide-image-with-overlap"),
        pytest.param(100, 64, 32, 0.0, id="tall-image-without-overlap"),
        pytest.param(16, 16, 32, 0.2, id="image-smaller-than-tile"),
    ],
)
def test_shared_window_enumerator_matches_tile_image(
    height: int, width: int, tile: int, overlap: float
) -> None:
    """The data-layer enumerator reproduces tile_image's coordinates exactly."""
    # Arrange / Act: run the inference-side tiler on a synthetic image.
    _, coordinates = tile_image(
        torch.zeros(3, height, width), overlap, tile, padding_mode="pad"
    )

    # Assert: its window origins equal the shared enumerator's output.
    assert [tuple(map(int, point)) for point in coordinates.tolist()] == (
        tile_window_starts(height, width, tile, overlap)
    )
