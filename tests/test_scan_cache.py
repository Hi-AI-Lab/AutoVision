# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Scan cache (design doc §4.8 D8): fingerprints, atomic store/load, and
dataset / CLI wiring on tmp_path fixtures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.cli.data_check import data_check
from autovision.data import scan_cache
from autovision.data.object_detection_dataset import ObjectDetectionDataset
from autovision.data.scan_cache import (
    CACHE_FILENAME,
    SCAN_CACHE_SCHEMA,
    load_scan_cache,
    manifest_records,
    scan_files,
    scan_fingerprint,
    store_scan_cache,
    summarize,
)
from autovision.runtime.env import Env


@dataclass
class _Args:
    """Minimal stand-in for the detection data args."""

    skip_if_annotations_missing: bool = False
    ignore_classes: set[int] = field(default_factory=set)


def _image(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)
    return path


def _row(path: Path, *, boxes=((0.5, 0.5, 0.2, 0.2),), labels=(0,)) -> dict:
    return {
        "image_path": str(path),
        "bboxes": json.dumps([list(box) for box in boxes]),
        "class_labels": json.dumps(list(labels)),
        "iscrowds": json.dumps([0] * len(labels)),
    }


@pytest.fixture(autouse=True)
def _default_env(monkeypatch: pytest.MonkeyPatch):
    """Normalize the toggle: tests must not depend on the developer env."""
    monkeypatch.delenv("AUTOVISION_SCAN_CACHE", raising=False)


# ---------------------------------------------------------------------------
# Fingerprint sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expect_equal"),
    [
        pytest.param(
            [("a.png", 10, 100)],
            [("a.png", 10, 200)],
            False,
            id="mtime-change-invalidates",
        ),
        pytest.param(
            [("a.png", 10, 100)],
            [("a.png", 20, 100)],
            False,
            id="size-change-invalidates",
        ),
        pytest.param(
            [("a.png", 10, 100)],
            [("a.png", 10, 100), ("b.png", 5, 100)],
            False,
            id="file-set-change-invalidates",
        ),
        pytest.param(
            [("a.png", 10, 100)],
            [("a.png", 10, 100)],
            True,
            id="identical-records-match",
        ),
        pytest.param(
            [("a.png", 10, 100), ("b.png", 5, 100)],
            [("b.png", 5, 100), ("a.png", 10, 100)],
            True,
            id="record-order-is-normalized",
        ),
    ],
)
def test_fingerprint_record_sensitivity(left, right, expect_equal):
    """File-record changes invalidate the fingerprint; order does not."""
    base = {"task": "object_detection"}
    assert (scan_fingerprint(left, base) == scan_fingerprint(right, base)) is expect_equal


def test_fingerprint_task_params_sensitivity():
    """Any task-parameter change forces a rescan via a new fingerprint."""
    records = [("a.png", 10, 100)]
    base = {"task": "object_detection", "num_channels": 3}
    assert scan_fingerprint(records, base) != scan_fingerprint(
        records, {"task": "object_detection", "num_channels": 1}
    )
    assert scan_fingerprint(records, base) != scan_fingerprint(
        records, {**base, "ignore_classes": [2]}
    )


def test_fingerprint_schema_sensitivity(monkeypatch: pytest.MonkeyPatch):
    """Bumping the schema version invalidates every existing fingerprint."""
    records = [("a.png", 10, 100)]
    params = {"task": "object_detection"}
    base = scan_fingerprint(records, params)
    assert SCAN_CACHE_SCHEMA == "1"  # guards the test's own assumption
    monkeypatch.setattr(scan_cache, "SCAN_CACHE_SCHEMA", "2")
    assert scan_fingerprint(records, params) != base


# ---------------------------------------------------------------------------
# store / load round-trip
# ---------------------------------------------------------------------------


def _payload(images: int = 2) -> dict:
    return {
        "per_file": {f"{index}.png": "ok" for index in range(images)},
        "counters": {"images": images, "backgrounds": 0, "corrupt": 0},
        "summary": f"{images} images, 0 backgrounds, 0 corrupt",
    }


def test_store_load_roundtrip_and_mismatches(tmp_path: Path):
    """Same fingerprint reloads the payload; unknown fp / wrong schema don't."""
    fingerprint = "f" * 64
    store_scan_cache(tmp_path, fingerprint, _payload())
    assert (tmp_path / CACHE_FILENAME).is_file()
    assert load_scan_cache(tmp_path, fingerprint) == _payload()
    assert load_scan_cache(tmp_path, "0" * 64) is None
    document = {"schema": "999", "reports": {fingerprint: _payload()}}
    (tmp_path / CACHE_FILENAME).write_text(json.dumps(document))
    assert load_scan_cache(tmp_path, fingerprint) is None


def test_store_preserves_other_fingerprints(tmp_path: Path):
    """Train/val share one root: a second store must not evict the first."""
    store_scan_cache(tmp_path, "a" * 64, _payload(1))
    store_scan_cache(tmp_path, "b" * 64, _payload(2))
    assert load_scan_cache(tmp_path, "a" * 64)["summary"].startswith("1 ")
    assert load_scan_cache(tmp_path, "b" * 64)["summary"].startswith("2 ")


def test_corrupt_cache_json_recovers(tmp_path: Path):
    """A hand-corrupted cache is skipped with a warning and rebuilt on store."""
    (tmp_path / CACHE_FILENAME).write_text("{not json at all")
    assert load_scan_cache(tmp_path, "a" * 64) is None
    store_scan_cache(tmp_path, "a" * 64, _payload())
    assert load_scan_cache(tmp_path, "a" * 64) == _payload()


# ---------------------------------------------------------------------------
# scan_files classification
# ---------------------------------------------------------------------------


def test_scan_files_classifications(tmp_path: Path):
    """ok / background / corrupt verdicts, counters, and the summary line."""
    good = _image(tmp_path / "images" / "good.png")
    empty = _image(tmp_path / "images" / "empty.png", (10, 200, 10))
    mismatched = _image(tmp_path / "images" / "mismatched.png", (60, 60, 10))
    truncated = tmp_path / "images" / "truncated.png"
    truncated.parent.mkdir(parents=True, exist_ok=True)
    truncated.write_bytes(b"not an image")
    rows = [
        _row(good),
        _row(empty, boxes=(), labels=()),
        _row(truncated),
        _row(tmp_path / "images" / "missing.png"),
        # box/label count mismatch is an incomplete annotation payload.
        _row(mismatched, boxes=((0.5, 0.5, 0.2, 0.2), (0.2, 0.2, 0.1, 0.1)), labels=(0,)),
    ]
    report = scan_files(tmp_path / "images", rows)
    per_file = report["per_file"]
    assert per_file["good.png"] == "ok"
    assert per_file["empty.png"] == "background"
    assert per_file["truncated.png"] == "corrupt"
    assert per_file["missing.png"] == "corrupt"
    assert per_file["mismatched.png"] == "corrupt"
    assert report["counters"] == {"images": 1, "backgrounds": 1, "corrupt": 3}
    assert report["summary"] == "1 images, 1 backgrounds, 3 corrupt"


def test_summarize_and_manifest_records(tmp_path: Path):
    """Census helper and stat records behave on present and missing files."""
    existing = _image(tmp_path / "a.png")
    records = manifest_records(tmp_path, [_row(tmp_path / "gone.png"), _row(existing)])
    assert records == sorted(records)
    sizes = {record[0]: (record[1], record[2]) for record in records}
    assert sizes["gone.png"] == (-1, -1)
    assert sizes["a.png"][0] == existing.stat().st_size
    counters, summary = summarize({"a.png": "ok", "b.png": "corrupt"})
    assert counters == {"images": 1, "backgrounds": 0, "corrupt": 1}
    assert summary == "1 images, 0 backgrounds, 1 corrupt"


# ---------------------------------------------------------------------------
# Dataset wiring
# ---------------------------------------------------------------------------


def _dataset_fixture(tmp_path: Path, *, num_images: int = 3) -> list[dict]:
    images_dir = tmp_path / "images"
    rows = []
    for index in range(num_images):
        path = _image(images_dir / f"img_{index}.png", (30 * index + 10, 30, 30))
        rows.append(_row(path))
    rows.append(_row(images_dir / "bg.png", boxes=(), labels=()))
    _image(images_dir / "bg.png", (10, 10, 200))
    return rows


def _count_scan_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = scan_cache.scan_files

    def counting(root, rows, *, num_channels: int = 3):
        calls.append(1)
        return original(root, rows, num_channels=num_channels)

    monkeypatch.setattr(scan_cache, "scan_files", counting)
    return calls


def test_dataset_scans_once_then_hits_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """First construction scans and stores; the second reuses the cache."""
    rows = _dataset_fixture(tmp_path)
    calls = _count_scan_calls(monkeypatch)

    first = ObjectDetectionDataset(_Args(), rows)
    assert calls == [1]
    assert first.scan_summary == "3 images, 1 backgrounds, 0 corrupt"
    assert first.scan_report is not None
    assert (tmp_path / "images" / CACHE_FILENAME).is_file()

    second = ObjectDetectionDataset(_Args(), rows)
    assert calls == [1]  # scan function did not run again
    assert second.scan_report == first.scan_report


def test_dataset_cache_invalidated_by_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Touching an image file changes mtime_ns and forces a rescan."""
    rows = _dataset_fixture(tmp_path)
    calls = _count_scan_calls(monkeypatch)
    ObjectDetectionDataset(_Args(), rows)
    ObjectDetectionDataset(_Args(), rows)
    assert calls == [1]
    os.utime(tmp_path / "images" / "img_0.png", None)
    ObjectDetectionDataset(_Args(), rows)
    assert calls == [1, 1]


def test_dataset_cache_invalidated_by_task_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A different num_channels is a different fingerprint: rescan, both kept."""
    rows = _dataset_fixture(tmp_path)
    calls = _count_scan_calls(monkeypatch)
    ObjectDetectionDataset(_Args(), rows, num_channels=3)
    ObjectDetectionDataset(_Args(), rows, num_channels=1)
    assert calls == [1, 1]
    document = json.loads((tmp_path / "images" / CACHE_FILENAME).read_text())
    assert document["schema"] == SCAN_CACHE_SCHEMA
    assert len(document["reports"]) == 2


def test_dataset_scan_cache_env_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AUTOVISION_SCAN_CACHE=0 disables the whole layer: no scan, no cache IO."""
    monkeypatch.setenv("AUTOVISION_SCAN_CACHE", "0")
    assert Env.SCAN_CACHE.value is False
    rows = _dataset_fixture(tmp_path)
    calls = _count_scan_calls(monkeypatch)
    dataset = ObjectDetectionDataset(_Args(), rows)
    assert calls == []
    assert dataset.scan_report is None
    assert not (tmp_path / "images" / CACHE_FILENAME).exists()


def test_dataset_scan_cache_env_off_ignores_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With the toggle off, an existing cache file is neither read nor rewritten."""
    rows = _dataset_fixture(tmp_path)
    ObjectDetectionDataset(_Args(), rows)
    cache_path = tmp_path / "images" / CACHE_FILENAME
    before = cache_path.read_text()
    monkeypatch.setenv("AUTOVISION_SCAN_CACHE", "0")
    calls = _count_scan_calls(monkeypatch)
    ObjectDetectionDataset(_Args(), rows)
    assert calls == []
    assert cache_path.read_text() == before


def test_dataset_use_scan_cache_false_bypasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """use_scan_cache=False bypasses the layer even with the toggle on."""
    rows = _dataset_fixture(tmp_path)
    calls = _count_scan_calls(monkeypatch)
    dataset = ObjectDetectionDataset(_Args(), rows, use_scan_cache=False)
    assert calls == []
    assert dataset.scan_report is None
    assert not (tmp_path / "images" / CACHE_FILENAME).exists()


def test_dataset_empty_manifest_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No rows (or rows without image paths): nothing scanned, nothing cached."""
    calls = _count_scan_calls(monkeypatch)
    empty = ObjectDetectionDataset(_Args(), [])
    assert calls == []
    assert empty.scan_report is None
    odd = ObjectDetectionDataset(_Args(), [{"other": "column"}])
    assert calls == []
    assert odd.scan_report is None
    assert not (tmp_path / CACHE_FILENAME).exists()


def test_dataset_getitem_unaffected_by_cache(tmp_path: Path):
    """Cache on: __getitem__ still yields the same image/boxes contract."""
    rows = _dataset_fixture(tmp_path)
    dataset = ObjectDetectionDataset(_Args(), rows)
    item = dataset[0]
    assert item["image"].shape[-3:] == (3, 48, 64)
    assert tuple(item["bboxes"].shape) == (1, 4)
    assert int(item["classes"][0]) == 0
    assert dataset.scan_report is not None


# ---------------------------------------------------------------------------
# data_check CLI wiring
# ---------------------------------------------------------------------------


def _coco_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny COCO dataset (train+val sharing one images dir) with a YAML."""
    images_dir = tmp_path / "images"
    categories = [{"id": 1, "name": "cat"}]
    splits: dict[str, list[str]] = {
        "train": ["train_a.png", "train_b.png"],
        "val": ["val.png"],
    }
    annotation_paths: dict[str, Path] = {}
    color_index = 0
    for split, file_names in splits.items():
        images = []
        annotations = []
        for index, file_name in enumerate(file_names):
            color_index += 1
            _image(images_dir / file_name, (30 * color_index, 60, 60))
            images.append(
                {"id": index + 1, "file_name": file_name, "width": 64, "height": 48}
            )
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "iscrowd": 0,
                }
            )
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps(
                {"images": images, "annotations": annotations, "categories": categories}
            )
        )
        annotation_paths[split] = path
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "format: coco\n"
        f"train:\n  annotations: {annotation_paths['train']}\n  images: {images_dir}\n"
        f"val:\n  annotations: {annotation_paths['val']}\n  images: {images_dir}\n"
    )
    return data_yaml, images_dir


def test_data_check_no_cache_bypass(tmp_path: Path):
    """no_cache=True runs the audit without reading or writing the cache."""
    data_yaml, images_dir = _coco_fixture(tmp_path)
    bypass = data_check(data_yaml, tmp_path / "qa-bypass", no_cache=True)
    assert not bypass.has_errors
    assert not (images_dir / CACHE_FILENAME).exists()

    seeded = data_check(data_yaml, tmp_path / "qa-seeded")
    assert not seeded.has_errors  # audit verdicts unchanged by the cache pass
    document = json.loads((images_dir / CACHE_FILENAME).read_text())
    assert document["schema"] == SCAN_CACHE_SCHEMA
    # Train and val share the images root: one report per split fingerprint.
    assert len(document["reports"]) == 2
    summaries = {
        load_scan_cache(images_dir, fingerprint)["summary"]
        for fingerprint in document["reports"]
    }
    assert summaries == {
        "2 images, 0 backgrounds, 0 corrupt",
        "1 images, 0 backgrounds, 0 corrupt",
    }


def test_cli_data_check_no_cache_flag(tmp_path: Path):
    """CLI-level smoke: the no_cache flag parses through the key=value shell."""
    data_yaml, images_dir = _coco_fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data-check",
            f"data={data_yaml}",
            f"out={tmp_path / 'qa-cli'}",
            "no_cache=true",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "report:" in result.output
    assert not (images_dir / CACHE_FILENAME).exists()
