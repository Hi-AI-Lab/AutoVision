# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO ``data.yaml`` snapshots and added-split drift detection.

Covers the DATA-01 (native Ultralytics data.yaml interpretation), DATA-02
(splits that appear after the snapshot fail verification) and DATA-22
(truncated snapshot schema fails loudly) hardening designs. Snapshot
files only need bytes — the
snapshot digests sizes, it never decodes images.
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from autovision.data.versioning import (
    dataset_snapshot,
    load_snapshot,
    verify_dataset_snapshot,
)
from autovision.errors import AutoVisionError


def _yolo_tree(root: Path) -> Path:
    """Ultralytics layout: <root>/images/<split> + mirrored labels."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(2):
            (root / "images" / split / f"img_{index}.jpg").write_bytes(b"fake-jpeg")
            (root / "labels" / split / f"img_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def _yolo_spec(root: Path, **overrides) -> dict:
    spec = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "dog", 1: "cat"},
    }
    spec.update(overrides)
    return spec


class TestYoloSpecSnapshot:
    """DATA-01: a standard YOLO data.yaml produces real train/val splits."""

    def test_string_split_values_become_named_splits(self, tmp_path):
        """train/val strings are split roots; names never becomes a path."""
        root = _yolo_tree(tmp_path / "dataset")
        payload = dataset_snapshot(_yolo_spec(root), tmp_path / "snap.json", strict=True)
        assert sorted(payload["splits"]) == ["train", "val"]
        # 2 images + 2 mirrored label files (review P3-8 pins the labels too)
        assert payload["splits"]["train"]["files"] == 4
        assert payload["names"] == {0: "dog", 1: "cat"}

    def test_names_strings_are_not_collected_as_paths(self, tmp_path):
        """strict=True must not choke on class-name strings as fake paths."""
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root, names={0: "does-not-exist", 1: "also-missing"})
        payload = dataset_snapshot(spec, tmp_path / "snap.json", strict=True)
        assert sorted(payload["splits"]) == ["train", "val"]

    def test_split_roots_anchor_to_the_path_key(self, tmp_path):
        """Relative roots resolve against `path`, not the spec directory."""
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root)
        payload = dataset_snapshot(spec, tmp_path / "snap.json")
        assert payload["splits"]["train"]["roots"] == [
            f"images/train:{root / 'images' / 'train'}",
            f"images/train@labels:{root / 'labels' / 'train'}",
        ]

    def test_list_roots_land_in_one_split(self, tmp_path):
        """`val: [a, b]` is one split with two labeled roots."""
        root = _yolo_tree(tmp_path / "dataset")
        (root / "images" / "val_extra").mkdir()
        (root / "images" / "val_extra" / "extra.jpg").write_bytes(b"fake-jpeg")
        spec = _yolo_spec(root, val=["images/val", "images/val_extra"])
        payload = dataset_snapshot(spec, tmp_path / "snap.json")
        # 3 images + 2 mirrored label files (labels/val exists; the extra
        # split dir has no labels counterpart)
        assert payload["splits"]["val"]["files"] == 5
        assert len(payload["splits"]["val"]["manifest"]) == 5

    def test_missing_split_error_names_only_split_paths(self, tmp_path):
        """strict errors reference train/val paths, never class names."""
        root = tmp_path / "dataset"
        (root / "images" / "train").mkdir(parents=True)
        spec = _yolo_spec(root)  # val/ and labels/ missing entirely
        with pytest.raises(AutoVisionError, match="missing path") as excinfo:
            dataset_snapshot(spec, tmp_path / "snap.json", strict=True)
        assert "val" in str(excinfo.value)
        assert "does-not-exist" not in str(excinfo.value)

    def test_format_key_opts_out_of_yolo_interpretation(self, tmp_path):
        """An explicit format key keeps the generic (shared) behavior.

        strict=False because the generic rules DO collect the class-name
        strings as (missing) paths — that pre-existing behavior is exactly
        why the YOLO interpretation exists; the format key is only an
        escape hatch, not a recommendation.
        """
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root, format="coco")
        payload = dataset_snapshot(spec, tmp_path / "snap.json", strict=False)
        assert sorted(payload["splits"]) == ["(shared)"]
        assert payload["names"] is None

    def test_non_string_split_value_fails_with_guidance(self, tmp_path):
        """A dict split under YOLO detection errors with the escape hatch."""
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root, train={"images": "images/train"})
        with pytest.raises(AutoVisionError, match="format"):
            dataset_snapshot(spec, tmp_path / "snap.json")

    def test_moved_tree_verifies_with_its_own_data_yaml(self, tmp_path):
        """Labels derive from raw spec strings: a moved copy verifies clean."""
        root = _yolo_tree(tmp_path / "dataset")
        spec_path = tmp_path / "data.yaml"
        spec_path.write_text(yaml.safe_dump(_yolo_spec(root)))
        dataset_snapshot(spec_path, tmp_path / "snap.json", deep=True)

        copy_root = tmp_path / "moved"
        shutil.copytree(root, copy_root)
        moved_spec_path = tmp_path / "moved.yaml"
        moved_spec_path.write_text(yaml.safe_dump(_yolo_spec(copy_root)))
        report = verify_dataset_snapshot(tmp_path / "snap.json", spec_override=moved_spec_path)
        assert report["ok"] is True, report["differences"]

    def test_yolo_label_edit_fails_verification(self, tmp_path):
        root = _yolo_tree(tmp_path / "dataset")
        dataset_snapshot(_yolo_spec(root), tmp_path / "snap.json", deep=True)
        victim = root / "images" / "train" / "img_0.jpg"
        victim.write_bytes(b"tampered")  # same size class change: new bytes
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False


class TestAddedSplitDetection:
    """DATA-02: splits that appear after the snapshot fail verification."""

    def test_generic_added_split_is_reported(self, tmp_path):
        root = tmp_path / "data"
        train_images = root / "images"
        train_images.mkdir(parents=True)
        (train_images / "a.jpg").write_bytes(b"fake-jpeg")
        spec = {
            "format": "coco",
            "train": {"images": str(train_images)},
            "val": {"images": str(root / "val_images")},  # missing at snapshot
        }
        dataset_snapshot(spec, tmp_path / "snap.json", strict=False)
        assert "val" not in load_snapshot(tmp_path / "snap.json")["splits"]

        val_images = root / "val_images"
        val_images.mkdir()
        (val_images / "b.jpg").write_bytes(b"fake-jpeg")
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False
        added = [d for d in report["differences"] if d["field"] == "added_split"]
        assert len(added) == 1 and added[0]["split"] == "val"
        assert "1 files" in added[0]["actual"]
        assert "val" in report["checked"]

    def test_yolo_added_split_is_reported(self, tmp_path):
        """The audit's exact scenario: snapshot took train, val appeared."""
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root)
        # Snapshot with val present, then simulate "val declared later" by
        # rewriting the stored spec: cheapest way is snapshot-with-val,
        # remove the recorded split, and re-verify against the same spec.
        dataset_snapshot(spec, tmp_path / "snap.json")
        payload = load_snapshot(tmp_path / "snap.json")
        del payload["splits"]["val"]
        (tmp_path / "trimmed.json").write_text(json.dumps(payload))

        report = verify_dataset_snapshot(tmp_path / "trimmed.json")
        assert report["ok"] is False
        assert any(
            d["field"] == "added_split" and d["split"] == "val"
            for d in report["differences"]
        )

    def test_empty_declared_split_is_not_reported_as_added(self, tmp_path):
        """A declared split with no surviving roots stays silent (strict=False)."""
        root = tmp_path / "data"
        images = root / "images"
        images.mkdir(parents=True)
        (images / "a.jpg").write_bytes(b"fake-jpeg")
        spec = {
            "format": "coco",
            "train": {"images": str(images)},
            "val": {"images": str(root / "never_created")},
        }
        dataset_snapshot(spec, tmp_path / "snap.json", strict=False)
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is True
        assert not [d for d in report["differences"] if d["field"] == "added_split"]


class TestSnapshotSchemaHardening:
    """DATA-22: truncated snapshots fail loudly instead of bare KeyErrors."""

    def test_missing_splits_mapping_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"kind": "autovision-dataset-snapshot"}))
        with pytest.raises(AutoVisionError, match="splits"):
            load_snapshot(bad)
        with pytest.raises(AutoVisionError, match="splits"):
            verify_dataset_snapshot(bad)

    def test_snapshot_without_spec_demands_override(self, tmp_path):
        bad = tmp_path / "nospec.json"
        bad.write_text(
            json.dumps({"kind": "autovision-dataset-snapshot", "splits": {}})
        )
        with pytest.raises(AutoVisionError, match="spec_override"):
            verify_dataset_snapshot(bad)


class TestReviewFollowUps:
    """Fixes from the post-implementation review round."""

    def test_yolo_snapshot_covers_mirrored_labels(self, tmp_path):
        """Review P3-8: the mirrored labels directory is part of the
        snapshot — editing a label after a deep snapshot must fail
        verification, not just the mmap fingerprint."""
        root = _yolo_tree(tmp_path / "dataset")
        dataset_snapshot(_yolo_spec(root), tmp_path / "snap.json", deep=True)
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is True

        victim = root / "labels" / "train" / "img_0.txt"
        original = victim.read_bytes()
        victim.write_bytes(bytes(reversed(original)))  # same length, new bytes
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False
        assert any(d["field"] == "content_digest" for d in report["differences"])

    def test_scalar_split_value_fails_with_guidance(self, tmp_path):
        """Review P3-1: `train: 5` under YOLO detection errors with the
        format-escape guidance, not a bare list() TypeError."""
        root = _yolo_tree(tmp_path / "dataset")
        spec = _yolo_spec(root, train=5)
        with pytest.raises(AutoVisionError, match="format"):
            dataset_snapshot(spec, tmp_path / "snap.json")

    def test_non_object_split_entry_rejected(self, tmp_path):
        """Review P3-2: a hand-edited non-object split entry fails loudly
        instead of a bare string-indices TypeError inside verification."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(
            {"kind": "autovision-dataset-snapshot", "splits": {"train": "garbage"}}
        ))
        with pytest.raises(AutoVisionError, match="not objects"):
            load_snapshot(bad)
