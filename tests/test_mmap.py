# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import json
import pickle

import pytest

from autovision.data.mmap_helpers import (
    _increment_ref_count,
    _ref_count_path,
    dataset_mmap_context,
    get_dataset_mmap_file,
    get_dataset_temp_mmap_path,
)
from autovision.data.serialize.memory_mapped_sequence import (
    MemoryMappedSequence,
    write_items_to_file,
)


def items(n: int):
    return [{"filenames": f"img_{i:03d}.jpg", "index": i} for i in range(n)]


def test_corrupt_refcount_is_reset(tmp_path):
    mmap = tmp_path / "data.mmap"
    mmap.write_bytes(b"")
    ref = _ref_count_path(mmap)
    ref.write_text("{not valid json")
    _increment_ref_count(mmap)
    assert json.loads(ref.read_text()) == 1


class TestMemoryMappedSequence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file(items(100), path)
        seq = MemoryMappedSequence.from_file(path)
        assert len(seq) == 100
        assert seq[0] == {"filenames": "img_000.jpg", "index": 0}
        assert seq[99]["index"] == 99

    def test_empty(self, tmp_path):
        path = tmp_path / "empty.mmap"
        write_items_to_file([], path)
        seq = MemoryMappedSequence.from_file(path)
        assert len(seq) == 0

    def test_slicing(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file(items(10), path)
        seq = MemoryMappedSequence.from_file(path)
        assert [row["index"] for row in seq[2:5]] == [2, 3, 4]
        assert [row["index"] for row in seq[8:2:-2]] == [8, 6, 4]
        assert seq[3:3] == []

    def test_pickle_carries_no_data(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file(items(5), path)
        seq = MemoryMappedSequence.from_file(path)
        state = pickle.dumps(seq)
        assert len(state) < 1000  # path + columns only
        restored = pickle.loads(state)
        assert len(restored) == 5
        assert restored[2] == seq[2]

    def test_lazy_table_per_pid(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file(items(3), path)
        seq = MemoryMappedSequence.from_file(path)
        assert seq._table is None  # from_file does not keep the table open
        assert seq[0]["index"] == 0  # table opened lazily on first access
        assert seq._pid is not None
        seq._table = None
        assert seq[1]["index"] == 1  # reopened lazily again

    def test_truncated_file_fails_with_rebuild_hint(self, tmp_path):
        """A truncated IPC file reports corruption, not a raw Arrow error."""
        path = tmp_path / "truncated.mmap"
        write_items_to_file(items(10), path)
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(RuntimeError, match="corrupt or truncated"):
            MemoryMappedSequence.from_file(path)


class TestMmapPath:
    def test_collision_guard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run1"
        path = get_dataset_temp_mmap_path(data="data", out_dir=out)
        path.touch()
        with pytest.raises(RuntimeError, match="multiple runs"):
            get_dataset_temp_mmap_path(data="data", out_dir=out)
        # resume/overwrite tolerate the existing file
        assert get_dataset_temp_mmap_path(data="data", out_dir=out, overwrite=True) == path
        assert (
            get_dataset_temp_mmap_path(data="data", out_dir=out, resume_interrupted=True)
            == path
        )

    def test_reuse_mode_ignores_out_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("AUTOVISION_MMAP_REUSE_FILE", "1")
        path1 = get_dataset_temp_mmap_path(data="shared-data", out_dir=tmp_path / "run1")
        path2 = get_dataset_temp_mmap_path(data="shared-data", out_dir=tmp_path / "run2")
        assert path1 == path2
        path3 = get_dataset_temp_mmap_path(data="other-data", out_dir=tmp_path / "run1")
        assert path3 != path1


class TestMmapContext:
    def test_write_and_cleanup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run"
        with dataset_mmap_context(lambda: items(7), data="d", out_dir=out) as seq:
            assert len(seq) == 7
            assert seq[6]["filenames"] == "img_006.jpg"
        # refcount dropped to zero -> file removed (reuse off)
        assert not seq.path.exists()

    def test_nested_contexts_share_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run"
        with dataset_mmap_context(lambda: items(7), data="d", out_dir=out) as s1:
            # File now exists; a second context on the same run reuses it.
            with dataset_mmap_context(lambda: items(7), data="d", out_dir=out) as s2:
                assert s1.path == s2.path
                assert len(s2) == 7
        # refcount decremented twice -> file removed
        assert not s1.path.exists()

    def test_premature_exit_still_cleans(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run"
        with pytest.raises(ValueError):
            with dataset_mmap_context(lambda: items(3), data="d", out_dir=out):
                raise ValueError("boom")
        cache = tmp_path / "cache"
        leftovers = [p for p in cache.iterdir() if p.suffix in (".mmap", ".ref_count")]
        assert leftovers == []


class TestMmapFingerprintHardening:
    def test_editing_a_yolo_label_changes_the_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir()
        labels.mkdir()
        label = labels / "a.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        out = tmp_path / "run"
        first = get_dataset_temp_mmap_path(data=str(images), out_dir=out)
        import os

        os.utime(label, (label.stat().st_atime, label.stat().st_mtime + 5))
        second = get_dataset_temp_mmap_path(data=str(images), out_dir=out)
        assert first != second

    def test_ignore_classes_participate_in_the_fingerprint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run"
        first = get_dataset_temp_mmap_path(
            data="data", out_dir=out, fingerprint_extra="[1]"
        )
        second = get_dataset_temp_mmap_path(
            data="data", out_dir=out, fingerprint_extra="[2]"
        )
        assert first != second

    def test_overwrite_rebuilds_existing_mmap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        out = tmp_path / "run"
        with dataset_mmap_context(lambda: items(1), data="d", out_dir=out) as seq:
            path = seq.path
            assert len(seq) == 1
        # Recreate the file as a stale leftover (no active refcount).
        write_items_to_file(items(1), path)
        with dataset_mmap_context(
            lambda: items(3), data="d", out_dir=out, overwrite=True
        ) as seq:
            assert len(seq) == 3
        assert not path.exists()


def _concurrent_opener(barrier, path, rows, results, index):
    """Child worker: synchronize on the barrier, then race the open path.

    H3-2g/H4-2: every process must succeed and take exactly one reference.
    """
    try:
        barrier.wait(timeout=60)
        seq = get_dataset_mmap_file(lambda: items(rows), path)
        results.put((index, len(seq), None))
    except Exception as error:  # noqa: BLE001 - reported to the parent verbatim
        results.put((index, None, repr(error)))


class TestConcurrentOpen:
    def test_double_process_open_race_keeps_refcount_exact(self, tmp_path):
        """H3-2g/H4-2: several processes racing the exists->open->increment
        sequence on one mmap file all succeed and the refcount ends at
        exactly the number of openers — no lost increments and no ghost
        references (a refcount pointing at a deleted file)."""
        import multiprocessing

        path = tmp_path / "shared.mmap"
        write_items_to_file(items(50), path)
        # One active holder in the parent so the cleanup path can never
        # remove the file mid-race; the refcount must end at 1 + N.
        _increment_ref_count(path)

        num_processes = 4
        ctx = multiprocessing.get_context("fork")
        barrier = ctx.Barrier(num_processes)
        results = ctx.Queue()
        processes = [
            ctx.Process(
                target=_concurrent_opener,
                args=(barrier, path, 50, results, index),
            )
            for index in range(num_processes)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
        assert all(process.exitcode == 0 for process in processes)

        outcomes = [results.get(timeout=10) for _ in range(num_processes)]
        for _, rows_seen, error in outcomes:
            assert error is None, error
            assert rows_seen == 50  # every opener actually opened the data

        ref = _ref_count_path(path)
        assert json.loads(ref.read_text()) == 1 + num_processes
        assert path.exists()  # no ghost reference: the file is still real


class TestDataFingerprintImageManifest:
    """DATA-03: image add/delete must invalidate the mmap row cache.

    The YOLO/pretrain loaders enumerate image files per mmap row, so the
    directory fingerprint has to cover the image manifest — not only the
    label files. Otherwise a newly added image silently never enters
    training and a deleted one crashes mid-epoch in ``open_image``.
    """

    def _tree(self, tmp_path):
        images = tmp_path / "train" / "images"
        labels = tmp_path / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "img_0.jpg").write_bytes(b"fake-jpeg")
        (labels / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        return images, labels

    def test_added_image_changes_the_fingerprint(self, tmp_path):
        from autovision.data.mmap_helpers import _data_fingerprint

        images, _labels = self._tree(tmp_path)
        before = _data_fingerprint(str(images))
        (images / "img_1.jpg").write_bytes(b"fake-jpeg")
        assert _data_fingerprint(str(images)) != before

    def test_deleted_image_changes_the_fingerprint(self, tmp_path):
        from autovision.data.mmap_helpers import _data_fingerprint

        images, _labels = self._tree(tmp_path)
        (images / "img_1.jpg").write_bytes(b"fake-jpeg")
        before = _data_fingerprint(str(images))
        (images / "img_1.jpg").unlink()
        assert _data_fingerprint(str(images)) != before

    def test_label_edit_still_changes_the_fingerprint(self, tmp_path):
        """The pre-existing label-content invalidation must not regress."""
        from autovision.data.mmap_helpers import _data_fingerprint

        images, labels = self._tree(tmp_path)
        before = _data_fingerprint(str(images))
        # A different LENGTH, not just different values: mtime granularity
        # can swallow a same-size rewrite inside one timestamp tick.
        (labels / "img_0.txt").write_text("0 0.40 0.4 0.2 0.2\n")
        assert _data_fingerprint(str(images)) != before

    def test_yolo_args_mmap_hash_tracks_the_image_set(self, tmp_path):
        from autovision.data.yolo_object_detection_dataset import (
            YOLOObjectDetectionDataArgs,
        )

        images, _labels = self._tree(tmp_path)
        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train=tmp_path / "train" / "images",
            val=tmp_path / "train" / "images",
            names={0: "dog"},
        )
        before = args.train_data_mmap_hash()
        (images / "img_1.jpg").write_bytes(b"fake-jpeg")
        assert args.train_data_mmap_hash() != before


class TestOverwriteDeletionOwnership:
    """DATA-12: overwrite deletes belong to rank zero only."""

    def test_rank_zero_overwrite_replaces_the_file(self, tmp_path, monkeypatch):
        import autovision.data.mmap_helpers as helpers

        monkeypatch.setattr(helpers, "is_local_rank_zero", lambda: True)
        path = tmp_path / "data.mmap"
        write_items_to_file(items(1), path)
        _ref_count_path(path).write_text(json.dumps(1))

        sequence = helpers.get_dataset_mmap_file(lambda: items(2), path, overwrite=True)
        assert len(sequence) == 2  # rebuilt with the new row set
        assert path.is_file()

    def test_non_zero_rank_overwrite_never_deletes(self, tmp_path, monkeypatch):
        """A slow rank used to unlink the file rank0 had JUST published;
        as a reader it must leave the mmap (and refcount) in place."""
        import autovision.data.mmap_helpers as helpers

        monkeypatch.setattr(helpers, "is_local_rank_zero", lambda: False)
        path = tmp_path / "data.mmap"
        write_items_to_file(items(3), path)
        _ref_count_path(path).write_text(json.dumps(1))

        sequence = helpers.get_dataset_mmap_file(lambda: items(3), path, overwrite=True)
        assert len(sequence) == 3  # opened the existing rows (reader role)
        assert path.is_file()
        assert json.loads(_ref_count_path(path).read_text()) == 2  # referenced, not deleted
