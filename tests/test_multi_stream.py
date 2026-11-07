# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-source runtime: shared batching, per-source artifacts, reconnect."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference.multi_stream import (
    MultiStreamRunner,
    StreamSource,
)
from autovision.inference.video import VideoInfo


def _frames(count: int, value: int = 0) -> list[np.ndarray]:
    return [
        np.full((48, 64, 3), (value + index) % 255, dtype=np.uint8)
        for index in range(count)
    ]


class FakeReader:
    def __init__(
        self,
        frames=None,
        *,
        fail: bool = False,
        fps: float = 10.0,
        fail_after: int | None = None,
    ):
        self.info = VideoInfo(
            width=64, height=48, fps=fps, frame_count=len(frames or [])
        )
        self._frames = frames or []
        self._fail = fail
        self._fail_after = fail_after
        self.closed = False

    def frames(self):
        if self._fail:
            raise RuntimeError("reader failed")
        for index, frame in enumerate(self._frames):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("reader failed mid-stream")
            yield index, frame

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self):
        self.batches: list[int] = []

    def run_batch(self, images, *, threshold=0.5):
        self.batches.append(len(images))
        return [
            {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class FailingEngine:
    def run_batch(self, images, *, threshold=0.5):
        raise RuntimeError("inference exploded")


class PoisoningEngine:
    """Good records for bright frames; an unserializable extra for dark ones.

    The extra ``set`` survives ``InferenceResult`` construction but fails
    ``json.dumps`` — the multi-stream serializer's failure mode (H3-1e).
    Per-packet isolation means mixed-source batches fail only the dark
    packets, which makes the failure accounting deterministic.
    """

    def run_batch(self, images, *, threshold=0.5):
        records = []
        for image in images:
            record = {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            if int(image[0, 0, 0]) < 50:
                record["unserializable"] = {1, 2}
            records.append(record)
        return records


class BatchingProbeEngine:
    """Records per-call batch sizes; sleeps so the reader can refill."""

    def __init__(self, delay: float = 0.02) -> None:
        self.batches: list[int] = []
        self._delay = delay

    def run_batch(self, images, *, threshold=0.5):
        time.sleep(self._delay)
        self.batches.append(len(images))
        return [
            {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


def test_two_sources_write_per_source_jsonl(tmp_path):
    readers = {
        "file1.mp4": FakeReader(_frames(4, 10)),
        "file2.mp4": FakeReader(_frames(4, 100)),
    }
    runner = MultiStreamRunner(
        [StreamSource("cam1", "file1.mp4"), StreamSource("cam2", "file2.mp4")],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=2,
        queue_size=4,
        render=False,
        reader_factory=lambda url: readers[url],
    )
    result = runner.run()
    assert result.sources["cam1"].frames_written == 4
    assert result.sources["cam2"].frames_written == 4
    for name in ("cam1", "cam2"):
        path = tmp_path / "out" / name / "predictions.jsonl"
        assert len(path.read_text().splitlines()) == 4
        record = json.loads(path.read_text().splitlines()[0])
        assert record["source"] == name
        assert record["predictions"]["labels"] == [0]
        assert (tmp_path / "out" / "summary.json").is_file()


def test_max_frames_per_source(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        max_frames_per_source=2,
        reader_factory=lambda url: FakeReader(_frames(5)),
    )
    result = runner.run()
    assert result.sources["cam"].frames_read == 2
    assert result.sources["cam"].frames_written == 2


def test_live_source_reconnects(tmp_path):
    calls: list[int] = []

    def factory(url: str):
        calls.append(1)
        if len(calls) == 1:
            return FakeReader(fail=True)
        return FakeReader(_frames(3, 50))

    runner = MultiStreamRunner(
        [StreamSource("live", "rtsp://fake", reconnect=True, max_reconnects=1)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        reader_factory=factory,
    )
    result = runner.run()
    stats = result.sources["live"]
    assert stats.reconnects == 1
    assert stats.frames_written == 3
    assert stats.errors


def test_inference_failures_are_counted_per_frame(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FailingEngine(),
        batch_size=2,
        render=False,
        reader_factory=lambda url: FakeReader(_frames(3)),
    )
    result = runner.run()
    stats = result.sources["cam"]
    assert stats.frames_written == 0
    assert stats.frames_failed == 3
    assert stats.errors
    assert result.to_dict()["sources"]["cam"]["frames_failed"] == 3
    assert "failed=3" in result.summary()


def test_stream_stats_redact_credentials(tmp_path):
    runner = MultiStreamRunner(
        [
            StreamSource(
                "cam", "rtsp://user:secret@cam.local:554/live", reconnect=False
            )
        ],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        reader_factory=lambda url: FakeReader(_frames(1)),
    )
    result = runner.run()
    url = result.to_dict()["sources"]["cam"]["url"]
    assert url == "rtsp://***@cam.local:554/live"
    assert "secret" not in url


def test_file_source_does_not_reconnect(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("file", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        render=False,
        reader_factory=lambda url: FakeReader(fail=True),
    )
    result = runner.run()
    stats = result.sources["file"]
    assert stats.reconnects == 0
    assert stats.frames_written == 0
    assert stats.errors


def test_render_writes_annotated_video(tmp_path):
    pytest.importorskip("cv2", reason="opencv not installed")
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=True,
        reader_factory=lambda url: FakeReader(_frames(3)),
    )
    result = runner.run()
    video = result.sources["cam"].video
    assert video is not None and Path(video).is_file()
    assert Path(video).stat().st_size > 0


def test_serialization_failure_is_isolated_per_source(tmp_path):
    """H3-1e: one source's unserializable records fail that source only; the
    other sources complete normally and the failure is counted per frame."""
    readers = {
        "bad.mp4": FakeReader(_frames(3, 10)),
        "good.mp4": FakeReader(_frames(4, 200)),
    }
    runner = MultiStreamRunner(
        [StreamSource("bad", "bad.mp4"), StreamSource("good", "good.mp4")],
        tmp_path / "out",
        engine=PoisoningEngine(),
        batch_size=1,
        queue_size=4,
        render=False,
        reader_factory=lambda url: readers[url],
    )
    result = runner.run()
    bad = result.sources["bad"]
    good = result.sources["good"]
    assert bad.frames_written == 0
    assert bad.frames_failed == 3
    assert any("serialization failed" in message for message in bad.errors)
    assert good.frames_written == 4
    assert good.frames_failed == 0
    good_lines = (
        tmp_path / "out" / "good" / "predictions.jsonl"
    ).read_text().splitlines()
    assert len(good_lines) == 4


def test_batch_size_fills_from_a_single_active_source(tmp_path):
    """H3-5a: batch assembly takes up to ``batch_size`` frames per source per
    round, so one fast source fills multi-frame batches while the other is
    idle — the parameter is no longer pinned to one frame per source."""
    readers = {
        "busy.mp4": FakeReader(_frames(40, 0)),
        "idle.mp4": FakeReader([]),
    }
    engine = BatchingProbeEngine()
    runner = MultiStreamRunner(
        [StreamSource("busy", "busy.mp4"), StreamSource("idle", "idle.mp4")],
        tmp_path / "out",
        engine=engine,
        batch_size=4,
        queue_size=64,  # larger than the frame count: no drops, exact totals
        render=False,
        reader_factory=lambda url: readers[url],
    )
    result = runner.run()
    busy = result.sources["busy"]
    idle = result.sources["idle"]
    assert busy.frames_read == 40
    assert busy.frames_written == 40
    assert busy.frames_dropped == 0
    assert idle.frames_written == 0
    assert engine.batches
    assert max(engine.batches) >= 2
    assert all(size <= 4 for size in engine.batches)
    assert sum(engine.batches) == 40


def test_validation_and_queue_drop_policy(tmp_path):
    with pytest.raises(AutoVisionError, match="at least one source"):
        MultiStreamRunner([], tmp_path, engine=FakeEngine())
    with pytest.raises(AutoVisionError, match="unique"):
        MultiStreamRunner(
            [StreamSource("x", "a"), StreamSource("x", "b")], tmp_path,
            engine=FakeEngine(),
        )
    with pytest.raises(AutoVisionError, match="drop_policy"):
        MultiStreamRunner(
            [StreamSource("x", "a")], tmp_path, engine=FakeEngine(),
            drop_policy="bogus",
        )
    with pytest.raises(AutoVisionError, match="run_batch"):
        MultiStreamRunner([StreamSource("x", "a")], tmp_path, engine=object())


def test_reconnect_budget_recovers_after_a_healthy_stretch(tmp_path):
    """H3-5d: the give-up budget counts reconnects since the last recovery,
    not the run total. Two drops separated by a healthy stretch both get a
    reconnect under ``max_reconnects=1`` (the old cumulative budget would
    have given up on the second), while ``reconnects`` keeps the run's
    cumulative history.

    ``fps=0.1`` makes the recovery threshold ``int(0.1 * 30) = 3`` healthy
    frames, so the middle segment's five frames prove recovery.
    """
    segments = [
        FakeReader(_frames(3, 0), fps=0.1, fail_after=2),
        FakeReader(_frames(5, 50), fps=0.1),
        FakeReader(_frames(3, 100), fps=0.1, fail_after=1),
    ]
    served: dict[str, int] = {"n": 0}

    def factory(url: str) -> FakeReader:
        reader = segments[served["n"]]
        served["n"] += 1
        return reader

    runner = MultiStreamRunner(
        [StreamSource("live", "rtsp://fake", reconnect=True, max_reconnects=1)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        queue_size=16,
        render=False,
        reader_factory=factory,
    )
    result = runner.run()
    stats = result.sources["live"]
    assert stats.reconnects == 2  # the second drop still reconnects
    assert stats.frames_read == 8  # 2 + 5 + 1 across the three segments
    assert stats.frames_written == 8


def test_resolution_change_rotates_writer_segment(tmp_path):
    """H3-5e: a mid-run resolution change rotates the writer file name
    (``annotated-0001.mp4``) instead of overwriting the earlier segment;
    the JSONL stream is unaffected and both segments survive."""
    pytest.importorskip("cv2", reason="opencv not installed")
    frames = _frames(3, 10) + [
        np.full((24, 32, 3), (100 + index) % 255, dtype=np.uint8)
        for index in range(2)
    ]
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=True,
        reader_factory=lambda url: FakeReader(frames),
    )
    result = runner.run()
    stats = result.sources["cam"]
    first = tmp_path / "out" / "cam" / "annotated.mp4"
    second = tmp_path / "out" / "cam" / "annotated-0001.mp4"
    assert first.is_file() and first.stat().st_size > 0
    assert second.is_file() and second.stat().st_size > 0
    assert stats.video == second  # the summary points at the newest segment
    lines = (
        tmp_path / "out" / "cam" / "predictions.jsonl"
    ).read_text().splitlines()
    assert len(lines) == 5  # JSONL unaffected by the writer rotation


def test_circuit_breaker_aborts_failing_source_and_others_finish(tmp_path):
    """H3-5f: 10 failures inside the 20-frame sliding window (>50% rate)
    trip the per-source breaker — the bad source is aborted and marked in
    the summary while the good source completes every frame.

    The failures come through the per-packet serialization branch (same
    ``_note_frame_failure`` accounting the batch-exception branch feeds),
    because a batch-level engine exception would also count the good
    source's frames sharing the batch.
    """
    readers = {
        "bad.mp4": FakeReader(_frames(30, 10)),
        "good.mp4": FakeReader(_frames(30, 200)),
    }
    runner = MultiStreamRunner(
        [StreamSource("bad", "bad.mp4"), StreamSource("good", "good.mp4")],
        tmp_path / "out",
        engine=PoisoningEngine(),
        batch_size=1,
        # Larger than the frame count: the breaker must trip on PROCESSED
        # failures, not be starved by read-side queue drops.
        queue_size=64,
        render=False,
        reader_factory=lambda url: readers[url],
    )
    result = runner.run()
    bad = result.sources["bad"]
    good = result.sources["good"]
    assert bad.aborted is True
    assert bad.frames_failed == 10  # trips on the 10th consecutive failure
    assert bad.frames_written == 0
    assert good.aborted is False
    assert good.frames_failed == 0
    assert good.frames_written == 30
    summary = result.to_dict()["sources"]
    assert summary["bad"]["aborted"] is True
    assert summary["good"]["aborted"] is False
    assert "failed frames in the last" in bad.errors[-1]  # breaker reason
    assert (
        tmp_path / "out" / "bad" / "predictions.jsonl"
    ).read_text() == ""  # closed mid-run, left partial
    assert len(
        (tmp_path / "out" / "good" / "predictions.jsonl")
        .read_text()
        .splitlines()
    ) == 30
