# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Threaded stream pipeline: stage accounting, backpressure and failures."""

from __future__ import annotations

import json
import logging
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference import run_stream
from autovision.inference.video import redact_source
from autovision.tracking import LineZone, PolygonZone

cv2 = pytest.importorskip("cv2", reason="opencv not installed")


def _clip(tmp_path, frames: int = 5) -> str:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), (index * 30) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


class _Engine:
    backend = "torch"

    def __init__(self, *, short: bool = False, fail: bool = False) -> None:
        self.short = short
        self.fail = fail

    def run_batch(self, images, threshold=0.5):
        if self.fail:
            raise RuntimeError("inference exploded")
        count = len(images) - 1 if self.short else len(images)
        return [
            {
                "bboxes": torch.tensor([[2, 2, 20, 18]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in range(count)
        ]


class _MovingEngine:
    """A consistently tracked box moving down the frame."""

    backend = "torch"

    def __init__(self) -> None:
        self.index = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            y = 2.0 + self.index * 3.0
            self.index += 1
            records.append(
                {
                    "bboxes": torch.tensor([[4, y, 20, y + 10]], dtype=torch.float32),
                    "scores": torch.tensor([0.95]),
                    "labels": torch.tensor([0]),
                }
            )
        return records


class _NanEngine(_Engine):
    def run_batch(self, images, threshold=0.5):
        records = super().run_batch(images, threshold=threshold)
        for record in records:
            record["scores"] = torch.tensor([float("nan")])
        return records


class _SlowFirstBatchEngine:
    """The first batch sleeps; every later batch is instant."""

    backend = "torch"

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._calls = 0

    def run_batch(self, images, threshold=0.5):
        if self._calls == 0:
            time.sleep(self._delay)
        self._calls += 1
        return [
            {
                "bboxes": torch.tensor([[2, 2, 20, 18]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class _BurstEngine:
    """Two-frame detection bursts every ``period`` frames; empty between.

    Each burst spawns one confirmed track and the empty stretch pushes it
    past ``track_buffer``, so every burst ends in exactly one retirement.
    """

    backend = "torch"

    def __init__(self, period: int = 40) -> None:
        self._period = period
        self._frame = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            if self._frame % self._period in (0, 1):
                records.append(
                    {
                        "bboxes": torch.tensor([[2.0, 2.0, 20.0, 18.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([0]),
                    }
                )
            else:
                records.append(
                    {
                        "bboxes": torch.zeros((0, 4)),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )
            self._frame += 1
        return records


class _ClassificationEngine:
    backend = "torch"

    def run_batch(self, images, threshold=0.5):
        return [
            {"scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
            for _ in images
        ]


class _ScriptedStream:
    """VideoFrameStream stand-in driven by a script of events.

    Events: ``("frame", array)`` yields the next ``(index, frame)``;
    ``("reconnect",)`` bumps ``reconnects_used``/``reconnect_generation``
    exactly as a successful live reconnect would (H3-5c);
    ``("drop_end",)`` ends the iteration the way a dropped live source
    does (``stream_ended_after_drop=True``).
    """

    def __init__(
        self,
        source,
        *,
        max_frames=None,
        reconnect_attempts=0,
        reconnect_delay_s=1.0,
        events=(),
    ) -> None:
        self.info = SimpleNamespace(fps=10.0)
        self.reconnects_used = 0
        self.reconnect_generation = 0
        self.stream_ended_after_drop = False
        self._events = list(events)

    def __iter__(self):
        index = 0
        for event in self._events:
            kind = event[0]
            if kind == "reconnect":
                self.reconnects_used += 1
                self.reconnect_generation += 1
            elif kind == "drop_end":
                self.stream_ended_after_drop = True
                return
            else:
                yield index, event[1]
                index += 1

    def close(self) -> None:
        return None


class _SlowEngine:
    """Sleeps per batch — a stand-in for a GPU-bound engine."""

    backend = "torch"

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def run_batch(self, images, threshold=0.5):
        time.sleep(self._delay)
        return [
            {
                "bboxes": torch.tensor([[2.0, 2.0, 20.0, 18.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class _ReconnectSceneEngine:
    """One 20px-tall box descending exactly 1px per frame, teleporting back
    up at the reconnect boundary.

    1px steps keep consecutive-frame IoU at ~0.9, above ByteTrack's 0.8
    match threshold, so each session holds ONE stable id — larger steps
    would thrash unconfirmed tracks that never publish (the confirmation
    gate needs two IoU-matched hits). Each session's descent crosses the
    line zone exactly once: bottom_center goes 44.3 -> 45.3 past y=45.
    """

    backend = "torch"

    def __init__(self, boundary: int = 10, total: int = 24) -> None:
        self._boundary = boundary
        self._total = total
        self._frame = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            index = self._frame
            self._frame += 1
            if index < self._boundary:
                y = 20.3 + index
            else:
                y = 14.3 + (index - self._boundary)
            records.append(
                {
                    "bboxes": torch.tensor([[4.0, y, 20.0, y + 20.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            )
        return records


class TestPipeline:
    def test_stage_timings_and_fps(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(),
            batch_size=2, render=True, prefetch=1, writer_buffer=1,
        )
        assert result.frames == 5
        assert result.fps > 0
        assert result.decode_s > 0
        assert result.infer_s > 0
        assert result.render_s > 0
        assert result.write_s > 0
        assert result.video_path is not None

    def test_render_false_skips_writer(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(), render=False
        )
        assert result.video_path is None
        assert result.write_s == 0.0
        assert result.render_s == 0.0
        assert result.frames == 5

    def test_jsonl_entries_stay_ordered(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(), batch_size=3
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert [record["frame"] for record in records] == list(range(5))
        assert all("predictions" in record for record in records)

    def test_short_engine_result_raises(self, tmp_path):
        with pytest.raises(AutoVisionError, match="records for"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(short=True)
            )

    def test_engine_failure_propagates(self, tmp_path):
        with pytest.raises(RuntimeError, match="inference exploded"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(fail=True)
            )

    def test_writer_failure_propagates_without_hanging(self, tmp_path, monkeypatch):
        import autovision.inference.stream as stream_module

        class _BadWriter:
            def __init__(self, *args, **kwargs):
                self.path = Path(args[0])

            def write(self, frame):
                raise RuntimeError("encoder exploded")

            def close(self):
                pass

        monkeypatch.setattr(stream_module, "AnnotatedVideoWriter", _BadWriter)
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="encoder exploded"):
            run_stream(
                _clip(tmp_path, frames=60),
                tmp_path / "out",
                engine=_Engine(),
                batch_size=1,
                render=True,
            )
        # The producer must notice the writer failure instead of processing
        # all 60 frames first.
        assert time.perf_counter() - started < 3.0

    def test_tracking_requires_a_detection_result(self, tmp_path):
        with pytest.raises(AutoVisionError, match="requires a detection model"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_ClassificationEngine(),
                track=True,
            )

    def test_prefetch_and_writer_buffer_validation(self, tmp_path):
        with pytest.raises(AutoVisionError, match="prefetch"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(), prefetch=0
            )
        with pytest.raises(AutoVisionError, match="writer_buffer"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(),
                writer_buffer=0,
            )

    def test_zones_require_tracking(self, tmp_path):
        with pytest.raises(AutoVisionError, match="track=True"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_Engine(),
                line_zones=[LineZone((0, 10), (32, 10))],
            )

    def test_smoothing_requires_tracking(self, tmp_path):
        with pytest.raises(AutoVisionError, match="smooth=True"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(), smooth=True
            )

    def test_duplicate_zone_names_are_rejected(self, tmp_path):
        with pytest.raises(AutoVisionError, match="unique"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_Engine(),
                track=True,
                line_zones=[LineZone((0, 10), (32, 10), name="z")],
                polygon_zones=[
                    PolygonZone(
                        [(0, 0), (32, 0), (32, 24), (0, 24)], name="z"
                    )
                ],
            )

    def test_line_and_polygon_counts_reach_jsonl_and_result(self, tmp_path):
        line = LineZone((0, 22), (32, 22), name="gate")
        polygon = PolygonZone(
            [(-100, -100), (100, -100), (100, 100), (-100, 100)],
            name="frame",
        )
        result = run_stream(
            _clip(tmp_path, frames=8),
            tmp_path / "out",
            engine=_MovingEngine(),
            track=True,
            line_zones=[line],
            polygon_zones=[polygon],
            render=True,
            batch_size=2,
        )
        assert result.zones["lines"]["gate"]["in"] >= 1
        assert result.zones["polygons"]["frame"]["total"] >= 1
        assert result.video_path is not None and result.video_path.exists()
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert all("zones" in record for record in records)
        assert "gate" in records[-1]["zones"]["lines"]

    def test_smoothing_does_not_change_jsonl_predictions(self, tmp_path):
        result = run_stream(
            _clip(tmp_path),
            tmp_path / "out",
            engine=_MovingEngine(),
            track=True,
            smooth=True,
            render=True,
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert len(records[0]["predictions"]["bboxes"][0]) == 4
        assert result.video_path is not None

    def test_non_finite_scores_become_null(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_NanEngine(), render=False
        )
        record = json.loads(result.predictions_path.read_text().splitlines()[0])
        assert record["predictions"]["scores"] == [None]

    def test_classification_records_render(self, tmp_path):
        result = run_stream(
            _clip(tmp_path),
            tmp_path / "out",
            engine=_ClassificationEngine(),
            class_names={0: "cat"},
            render=True,
        )
        assert result.frames == 5
        assert result.video_path is not None


def test_slow_first_batch_does_not_hang_on_a_full_read_queue(tmp_path):
    """H1-2 regression: the reader used to give up on its sentinel after 5s.

    With 3 frames and prefetch=2 the reader enqueues everything while the
    engine is still asleep, then faces a FULL queue with the sentinel —
    the exact reproduction of the permanent main-loop spin. The run must
    return within a wall-clock bound with a complete JSONL.
    """
    started = time.perf_counter()
    result = run_stream(
        _clip(tmp_path, frames=3),
        tmp_path / "out",
        engine=_SlowFirstBatchEngine(delay=0.5),
        batch_size=1,
        prefetch=2,
        render=False,
    )
    assert time.perf_counter() - started < 10.0
    assert result.frames == 3
    lines = result.predictions_path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["frame"] for line in lines] == [0, 1, 2]


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)
def test_reader_death_without_sentinel_ends_the_run(tmp_path, monkeypatch, caplog):
    """H1-2 guard: a reader that dies cleanly without delivering a sentinel
    must end the run (with a warning) instead of spinning the main loop.

    The poisoned sentinel put kills the reader thread by design; the
    thread-exception warning it raises is filtered above.
    """
    real_put = queue.Queue.put

    def poisoned_put(q, item, *args, **kwargs):
        # Only the sentinel is blocked: the reader thread dies on its very
        # first delivery attempt, after every frame is already enqueued.
        if item is None:
            raise ValueError("sentinel poisoned")
        return real_put(q, item, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "put", poisoned_put)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        result = run_stream(
            _clip(tmp_path, frames=3),
            tmp_path / "out",
            engine=_Engine(),
            render=False,
        )
    assert result.frames == 3
    assert len(result.predictions_path.read_text().splitlines()) == 3
    assert "sentinel" in caplog.text


def test_reader_error_propagates_without_hanging(tmp_path, monkeypatch):
    """H1-2: a reader that crashes mid-iteration surfaces its error through
    the run's existing reader_error semantics instead of hanging the run."""

    class _ExplodingStream:
        def __init__(self, source, **kwargs) -> None:
            self.info = SimpleNamespace(fps=10.0)
            self.reconnects_used = 0
            self.reconnect_generation = 0
            self.stream_ended_after_drop = False

        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("decoder exploded")

        def close(self) -> None:
            return None

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", _ExplodingStream)
    with pytest.raises(RuntimeError, match="decoder exploded"):
        run_stream(
            _clip(tmp_path, frames=3),
            tmp_path / "out",
            engine=_Engine(),
            render=False,
        )


def test_run_stream_drains_retired_tracks_into_stats(tmp_path):
    """H1-3: long runs drain retired tracks per flush so the tracker's
    ``_removed`` backlog stays bounded; the count lands in
    ``StreamRunResult.tracks_removed`` (8 bursts -> 8 retirements)."""
    result = run_stream(
        _clip(tmp_path, frames=300),
        tmp_path / "out",
        engine=_BurstEngine(period=40),
        track=True,
        render=False,
        batch_size=1,
    )
    assert result.frames == 300
    lines = result.predictions_path.read_text().splitlines()
    assert len(lines) == 300
    assert result.tracks == 8
    assert result.tracks_removed == 8


def test_redact_source_hides_url_credentials():
    assert (
        redact_source("rtsp://user:pass@cam.local:554/live")
        == "rtsp://***@cam.local:554/live"
    )
    assert redact_source("/data/clip.mp4") == "/data/clip.mp4"


def test_max_lag_frames_drops_oldest_and_accounts_every_frame(tmp_path, monkeypatch):
    """H3-5b: with ``max_lag_frames`` set, a reader that outruns a slow
    engine drops the oldest queued frames and accounts for every one —
    frames + frames_dropped == frames decoded, and the drop only touches
    the read side. The default (None) keeps blocking backpressure and
    delivers every frame."""
    total = 40
    events = [
        ("frame", np.full((24, 32, 3), (index * 30) % 255, dtype=np.uint8))
        for index in range(total)
    ]

    def factory(*args, **kwargs):
        return _ScriptedStream(*args, events=events, **kwargs)

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", factory)

    result = run_stream(
        "fake://cam", tmp_path / "dropping", engine=_SlowEngine(0.005),
        batch_size=1, prefetch=2, max_lag_frames=2, render=False,
    )
    assert result.frames_dropped > 0
    assert result.frames + result.frames_dropped == total

    full = run_stream(
        "fake://cam", tmp_path / "blocking", engine=_SlowEngine(0.005),
        batch_size=1, prefetch=2, render=False,
    )
    assert full.frames_dropped == 0
    assert full.frames == total


def test_max_lag_frames_validation(tmp_path):
    """H3-5b: the lag cap must be a positive frame count when given."""
    with pytest.raises(AutoVisionError, match="max_lag_frames"):
        run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(),
            max_lag_frames=0,
        )


def test_reconnect_generation_resets_tracker_and_zone_state(tmp_path, monkeypatch):
    """H3-5c: a live reconnect rebuilds tracking state — zone counts
    accumulated before the drop are wiped and only the fresh session's
    crossing is counted (without the reset branch the run would report
    two), and the post-reconnect identity is one stable track, not a
    continuation of the old session's ids.

    The reset fires on whichever flush first observes the generation bump,
    which the reader's lookahead makes land within a 3-frame window; the
    scene places both crossings outside that window so the assertions hold
    for any landing.
    """
    boundary = 10
    total = 24
    events = [
        ("frame", np.zeros((48, 64, 3), dtype=np.uint8)) for _ in range(boundary)
    ] + [("reconnect",)] + [
        ("frame", np.zeros((48, 64, 3), dtype=np.uint8))
        for _ in range(total - boundary)
    ]

    def factory(*args, **kwargs):
        return _ScriptedStream(*args, events=events, **kwargs)

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", factory)

    result = run_stream(
        "rtsp://fake/cam", tmp_path / "out",
        engine=_ReconnectSceneEngine(boundary=boundary, total=total),
        track=True,
        line_zones=[LineZone((0, 45), (32, 45), name="gate")],
        batch_size=1,
        prefetch=1,
        render=False,
    )
    assert result.frames == total
    records = [
        json.loads(line)
        for line in result.predictions_path.read_text().splitlines()
    ]
    ids_after = {
        track["track_id"] for record in records[-6:] for track in record["tracks"]
    }
    # One stable identity after the boundary; whether the reset landed on
    # the boundary frame (id 1 reactivated) or earlier (the teleport spawns
    # a confirming id 2) decides WHICH number, not how many.
    assert len(ids_after) == 1
    assert ids_after <= {1, 2}
    # Session 1's crossing (frame 5) was wiped by the reset; session 2's
    # (frame 21) survives. Without the reset branch this reads 2.
    assert result.zones["lines"]["gate"]["in"] == 1
    # The teleport is IoU-disconnected, so no id is tracked across it and
    # the upward jump must not register an "out" crossing.
    assert result.zones["lines"]["gate"]["out"] == 0


def test_dropped_live_stream_without_reconnect_warns(tmp_path, monkeypatch, caplog):
    """H3-5c: a live source that ends on a drop with reconnects=0 is
    flagged with a warning — the result alone cannot distinguish it from
    a clean end-of-stream."""
    events = [
        ("frame", np.zeros((24, 32, 3), dtype=np.uint8)) for _ in range(3)
    ] + [("drop_end",)]

    def factory(*args, **kwargs):
        return _ScriptedStream(*args, events=events, **kwargs)

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", factory)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        result = run_stream(
            "rtsp://fake/cam", tmp_path / "out", engine=_Engine(), render=False
        )
    assert result.frames == 3
    assert result.reconnects == 0
    assert "not a clean end-of-stream" in caplog.text


def test_clean_live_eof_does_not_warn(tmp_path, monkeypatch, caplog):
    """H3-5c: a stream that stops without a drop stays silent — the
    warning must not fire on ordinary end-of-stream."""
    events = [
        ("frame", np.zeros((24, 32, 3), dtype=np.uint8)) for _ in range(2)
    ]

    def factory(*args, **kwargs):
        return _ScriptedStream(*args, events=events, **kwargs)

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", factory)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        result = run_stream(
            "rtsp://fake/cam", tmp_path / "out", engine=_Engine(), render=False
        )
    assert result.frames == 2
    assert "not a clean end-of-stream" not in caplog.text


def test_frame_size_change_raises_with_guidance(tmp_path, monkeypatch):
    """H3-5e: run_stream requires a fixed resolution — a frame size change
    after the first frame fails loudly with guidance instead of writing
    inconsistent pixel geometry."""
    events = [
        ("frame", np.zeros((24, 32, 3), dtype=np.uint8)),
        ("frame", np.zeros((48, 64, 3), dtype=np.uint8)),
    ]

    def factory(*args, **kwargs):
        return _ScriptedStream(*args, events=events, **kwargs)

    import autovision.inference.stream as stream_module

    monkeypatch.setattr(stream_module, "VideoFrameStream", factory)
    with pytest.raises(AutoVisionError, match="fixed resolution"):
        run_stream(
            "fake://cam", tmp_path / "out", engine=_Engine(), render=False
        )
