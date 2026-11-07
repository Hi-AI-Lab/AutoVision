# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Long-run soak suite (H4-1): unbounded-growth and timing-hang guards.

Short fixtures cannot expose the failure class the hardening plan calls out
for the inference runtime: retired-track backlog growth on long tracking
runs (H1-3), end-of-stream timing hangs (H1-2) and per-source accounting
drift on multi-source runs. These tests replay hundreds of frames and
assert boundedness (backlog, wall clock, complete output) rather than
exact interior states.

The suite is opt-in: marked ``soak`` and skipped unless
``AUTOVISION_RUN_SOAK=1`` is set (nightly gate). The skip decision is made
at collection time — before ``tests/conftest.py`` scrubs AUTOVISION_* from
the environment — so the module-level ``skipif`` is reliable without
touching conftest.

Run:
    AUTOVISION_RUN_SOAK=1 pytest tests/test_soak.py -q
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import torch

from autovision.inference import run_stream
from autovision.inference.multi_stream import MultiStreamRunner, StreamSource
from autovision.inference.video import VideoInfo

pytestmark = [
    pytest.mark.soak,
    pytest.mark.skipif(
        os.environ.get("AUTOVISION_RUN_SOAK") != "1",
        reason="soak suite is opt-in: set AUTOVISION_RUN_SOAK=1 (H4-1)",
    ),
]

_BLINK_PERIOD = 40
_BLINK_FRAMES = 1000
#: 1000 frames / period-40 cycles = one confirmed + one retired track each.
_BLINK_CYCLES = _BLINK_FRAMES // _BLINK_PERIOD


def _write_clip(tmp_path, frames: int, name: str = "clip.mp4") -> str:
    """Write a deterministic 10 fps mp4 clip with ``frames`` solid frames."""
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")
    path = tmp_path / name
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((24, 32, 3), (index * 7) % 255, dtype=np.uint8))
    writer.release()
    return str(path)


def _frames(count: int, value: int = 0) -> list[np.ndarray]:
    """``count`` distinct solid frames, as for the multi-stream fakes."""
    return [
        np.full((48, 64, 3), (value + index) % 255, dtype=np.uint8)
        for index in range(count)
    ]


class _BlinkEngine:
    """High-score detections reappearing at a far position every cycle.

    Each cycle is two adjacent frames with the same box (one confirmed
    track) followed by a 38-frame gap that pushes the track past the
    retirement buffer; the box jumps across cycles so association can
    never bridge them. Every cycle therefore adds exactly one retired
    track — the accumulation that used to grow ``_removed`` without bound
    (H1-3) and that the per-flush drain must keep bounded.
    """

    backend = "torch"

    def __init__(self, period: int = _BLINK_PERIOD) -> None:
        self._period = period
        self._frame = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            if self._frame % self._period in (0, 1):
                x = 4.0 if (self._frame // self._period) % 2 == 0 else 20.0
                records.append(
                    {
                        "bboxes": torch.tensor([[x, 2.0, x + 8.0, 18.0]]),
                        "scores": torch.tensor([0.95]),
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


class _SlowFirstBatchEngine:
    """Sleeps once on the first batch; every later batch is instant."""

    backend = "torch"

    def __init__(self, delay: float = 0.3) -> None:
        self._delay = delay
        self._calls = 0

    def run_batch(self, images, threshold=0.5):
        if self._calls == 0:
            time.sleep(self._delay)
        self._calls += 1
        return [
            {
                "bboxes": torch.tensor([[2.0, 2.0, 20.0, 18.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class _OneBoxEngine:
    """Deterministic single-box engine for multi-source soak volume."""

    def run_batch(self, images, threshold=0.5):
        return [
            {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class _SoakReader:
    """Fake reader mirroring ``tests/test_multi_stream.py``'s injection."""

    def __init__(self, frames: list[np.ndarray], *, fps: float = 25.0) -> None:
        self.info = VideoInfo(
            width=64, height=48, fps=fps, frame_count=len(frames)
        )
        self._frames = frames
        self.closed = False

    def frames(self):
        for index, frame in enumerate(self._frames):
            yield index, frame

    def close(self) -> None:
        self.closed = True


def _instrument_tracker(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Capture the tracker run_stream builds; record peak ``_removed`` size.

    ``run_stream`` imports ``BYTETracker`` from :mod:`autovision.tracking`
    at call time, so patching the module attribute intercepts the instance
    without touching the production path.
    """
    import autovision.tracking as tracking_module

    real_cls = tracking_module.BYTETracker
    captured: list[object] = []

    class _InstrumentedTracker(real_cls):  # type: ignore[misc, valid-type]
        """Subclass recording the peak undrained ``_removed`` backlog."""

        max_removed_backlog: int

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.max_removed_backlog = 0
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            captured.append(self)

        def update(self, *args: object, **kwargs: object):
            tracks = super().update(*args, **kwargs)  # type: ignore[arg-type]
            self.max_removed_backlog = max(
                self.max_removed_backlog, len(self._removed)
            )
            return tracks

    monkeypatch.setattr(tracking_module, "BYTETracker", _InstrumentedTracker)
    return captured


def test_blink_1000_frames_tracker_backlog_and_output_stay_bounded(
    tmp_path, monkeypatch
):
    """H4-1/H1-3: 1000 blink frames under track=True accumulate nothing.

    Every cycle retires one track; the tracker's ``_removed`` backlog must
    stay bounded (drained per flush), the JSONL must stay complete and in
    stream order, and the wall clock must stay bounded.
    """
    clip = _write_clip(tmp_path, _BLINK_FRAMES)
    captured = _instrument_tracker(monkeypatch)
    batch_size = 4
    started = time.perf_counter()
    result = run_stream(
        clip,
        tmp_path / "out",
        engine=_BlinkEngine(),
        track=True,
        batch_size=batch_size,
        render=False,
        prefetch=4,
        writer_buffer=4,
    )
    elapsed = time.perf_counter() - started

    assert result.frames == _BLINK_FRAMES
    records = [
        json.loads(line)
        for line in result.predictions_path.read_text().splitlines()
    ]
    assert [record["frame"] for record in records] == list(range(_BLINK_FRAMES))
    assert len(captured) == 1
    tracker = captured[0]
    # The backlog is drained once per flush: it can never exceed one batch,
    # and it must be empty once the run ends (H1-3).
    assert tracker.max_removed_backlog <= batch_size
    assert len(tracker._removed) == 0
    # One retirement per blink cycle: bounded, not growing with time.
    assert result.tracks == _BLINK_CYCLES
    assert result.tracks_removed == _BLINK_CYCLES
    assert elapsed < 60.0, f"soak run took {elapsed:.1f}s for {_BLINK_FRAMES} frames"


def test_multi_stream_three_sources_per_source_stats_close(tmp_path):
    """H4-1: 3 sources x 100 frames — per-source accounting must close.

    ``frames_read`` must equal written + failed + dropped for every source;
    the clean run drops and fails nothing, every per-source JSONL stays
    complete and in order, and the summary artifact is written.
    """
    sources = [
        StreamSource(f"cam{index}", f"file{index}.mp4") for index in range(3)
    ]
    runner = MultiStreamRunner(
        sources,
        tmp_path / "out",
        engine=_OneBoxEngine(),
        batch_size=2,
        queue_size=16,
        drop_policy="block",
        render=False,
        reader_factory=lambda url: _SoakReader(
            _frames(100, 10 * int(url[-1]))
        ),
    )
    result = runner.run()
    assert set(result.sources) == {"cam0", "cam1", "cam2"}
    for name, stats in result.sources.items():
        assert stats.frames_read == 100, name
        assert stats.frames_written == 100, name
        assert stats.frames_failed == 0, name
        assert stats.frames_dropped == 0, name
        assert stats.frames_read == (
            stats.frames_written + stats.frames_failed + stats.frames_dropped
        ), name
        lines = stats.jsonl.read_text().splitlines()
        assert [json.loads(line)["frame_index"] for line in lines] == list(
            range(100)
        ), name
    assert (tmp_path / "out" / "summary.json").is_file()
    assert result.elapsed_s < 60.0


def test_slow_first_batch_returns_within_wall_clock(tmp_path):
    """H4-1/H1-2: a slow first batch must not turn EOF into a hang.

    With prefetch=2 and batch_size=1 over a 3-frame source, the reader
    enqueues everything while the engine sleeps, then faces a FULL queue
    with its sentinel — the permanent-spin reproduction. The run must
    return inside a wall-clock bound with a complete JSONL.
    """
    clip = _write_clip(tmp_path, frames=3, name="short.mp4")
    started = time.perf_counter()
    result = run_stream(
        clip,
        tmp_path / "out",
        engine=_SlowFirstBatchEngine(delay=0.3),
        batch_size=1,
        prefetch=2,
        render=False,
    )
    assert time.perf_counter() - started < 10.0
    assert result.frames == 3
    lines = result.predictions_path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["frame"] for line in lines] == [0, 1, 2]
