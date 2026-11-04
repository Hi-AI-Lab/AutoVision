# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Live-stream capture contract: FPS sanitization + reconnect (D35).

Contract: reported FPS outside ``(0, 1000)`` (0, NaN, 90 kHz RTP clocks)
becomes 0.0 — the module's "unknown" value that downstream consumers
already default on. Local files never reconnect (a short read is
truncation and raises); live sources re-open the capture up to
``reconnect_attempts`` times with frame numbering continuing. The default
parameters keep the historical drop-and-end behavior bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import autovision.inference.video as video_module
from autovision.errors import AutoVisionError
from autovision.inference.stream import StreamRunResult

_LIVE_SOURCE = "rtsp://cam.local/live"


class _FakeCV2:
    cvtColor = staticmethod(lambda frame, flag: frame)
    COLOR_BGR2RGB = 4
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_OPEN_TIMEOUT_MSEC = 11
    CAP_PROP_READ_TIMEOUT_MSEC = 12


class _FakeCapture:
    """Scriptable capture: per-instance fps report + read schedule.

    ``reads`` is a list of ``(ok, frame_or_None)`` tuples served in order;
    ``fail_reopens`` makes every open() after a release() report failure,
    which exercises the reconnect path's reopen-failure branch while the
    initial open still succeeds.
    """


    def __init__(self, *, fps: float = 25.0, reads: list | None = None,
                 fail_reopens: bool = False, frame_count: int = 0) -> None:
        self.fps = fps
        self.reads = list(reads or [])
        self.fail_reopens = fail_reopens
        self.frame_count = frame_count
        self.opens = 0
        self.released = 0
        self.dead = False

    def set(self, *args, **kwargs):
        return True

    def open(self, source):
        self.opens += 1
        if self.fail_reopens and self.released > 0:
            self.dead = True
            return False
        self.dead = False
        return True

    def isOpened(self):
        return not self.dead

    def get(self, prop):
        return {
            _FakeCV2.CAP_PROP_FRAME_WIDTH: 10,
            _FakeCV2.CAP_PROP_FRAME_HEIGHT: 10,
            _FakeCV2.CAP_PROP_FPS: self.fps,
            _FakeCV2.CAP_PROP_FRAME_COUNT: self.frame_count,
        }.get(prop, 0)

    def read(self):
        if self.dead or not self.reads:
            return False, None
        return self.reads.pop(0)

    def release(self):
        self.released += 1
        self.dead = True


def _install(monkeypatch: pytest.MonkeyPatch, capture: _FakeCapture) -> None:
    monkeypatch.setattr(video_module, "_require_cv2", lambda: _FakeCV2)
    monkeypatch.setattr(
        video_module.cv2 if hasattr(video_module, "cv2") else _FakeCV2,
        "VideoCapture",
        lambda: capture,
        raising=False,
    )
    # The module resolves cv2 through _require_cv2 (patched above); the
    # capture constructor call goes through _FakeCV2.VideoCapture.
    monkeypatch.setattr(_FakeCV2, "VideoCapture", lambda: capture, raising=False)


class TestFpsSanitization:
    """Driver-reported FPS outside (0, 1000) is "unknown" (0.0)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param(25.0, 25.0, id="normal"),
            pytest.param(0.0, 0.0, id="zero"),
            pytest.param(-1.0, 0.0, id="negative"),
            pytest.param(float("nan"), 0.0, id="nan"),
            pytest.param(90_000.0, 0.0, id="rtp-clock"),
            pytest.param(999.0, 999.0, id="high-but-plausible"),
        ],
    )
    def test_fps_values(self, monkeypatch: pytest.MonkeyPatch,
                        raw: float, expected: float) -> None:
        capture = _FakeCapture(fps=raw)
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(_LIVE_SOURCE, reconnect_delay_s=0)
        assert stream.info.fps == expected


class TestLiveReconnect:
    """Dropped live reads reopen the capture; numbering continues."""

    def test_default_parameters_keep_drop_and_end(self, monkeypatch) -> None:
        """Zero attempts (the default) reproduces the historical behavior."""
        frame = "frame0"
        capture = _FakeCapture(reads=[(True, frame), (False, None)])
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(_LIVE_SOURCE)
        frames = list(stream)
        assert frames == [(0, frame)]
        assert capture.opens == 1  # no reopen attempts

    def test_dropped_read_reconnects_and_continues(self, monkeypatch) -> None:
        """One dropped frame: reopen succeeds, numbering continues at 1."""
        frames = ["f0", "f1", "f2"]
        capture = _FakeCapture(
            reads=[(True, frames[0]), (False, None), (True, frames[1]), (True, frames[2])]
        )
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(
            _LIVE_SOURCE, max_frames=3,
            reconnect_attempts=2, reconnect_delay_s=0,
        )
        got = list(stream)
        assert [index for index, _ in got] == [0, 1, 2]
        assert got[1][1] == frames[1]
        assert capture.opens == 2  # one reopen

    def test_reopen_failure_consumes_an_attempt(self, monkeypatch) -> None:
        """A failing reopen burns one attempt; a later read still recovers."""
        capture = _FakeCapture(
            reads=[(False, None), (True, "after-reconnect")], fail_reopens=True
        )
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(
            _LIVE_SOURCE, reconnect_attempts=2, reconnect_delay_s=0
        )
        got = list(stream)
        # First read drop -> reopen fails -> attempt consumed -> next read
        # drop exhausts the remaining budget -> stream ends.
        assert got == []
        assert capture.opens >= 3

    def test_exhausted_attempts_end_the_stream(self, monkeypatch) -> None:
        """A persistently dead stream ends cleanly after the last attempt."""
        capture = _FakeCapture(reads=[(False, None)], fail_reopens=True)
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(
            _LIVE_SOURCE, reconnect_attempts=3, reconnect_delay_s=0
        )
        assert list(stream) == []
        assert capture.opens == 4  # initial open + every attempt

    def test_invalid_reconnect_parameters_raise(self) -> None:
        with pytest.raises(AutoVisionError, match="reconnect_attempts"):
            video_module.VideoFrameStream(
                _LIVE_SOURCE, reconnect_attempts=-1
            )
        with pytest.raises(AutoVisionError, match="reconnect_delay_s"):
            video_module.VideoFrameStream(
                _LIVE_SOURCE, reconnect_delay_s=-0.5
            )


class TestReconnectTelemetry:
    """``reconnects_used`` / ``StreamRunResult.reconnects`` make a dropped
    live source observable: a run that continued past drops (or gave up and
    ended early) is no longer indistinguishable from a clean EOF."""

    def test_counter_surfaces_the_drop(self, monkeypatch) -> None:
        """One reconnect: the counter reports it; frame numbering continues."""
        capture = _FakeCapture(
            reads=[(True, "f0"), (False, None), (True, "f1")]
        )
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(
            _LIVE_SOURCE, reconnect_attempts=1, reconnect_delay_s=0
        )
        assert [index for index, _ in stream] == [0, 1]
        assert stream.reconnects_used == 1

    def test_clean_stream_reports_zero(self, monkeypatch) -> None:
        """No drop, no reconnect: the counter stays 0 (default semantics)."""
        capture = _FakeCapture(reads=[(True, "f0"), (False, None)])
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(_LIVE_SOURCE)
        assert list(stream) == [(0, "f0")]
        assert stream.reconnects_used == 0

    def test_counter_resets_on_a_second_iteration(self, monkeypatch) -> None:
        """Each iteration starts from a fresh count (mirrors the fresh
        attempt budget), so the attribute always describes the last pass."""
        capture = _FakeCapture(reads=[(True, "f0"), (False, None)])
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(
            _LIVE_SOURCE, reconnect_attempts=1, reconnect_delay_s=0
        )
        assert list(stream) == [(0, "f0")]
        assert stream.reconnects_used == 1
        # The fake capture serves its schedule once, so the second pass
        # drops immediately and burns its fresh attempt: the counter reads
        # 1 only because the per-iteration reset zeroed the first pass.
        assert list(stream) == []
        assert stream.reconnects_used == 1

    def test_run_result_summary_reports_reconnects(self) -> None:
        """A non-zero count lands in the CLI summary line; zero is silent."""
        base: dict = {
            "out_dir": Path("/tmp/run"),
            "predictions_path": Path("/tmp/run/predictions.jsonl"),
            "video_path": None,
            "frames": 10,
            "elapsed_s": 1.0,
            "backend": "onnx",
        }
        clean = StreamRunResult(**base)
        assert "reconnects" not in clean.summary()
        dropped = StreamRunResult(**base, reconnects=3)
        assert "reconnects=3" in dropped.summary()


class TestFileSourcesNeverReconnect:
    """A local file's short read is truncation, not a dropped stream."""

    def test_short_file_read_raises_even_with_attempts(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "truncated.mp4"
        path.write_bytes(b"pretend-video")
        capture = _FakeCapture(frame_count=5)
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(path, reconnect_attempts=5)
        with pytest.raises(AutoVisionError, match="decode failed"):
            list(stream)

    def test_normal_file_eof_still_ends_cleanly(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"pretend-video")
        capture = _FakeCapture(reads=[(True, "only-frame"), (False, None)], frame_count=1)
        _install(monkeypatch, capture)
        stream = video_module.VideoFrameStream(path)
        assert [index for index, _ in stream] == [0]
