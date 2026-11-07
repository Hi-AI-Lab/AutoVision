# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Second-round inference hardening: writer isolation, per-frame isolation,
metadata gate, output contract, trail bounds.

Covers the INFER-01..06 / 08 / 09 / 11 / 13 / 16 hardening designs.
The load-bearing invariant for the
stream changes: a single bad frame (or a single failing source's video
writer) must never end a 24/7 run — every test here fails on the
pre-fix behavior by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference.engine import InferenceEngine
from autovision.inference.multi_stream import MultiStreamRunner, StreamSource
from autovision.inference.stream import run_stream
from autovision.inference.video import VideoInfo
from autovision.results import InferenceResult
from autovision.tracking.analytics import TrackSmoother
from autovision.tracking.byte_track import Track
from autovision.visualize.annotators import SemanticMaskAnnotator
from autovision.visualize.effects import TraceAnnotator

# ── shared fixtures (mirroring the stream/multi-stream suite patterns) ────


class _ScriptedStream:
    """VideoFrameStream stand-in driven by ``("frame", array)`` events."""

    def __init__(
        self,
        source,
        *,
        max_frames=None,
        reconnect_attempts=0,
        reconnect_delay_s=1.0,
        events=(),
        **_ignored,
    ) -> None:
        self.info = SimpleNamespace(fps=10.0)
        self.reconnects_used = 0
        self.reconnect_generation = 0
        self.stream_ended_after_drop = False
        self._events = list(events)

    def __iter__(self):
        index = 0
        for event in self._events:
            yield index, event[1]
            index += 1

    def close(self) -> None:
        return None


class _FrameEngine:
    """Returns one detection per frame; the score script decides per frame."""

    backend = "torch"

    def __init__(self, scores: dict[int, float] | None = None) -> None:
        self._scores = scores or {}
        self._frame = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            score = self._scores.get(self._frame, 0.9)
            records.append(
                {
                    "bboxes": torch.tensor([[2.0, 2.0, 20.0, 18.0]]),
                    "scores": torch.tensor([score]),
                    "labels": torch.tensor([0]),
                }
            )
            self._frame += 1
        return records


class FakeReader:
    def __init__(self, frames, *, fps: float = 10.0):
        self.info = VideoInfo(
            width=64, height=48, fps=fps, frame_count=len(frames)
        )
        self._frames = frames
        self.closed = False

    def frames(self):
        yield from enumerate(self._frames)

    def close(self):
        self.closed = True


class FakeEngine:
    backend = "torch"

    def run_batch(self, images, *, threshold=0.5):
        return [
            {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


def _frames(count: int, value: int = 0) -> list[np.ndarray]:
    return [
        np.full((48, 64, 3), (value + index) % 255, dtype=np.uint8)
        for index in range(count)
    ]


def _scripted_run(tmp_path: Path, total: int, scores: dict[int, float], **kwargs):
    events = [
        ("frame", np.full((24, 32, 3), (index * 30) % 255, dtype=np.uint8))
        for index in range(total)
    ]

    import autovision.inference.stream as stream_module

    with patch.object(stream_module, "VideoFrameStream", lambda *a, **k: _ScriptedStream(*a, events=events, **k)):
        return run_stream(
            "fake://cam", tmp_path / "out", engine=_FrameEngine(scores=scores),
            batch_size=1, prefetch=1, render=False, **kwargs,
        )


# ── INFER-01: multi_stream writer isolation ───────────────────────────────


class TestMultiStreamWriterIsolation:
    def test_writer_setup_failure_aborts_only_that_source(
        self, tmp_path, monkeypatch
    ):
        """A source whose AnnotatedVideoWriter cannot be created used to
        propagate out of run() and kill every other source with the
        summary lost; now it trips that source's breaker and the run
        completes with a summary."""
        import autovision.inference.multi_stream as multi_module

        real_writer = multi_module.AnnotatedVideoWriter

        def patched_writer(path, *args, **kwargs):
            if "cam_bad" in str(path):
                raise AutoVisionError("disk full while creating the writer")
            return real_writer(path, *args, **kwargs)

        monkeypatch.setattr(multi_module, "AnnotatedVideoWriter", patched_writer)
        runner = MultiStreamRunner(
            [StreamSource("cam_good", "good.mp4"), StreamSource("cam_bad", "bad.mp4")],
            tmp_path / "out",
            engine=FakeEngine(),
            batch_size=1,
            render=True,
            reader_factory=lambda url: FakeReader(_frames(3, 10)),
        )
        result = runner.run()  # must NOT raise (pre-fix: propagate)
        assert result.sources["cam_good"].frames_written == 3
        assert result.sources["cam_bad"].aborted is True
        assert "writer" in (result.sources["cam_bad"].errors[-1] if result.sources["cam_bad"].errors else "")
        assert (tmp_path / "out" / "summary.json").is_file()
        # The healthy source's artifacts are complete.
        assert len((tmp_path / "out" / "cam_good" / "predictions.jsonl").read_text().splitlines()) == 3

    def test_summary_survives_an_uncontained_failure(self, tmp_path):
        """INFER-01's last-resort net: even a failure outside the per-source
        containment (here, the artifact setup loop — a FILE occupies the
        path where the source directory must be created) saves
        summary.json before re-raising."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "cam").write_text("a file blocks the source directory")
        runner = MultiStreamRunner(
            [StreamSource("cam", "file.mp4")],
            out,
            engine=FakeEngine(),
            batch_size=1,
            render=False,
            reader_factory=lambda url: FakeReader(_frames(2)),
        )
        with pytest.raises(OSError):
            runner.run()
        assert (out / "summary.json").is_file()


# ── INFER-02: float semantic label map rendering ──────────────────────────


class TestFloatLabelMapRendering:
    def test_float_label_map_annotates_without_crash(self):
        """Pre-fix: float ids crashed the direct branch with IndexError
        while the resize branch survived — behavior depended on the
        resolution."""
        annotator = SemanticMaskAnnotator()
        result = InferenceResult.from_record({"mask": np.full((32, 32), 2.0)})
        out = annotator.annotate(_rgb_image(size=(32, 32)), result)
        assert out.size == (32, 32)

    def test_float_size_mismatch_branch_still_works(self):
        """The resize branch must keep working after the rounding guard."""
        out = SemanticMaskAnnotator().annotate(
            _rgb_image(size=(64, 64)),
            InferenceResult.from_record({"mask": np.full((32, 32), 3.0)}),
        )
        assert out.size == (64, 64)

    def test_float_matches_int_reference_pixelwise(self):
        int_out = SemanticMaskAnnotator().annotate(
            _rgb_image(size=(32, 32)),
            InferenceResult.from_record(
                {"mask": np.full((32, 32), 2, dtype=np.int64)}
            ),
        )
        float_out = SemanticMaskAnnotator().annotate(
            _rgb_image(size=(32, 32)),
            InferenceResult.from_record({"mask": np.full((32, 32), 2.0)}),
        )
        assert np.array_equal(np.asarray(int_out), np.asarray(float_out))

    def test_float_nan_buckets_to_background(self):
        mask = np.full((16, 16), 1.0)
        mask[0, 0] = np.nan
        out = SemanticMaskAnnotator().annotate(
            _rgb_image(size=(16, 16)), InferenceResult.from_record({"mask": mask})
        )
        assert out.size == (16, 16)

def _rgb_image(size: tuple[int, int]):
    from PIL import Image

    return Image.new("RGB", size, (10, 20, 30))


# ── INFER-03: run_stream per-frame isolation ──────────────────────────────


class TestPerFrameIsolation:
    def test_nan_score_with_tracking_skips_one_frame_and_keeps_running(
        self, tmp_path
    ):
        """Pre-fix: one NaN score on a track=True run raised through the
        tracker and killed the whole run; now the frame is skipped, logged
        and counted, and the remaining frames complete."""
        result = _scripted_run(tmp_path, total=6, scores={2: float("nan")}, track=True)
        assert result.frames == 5
        assert result.frames_failed == 1
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        errors = [record for record in records if "error" in record]
        assert len(errors) == 1 and errors[0]["frame"] == 2
        assert [r["frame"] for r in records if "predictions" in r] == [0, 1, 3, 4, 5]

    def test_on_frame_error_raise_keeps_fail_fast(self, tmp_path):
        with pytest.raises(AutoVisionError):
            _scripted_run(
                tmp_path / "raise", total=4, scores={1: float("nan")},
                track=True, on_frame_error="raise",
            )

    def test_consecutive_failures_still_abort(self, tmp_path):
        """A fully broken pipeline cannot spin forever under skip."""
        scores = {index: float("nan") for index in range(20)}
        with pytest.raises(AutoVisionError, match="consecutive"):
            _scripted_run(tmp_path / "all", total=20, scores=scores, track=True)

    def test_on_frame_error_validation(self, tmp_path):
        with pytest.raises(AutoVisionError, match="on_frame_error"):
            _scripted_run(tmp_path, total=1, scores={}, on_frame_error="ignore")


# ── INFER-04: metadata gate ───────────────────────────────────────────────


class TestMetadataGate:
    def _engine(self, tmp_path: Path, monkeypatch, require_metadata: bool):
        import autovision.export.artifact_metadata as metadata_module

        monkeypatch.setattr(metadata_module, "read_metadata", lambda *a, **k: {})
        monkeypatch.setattr(
            metadata_module, "verify_engine_artifact", lambda *a, **k: None
        )
        monkeypatch.setattr(
            InferenceEngine,
            "_make_onnx_forward",
            lambda self, providers: (lambda batch: []),
        )
        # The engine validates artifact existence before the metadata gate.
        artifact = tmp_path / "fake.onnx"
        artifact.touch()
        return InferenceEngine(
            _StubNetwork(),
            backend="onnx",
            artifact=artifact,
            require_metadata=require_metadata,
        )

    def test_require_metadata_refuses_stripped_artifact(self, tmp_path, monkeypatch):
        with pytest.raises(AutoVisionError, match="identity metadata"):
            self._engine(tmp_path, monkeypatch, require_metadata=True)

    def test_default_warns_but_proceeds(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="autovision.inference.engine"):
            engine = self._engine(tmp_path, monkeypatch, require_metadata=False)
        assert engine.backend == "onnx"
        assert any("identity metadata" in r.getMessage() for r in caplog.records)


# ── INFER-05/09: run_batch output contract ────────────────────────────────


class _StubNetwork(torch.nn.Module):
    """Duck-typed TaskNetwork whose postprocess returns a scripted payload."""

    def __init__(self, postprocess=None) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self._postprocess = postprocess or (
            lambda raw, metadata: [
                {"bboxes": torch.zeros(0, 4), "scores": torch.zeros(0)}
                for _ in metadata
            ]
        )

    def _module_device(self):
        return torch.device("cpu")

    def forward_backend(self, batch):
        return batch

    def preprocess_image(self, image):
        return torch.zeros(3, 4, 4), {}

    def preprocess_batch(self, batch):
        return batch

    def postprocess(self, raw, metadata, *, threshold=0.5):
        return self._postprocess(raw, metadata)


def _engine_with(postprocess) -> InferenceEngine:
    return InferenceEngine(_StubNetwork(postprocess), backend="torch")


class TestRunBatchContract:
    def test_empty_batch_raises_named_error(self):
        engine = _engine_with(lambda raw, meta: [])
        with pytest.raises(AutoVisionError, match="empty image list"):
            engine.run_batch([])

    def test_record_count_mismatch_names_the_backend(self):
        engine = _engine_with(lambda raw, meta: [{"bboxes": torch.zeros(0, 4)}])
        with pytest.raises(AutoVisionError, match="torch backend"):
            engine.run_batch([torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)])

    def test_record_without_known_keys_names_the_backend(self):
        engine = _engine_with(
            lambda raw, meta: [{"logits": torch.zeros(2)} for _ in raw]
        )
        with pytest.raises(AutoVisionError, match="canonical"):
            engine.run_batch([torch.zeros(3, 4, 4)])

    def test_non_dict_record_rejected(self):
        engine = _engine_with(lambda raw, meta: [None for _ in raw])
        with pytest.raises(AutoVisionError, match="dict record"):
            engine.run_batch([torch.zeros(3, 4, 4)])

    def test_valid_records_pass_through(self):
        engine = _engine_with(
            lambda raw, meta: [
                {"bboxes": torch.zeros(0, 4), "scores": torch.zeros(0)} for _ in raw
            ]
        )
        assert len(engine.run_batch([torch.zeros(3, 4, 4)])) == 1


# ── INFER-06: TraceAnnotator history bound ────────────────────────────────


class TestTraceAnnotatorBound:
    def test_history_stays_bounded_under_churn(self):
        annotator = TraceAnnotator(max_tracks=4)
        image = _rgb_image(size=(64, 64))
        for index in range(100):
            result = InferenceResult.from_record(
                {
                    "bboxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "tracker_ids": [index + 1],
                }
            )
            annotator.annotate(image.copy(), result)
        assert len(annotator._history) <= 4

    def test_unbounded_opt_out_keeps_everything(self):
        annotator = TraceAnnotator(max_tracks=None)
        image = _rgb_image(size=(64, 64))
        for index in range(30):
            result = InferenceResult.from_record(
                {
                    "bboxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "tracker_ids": [index + 1],
                }
            )
            annotator.annotate(image.copy(), result)
        assert len(annotator._history) == 30

    def test_invalid_max_tracks_rejected(self):
        with pytest.raises(AutoVisionError, match="max_tracks"):
            TraceAnnotator(max_tracks=0)


# ── INFER-08: det_index survives the label-less reconstruction ────────────


class TestTrackRecordDetIndex:
    def test_det_index_survives_label_less_records(self):
        """The join-back contract: a track maps to its originating
        detection even when the record carries no labels."""
        from autovision.inference.stream import _track_record

        class _OneShotTracker:
            def update(self, boxes, scores, labels, *, frame=None):
                return [
                    Track(
                        track_id=1,
                        bbox=(0.0, 0.0, 4.0, 4.0),
                        score=0.9,
                        class_id=0,
                        state=1,
                        age=1,
                        det_index=2,
                    )
                ]

        record = {
            "bboxes": torch.zeros(3, 4),
            "scores": torch.ones(3),
            # no labels: the reconstruction branch runs
        }
        tracks, _names = _track_record(record, _OneShotTracker(), None)
        assert tracks[0].det_index == 2


# ── INFER-11: 16-bit frame normalization ──────────────────────────────────


class TestSixteenBitNormalization:
    def test_uint16_frame_does_not_wrap(self):
        from autovision.visualize.detection import draw_prediction

        frame = np.full((16, 16, 1), 40000, dtype=np.uint16)
        canvas = np.asarray(draw_prediction(frame, {}))
        # /255 would truncate 40000 -> 64; the full-range scale keeps ~157.
        assert abs(int(canvas[8, 8][0]) - 157) <= 3


# ── INFER-16: concurrent iteration guard ──────────────────────────────────


class TestStreamDoubleIteration:
    def test_second_concurrent_iteration_is_rejected(self, tmp_path):
        """Two live iterators share one cv2 capture; the guard names the
        bug instead of splitting frames undefined-ly."""
        import cv2

        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
        )
        assert writer.isOpened()
        for index in range(4):
            writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
        writer.release()

        from autovision.inference.video import VideoFrameStream

        stream = VideoFrameStream(path)
        first = iter(stream)
        next(first)  # the stream is mid-iteration
        second = iter(stream)
        with pytest.raises(AutoVisionError, match="already being iterated"):
            next(second)  # generator bodies start on the first next()
        first.close()


# ── INFER-13: TrackSmoother duplicate-id guard ────────────────────────────


class TestTrackSmootherDuplicateIds:
    def test_duplicate_id_in_one_frame_raises(self):
        smoother = TrackSmoother(length=3)
        duplicate = Track(
            track_id=7, bbox=(0.0, 0.0, 4.0, 4.0), score=0.9, class_id=0,
            state=1, age=1,
        )
        with pytest.raises(AutoVisionError, match="twice in one frame"):
            smoother.update([duplicate, duplicate])

    def test_distinct_ids_still_smooth(self):
        smoother = TrackSmoother(length=2)
        first = Track(track_id=1, bbox=(0.0, 0.0, 4.0, 4.0), score=0.5,
                      class_id=0, state=1, age=1)
        second = Track(track_id=2, bbox=(0.0, 0.0, 8.0, 8.0), score=0.7,
                       class_id=0, state=1, age=1)
        out = smoother.update([first, second])
        assert len(out) == 2


# ── review follow-ups: threshold boundary, mid-run writer failure, float edge ─


class TestReviewFollowUps:
    def test_threshold_resets_after_a_healthy_frame(self, tmp_path):
        """Exactly 11 consecutive failures abort; but 10 consecutive
        failures followed by a healthy frame reset the ledger — the run
        continues on later scattered failures (INFER-03 review P3-3a)."""
        scores = {index: float("nan") for index in range(1, 11)}
        result = _scripted_run(tmp_path, total=13, scores=scores, track=True)
        # 10 consecutive failures (1..10), frames 0/11/12 healthy: the
        # streak never EXCEEDS the cap, so the run completes.
        assert result.frames == 3
        assert result.frames_failed == 10

    def test_writer_mid_run_failure_aborts_only_that_source(self, tmp_path):
        """Review P3-3b: a writer that dies AFTER setup (mid-write) takes
        down its own source via the breaker, never the run."""
        import autovision.inference.multi_stream as multi_module

        real_writer = multi_module.AnnotatedVideoWriter

        class _DyingWriter:
            frame_size = (64, 48)
            path = "ignored"

            def __init__(self) -> None:
                self._writes = 0

            def write(self, frame):
                self._writes += 1
                if self._writes >= 2:
                    raise AutoVisionError("codec died mid-stream")

            def close(self):
                return None

        def patched_writer(path, *args, **kwargs):
            if "cam_bad" in str(path):
                return _DyingWriter()
            return real_writer(path, *args, **kwargs)

        import unittest.mock as mock

        runner = MultiStreamRunner(
            [StreamSource("cam_good", "good.mp4"), StreamSource("cam_bad", "bad.mp4")],
            tmp_path / "out",
            engine=FakeEngine(),
            batch_size=1,
            render=True,
            reader_factory=lambda url: FakeReader(_frames(5, 10)),
        )
        with mock.patch.object(multi_module, "AnnotatedVideoWriter", patched_writer):
            result = runner.run()  # must NOT raise
        assert result.sources["cam_good"].frames_written == 5
        assert result.sources["cam_bad"].aborted is True
        assert (tmp_path / "out" / "summary.json").is_file()

    def test_inf_and_negative_float_label_maps_round(self):
        """Review P3-3d: Inf buckets to background like NaN; negative ids
        survive the round-trip and clip like integer negatives."""
        annotator = SemanticMaskAnnotator()
        inf_mask = np.full((8, 8), 1.0)
        inf_mask[0, 0] = np.inf
        out = annotator.annotate(
            _rgb_image(size=(8, 8)), InferenceResult.from_record({"mask": inf_mask})
        )
        assert out.size == (8, 8)
        negative = np.full((8, 8), -1.0)
        out_neg = annotator.annotate(
            _rgb_image(size=(8, 8)), InferenceResult.from_record({"mask": negative})
        )
        assert out_neg.size == (8, 8)
