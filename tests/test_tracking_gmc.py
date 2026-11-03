# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GMC camera-motion compensation (design D6) and BoT-SORT (design D7).

GMC estimation is exercised on synthetic frames (seeded noise textures
translated by a known offset); BoT-SORT is exercised through its motion
model contract and a stable-tracking smoke, on top of the shared golden
lifecycle that pins the ByteTrack behavior both trackers inherit.
"""

from __future__ import annotations

import numpy as np
import pytest

from autovision.errors import AutoVisionError
from autovision.tracking import BOTSORT, BYTETracker, GMCEstimator


def _textured_frame(seed: int, size: int = 128) -> np.ndarray:
    """Deterministic high-texture frame (goodFeaturesToTrack needs corners)."""
    rng = np.random.default_rng(seed)
    frame = (rng.integers(0, 256, size=(size, size), dtype=np.uint8))
    return frame


def _shifted(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a frame with border fill (the camera panned by dx, dy)."""
    shifted = np.roll(frame, shift=(dy, dx), axis=(0, 1))
    if dx > 0:
        shifted[:, :dx] = 0
    if dx < 0:
        shifted[:, dx:] = 0
    if dy > 0:
        shifted[:dy, :] = 0
    if dy < 0:
        shifted[dy:, :] = 0
    return shifted


# ── GMCEstimator ────────────────────────────────────────────────────────


def test_gmc_first_frame_has_no_estimate() -> None:
    """The estimator stores the first frame and reports no motion yet."""
    estimator = GMCEstimator(downscale=1)

    assert estimator.apply(_textured_frame(0)) is None


def test_gmc_sparse_flow_recovers_a_known_translation() -> None:
    """A panned texture yields a translation estimate close to the truth."""
    estimator = GMCEstimator(method="sparse_opt_flow", downscale=1)
    frame = _textured_frame(7)
    estimator.apply(frame)

    transform = estimator.apply(_shifted(frame, 6, 4))

    assert transform is not None
    assert transform[0, 2] == pytest.approx(6.0, abs=1.0)
    assert transform[1, 2] == pytest.approx(4.0, abs=1.0)


def test_gmc_translation_is_rescaled_from_downscaled_coordinates() -> None:
    """downscale=2 halves the working resolution; translation doubles back."""
    estimator = GMCEstimator(method="sparse_opt_flow", downscale=2)
    frame = _textured_frame(11)
    estimator.apply(frame)

    transform = estimator.apply(_shifted(frame, 8, 0))

    assert transform is not None
    assert transform[0, 2] == pytest.approx(8.0, abs=2.0)


def test_gmc_reset_forgets_the_previous_frame() -> None:
    """After reset, the next apply is a first-frame call (None)."""
    estimator = GMCEstimator(downscale=1)
    frame = _textured_frame(3)
    estimator.apply(frame)
    estimator.reset()

    assert estimator.apply(frame) is None


def test_gmc_rejects_unknown_methods() -> None:
    """Method names are validated eagerly, not at the first frame."""
    with pytest.raises(AutoVisionError, match="Unknown GMC method"):
        GMCEstimator(method="magic_flow")


def test_gmc_uniform_frames_degrade_to_none_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A featureless frame yields no keypoints: identity, warned once."""
    estimator = GMCEstimator(method="sparse_opt_flow", downscale=1)
    flat = np.full((64, 64), 127, dtype=np.uint8)

    assert estimator.apply(flat) is None
    with caplog.at_level("WARNING"):
        assert estimator.apply(flat) is None
    assert "could not estimate motion" in caplog.text


def test_tracker_with_gmc_tracks_a_panned_scene() -> None:
    """End-to-end: the warped prediction keeps one stable id across pans."""
    tracker = BYTETracker(motion_compensation="sparse_opt_flow")
    base = _textured_frame(21)
    ids: set[int] = set()
    for step in range(4):
        dx = 6 * step
        frame = _shifted(base, dx, 0) if step else base
        boxes = np.array([[30.0 + dx, 30.0, 60.0 + dx, 60.0]])
        tracks = tracker.update(
            boxes, np.array([0.9]), np.array([0]), frame=frame
        )
        ids.update(track.track_id for track in tracks)
    assert ids == {1}


def test_tracker_without_gmc_ignores_the_frame_argument() -> None:
    """motion_compensation='none' + frame= is exactly the plain behavior."""
    tracker = BYTETracker()
    boxes = np.array([[30.0, 30.0, 60.0, 60.0]])
    first = tracker.update(boxes, np.array([0.9]), np.array([0]))
    second = tracker.update(
        boxes + 4, np.array([0.9]), np.array([0]),
        frame=_textured_frame(5),
    )

    assert [track.track_id for track in first + second] == [1, 1]


# ── BOTSORT ─────────────────────────────────────────────────────────────


def test_bot_sort_xywh_state_round_trips_boxes() -> None:
    """The XYWH layout converts boxes losslessly (the D7 hook contract)."""
    from autovision.tracking.bot_sort import _KalmanFilterXYWH

    kalman = _KalmanFilterXYWH()
    box = np.array([10.0, 20.0, 50.0, 80.0])

    assert np.allclose(kalman.to_box(kalman.to_state(box)), box)


def test_bot_sort_defaults_to_gmc_and_tracks_stably() -> None:
    """BoT-SORT enables compensation by default and holds one id."""
    tracker = BOTSORT()
    assert tracker._gmc is not None  # noqa: SLF001 - the contract under test

    base = _textured_frame(31)
    ids: set[int] = set()
    for step in range(4):
        dx = 5 * step
        frame = _shifted(base, dx, 0) if step else base
        tracks = tracker.update(
            np.array([[30.0 + dx, 30.0, 60.0 + dx, 60.0]]),
            np.array([0.9]),
            np.array([0]),
            frame=frame,
        )
        ids.update(track.track_id for track in tracks)
    assert ids == {1}


def test_bot_sort_reported_boxes_match_the_input_geometry() -> None:
    """The XYWH filter reports the box it was given (no aspect drift)."""
    tracker = BOTSORT(track_thresh=0.5)
    tracks = tracker.update(
        np.array([[10.0, 10.0, 110.0, 40.0]]), np.array([0.9]), np.array([0])
    )

    assert tracks and tracks[0].bbox == (10.0, 10.0, 110.0, 40.0)


def test_bot_sort_reset_restarts_ids() -> None:
    """The lifecycle (D2) carries over: reset() restarts at id 1."""
    tracker = BOTSORT()
    tracker.update(
        np.array([[10.0, 10.0, 40.0, 40.0]]), np.array([0.9]), np.array([0])
    )

    tracker.reset()
    fresh = tracker.update(
        np.array([[10.0, 10.0, 40.0, 40.0]]), np.array([0.9]), np.array([0])
    )

    assert fresh and fresh[0].track_id == 1
