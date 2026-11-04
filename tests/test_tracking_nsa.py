# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""NSA observation-noise scaling for the tracking Kalman (design D36).

Contract: ``confidence=None`` keeps the official ByteTrack fixed-noise
numerics bit-exact (the golden lifecycle fixture anchors that); a score
scales the innovation covariance by ``max(1 - score, 0.05)`` so a
high-confidence detection pulls the state harder than a low-confidence
one. Opt-in via ``BYTETracker(confidence_scaling=True)``; BoT-SORT
inherits through the D7 motion-model hook.
"""

from __future__ import annotations

import numpy as np
import pytest

from autovision.tracking.bot_sort import BOTSORT
from autovision.tracking.byte_track import BYTETracker, _KalmanFilter


def _tracker_frame(
    tracker: BYTETracker,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray | None = None,
) -> list:
    return tracker.update(boxes, scores, classes)


class TestProjectScaling:
    """The innovation-covariance scaling itself."""

    def test_none_confidence_keeps_official_noise(self) -> None:
        """confidence=None is numerically identical to the legacy project()."""
        kalman = _KalmanFilter()
        mean, covariance = kalman.initiate(np.array([5.0, 5.0, 0.5, 10.0]))
        mean = kalman._motion @ mean
        covariance = kalman._motion @ covariance @ kalman._motion.T
        _, without = kalman.project(mean, covariance)
        _, with_none = kalman.project(mean, covariance, confidence=None)
        assert np.array_equal(without, with_none)

    @pytest.mark.parametrize(
        ("confidence", "factor"),
        [
            pytest.param(1.0, 0.05, id="certain-floored"),
            pytest.param(0.3, 0.7, id="mid"),
            pytest.param(0.5, 0.5, id="half"),
            pytest.param(-0.4, 1.4, id="negative-score-widens"),
        ],
    )
    def test_confidence_scales_the_noise(self, confidence: float, factor: float) -> None:
        """projected covariance grows by exactly max(1 - conf, 0.05)."""
        kalman = _KalmanFilter()
        mean, covariance = kalman.initiate(np.array([5.0, 5.0, 0.5, 10.0]))
        mean = kalman._motion @ mean
        covariance = kalman._motion @ covariance @ kalman._motion.T
        _, base = kalman.project(mean, covariance)
        _, scaled = kalman.project(mean, covariance, confidence=confidence)
        # The innovation-covariance term contributes only to the diagonal of
        # the projected covariance; the process-noise term is unchanged.
        process_term = (
            kalman._update_matrix @ covariance @ kalman._update_matrix.T
        )
        assert np.allclose(
            scaled - process_term, (base - process_term) * factor
        )


class TestUpdateDirection:
    """A confident observation pulls the posterior closer than a weak one."""

    def _posterior_distance(self, score: float) -> float:
        tracker = BYTETracker(track_thresh=0.1, confidence_scaling=True)
        box = np.array([[10.0, 10.0, 30.0, 30.0]])
        first = _tracker_frame(tracker, box, np.array([score]))
        assert first, "frame 1 activates every spawned track"
        # The object moved: the posterior lands between the prediction and
        # the new observation, nearer to it under a higher confidence.
        moved = box + np.array([2.0, 0.0, 2.0, 0.0])
        second = _tracker_frame(tracker, moved, np.array([score]))
        assert second
        return abs(second[0].bbox[0] - 12.0) + abs(second[0].bbox[2] - 42.0)

    def test_high_confidence_tracks_the_observation_more_tightly(self) -> None:
        """A 0.9 detection drags the posterior nearer than a 0.3 one."""
        assert self._posterior_distance(0.9) < self._posterior_distance(0.3)


class TestDefaultOff:
    """Default construction keeps the official fixed-noise semantics."""

    def test_default_tracker_has_scaling_off(self) -> None:
        assert BYTETracker().confidence_scaling is False

    def test_disabled_tracker_uses_none_confidence(self) -> None:
        """The stage hooks pass ``None`` to the Kalman when scaling is off."""
        tracker = BYTETracker()
        box = np.array([[10.0, 10.0, 30.0, 30.0]])
        tracker.update(box, np.array([0.9]))
        track = tracker.tracks[0]
        mean_before = track.mean.copy()
        covariance_before = track.covariance.copy()
        # Direct predict+fixed-noise-update for comparison — the tracker
        # frame 2 path is predict() then update(confidence=None).
        kalman = _KalmanFilter()
        mean_pred, covariance_pred = kalman.predict(mean_before, covariance_before)
        expected_mean, expected_covariance = kalman.update(
            mean_pred, covariance_pred, kalman.to_state(box[0]),
        )
        tracker.update(box, np.array([0.9]))
        assert np.allclose(track.mean, expected_mean)
        assert np.allclose(track.covariance, expected_covariance)


class TestBotSortInheritance:
    """The D7 motion-model hook carries NSA scaling to BoT-SORT."""

    def test_bot_sort_accepts_confidence_scaling(self) -> None:
        tracker = BOTSORT(confidence_scaling=True)
        assert tracker.confidence_scaling is True
        box = np.array([[10.0, 10.0, 30.0, 30.0]])
        for _ in range(3):
            tracks = tracker.update(box, np.array([0.9]))
        assert tracks, "BoT-SORT publishes the activated track"
