# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ByteTrack: per-frame detection association with a Kalman motion model (A3).

Reference implementation of the ByteTrack association strategy (Zhang et
al., ECCV 2022 — "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box"), following the official ``BYTETracker`` (yolox) semantics:

- a constant-velocity Kalman filter per track over the ``(x, y, aspect,
  height)`` state, with the official process/measurement noise weights;
- every frame, detections split at ``track_thresh`` into the HIGH pool and
  the LOW-score pool ``0.1 < score < track_thresh`` — ByteTrack's core
  idea is that low-confidence boxes still carry motion evidence and must
  not be dropped;
- THREE association passes, with the official thresholds (IoU similarity):
  1. activated tracks (tracked + lost) vs the high pool —
     ``sim >= 1 - match_thresh`` with the detection score fused into the
     similarity (``sim = IoU * score``, the official non-mot20 path);
  2. still-tracked leftovers vs the low pool — plain IoU ``>= 0.5``;
  3. unconfirmed one-frame tracks vs the high detections left over —
     ``sim >= 0.3``; an unmatched unconfirmed track is removed WITHOUT
     ever being published (the confirmation gate);
- a detection starts a track only with ``score >= track_thresh + 0.1``
  (``det_thresh``), and the new track stays unconfirmed until its first
  successful match — the first frame is the official exception, where
  every new track is activated at once.

Differences from the official MOT17 implementation, documented rather
than hidden:

- **Class-aware**: boxes are associable only when their class ids match
  (the official tracker is class-agnostic for single-class MOT; the
  restriction stops ids swapping across classes on multi-class output).
- **Per-frame API** over absolute-pixel ``xyxy`` boxes (the framework's
  native detection output) instead of the official ``tlwh``; the state
  conversion happens inside.
- The ``mot20`` variant (unfused stage-1 cost) is not implemented — the
  official non-mot20 behavior is the only mode.

Deterministic: given the same inputs and ``frame_rate`` the ids are stable
(no RNG anywhere; one id counter per tracker instance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from autovision.errors import AutoVisionError

__all__ = ["BYTETracker", "Track"]

logger = logging.getLogger(__name__)

#: ByteTrack's Kalman std weights (official values).
_STD_WEIGHT_POSITION = 1.0 / 20.0
_STD_WEIGHT_VELOCITY = 1.0 / 160.0

#: Official gates (BYTETracker, non-mot20 defaults).
_DET_THRESH_MARGIN = 0.1  # new tracks need score >= track_thresh + 0.1
_LOW_SCORE_FLOOR = 0.1  # the low pool is 0.1 < score < track_thresh
_SECOND_MATCH_IOU = 0.5  # stage 2: plain-IoU similarity threshold
_UNCONFIRMED_DIST_THRESH = 0.7  # stage 3: fused similarity >= 1 - 0.7

_NEW = "new"
_TRACKED = "tracked"
_LOST = "lost"
_REMOVED = "removed"


@dataclass
class Track:
    """One live track as reported to the caller."""

    track_id: int
    bbox: tuple[float, float, float, float]  # xyxy absolute pixels
    score: float
    class_id: int
    state: str = _TRACKED
    #: Frames since the track was created (``frame_id - start_frame``).
    age: int = 0
    #: Index of the detection this track matched in the frame's input
    #: arrays (design D2); ``-1`` when the track matched nothing this
    #: frame (predicted position only). Lets callers join tracker output
    #: back onto their own detection records without re-doing the IoU.
    det_index: int = -1


class _KalmanFilter:
    """Constant-velocity filter over the ``(x, y, aspect, height)`` state.

    Also owns the box<->state conversion and the camera-motion warp, so a
    motion-model variant (BoT-SORT's XYWH, design D7) subclasses and swaps
    only these — the association stages never see the state layout.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim, dtype=np.float64)
        for index in range(ndim):
            self._motion[index, ndim + index] = dt
        self._update_matrix = np.eye(ndim, 2 * ndim, dtype=np.float64)

    # ── state layout contract ───────────────────────────────────────
    def to_state(self, box: np.ndarray) -> np.ndarray:
        """xyxy box -> the filter's 4-state measurement."""
        return _xyxy_to_xyah(box)

    def to_box(self, mean: np.ndarray) -> np.ndarray:
        """The filter's 4-state mean -> xyxy box."""
        return _xyah_to_xyxy(mean)

    def warp_state(self, mean: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Warp the state's positional components by a 2x3 affine."""
        warped = mean.copy()
        x, y = warped[0], warped[1]
        warped[0] = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]
        warped[1] = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]
        return warped

    # ── filter dynamics (official ByteTrack parameterization) ───────
    def _std(self, position: float, velocity: float, height: float) -> np.ndarray:
        return np.array(
            [
                position * height, position * height, 1e-2, position * height,
                velocity * height, velocity * height, 1e-5, velocity * height,
            ],
            dtype=np.float64,
        )

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        std = self._std(_STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY, float(measurement[3]))
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std = self._std(_STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY, float(mean[3]))
        motion_cov = np.diag(np.square(std))
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + motion_cov
        return mean, covariance

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        confidence: float | None = None,
    ):
        # Measurement noise is 4-dimensional (position-only weights, 1e-1 for
        # the aspect term) — the 8-dimensional vector above is the PROCESS
        # noise of ``predict`` (official ByteTrack parameterization).
        std = np.array(
            [
                _STD_WEIGHT_POSITION * mean[3],
                _STD_WEIGHT_POSITION * mean[3],
                1e-1,
                _STD_WEIGHT_POSITION * mean[3],
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:
            # NSA (StrongSORT): scale the observation noise by the detection
            # confidence so high-confidence boxes pull the state harder; the
            # 0.05 floor keeps a near-certain detection from collapsing the
            # innovation covariance to singular (design D36).
            innovation_cov *= max(1.0 - float(confidence), 0.05)
        projected_mean = self._update_matrix @ mean
        projected_cov = self._update_matrix @ covariance @ self._update_matrix.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float | None = None,
    ):
        projected_mean, projected_cov = self.project(
            mean, covariance, confidence=confidence
        )
        # Kalman gain via an explicit solve (no scipy.linalg dependency).
        gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._update_matrix.T).T
        ).T
        new_mean = mean + gain @ (measurement - projected_mean)
        new_covariance = covariance - gain @ projected_cov @ gain.T
        return new_mean, new_covariance


def _xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return np.array(
        [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, width / max(height, 1e-6), height],
        dtype=np.float64,
    )


def _xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    height = state[3]
    width = state[2] * height
    return np.array(
        [
            state[0] - width / 2.0,
            state[1] - height / 2.0,
            state[0] + width / 2.0,
            state[1] + height / 2.0,
        ]
    )


def _xyxy_tuple(box: np.ndarray) -> tuple[float, float, float, float]:
    """A 4-vector as a strict xyxy tuple (``Track.bbox``'s exact type)."""
    x1, y1, x2, y2 = box.tolist()
    return x1, y1, x2, y2


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every ``a[i]``/``b[j]`` pair (``(N, M)``, numpy only)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = (np.minimum(ax2, bx2) - np.maximum(ax1, bx1)).clip(0) * (
        np.minimum(ay2, by2) - np.maximum(ay1, by1)
    ).clip(0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


class _STrack:
    """One Kalman track's bookkeeping (``new`` → ``tracked`` ⇄ ``lost``)."""

    def __init__(
        self,
        box: np.ndarray,
        score: float,
        class_id: int,
        *,
        frame_id: int,
        kalman: "_KalmanFilter",
        det_index: int = -1,
    ) -> None:
        self._kalman = kalman
        self.mean, self.covariance = kalman.initiate(kalman.to_state(box))
        self.score = float(score)
        self.class_id = int(class_id)
        self.track_id = 0
        self.det_index = int(det_index)
        #: Unconfirmed until the first successful match (frame 1 excepted).
        self.state = _NEW
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = int(frame_id)

    @property
    def box(self) -> np.ndarray:
        return self._kalman.to_box(self.mean)

    def activate(self, track_id: int) -> None:
        self.track_id = track_id
        self.state = _TRACKED

    def predict(self) -> None:
        self.mean, self.covariance = self._kalman.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(
        self,
        box: np.ndarray,
        score: float,
        class_id: int,
        det_index: int = -1,
        confidence: float | None = None,
    ) -> None:
        """Kalman update from a matched detection.

        Carries the state transitions too: a ``new`` track becomes
        ``tracked`` (activation) and a ``lost`` track is re-found.
        ``confidence`` (design D36) scales the observation noise by the
        matched detection's score when the tracker enabled NSA scaling;
        ``None`` keeps the official fixed-noise behavior.
        """
        self.mean, self.covariance = self._kalman.update(
            self.mean, self.covariance, self._kalman.to_state(box),
            confidence=confidence,
        )
        self.score = float(score)
        self.class_id = int(class_id)
        self.det_index = int(det_index)
        self.hits += 1
        self.time_since_update = 0
        self.state = _TRACKED

    def warp_affine(self, transform: np.ndarray) -> None:
        """Warp the state's position by a 2x3 frame-to-frame motion matrix
        (design D6): the camera moved, so the PREDICTED position must move
        with it before association. Only the center is warped — camera
        rotation/scale changes to apparent size are second-order for the
        sparse-flow estimate and stay on the Kalman process noise."""
        self.mean = self._kalman.warp_state(self.mean, transform)


@dataclass(frozen=True)
class _FrameDetections:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    #: Position of each row in the frame's ORIGINAL detection arrays, so a
    #: match can be reported as ``Track.det_index`` (design D2).
    det_indices: np.ndarray

    def take(self, indices: list[int]) -> "_FrameDetections":
        return _FrameDetections(
            self.boxes[indices],
            self.scores[indices],
            self.classes[indices],
            self.det_indices[indices],
        )


class BYTETracker:
    """ByteTrack association over a detection stream.

    ``update(boxes, scores, classes)`` takes absolute-pixel ``xyxy`` boxes
    for ONE frame and returns the ACTIVATED tracks (``Track``). New tracks
    need ``score >= track_thresh + 0.1`` and stay unconfirmed until their
    first successful match (frame 1 excepted) — a one-frame blip is never
    published. Parameters follow the official API: ``track_thresh`` splits
    the high/low detection pools, ``match_thresh`` is the stage-1
    similarity ceiling, ``track_buffer`` (in frames at ``frame_rate``)
    bounds how long a lost track survives.

    State lifecycle (design D2): each track reports the index of the
    detection it matched (``Track.det_index``); tracks retired by the
    buffer are published once through :meth:`drain_removed_tracks` so
    per-id downstream state (zone histories, smoothing buffers) can be
    dropped; :meth:`reset` restarts the tracker for a new source. One
    tracker instance belongs to one stream — never share it across
    streams or cache it outside the stream loop.
    """

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        class_aware: bool = True,
        motion_compensation: str = "none",
        confidence_scaling: bool = False,
    ) -> None:
        if not 0.0 <= track_thresh <= 1.0:
            raise AutoVisionError(f"track_thresh must be in [0, 1], got {track_thresh}.")
        if not 0.0 <= match_thresh <= 1.0:
            raise AutoVisionError(f"match_thresh must be in [0, 1], got {match_thresh}.")
        if track_buffer < 0:
            raise AutoVisionError(f"track_buffer must be >= 0, got {track_buffer}.")
        if frame_rate < 1:
            raise AutoVisionError(f"frame_rate must be >= 1, got {frame_rate}.")
        self.track_thresh = float(track_thresh)
        self.det_thresh = self.track_thresh + _DET_THRESH_MARGIN
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate = int(frame_rate)
        self.class_aware = bool(class_aware)
        # NSA observation-noise scaling (design D36): opt-in — the default
        # keeps the official ByteTrack fixed-noise numerics bit-exact.
        self.confidence_scaling = bool(confidence_scaling)
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        #: Per-instance (not module-level) so a tracker's matrices can never
        #: become shared mutable state across streams or resets.
        self._kalman = self._make_kalman_filter()
        #: Camera-motion compensation (design D6): ``None`` disables it;
        #: otherwise a lazily-imported :class:`GMCEstimator` (cv2 lives in
        #: the ``video`` extra).
        self._gmc = None
        if motion_compensation != "none":
            from autovision.tracking.gmc import GMCEstimator

            self._gmc = GMCEstimator(motion_compensation)
        #: Every live track (the ``state`` field carries new/tracked/lost);
        #: the tracker owns the list so a re-found track cannot linger in a
        #: stale pool.
        self.tracks: list[_STrack] = []
        #: Tracks retired this frame, published once by
        #: :meth:`drain_removed_tracks` (tracks removed before their first
        #: successful match were never published and are NOT queued here).
        self._removed: list[Track] = []
        self._next_id = 1
        self.frame_id = 0

    def _make_kalman_filter(self) -> "_KalmanFilter":
        """The motion model (design D7 hook): BoT-SORT swaps in XYWH."""
        return _KalmanFilter()

    def _kalman_confidence(self, score: Any) -> float | None:
        """The matched score as an NSA observation weight (design D36).

        ``None`` keeps the official fixed-noise update; a score scales the
        innovation covariance by ``max(1 - score, 0.05)`` when the tracker
        was built with ``confidence_scaling``.
        """
        return float(score) if self.confidence_scaling else None

    def reset(self) -> None:
        """Clear all state: ids restart at 1, as for a brand-new source."""
        self.tracks.clear()
        self._removed.clear()
        if self._gmc is not None:
            self._gmc.reset()
        self._next_id = 1
        self.frame_id = 0

    def drain_removed_tracks(self) -> list[Track]:
        """Return (and clear) the tracks retired so far (design D2).

        Consumers keeping per-track-id state (zone histories, smoothers)
        call this after each ``update`` to drop state for ids that will
        never come back. Tracks removed before ever being published
        (unconfirmed) are not included.
        """
        removed, self._removed = self._removed, []
        return removed

    # ── association ─────────────────────────────────────────────────────
    def _match(
        self,
        tracks: list[_STrack],
        detections: _FrameDetections,
        similarity_threshold: float,
        *,
        fuse_score: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian over the similarity cost.

        Returns ``(pairs, unmatched_tracks, unmatched_dets)``. With
        ``fuse_score`` the detection score multiplies the IoU (the official
        ``fuse_score``, non-mot20 path); a class mismatch zeroes the pair
        under ``class_aware``. Matching is 1:1 — every detection and every
        track is consumed at most once.
        """
        from scipy.optimize import linear_sum_assignment

        if not tracks or len(detections.boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(detections.boxes)))
        boxes = np.stack([track.box for track in tracks])
        similarity = _iou_matrix(boxes, detections.boxes)
        if self.class_aware:
            track_classes = np.array([track.class_id for track in tracks])[:, None]
            similarity = np.where(
                track_classes == detections.classes[None, :], similarity, 0.0
            )
        if fuse_score:
            similarity = similarity * detections.scores[None, :]
        cost = 1.0 - similarity
        rows, cols = linear_sum_assignment(cost)
        pairs, matched_tracks, matched_dets = [], set(), set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if similarity[row, col] >= similarity_threshold:
                pairs.append((row, col))
                matched_tracks.add(row)
                matched_dets.add(col)
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections.boxes)) if i not in matched_dets]
        return pairs, unmatched_tracks, unmatched_dets

    # ── per-frame API ───────────────────────────────────────────────────
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Associate one frame's detections; returns the activated tracks.

        ``Track.det_index`` on each result indexes the arrays as passed in
        (degenerate boxes dropped by validation keep their original
        positions, design D2). ``frame`` (HxW or HxWx3) is optional and
        only consumed when the tracker was built with
        ``motion_compensation`` (design D6); the fixed template below is
        the official ByteTrack schedule, decomposed into overridable
        hooks (design D7) so variants like BoT-SORT replace stages
        without copying the orchestration.
        """
        self.frame_id += 1
        boxes, scores, classes, source_index = self._validate(boxes, scores, classes)
        high, low = self._split_detections(boxes, scores, classes, source_index)
        self._predict_tracks()
        self._pre_association(frame, boxes)
        pool = [track for track in self.tracks if track.state in (_TRACKED, _LOST)]
        unconfirmed = [track for track in self.tracks if track.state == _NEW]
        unmatched_pool, remaining = self._first_association(pool, high)
        self._second_association(pool, unmatched_pool, low)
        remaining = self._unconfirmed_association(unconfirmed, remaining)
        self._spawn_new_tracks(remaining)
        self._retire_tracks()
        return self._published()

    # ── stage hooks (template methods; overridable, design D7) ─────────
    def _split_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        source_index: np.ndarray,
    ) -> tuple[_FrameDetections, _FrameDetections]:
        """Split the frame's detections into the HIGH and LOW pools."""
        high_mask = scores > self.track_thresh
        low_mask = (scores > _LOW_SCORE_FLOOR) & (scores < self.track_thresh)
        high = _FrameDetections(
            boxes[high_mask], scores[high_mask], classes[high_mask],
            source_index[high_mask],
        )
        low = _FrameDetections(
            boxes[low_mask], scores[low_mask], classes[low_mask],
            source_index[low_mask],
        )
        return high, low

    def _predict_tracks(self) -> None:
        """Kalman-predict every track; matched detections overwrite the
        ``det_index`` reset performed here."""
        for track in self.tracks:
            track.predict()
            track.det_index = -1

    def _pre_association(
        self, frame: np.ndarray | None, det_boxes: np.ndarray
    ) -> None:
        """Camera-motion compensation hook (design D6): warp live tracks
        by the estimated frame-to-frame motion before any association."""
        if self._gmc is None or frame is None:
            return
        transform = self._gmc.apply(frame, det_boxes=det_boxes)
        if transform is None:
            return
        for track in self.tracks:
            track.warp_affine(transform)

    def _first_association(
        self, pool: list[_STrack], high: _FrameDetections
    ) -> tuple[list[int], _FrameDetections]:
        """Stage 1: activated tracks (tracked + lost) vs the high-score pool.

        Returns ``(unmatched_pool_indices, leftover_high)``.
        """
        pairs, unmatched_pool, unmatched_high = self._match(
            pool, high, 1.0 - self.match_thresh, fuse_score=True
        )
        for track_index, det_index in pairs:
            pool[track_index].update(
                high.boxes[det_index], float(high.scores[det_index]),
                int(high.classes[det_index]), det_index=int(high.det_indices[det_index]),
                confidence=self._kalman_confidence(high.scores[det_index]),
            )
        return unmatched_pool, high.take(unmatched_high)

    def _second_association(
        self,
        pool: list[_STrack],
        unmatched_pool: list[int],
        low: _FrameDetections,
    ) -> None:
        """Stage 2: still-tracked leftovers vs the LOW pool — ByteTrack's
        "associate every detection box"; leftovers drop to ``lost``."""
        tracked_left = [pool[i] for i in unmatched_pool if pool[i].state == _TRACKED]
        pairs, unmatched_tracked, _ = self._match(tracked_left, low, _SECOND_MATCH_IOU)
        for track_index, det_index in pairs:
            tracked_left[track_index].update(
                low.boxes[det_index], float(low.scores[det_index]),
                int(low.classes[det_index]), det_index=int(low.det_indices[det_index]),
                confidence=self._kalman_confidence(low.scores[det_index]),
            )
        for track_index in unmatched_tracked:
            tracked_left[track_index].state = _LOST

    def _unconfirmed_association(
        self, unconfirmed: list[_STrack], remaining: _FrameDetections
    ) -> _FrameDetections:
        """Stage 3: unconfirmed tracks vs the leftovers of stage 1; an
        unmatched unconfirmed track is removed WITHOUT ever being
        published (the confirmation gate). Returns ``remaining``."""
        pairs, unmatched_new, _ = self._match(
            unconfirmed, remaining, 1.0 - _UNCONFIRMED_DIST_THRESH, fuse_score=True
        )
        for track_index, det_index in pairs:
            unconfirmed[track_index].update(
                remaining.boxes[det_index],
                float(remaining.scores[det_index]),
                int(remaining.classes[det_index]),
                det_index=int(remaining.det_indices[det_index]),
                confidence=self._kalman_confidence(remaining.scores[det_index]),
            )
        for track_index in unmatched_new:
            unconfirmed[track_index].state = _REMOVED
        return remaining

    def _spawn_new_tracks(self, remaining: _FrameDetections) -> None:
        """New tracks from the high detections nobody claimed; gated on
        ``det_thresh`` and activated at once on frame 1 (the official
        exception)."""
        for det_index in range(len(remaining.boxes)):
            if remaining.scores[det_index] < self.det_thresh:
                continue
            track = _STrack(
                remaining.boxes[det_index],
                float(remaining.scores[det_index]),
                int(remaining.classes[det_index]),
                frame_id=self.frame_id,
                kalman=self._kalman,
                det_index=int(remaining.det_indices[det_index]),
            )
            track.track_id = self._next_id
            self._next_id += 1
            if self.frame_id == 1:
                track.state = _TRACKED
            self.tracks.append(track)

    def _retire_tracks(self) -> None:
        """Retire tracks missing beyond the buffer. A retired track HAS
        been published before, so it is queued for
        ``drain_removed_tracks`` (design D2); unconfirmed removals were
        never published and are dropped silently."""
        surviving: list[_STrack] = []
        for track in self.tracks:
            if track.state == _LOST and track.time_since_update > self.max_time_lost:
                track.state = _REMOVED
                self._removed.append(
                    Track(
                        track_id=track.track_id,
                        bbox=_xyxy_tuple(track.box),
                        score=track.score,
                        class_id=track.class_id,
                        state=_REMOVED,
                        age=self.frame_id - track.start_frame,
                        det_index=-1,
                    )
                )
            else:
                surviving.append(track)
        self.tracks = surviving

    def _published(self) -> list[Track]:
        """The frame's ACTIVATED tracks as public ``Track`` records."""
        return [
            Track(
                track_id=track.track_id,
                bbox=_xyxy_tuple(track.box),
                score=track.score,
                class_id=track.class_id,
                state=track.state,
                age=self.frame_id - track.start_frame,
                det_index=track.det_index,
            )
            for track in self.tracks
            if track.state == _TRACKED
        ]

    # ── validation ──────────────────────────────────────────────────────
    def _validate(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Strict input checks — bad geometry fails here, not in the Kalman.

        Returns the validated arrays plus each surviving row's index in the
        ARRAYS AS PASSED (``source_index``), so a match can be reported as
        ``Track.det_index`` against the caller's own detection records even
        when degenerate boxes were dropped.
        """
        boxes_array = np.asarray(boxes, dtype=np.float64)
        if boxes_array.size:
            if boxes_array.ndim == 1:
                boxes_array = boxes_array.reshape(1, -1)
            if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
                raise AutoVisionError(
                    "tracker.update: boxes must be (N, 4) xyxy, got shape "
                    f"{boxes_array.shape}."
                )
            if not np.isfinite(boxes_array).all():
                raise AutoVisionError("tracker.update: boxes contain non-finite values.")
        else:
            boxes_array = boxes_array.reshape(0, 4)
        scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores_array) != len(boxes_array):
            raise AutoVisionError(
                f"tracker.update: {len(boxes_array)} boxes but {len(scores_array)} scores."
            )
        if not np.isfinite(scores_array).all():
            raise AutoVisionError("tracker.update: scores contain non-finite values.")
        classes_array: np.ndarray
        if classes is None:
            classes_array = np.zeros(len(boxes_array), dtype=np.int64)
        else:
            classes_array = np.asarray(classes).reshape(-1)
            if len(classes_array) != len(boxes_array):
                raise AutoVisionError(
                    f"tracker.update: {len(boxes_array)} boxes but "
                    f"{len(classes_array)} classes."
                )
            if classes_array.dtype.kind not in "iu":
                raise AutoVisionError(
                    "tracker.update: classes must be integer-valued, got dtype "
                    f"{classes_array.dtype}."
                )
        # Zero/negative-area boxes make the Kalman innovation covariance
        # singular. Drop them at the boundary instead of failing mid-stream;
        # survivors keep their original positions via ``source_index``.
        widths = boxes_array[:, 2] - boxes_array[:, 0]
        heights = boxes_array[:, 3] - boxes_array[:, 1]
        valid = (widths > 0) & (heights > 0)
        source_index = np.flatnonzero(valid).astype(np.int64)
        if not valid.all():
            logger.warning(
                "tracker.update: dropped %d degenerate boxes (non-positive "
                "width/height).",
                int((~valid).sum()),
            )
            boxes_array = boxes_array[valid]
            scores_array = scores_array[valid]
            classes_array = classes_array[valid]
        return boxes_array, scores_array, classes_array.astype(np.int64), source_index
