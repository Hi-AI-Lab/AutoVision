# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ByteTrack golden lifecycle snapshot (design D7 refactor anchor).

Three deterministic streams exercise every association path — high/low
pools with gaps and a degenerate box, class-swap respawns, buffer
retirement with an unconfirmed blip — including the per-frame
``det_index`` and ``drain_removed_tracks`` output (design D2). The
snapshot pins the EXACT published/drain sequences so the hook refactor
(design D7) and any later tracker change that shifts behavior is caught,
not discovered downstream.

Delete ``tests/golden/byte_track_lifecycle.json`` to re-record after an
*intentional* behavior change.

Recording is fail-fast (H4-5): a missing golden is written and then
*fails* the run — the second run performs the comparison. Record-and-skip
would let a deleted golden pass CI on its first round.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autovision.tracking.byte_track import BYTETracker

GOLDEN_PATH = Path(__file__).parent / "golden" / "byte_track_lifecycle.json"


def _empty() -> tuple[np.ndarray, np.ndarray, None]:
    return np.zeros((0, 4)), np.zeros(0), None


def _scenario_pools_and_gaps() -> tuple[dict, list]:
    """Two objects: steady high, alternating high/low, gaps, gated blip."""
    frames = []
    for frame in range(12):
        boxes, scores, classes = [], [], []
        if frame not in (5, 6):
            boxes.append([10.0 + 4 * frame, 10.0, 40.0 + 4 * frame, 40.0])
            scores.append(0.9)
            classes.append(0)
        boxes.append([300.0 + 2 * frame, 200.0, 330.0 + 2 * frame, 230.0])
        scores.append(0.7 if frame % 2 == 0 else 0.35)
        classes.append(1)
        if frame == 3:
            boxes.append([0.0, 0.0, 0.0, 10.0])  # degenerate
            scores.append(0.8)
            classes.append(0)
        if frame == 4:
            boxes.append([500.0, 400.0, 530.0, 430.0])  # below det_thresh
            scores.append(0.55)
            classes.append(0)
        frames.append((
            np.array(boxes, dtype=float).reshape(-1, 4),
            np.array(scores),
            np.array(classes),
        ))
    return (
        dict(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30),
        frames,
    )


def _scenario_class_swap() -> tuple[dict, list]:
    """One box changing class mid-stream: class-aware respawn + re-found."""
    frames = []
    for frame, class_id in enumerate([0, 0, 1, 1, 0, 0]):
        boxes = np.array([[100.0 + 3 * frame, 50.0, 130.0 + 3 * frame, 80.0]])
        frames.append((boxes, np.array([0.9]), np.array([class_id])))
    return (
        dict(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30),
        frames,
    )


def _scenario_retire_reappear() -> tuple[dict, list]:
    """Disappear past the buffer (retire+drain), then an unconfirmed blip."""
    frames = []
    for frame in range(20):
        if frame <= 2 or frame == 17:
            boxes = np.array([[60.0 + 2 * frame, 80.0, 90.0 + 2 * frame, 110.0]])
            frames.append((boxes, np.array([0.9]), np.array([0])))
        elif frame == 12:
            boxes = np.array([[500.0, 400.0, 530.0, 430.0]])
            frames.append((boxes, np.array([0.9]), np.array([0])))
        else:
            frames.append(_empty())
    return (
        dict(track_thresh=0.5, match_thresh=0.8, track_buffer=5, frame_rate=30),
        frames,
    )


def _track_record(track, *, drained: bool) -> dict:
    record = {
        "track_id": track.track_id,
        "score": round(track.score, 4),
        "class_id": int(track.class_id),
        "state": track.state,
        "age": int(track.age),
        "det_index": int(track.det_index),
    }
    if not drained:
        record["bbox"] = [round(value, 4) for value in track.bbox]
    return record


def _run_stream(kwargs: dict, frames: list) -> dict:
    tracker = BYTETracker(**kwargs)
    timeline = []
    for boxes, scores, classes in frames:
        published = [
            _track_record(track, drained=False)
            for track in tracker.update(boxes, scores, classes)
        ]
        drained = [
            _track_record(track, drained=True)
            for track in tracker.drain_removed_tracks()
        ]
        timeline.append({"published": published, "drained": drained})
    return {"kwargs": kwargs, "frames": timeline}


def _snapshot() -> dict:
    return {
        "pools_and_gaps": _run_stream(*_scenario_pools_and_gaps()),
        "class_swap": _run_stream(*_scenario_class_swap()),
        "retire_reappear": _run_stream(*_scenario_retire_reappear()),
    }


class TestGoldenLifecycle:
    """The refactor anchor: exact published/drain sequences per frame."""

    def test_lifecycle_matches_snapshot(self):
        snapshot = _snapshot()
        if not GOLDEN_PATH.exists():
            # Record-then-fail (H4-5): a missing golden must never pass its
            # first CI round — write it, fail, and let the re-run compare.
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(snapshot, indent=2))
            pytest.fail("golden recorded; re-run to verify")
        golden = json.loads(GOLDEN_PATH.read_text())
        assert golden == snapshot, (
            "Tracker behavior diverged from the golden snapshot; if this "
            f"change is intentional, delete {GOLDEN_PATH} to re-record."
        )
