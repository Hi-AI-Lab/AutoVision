#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tracking subsystem smoke test (``tracking/``).

Exercises the tracking primitives without any video file: a synthetic frame
sequence (one box sweeping left→right across a vertical line + a polygon, and
a second box appearing in the second half) is fed frame-by-frame to
``BYTETracker``. The returned activated tracks drive ``LineZone`` and
``PolygonZone`` counting and ``TrackSmoother``. The demo asserts that the
sweeping box keeps a STABLE track id across frames (association works) and
that the line zone records a crossing and the polygon a count.

    python examples/tasks/tracking_demo.py
"""

from __future__ import annotations

import argparse

import numpy as np

from autovision.tracking import (
    BYTETracker,
    LineZone,
    PolygonZone,
    TrackSmoother,
)

WIDTH, HEIGHT = 640, 480
NUM_FRAMES = 30


def _synthetic_sequence() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-frame (boxes, scores, classes) for a deterministic scene.

    Box A (160px) sweeps left→right with ~14px/frame motion, so its frame-to-
    frame IoU stays above the 0.8 association ceiling and ByteTrack keeps ONE
    stable id; it crosses the vertical line at x=320 and enters the polygon
    band. Box B appears only in the second half (a distinct id).
    """
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    half = 80.0  # 160px box -> frame-to-frame IoU ~0.84 at 14px/frame
    for frame in range(NUM_FRAMES):
        boxes, scores, classes = [], [], []
        # Box A: center sweeping x in [120, 520] at y=240.
        cx = 120.0 + (520.0 - 120.0) * frame / (NUM_FRAMES - 1)
        boxes.append([cx - half, 240 - half, cx + half, 240 + half])
        scores.append(0.9)
        classes.append(0)
        # Box B: appears from frame 15, fixed top-left, class 1.
        if frame >= 15:
            boxes.append([40, 40, 200, 200])
            scores.append(0.85)
            classes.append(1)
        frames.append(
            (
                np.asarray(boxes, dtype=np.float32),
                np.asarray(scores, dtype=np.float32),
                np.asarray(classes, dtype=np.int64),
            )
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    frames = _synthetic_sequence()[: args.frames]

    tracker = BYTETracker(track_thresh=0.5, frame_rate=30, class_aware=True)
    line_zone = LineZone(start=(320, 0), end=(320, HEIGHT), name="midline")
    polygon_zone = PolygonZone(
        polygon=[(300, 150), (600, 150), (600, 330), (300, 330)],
        name="center-band",
    )
    smoother = TrackSmoother(length=5)

    seen_ids: set[int] = set()
    smoothed_ids: set[int] = set()
    for frame, (boxes, scores, classes) in enumerate(frames):
        tracks = tracker.update(boxes, scores, classes)
        tracks = smoother.update(tracks)  # smoothed boxes/score, same ids
        tracker.drain_removed_tracks()  # drop per-id state for retired tracks
        line_zone.update(tracks)
        polygon_zone.update(tracks)
        for track in tracks:
            seen_ids.add(int(track.track_id))
            smoothed_ids.add(int(track.track_id))
        print(
            f"frame {frame:2d}: {len(tracks)} active track(s) -> "
            f"ids={sorted(int(t.track_id) for t in tracks)}"
        )

    print("\nunique track ids seen :", sorted(seen_ids), "(expect 2: A and B)")
    print("smoother preserved ids:", sorted(smoothed_ids))
    print("line zone counts      :", line_zone.counts.as_dict())
    print("polygon zone counts   :", polygon_zone.counts.as_dict())

    # The sweeping box must keep ONE stable id and cross the line / enter the
    # polygon; the late box is a distinct id.
    assert len(seen_ids) == 2, f"expected 2 track ids, got {seen_ids}"
    assert line_zone.counts.in_count + line_zone.counts.out_count >= 1
    assert polygon_zone.counts.total >= 1
    print("\nOK tracking primitives (BYTETracker + zones + smoother)")


if __name__ == "__main__":
    main()
