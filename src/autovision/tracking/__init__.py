# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-object tracking for the inference side (roadmap A3).

ByteTrack association over per-frame detections:
:class:`~autovision.tracking.byte_track.BYTETracker`. Attached to the
reference inference runtime via ``run_stream(..., track=True)``.
"""

from autovision.tracking.analytics import (
    LineZone,
    LineZoneCounts,
    PolygonZone,
    TrackSmoother,
    ZoneCounts,
)
from autovision.tracking.bot_sort import BOTSORT
from autovision.tracking.byte_track import BYTETracker, Track
from autovision.tracking.gmc import SUPPORTED_GMC_METHODS, GMCEstimator

__all__ = [
    "BOTSORT",
    "BYTETracker",
    "GMCEstimator",
    "LineZone",
    "LineZoneCounts",
    "PolygonZone",
    "SUPPORTED_GMC_METHODS",
    "Track",
    "TrackSmoother",
    "ZoneCounts",
]
