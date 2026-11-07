# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""BoT-SORT: ByteTrack association with a full-state motion model (design D7).

BoT-SORT's contribution over ByteTrack is the camera-motion-aware,
box-size-aware motion model: camera compensation runs before association
(:meth:`BYTETracker._pre_association`, design D6, on by default here) and
the Kalman state carries ``(x, y, width, height)`` so size changes are
filtered instead of being squeezed through an aspect-ratio term.

Algorithm reference: BoT-SORT (Aharon et al., arXiv 2022) stage
structure; clean-room implementation on this repository's tracker hooks
(see NOTICE, "architecture references"). ReID / appearance matching is
deliberately NOT implemented (no embedding infrastructure and upstream
ReID weights need separate license review): a future variant overrides
``_first_association`` to fuse appearance distances into the stage-1
cost — the hook split is the contract.
"""

from __future__ import annotations

import numpy as np

from autovision.tracking.byte_track import BYTETracker, _KalmanFilter

__all__ = ["BOTSORT"]


class _KalmanFilterXYWH(_KalmanFilter):
    """Constant-velocity filter over the ``(x, y, width, height)`` state.

    Only the state-layout contract is overridden (design D7): the filter
    dynamics and the camera-motion warp are inherited — the warped
    components are the CENTER coordinates in both layouts.
    """

    def to_state(self, box: np.ndarray) -> np.ndarray:
        """xyxy box -> (center_x, center_y, width, height)."""
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        return np.array(
            [
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
                width,
                height,
            ],
            dtype=np.float64,
        )

    def to_box(self, mean: np.ndarray) -> np.ndarray:
        """(center_x, center_y, width, height) -> xyxy box."""
        width = mean[2]
        height = mean[3]
        return np.array(
            [
                mean[0] - width / 2.0,
                mean[1] - height / 2.0,
                mean[0] + width / 2.0,
                mean[1] + height / 2.0,
            ]
        )


class BOTSORT(BYTETracker):
    """BoT-SORT tracker: drop-in ``BYTETracker`` alternative for moving
    cameras.

    Inherits the full ByteTrack association schedule (confirmation gate,
    high/low pools, buffer retirement, ``det_index`` join-back and the
    ``drain_removed_tracks`` lifecycle). Differences by default:
    ``motion_compensation="sparse_opt_flow"`` (pass ``frame=`` to
    ``update``) and an XYWH Kalman state.
    """

    def __init__(
        self, *, motion_compensation: str = "sparse_opt_flow", **kwargs
    ) -> None:
        super().__init__(motion_compensation=motion_compensation, **kwargs)

    def _make_kalman_filter(self) -> _KalmanFilter:
        return _KalmanFilterXYWH()
