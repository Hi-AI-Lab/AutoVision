# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Camera-motion compensation for trackers (design D6).

When the CAMERA moves (drone, vehicle, handheld), a track's predicted
position drifts by the camera delta and association degrades into ID
switches. A :class:`GMCEstimator` estimates the frame-to-frame 2x3
motion from the IMAGE (detection boxes optionally masked out as
foreground) so the tracker can warp its predictions before matching.

Algorithm reference: the standard GMC repertoire (sparse Lucas–Kanade
optical flow, ORB/SIFT + RANSAC, ECC refinement) as surveyed in the
BoT-SORT / ByteTrack lineage; implemented here on OpenCV primitives from
this repository's design document, clean-room (see NOTICE, "architecture
references").

``cv2`` lives in the ``video`` extra (AGENTS.md optional-dependency
rule): the import happens inside :meth:`apply` and raises an
``AutoVisionError`` naming the extra, so a CPU-only install without
cv2 never fails at import time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from autovision.errors import AutoVisionError

__all__ = ["GMCEstimator", "SUPPORTED_GMC_METHODS"]

logger = logging.getLogger(__name__)

#: Estimation methods. ``sparse_opt_flow`` is the fast default; ``ecc``
#: is the most accurate and the slowest; ``orb``/``sift`` sit between.
SUPPORTED_GMC_METHODS = ("none", "sparse_opt_flow", "orb", "sift", "ecc")

#: Minimum surviving point/descriptor correspondences for a usable fit.
_MIN_CORRESPONDENCES = 4


class GMCEstimator:
    """Frame-to-frame 2x3 motion estimate for one video stream.

    Owns the previous frame's (downscaled) grayscale image — one instance
    per stream, reset when the stream restarts. ``apply`` returns the
    transform that maps previous-frame coordinates to the current frame,
    or ``None`` when no confident estimate exists (first frame, too few
    correspondences, estimation failure); ``None`` means "keep the
    prediction as-is", i.e. an identity warp.
    """

    def __init__(self, method: str = "sparse_opt_flow", *, downscale: int = 2) -> None:
        if method not in SUPPORTED_GMC_METHODS:
            raise AutoVisionError(
                f"Unknown GMC method '{method}'; expected one of "
                f"{list(SUPPORTED_GMC_METHODS)}."
            )
        if downscale < 1:
            raise AutoVisionError(f"GMC downscale must be >= 1, got {downscale}.")
        self.method = method
        self.downscale = int(downscale)
        self._prev_gray: np.ndarray | None = None
        self._warned = False

    def reset(self) -> None:
        """Forget the previous frame (stream restart / tracker reset)."""
        self._prev_gray = None

    def apply(
        self, frame: np.ndarray, det_boxes: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Estimate motion from ``frame`` relative to the previous call.

        ``frame`` is HxW or HxWx3 (channel order only shifts the gray
        weighting — irrelevant for motion). ``det_boxes`` are absolute-pixel
        xyxy foreground boxes; keypoints inside them are excluded so the
        estimate describes CAMERA motion, not object motion.
        """
        cv2 = self._import_cv2()
        gray = self._to_gray(cv2, frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None
        try:
            if self.method == "sparse_opt_flow":
                transform = self._sparse_opt_flow(cv2, gray, det_boxes)
            elif self.method in ("orb", "sift"):
                transform = self._descriptor_match(cv2, gray, det_boxes)
            else:
                transform = self._ecc(cv2, gray)
        except cv2.error as error:
            logger.debug("GMC estimation failed: %s", error)
            transform = None
        self._prev_gray = gray
        if transform is None or not np.isfinite(transform).all():
            self._warn_once()
            return None
        # The estimate lives in DOWNSCALED coordinates: the linear part is
        # dimensionless, but the translation column must come back to
        # full-resolution pixels before warping absolute-pixel boxes.
        transform = np.asarray(transform, dtype=np.float64).copy()
        transform[:, 2] *= self.downscale
        return transform

    # ── internals ───────────────────────────────────────────────────────
    def _import_cv2(self) -> Any:
        try:
            import cv2
        except ImportError as error:
            raise AutoVisionError(
                "motion_compensation requires the 'cv2' package; install it "
                "with: pip install -e '.[video]'."
            ) from error
        return cv2

    def _to_gray(self, cv2: Any, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        if self.downscale > 1:
            image = cv2.resize(
                image, (max(1, width // self.downscale), max(1, height // self.downscale))
            )
        return image

    def _foreground_mask(
        self, shape: tuple[int, int], det_boxes: np.ndarray | None
    ) -> np.ndarray | None:
        """uint8 mask zeroed inside the detection boxes (foreground)."""
        mask: np.ndarray | None = None
        if det_boxes is None or len(det_boxes) == 0:
            return mask
        mask = np.full(shape, 255, dtype=np.uint8)
        scale = 1.0 / self.downscale
        for x1, y1, x2, y2 in np.asarray(det_boxes, dtype=np.float64):
            left = int(x1 * scale)
            top = int(y1 * scale)
            right = int(x2 * scale)
            bottom = int(y2 * scale)
            mask[max(0, top):max(0, bottom), max(0, left):max(0, right)] = 0
        return mask

    def _sparse_opt_flow(
        self, cv2: Any, gray: np.ndarray, det_boxes: np.ndarray | None
    ) -> np.ndarray | None:
        assert self._prev_gray is not None
        mask = self._foreground_mask(self._prev_gray.shape, det_boxes)
        points = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=10, mask=mask,
        )
        if points is None or len(points) < _MIN_CORRESPONDENCES:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, points, None
        )
        good_old = points[status.reshape(-1) == 1]
        good_new = tracked[status.reshape(-1) == 1]
        if len(good_old) < _MIN_CORRESPONDENCES:
            return None
        transform, _inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        return None if transform is None else np.asarray(transform, dtype=np.float64)

    def _descriptor_match(
        self, cv2: Any, gray: np.ndarray, det_boxes: np.ndarray | None
    ) -> np.ndarray | None:
        assert self._prev_gray is not None
        detector = (
            cv2.ORB_create(nfeatures=500)
            if self.method == "orb"
            else cv2.SIFT_create(nfeatures=500)
        )
        mask = self._foreground_mask(self._prev_gray.shape, det_boxes)
        keypoints_prev, descriptors_prev = detector.detectAndCompute(
            self._prev_gray, mask
        )
        keypoints_current, descriptors_current = detector.detectAndCompute(gray, None)
        if descriptors_prev is None or descriptors_current is None:
            return None
        if (
            len(descriptors_prev) < _MIN_CORRESPONDENCES
            or len(descriptors_current) < _MIN_CORRESPONDENCES
        ):
            return None
        norm = cv2.NORM_HAMMING if self.method == "orb" else cv2.NORM_L2
        matches = cv2.BFMatcher(norm).knnMatch(
            descriptors_prev, descriptors_current, k=2
        )
        # Lowe's ratio test: keep unambiguous correspondences only.
        good = [
            first
            for first, second in (pair for pair in matches if len(pair) == 2)
            if first.distance < 0.75 * second.distance
        ]
        if len(good) < _MIN_CORRESPONDENCES:
            return None
        source = np.float32(
            [keypoints_prev[match.queryIdx].pt for match in good]
        )
        target = np.float32(
            [keypoints_current[match.trainIdx].pt for match in good]
        )
        transform, _inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        return None if transform is None else np.asarray(transform, dtype=np.float64)

    def _ecc(self, cv2: Any, gray: np.ndarray) -> np.ndarray | None:
        assert self._prev_gray is not None
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4,
        )
        _correlation, warp = cv2.findTransformECC(
            self._prev_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 1
        )
        return np.asarray(warp, dtype=np.float64)

    def _warn_once(self) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "GMC method '%s' could not estimate motion; continuing "
                "without compensation (this warning is logged once per "
                "estimator).", self.method,
            )
