# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Polygon primitives shared by the tiling and augmentation surfaces.

NumPy-only (the geometry package convention): :func:`clip_polygon_to_canvas`
is the single source of polygon-vs-rectangle clipping — training-side
tiling (design D9) uses it for window-target assignment, and any future
oriented-box affine path would reuse it instead of re-deriving Sutherland–
Hodgman inline. :func:`is_valid_polygon` / :func:`polygon_from_bbox`
carry the COCO-import sanitization contract (design D19).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "clip_polygon_to_canvas",
    "is_valid_polygon",
    "polygon_area",
    "polygon_from_bbox",
]


def polygon_area(poly: np.ndarray) -> float:
    """Absolute shoelace area of an ``(N, 2)`` polygon (open or closed)."""
    points = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def is_valid_polygon(values: Any) -> bool:
    """True when ``values`` is a usable flat polygon (design doc D19).

    Usable means: a 1-D sequence of at least 3 ``(x, y)`` points (so even
    length >= 6) whose entries are all finite numbers — the whitelist that
    downstream rasterizers (``pycocotools`` ``frPyObjects``) silently
    assume. Anything else (ragged, odd length, NaN, non-numeric) fails
    here instead of deep inside a rasterizer.

    Args:
        values: Flat polygon candidate (list/tuple/ndarray of numbers).

    Returns:
        True when the candidate can be handed to a rasterizer as-is.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.size < 6 or values.size % 2 != 0:
            return False
        try:
            flat: np.ndarray = values.astype(np.float64)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(flat).all())
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) < 6 or len(values) % 2 != 0:
        return False
    try:
        flat = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(flat).all())


def polygon_from_bbox(bbox: Sequence[float]) -> np.ndarray | None:
    """Axis-aligned ``(4, 2)`` rectangle polygon from a COCO ``bbox``.

    The fallback shape for malformed segmentation polygons (design D19):
    the instance keeps its box geometry instead of being dropped.

    Args:
        bbox: COCO ``[x, y, width, height]``.

    Returns:
        The rectangle polygon, or ``None`` when ``bbox`` itself is not a
        usable 4-entry all-finite sequence — a NaN rectangle would only
        move the crash from the polygon rasterizer to here.
    """
    if isinstance(bbox, np.ndarray):
        if bbox.ndim != 1 or bbox.size != 4:
            return None
    elif not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x, y, width, height)):
        return None
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64,
    )


def clip_polygon_to_canvas(
    poly: np.ndarray, canvas_hw: tuple[int, int]
) -> np.ndarray | None:
    """Clip an ``(N, 2)`` polygon to ``[0, w] x [0, h]`` (Sutherland–Hodgman).

    Returns the clipped polygon with at least 3 vertices, or ``None`` when
    the intersection degenerates (fully outside, or reduced to a line).
    """
    points = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    height, width = canvas_hw
    # (axis, sign, bound) for the four canvas edges; a point is inside the
    # edge when ``sign * coord >= bound`` (right/top use the negated bound
    # so ``-x >= -w`` means ``x <= w``).
    edges = (
        (0, 1, 0.0),
        (0, -1, -float(width)),
        (1, 1, 0.0),
        (1, -1, -float(height)),
    )
    output = points
    for axis, sign, bound in edges:
        output = _clip_against_edge(output, axis=axis, sign=sign, bound=bound)
        if output is None:
            return None
    return output


def _clip_against_edge(
    poly: np.ndarray, *, axis: int, sign: int, bound: float
) -> np.ndarray | None:
    """Clip one polygon against a single half-plane, in closed form."""
    survivors: list[np.ndarray] = []
    count = poly.shape[0]
    for index in range(count):
        current = poly[index]
        previous = poly[(index + count - 1) % count]
        current_inside = sign * float(current[axis]) >= bound
        previous_inside = sign * float(previous[axis]) >= bound
        if current_inside != previous_inside:
            survivors.append(_edge_intersection(previous, current, axis, sign, bound))
        if current_inside:
            survivors.append(current)
    if len(survivors) < 3:
        return None
    return np.stack(survivors, axis=0)


def _edge_intersection(
    outside: np.ndarray, inside: np.ndarray, axis: int, sign: int, bound: float
) -> np.ndarray:
    """Point on segment ``outside -> inside`` crossing the half-plane edge."""
    span = sign * float(inside[axis]) - sign * float(outside[axis])
    step = (bound - sign * float(outside[axis])) / span
    return outside + step * (inside - outside)
