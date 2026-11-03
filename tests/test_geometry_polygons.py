# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""geometry.polygon primitives: shoelace area + canvas clipping."""

from __future__ import annotations

import numpy as np
import pytest

from autovision.geometry.polygons import clip_polygon_to_canvas, polygon_area


def test_polygon_area_matches_the_shoelace_value() -> None:
    """Axis-aligned rectangle and triangle give their textbook areas."""
    rectangle = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    triangle = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 4.0]])

    assert polygon_area(rectangle) == pytest.approx(50.0)
    assert polygon_area(triangle) == pytest.approx(12.0)
    assert polygon_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0


def test_fully_inside_polygon_is_unchanged() -> None:
    """A polygon strictly inside the canvas comes back identical."""
    poly = np.array([[2.0, 2.0], [8.0, 2.0], [8.0, 6.0], [2.0, 6.0]])

    clipped = clip_polygon_to_canvas(poly, (10, 10))

    assert clipped is not None
    assert np.allclose(clipped, poly)


def test_partially_outside_polygon_is_cut_at_the_border() -> None:
    """The outside half is replaced by the crossing points on the edge."""
    poly = np.array([[-4.0, 2.0], [6.0, 2.0], [6.0, 6.0], [-4.0, 6.0]])

    clipped = clip_polygon_to_canvas(poly, (10, 10))

    assert clipped is not None
    assert np.allclose(clipped[:, 0].min(), 0.0)
    assert polygon_area(clipped) == pytest.approx(polygon_area(poly) * 0.6)


def test_fully_outside_polygon_degenerates_to_none() -> None:
    """A polygon entirely past the right edge has no intersection."""
    poly = np.array([[12.0, 2.0], [18.0, 2.0], [18.0, 6.0], [12.0, 6.0]])

    assert clip_polygon_to_canvas(poly, (10, 10)) is None


def test_diagonal_overlap_keeps_the_triangle_fan() -> None:
    """A corner-crossing polygon keeps >=3 vertices after clipping."""
    poly = np.array([[8.0, 8.0], [14.0, 8.0], [14.0, 14.0]])

    clipped = clip_polygon_to_canvas(poly, (10, 10))

    assert clipped is not None
    assert len(clipped) >= 3
    assert (clipped >= 0.0).all() and (clipped[:, 0] <= 10.0).all()
    assert (clipped[:, 1] <= 10.0).all()


def test_degenerate_inputs_return_none() -> None:
    """Fewer than three vertices never enter the clip loop."""
    assert clip_polygon_to_canvas(np.array([[0.0, 0.0], [5.0, 5.0]]), (10, 10)) is None
