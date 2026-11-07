# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""box_candidates degradation gate (design D3) and wiring defaults."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from autovision.augment.detection.args import RTDETRDetectionTransformArgs
from autovision.augment.detection.box_candidates import box_candidates


def test_unchanged_boxes_survive() -> None:
    """A no-op transform keeps every legitimate target."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 40.0], [100.0, 20.0, 180.0, 60.0]])

    assert bool(box_candidates(boxes, boxes.clone()).all())


def test_shrunk_sliver_is_rejected() -> None:
    """A rotation/crop that collapses a box past the area floor drops it."""
    before = torch.tensor([[10.0, 10.0, 60.0, 60.0]])  # 2500 px²
    after = torch.tensor([[10.0, 10.0, 12.0, 30.0]])  # 40 px², ratio 0.016

    assert not bool(box_candidates(before, after).any())


def test_moderate_scale_down_survives() -> None:
    """The uniform 0.5x mosaic scale is not 'degeneration' (ratio 0.25)."""
    before = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    after = torch.tensor([[0.0, 0.0, 50.0, 50.0]])

    assert bool(box_candidates(before, after).all())


def test_min_wh_floor_drops_subpixel_strips() -> None:
    """A tall-but-2px-wide strip fails the per-side floor, not the area."""
    before = torch.tensor([[10.0, 10.0, 60.0, 60.0]])
    after = torch.tensor([[10.0, 10.0, 11.9, 210.0]])  # ~400 px², w < 2

    assert not bool(box_candidates(before, after).any())


def test_aspect_ratio_ceiling_checks_both_sides() -> None:
    """A pre-transform sliver cannot pass by staying a sliver."""
    sliver = torch.tensor([[0.0, 0.0, 500.0, 4.0]])  # 125:1 already
    normal = torch.tensor([[0.0, 0.0, 30.0, 30.0]])

    assert not bool(box_candidates(sliver, sliver.clone()).any())
    assert bool(box_candidates(normal, normal.clone()).any())


def test_empty_input_yields_empty_mask() -> None:
    """Zero-target batches pass through without shape errors."""
    empty = torch.zeros((0, 4))

    assert box_candidates(empty, empty.clone()).shape == (0,)


def test_recipe_defaults_match_the_design() -> None:
    """The recipe exposes the gate with the documented lenient defaults."""
    args = RTDETRDetectionTransformArgs()

    assert args.box_min_wh_px == 2.0
    assert args.box_max_aspect_ratio == 100.0
    assert args.box_min_area_ratio == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"box_min_wh_px": -1.0},
        {"box_max_aspect_ratio": 0.5},
        {"box_min_area_ratio": 1.5},
    ],
)
def test_recipe_thresholds_are_validated(kwargs: dict[str, float]) -> None:
    """Pydantic guards keep the recipe knobs in meaningful ranges."""
    with pytest.raises(ValidationError):
        RTDETRDetectionTransformArgs(**kwargs)  # type: ignore[arg-type]
