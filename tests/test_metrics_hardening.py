# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Metrics hardening regressions (H3-6c).

Anchors the map evaluator's batch-3 hardening items: the ``auto_raise``
per-call cap lift must not mutate evaluator state across ``compute()``
rounds, the returned metric dict stays float-only (the engine rounds every
``val_metric/*`` entry), and the headline protocol is declared on the
output side (``HEADLINE_PROTOCOL``) instead of smuggled into the dict.
"""

from __future__ import annotations

import logging

import pytest
import torch

from autovision.metrics.map_evaluator import HEADLINE_PROTOCOL, COCOMAPEvaluator

CATEGORIES = {2: "zebra", 1: "ant"}  # deliberately not in sorted-name order


def _dense_evaluator(**kwargs) -> COCOMAPEvaluator:
    """An evaluator whose densest image (150 instances) exceeds maxDets=100.

    Targets are normalized cxcywh on a 400x400 frame: alternating 20x20 px
    boxes at x-offsets 10 / 30. The two predictions repeat the first two GT
    boxes verbatim in absolute xyxy, so the evaluation runs the real
    pycocotools path with a genuinely triggered auto-raise and perfect
    per-class detections.
    """
    evaluator = COCOMAPEvaluator(CATEGORIES, max_dets=100, auto_raise=True, **kwargs)
    boxes = torch.zeros(150, 4)
    boxes[::2] = torch.tensor([0.05, 0.05, 0.05, 0.05])  # xyxy (10, 10, 30, 30)
    boxes[1::2] = torch.tensor([0.10, 0.05, 0.05, 0.05])  # xyxy (30, 10, 50, 30)
    labels = torch.zeros(150, dtype=torch.long)
    labels[::2] = 0  # internal 0 -> category 1 (ant)
    labels[1::2] = 1  # internal 1 -> category 2 (zebra)
    evaluator.update_targets([{"boxes": boxes, "labels": labels}], [(400, 400)])
    evaluator.update(
        [
            {
                "bboxes": [[10.0, 10.0, 30.0, 30.0], [30.0, 10.0, 50.0, 30.0]],
                "scores": [0.9, 0.8],
                "labels": [0, 1],
            }
        ]
    )
    return evaluator


class TestAutoRaiseNoStateLeak:
    """H3-6c: the cap lift is a per-call adjustment, not a state mutation."""

    def test_two_computes_are_identical_and_max_dets_restored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dense data + auto_raise: two compute() rounds agree bitwise and
        ``max_dets`` still reads the caller's value afterwards (H3-6c)."""
        evaluator = _dense_evaluator()
        with caplog.at_level(logging.WARNING):
            first = evaluator.compute()
            second = evaluator.compute()
        assert evaluator.max_dets == 100
        assert first == second, "repeated compute() readings must be comparable"
        assert "raised to 150" in caplog.text
        assert caplog.text.count("raised to 150") == 2  # each round warns once

    def test_auto_raise_does_not_touch_explicit_protocol_cap(self) -> None:
        """Without a trigger the cap value itself is never rewritten (H3-6c)."""
        evaluator = COCOMAPEvaluator(CATEGORIES, max_dets=500, auto_raise=True)
        evaluator.update_targets(
            [{"boxes": torch.zeros(3, 4), "labels": torch.zeros(3, dtype=torch.long)}],
            [(100, 100)],
        )
        evaluator.compute()
        assert evaluator.max_dets == 500


class TestComputeContract:
    """H3-6c: the metric dict stays float-only; the protocol is declared
    beside the numbers, not inside them."""

    def test_compute_values_are_all_floats(self) -> None:
        """Every compute() value is a float — the engine's val logging
        rounds each ``val_metric/*`` entry, so a str value would crash
        training; ``headline_protocol`` must not become a dict key until
        that consumer normalizes non-floats (H3-6c)."""
        metrics = _dense_evaluator().compute()
        assert metrics, "the dense fixture must produce a full evaluation"
        assert all(isinstance(value, float) for value in metrics.values())

    def test_headline_protocol_declares_argmax_f1_reading(self) -> None:
        """The output-side declaration names the single operating point
        (METRIC-04 design A) and disclaims the COCO summary protocol."""
        lowered = HEADLINE_PROTOCOL.lower()
        assert "one operating point" in lowered
        assert "headline_score_threshold" in lowered
        assert "ap/ar remain the coco protocol" in lowered
