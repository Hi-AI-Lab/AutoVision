# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Benchmark timing methodology (design D5): sigma clipping + robust mode."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autovision.export.benchmark import benchmark_inference
from autovision.export.timing import adaptive_run_count, iterative_sigma_clip


class _PassthroughModel(nn.Module):
    """Minimal model exposing the ``forward_backend`` benchmark contract."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward_backend(self, batch: torch.Tensor) -> torch.Tensor:
        return batch


# ── pure primitives ─────────────────────────────────────────────────────


def test_sigma_clip_removes_scheduler_spikes() -> None:
    """One 10x outlier among tight samples is dropped; the rest survive."""
    samples = [10.0, 10.1, 9.9, 10.05, 9.95, 10.0, 100.0]

    kept = iterative_sigma_clip(samples)

    assert 100.0 not in kept
    assert len(kept) == len(samples) - 1
    assert kept.mean() == pytest.approx(10.0, abs=0.1)


def test_sigma_clip_is_stable_on_clean_samples() -> None:
    """Clean input survives unchanged (no over-clipping)."""
    samples = [10.0 + 0.01 * i for i in range(30)]

    kept = iterative_sigma_clip(samples)

    assert len(kept) == len(samples)


def test_sigma_clip_never_fewer_than_two_samples() -> None:
    """Pathological inputs degrade to the raw pair, not a crash."""
    assert len(iterative_sigma_clip([1.0, 100.0])) == 2
    assert len(iterative_sigma_clip([42.0])) == 1
    assert len(iterative_sigma_clip([])) == 0


def test_adaptive_run_count_hits_the_target_window() -> None:
    """The step count follows the probe estimate within the clamps."""
    assert adaptive_run_count(0.01, target_seconds=10.0) == 500  # clamped high
    assert adaptive_run_count(1.0, target_seconds=10.0) == 20  # clamped low
    assert adaptive_run_count(0.05, target_seconds=10.0) == 200
    assert adaptive_run_count(0.0) == 20  # degenerate probe -> floor


# ── benchmark wiring ────────────────────────────────────────────────────


def test_robust_mode_reports_clipped_statistics() -> None:
    """robust=True fills the extra block; headline fields stay compatible."""
    model = _PassthroughModel()
    result = benchmark_inference(
        model,
        {"images": torch.zeros(2, 3, 8, 8)},
        warmup_steps=1,
        steps=5,
        robust=True,
        min_measure_seconds=0.01,
        forward=model.forward_backend,
    )

    assert "latency_ms_mean_robust" in result.extra
    assert "latency_ms_std" in result.extra
    assert "outliers_dropped" in result.extra
    assert result.extra["latency_ms_mean_robust"] > 0.0
    # The adaptive window replaced the fixed step count (bounded, non-trivial).
    assert result.num_steps >= 20
    assert result.extra["backend"] == "torch"


def test_default_mode_keeps_fixed_step_semantics() -> None:
    """robust=False (the default) reproduces the pre-D5 behavior exactly."""
    model = _PassthroughModel()
    result = benchmark_inference(
        model,
        {"images": torch.zeros(1, 3, 8, 8)},
        warmup_steps=1,
        steps=7,
        forward=model.forward_backend,
    )

    assert result.num_steps == 7
    assert "latency_ms_mean_robust" not in result.extra
