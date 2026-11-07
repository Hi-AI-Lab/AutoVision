# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Latency-sample statistics for reproducible benchmarks (design D5).

Two pure primitives behind ``benchmark_inference(robust=True)``:

- :func:`iterative_sigma_clip` discards host-scheduler/thermal outliers
  (iterative 2-sigma), so a reported mean describes the MODEL, not the
  machine's worst millisecond;
- :func:`adaptive_run_count` sizes the measurement loop from a probe run
  so every benchmark spends about the same measurement window regardless
  of per-step latency — fixed step counts over-measure fast models and
  under-measure slow ones.

Reported numbers must be reproducible; these two functions are the whole
methodology, so they stay dependency-free and trivially testable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adaptive_run_count", "iterative_sigma_clip"]


def iterative_sigma_clip(
    samples: list[float] | np.ndarray,
    *,
    sigma: float = 2.0,
    max_rounds: int = 5,
) -> np.ndarray:
    """Drop outliers beyond ``sigma`` standard deviations, iteratively.

    Each round recomputes mean/std over the SURVIVORS and re-flags; the
    loop stops when no sample is flagged, ``max_rounds`` is exhausted, or
    the std collapses (fewer than 2 distinct samples). Returns the kept
    samples in input order — never fewer than 2 samples.
    """
    kept = np.asarray(samples, dtype=np.float64).reshape(-1)
    if kept.size <= 2:
        return kept
    for _ in range(max_rounds):
        mean = kept.mean()
        std = kept.std()
        if std <= 0.0:
            break
        mask = np.abs(kept - mean) <= sigma * std
        if mask.all():
            break
        candidate = kept[mask]
        if candidate.size < 2:
            break
        kept = candidate
    return kept


def adaptive_run_count(
    single_run_seconds: float,
    *,
    target_seconds: float = 10.0,
    floor: int = 20,
    ceiling: int = 500,
) -> int:
    """Steps needed so the timed window spans about ``target_seconds``.

    ``single_run_seconds`` comes from a small probe run (3 steps). The
    result is clamped to ``[floor, ceiling]``: the floor keeps statistics
    meaningful for very fast steps, the ceiling bounds the worst-case
    wall time for very slow ones.
    """
    if single_run_seconds <= 0:
        return floor
    estimate = int(target_seconds / single_run_seconds)
    return max(floor, min(ceiling, estimate))
