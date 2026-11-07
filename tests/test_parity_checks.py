# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared export-parity helpers: row matching, sum invariants, strict gates."""

from __future__ import annotations

import pytest
import torch

from autovision.export.parity_checks import (
    assert_dense_close,
    assert_outputs_close_sum,
    assert_query_outputs_close,
)


def _query_outputs(batch: int = 2, rows: int = 8, width: int = 4, seed: int = 0):
    """Distinct well-separated rows so the greedy match is unambiguous."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, rows, 2, generator=generator) * 3
    boxes = torch.randn(batch, rows, width, generator=generator)
    return (logits, boxes)


def test_identical_outputs_pass():
    """The common fp32 case: identical outputs compare clean."""
    reference = _query_outputs()
    assert_query_outputs_close(reference, reference)


def test_permuted_row_set_passes():
    """A reordered-but-equal row set (the legitimate top-k tie flip) must
    still pass: the greedy match realigns it element-wise."""
    reference = _query_outputs()
    order = torch.tensor([[7, 0, 3, 1, 2, 4, 5, 6], [5, 4, 3, 2, 1, 0, 6, 7]])
    candidate = (
        reference[0].gather(1, order.unsqueeze(-1).expand(-1, -1, 2)),
        reference[1].gather(1, order.unsqueeze(-1).expand(-1, -1, 4)),
    )
    assert_query_outputs_close(reference, candidate)


def test_row_local_drift_fails_where_row_sum_would_pass():
    """The regression the strict gate exists for: a ±0.5 pair inside one row
    cancels exactly in the row sum (the historical hook passes) while each
    element sits far outside the tolerance band."""
    reference = _query_outputs()
    drifted = list(reference)
    drifted[0] = drifted[0].clone()
    # +0.5 / -0.5 across two rows, same class: the class channel's row-sum
    # is unchanged (the historical hook passes) while both rows are far
    # outside the element-wise band.
    drifted[0][:, 0, 0] += 0.5
    drifted[0][:, 1, 0] -= 0.5
    assert_outputs_close_sum(reference, drifted)  # the old gate passes
    with pytest.raises(AssertionError):
        assert_query_outputs_close(reference, tuple(drifted))


def test_small_numeric_noise_passes():
    """fp32 backend noise (~1e-5 observed ceiling) sits far inside the
    tolerance band."""
    reference = _query_outputs()
    noisy = tuple(t + torch.randn_like(t) * 1e-5 for t in reference)
    assert_query_outputs_close(reference, noisy)


def test_nan_in_candidate_fails():
    """equal_nan excuses NaN only on both sides: a graph emitting NaN where
    the reference is finite must fail."""
    reference = _query_outputs()
    nan_candidate = list(reference)
    nan_candidate[0] = nan_candidate[0].clone()
    nan_candidate[0][:, 2, 0] = float("nan")
    with pytest.raises(AssertionError):
        assert_query_outputs_close(reference, tuple(nan_candidate))


def test_duplicate_rows_are_ambiguous_but_consistent():
    """Identical rows make the assignment ambiguous — any assignment yields
    the same element-wise values, so the gate stays green."""
    reference = _query_outputs()
    reference[1][:, 5] = reference[1][:, 4]  # duplicate a row
    candidate = list(reference)
    # swap the duplicated rows in the candidate: value-identical
    candidate[1] = candidate[1].clone()
    candidate[1][:, [4, 5]] = candidate[1][:, [5, 4]]
    assert_query_outputs_close(reference, tuple(candidate))


def test_sum_helper_preserves_legacy_semantics():
    """The shared sum helper matches the historical per-hook idiom exactly
    (same dim, tolerances) so switching hooks onto it is a no-op refactor."""
    reference = _query_outputs()
    assert_outputs_close_sum(reference, reference)
    with pytest.raises(AssertionError):
        assert_outputs_close_sum(reference, tuple(t + 1.0 for t in reference))


def test_dense_close_rejects_local_drift():
    """Dense maps (depth / semseg) compare element-wise: a ±0.5 pair that
    cancels in a spatial sum still fails the strict compare."""
    reference = torch.rand(2, 8, 8)
    drifted = reference.clone()
    # +0.5 / -0.5 in the same width bucket, different rows: cancels in the
    # dim-1 sum, far outside the element-wise band.
    drifted[:, 0, 0] += 0.5
    drifted[:, 1, 0] -= 0.5
    assert_outputs_close_sum((reference,), (drifted,))  # old gate passes
    with pytest.raises(AssertionError):
        assert_dense_close(reference, drifted)
    assert_dense_close(reference, reference)
