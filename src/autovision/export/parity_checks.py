# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Numerical verification helpers shared by the task families' export hooks.

The hooks must answer two different questions, and the graph's decode shape
decides which one is even well-posed:

- Dense outputs (depth maps, segmentation logits) and selection-free
  query outputs (the GTR pose decoder emits every query in a fixed order)
  can be compared element-wise once rows are aligned — strict, and safe
  because no in-graph decode can legitimately reorder rows.
- Graphs that select top-k queries in-graph (D-FINE-style det / OBB
  heads) legitimately swap whole rows when two candidates' scores sit
  within numeric noise: no bijection aligns those outputs element-wise,
  so the honest invariant is the row-sum one. The same applies to fp16
  graphs on degenerate weights, where near-tied scores flip decode
  selections outright (observed ~0.8 element-wise drift on untrained GTR
  pose weights that NO element-wise gate can tolerate).
"""

from __future__ import annotations

import torch

__all__ = [
    "assert_dense_close",
    "assert_outputs_close_sum",
    "assert_query_outputs_close",
]

#: Tolerances shared by every verify hook (fp32-diff observed ceiling is
#: ~5e-5; the rtol term carries fp16-CUDA graphs that still run this path).
_ATOL = 5e-3
_RTOL = 1e-1

_ASSERT_CLOSE = {
    "atol": _ATOL,
    "rtol": _RTOL,
    "equal_nan": True,
    "check_device": False,
    "check_dtype": False,
}


def _to_tensor(output: object) -> torch.Tensor:
    return output.detach().float().cpu() if isinstance(output, torch.Tensor) else torch.as_tensor(output).float()


def assert_dense_close(reference: object, candidate: object) -> None:
    """Element-wise verification for dense outputs (depth maps, per-pixel
    logits) — no decode, so no legitimate reordering can hide behind."""
    torch.testing.assert_close(_to_tensor(reference), _to_tensor(candidate), **_ASSERT_CLOSE)


def assert_outputs_close_sum(
    torch_outputs: "tuple | list",
    onnx_outputs: "tuple | list",
    *, dim: int = 1,
) -> None:
    """Row-sum verification (the historical hook idiom).

    permutation-tolerant and drift-cancelling by construction — the right
    invariant for in-graph top-k decode and for fp16 graphs on degenerate
    weights, too weak for everything else (see module docstring).
    """
    for reference, candidate in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(
            _to_tensor(reference).sum(dim=dim),
            _to_tensor(candidate).sum(dim=dim),
            **_ASSERT_CLOSE,
        )


def _greedy_row_permutation(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Greedy nearest-row assignment (L1) — ``(B, N, ...)`` rows against
    ``(B, N, ...)`` rows, returned as ``(B, N)`` indices into ``candidate``.

    Rows that already agree map to themselves (identity in the common
    fp32 case); a reordered-but-equal row set maps to its true partners.
    NaNs cannot steer the match (the cost is nan-sanitized), so a graph
    that emits garbage still gets a complete assignment — and then fails
    the element-wise compare.
    """
    batch, rows = reference.shape[0], reference.shape[1]
    flat_ref = reference.reshape(batch, rows, -1)
    flat_candidate = candidate.reshape(batch, rows, -1)
    permutation = torch.empty(batch, rows, dtype=torch.long)
    for b in range(batch):
        cost = torch.nan_to_num(
            torch.cdist(flat_ref[b].unsqueeze(0), flat_candidate[b].unsqueeze(0), p=1)[0],
            nan=torch.finfo(torch.float32).max,
        )
        available = list(range(rows))
        for i in range(rows):
            best = min(available, key=lambda j: float(cost[i, j]))
            permutation[b, i] = best
            available.remove(best)
    return permutation


def _gather_rows(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Reorder the ``dim=1`` rows of a ``(B, N, ...)`` tensor by index."""
    index = permutation
    for _ in range(tensor.dim() - 2):
        index = index.unsqueeze(-1)
    # -1 keeps a dim's size, so the trailing dims must be explicit to grow
    # the unsqueezed size-1 axes back to the tensor's shape.
    return tensor.gather(1, index.expand(-1, *tensor.shape[1:]))


def assert_query_outputs_close(
    torch_outputs: "tuple | list",
    onnx_outputs: "tuple | list",
) -> None:
    """Strict verification for selection-free query-structured outputs.

    Rows are matched greedily on the FIRST output (the scores that drive
    any downstream decode), then every output is compared element-wise
    under that permutation: a drifted row fails the gate even when the
    row-sum tolerance would absorb it, while a legitimately reordered row
    set still passes.
    """
    references = [_to_tensor(o) for o in torch_outputs]
    candidates = [_to_tensor(o) for o in onnx_outputs]
    permutation = _greedy_row_permutation(references[0], candidates[0])
    for reference, candidate in zip(references, candidates):
        torch.testing.assert_close(
            reference, _gather_rows(candidate, permutation), **_ASSERT_CLOSE
        )
