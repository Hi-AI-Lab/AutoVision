# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for D23c (model inventory/GFLOPs) and D23d (Conv+BN fusion)."""

from __future__ import annotations

import pytest
import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, ReLU, Sequential

from autovision.errors import AutoVisionError
from autovision.runtime.fuse import (
    assert_fused_parity,
    fuse_conv_bn_model,
    fuse_conv_bn_pair,
)
from autovision.runtime.model_info import (
    estimate_gflops,
    format_param_count,
    model_info,
)


class _Block(Module):
    """conv -> bn -> relu -> conv -> bn: one fusible run in eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(3, 8, 3, padding=1),
            BatchNorm2d(8),
            ReLU(),
            Conv2d(8, 4, 3, padding=1),
            BatchNorm2d(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestModelInfo:
    """Parameter accounting and GFLOPs estimation (D23c)."""

    def test_counts_trainable_and_frozen(self) -> None:
        """Frozen parameters count in total but not in trainable."""
        model = Sequential(Linear(4, 8), Linear(8, 2))
        model[1].weight.requires_grad_(False)
        info = model_info(model)
        total = sum(p.numel() for p in model.parameters())
        assert info["params_total"] == total
        assert info["params_trainable"] == total - 8 * 2
        assert info["params_frozen"] == 16
        assert info["modules"] == 2  # two Linear leaves

    def test_gflops_none_without_example_or_spec(self) -> None:
        """No probe input and no model_input_spec: report n/a, not a guess."""
        assert estimate_gflops(Linear(4, 2)) is None

    def test_gflops_with_explicit_example(self) -> None:
        """A real profile returns a positive float; failures stay None."""
        pytest.importorskip("thop")
        result = estimate_gflops(Linear(4, 2), example=torch.randn(1, 4))
        assert result is not None and result > 0.0

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            pytest.param(21_300_000, "21.3M", id="millions"),
            pytest.param(1_400_000_000, "1.40B", id="billions"),
            pytest.param(980, "980", id="small"),
        ],
    )
    def test_format_param_count(self, count: int, expected: str) -> None:
        """Banner formatting covers the K/M/B ranges."""
        assert format_param_count(count) == expected


class TestFuseConvBnPair:
    """Single-pair folding math and rejection of mismatched pairs."""

    def test_pair_fold_is_numerically_equivalent(self) -> None:
        """Folded conv reproduces bn(conv(x)) in eval mode within fp32 noise."""
        conv = Conv2d(3, 8, 3, padding=1)
        norm = BatchNorm2d(8)
        norm.eval()
        conv.eval()
        example = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            reference = norm(conv(example))
        fused = fuse_conv_bn_pair(conv, norm)
        fused.eval()
        with torch.no_grad():
            candidate = fused(example)
        torch.testing.assert_close(reference, candidate, rtol=1e-4, atol=1e-5)

    def test_mismatched_ranks_rejected(self) -> None:
        """A 2-d conv paired with a 1-d batch norm is a readable error."""
        from torch.nn import BatchNorm1d, Conv2d

        with pytest.raises(ValueError, match="same-rank"):
            fuse_conv_bn_pair(Conv2d(3, 4, 1), BatchNorm1d(4))

    def test_bias_absorbed_in_fold(self) -> None:
        """A conv with its own bias folds correctly (bias math included)."""
        conv = Conv2d(2, 4, 3, padding=1, bias=True)
        norm = BatchNorm2d(4)
        conv.eval()
        norm.eval()
        example = torch.randn(1, 2, 6, 6)
        with torch.no_grad():
            reference = norm(conv(example))
        fused = fuse_conv_bn_pair(conv, norm).eval()
        with torch.no_grad():
            torch.testing.assert_close(fused(example), reference, rtol=1e-4, atol=1e-5)


class TestFuseConvBnModel:
    """Whole-model Sequential rewriting with a parity gate."""

    def test_blocks_fold_and_parity_holds(self) -> None:
        """Both conv+bn pairs fold; outputs track the unfused model."""
        model = _Block()
        model.eval()
        example = torch.randn(1, 3, 8, 8)
        fused = fuse_conv_bn_model(model)
        convs = [
            m for m in fused.modules() if isinstance(m, Conv2d)
        ]
        norms = [
            m for m in fused.modules() if isinstance(m, BatchNorm2d)
        ]
        assert len(convs) == 2
        assert len(norms) == 0
        assert_fused_parity(model, fused, example)  # raises on drift

    def test_original_model_untouched(self) -> None:
        """Fusion deep-copies: the input keeps its batch norms."""
        model = _Block().eval()
        _ = fuse_conv_bn_model(model)
        assert len([m for m in model.modules() if isinstance(m, BatchNorm2d)]) == 2

    def test_parity_gate_rejects_drift(self) -> None:
        """A diverged "fused" model fails the parity assertion loudly."""
        model = _Block().eval()
        fused = fuse_conv_bn_model(model)
        # Corrupt the copy's first conv so outputs diverge.
        first_conv = next(m for m in fused.modules() if isinstance(m, Conv2d))
        first_conv.weight.data.add_(1.0)
        with pytest.raises(AutoVisionError, match="drifted"):
            assert_fused_parity(model, fused, torch.randn(1, 3, 8, 8))
