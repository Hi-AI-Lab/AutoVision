# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the SIGReg regularizer (design doc 16, LeVJEPA R1-A1).

Covers the Epps–Pulley statistic's distribution sensitivity, the raw-output
tap on the projection head, and the config-surface contracts: SIGReg
requires an active mode, the two_crop view, and the matched-pool instance
source.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from autovision._task_models.ltdetr_object_detection.config import LTDETRSSLArgs
from autovision._task_models.object_detection_components.ssl_contrastive import (
    SSLProjectionHead,
    epps_pulley_sigreg,
)


def _gaussian(n: int = 256, k: int = 16) -> Tensor:
    return torch.randn(n, k, generator=torch.Generator().manual_seed(0))


class TestEppsPulleySigreg:
    """Distribution sensitivity of the sketched Epps–Pulley statistic."""

    def test_standard_gaussian_is_near_zero(self):
        value = epps_pulley_sigreg(_gaussian(), num_slices=256)
        assert value.item() < 0.05

    def test_collapsed_embeddings_score_high_but_finite(self):
        collapsed = torch.ones(64, 16) * 3.0
        value = epps_pulley_sigreg(collapsed, num_slices=64)
        assert 0.1 < value.item() < 100.0
        assert torch.isfinite(value)

    def test_shifted_gaussian_scores_above_standard(self):
        gen = torch.Generator().manual_seed(1)
        standard = torch.randn(128, 16, generator=gen)
        shifted = standard + 5.0
        v_standard = epps_pulley_sigreg(standard, num_slices=256, generator=gen)
        v_shifted = epps_pulley_sigreg(shifted, num_slices=256, generator=gen)
        assert v_shifted > v_standard + 0.1

    def test_generator_reproducibility(self):
        samples = _gaussian()
        a = epps_pulley_sigreg(
            samples, num_slices=128, generator=torch.Generator().manual_seed(5)
        )
        b = epps_pulley_sigreg(
            samples, num_slices=128, generator=torch.Generator().manual_seed(5)
        )
        assert torch.equal(a, b)

    def test_degenerate_pool_returns_connected_zero(self):
        samples = torch.randn(1, 8, requires_grad=True)
        value = epps_pulley_sigreg(samples, num_slices=64)
        assert value.item() == 0.0
        value.backward()
        assert samples.grad is not None and bool(torch.isfinite(samples.grad).all())

    def test_gradient_flows_and_is_finite(self):
        samples = (
            torch.randn(64, 16, generator=torch.Generator().manual_seed(2)) * 2 + 1
        ).requires_grad_(True)
        value = epps_pulley_sigreg(samples, num_slices=128)
        value.backward()
        assert torch.isfinite(samples.grad).all()
        assert samples.grad.abs().max() > 0

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            epps_pulley_sigreg(torch.randn(2, 3, 8))


class TestForwardWithRaw:
    """The raw tap: SIGReg regularizes pre-normalization projector outputs."""

    def test_returns_normalized_and_raw(self):
        head = SSLProjectionHead(hidden_dim=32, projection_dim=16, dropout=0.0)
        features = torch.randn(8, 32)
        z, s = head.forward_with_raw(features)
        assert z.shape == (8, 16) and s.shape == (8, 16)
        torch.testing.assert_close(z, torch.nn.functional.normalize(s, dim=-1))
        assert torch.allclose(z.pow(2).sum(-1), torch.ones(8), atol=1e-5)

    def test_forward_matches_forward_with_raw_z(self):
        head = SSLProjectionHead(hidden_dim=32, projection_dim=16, dropout=0.0)
        head.eval()
        features = torch.randn(4, 32)
        z_direct = head(features)
        z_pair, _ = head.forward_with_raw(features)
        torch.testing.assert_close(z_direct, z_pair)

    def test_sigreg_on_raw_not_normalized(self):
        """Unit-norm rows are far from N(0, I) — the raw space is required."""
        head = SSLProjectionHead(hidden_dim=32, projection_dim=16, dropout=0.0)
        _, s = head.forward_with_raw(torch.randn(64, 32))
        v_raw = epps_pulley_sigreg(s, num_slices=128)
        v_norm = epps_pulley_sigreg(
            torch.nn.functional.normalize(s, dim=-1), num_slices=128
        )
        assert v_raw < v_norm  # normalized rows look shell-distributed, not Gaussian


class TestConfigSurface:
    """SIGReg config contracts (design doc 16 §4.2)."""

    def test_defaults_are_off(self):
        args = LTDETRSSLArgs()
        assert args.sigreg_coef == 0.0
        assert args.to_ssl_params().sigreg_coef == 0.0

    def test_params_mapping(self):
        args = LTDETRSSLArgs(sigreg_coef=0.3, sigreg_slices=512, sigreg_knots=21)
        params = args.to_ssl_params()
        assert params.sigreg_coef == 0.3
        assert params.sigreg_slices == 512
        assert params.sigreg_knots == 21

    def test_sigreg_requires_active_mode(self):
        with pytest.raises(ValueError, match="active mode"):
            LTDETRSSLArgs(mode="off", sigreg_coef=0.3)

    def test_sigreg_requires_two_crop(self):
        with pytest.raises(ValueError, match="two_crop"):
            LTDETRSSLArgs(view="dropout", sigreg_coef=0.3)

    def test_sigreg_requires_matched_gt_pool(self):
        # encoder_candidates forces view="dropout" first, so pass it to
        # reach the SIGReg-specific constraint.
        with pytest.raises(ValueError, match="matched_gt"):
            LTDETRSSLArgs(
                instance_source="encoder_candidates",
                view="dropout",
                sigreg_coef=0.3,
            )
