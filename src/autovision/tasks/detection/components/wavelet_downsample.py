# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Wavelet downsampling for the FSDC-DETR family (design doc 22 §4.2).

Implements FSD-Down (paper Eq.12-13) on top of a Haar DWT realized as a
stride-2 grouped convolution — no ``pywt`` dependency, export-friendly.

- :class:`HaarDWT2d` — orthonormal 2D Haar transform as a fixed or
  learnable (Haar-initialized, G5) depthwise stride-2 conv. Output channel
  order per input channel: ``[LL, LH, HL, HH]``.
- :func:`haar_idwt2` — the matching inverse (test-only reference; perfect
  reconstruction on even sizes).
- :class:`FSDDown` — the paper's downsampling block: an 8C shared channel
  gate (Eq.12) modulates the DWT subband path (4C, all high-frequency
  subbands kept) and a learnable grouped-conv path, combined by Eq.13's
  single output projection with a learnable scalar ``rho`` on the DWT arm.

Component spec source: ``docs/design/22-fsdc-detr-native-dual-branch.md``
§4.2 (E5/G5); shared with the AV-DETR route (doc 21 §4.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FSDDown", "HaarDWT2d", "haar_idwt2"]


def _haar_basis() -> Tensor:
    """The four orthonormal Haar 2x2 analysis filters, ordered [LL, LH, HL, HH]."""
    half = 0.5
    return torch.tensor(
        [
            [[half, half], [half, half]],      # LL: all-passage average
            [[half, half], [-half, -half]],    # LH: horizontal low, vertical high
            [[half, -half], [half, -half]],    # HL: horizontal high, vertical low
            [[half, -half], [-half, half]],    # HH: diagonal detail
        ]
    )


class HaarDWT2d(nn.Module):
    """2D Haar DWT as a stride-2 grouped conv.

    ``freeze_wavelet=False`` (default) keeps the filters as an
    nn.Parameter initialized to the orthonormal Haar basis — the paper's
    "learnable wavelet decomposition" reading (G5); ``True`` registers a
    fixed buffer (the classical HWD-style ablation arm). Either way the
    tensor is named ``filters`` so state dicts round-trip across the flag.
    """

    def __init__(self, channels: int, freeze_wavelet: bool = False) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        self.channels = channels
        filters = _haar_basis().unsqueeze(1).repeat(channels, 1, 1, 1)  # (4C, 1, 2, 2)
        if freeze_wavelet:
            self.register_buffer("filters", filters)
        else:
            self.filters = nn.Parameter(filters)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if height % 2 or width % 2:
            # ceil semantics: zero-pad one row/column so the output is
            # ceil(H/2) x ceil(W/2) — chains with the encoder's stride grid.
            x = F.pad(x, (0, width % 2, 0, height % 2))
        return F.conv2d(x, self.filters, stride=2, groups=self.channels)


def haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Inverse Haar transform of the four subbands (test reference).

    Assumes even input sizes (caller crops the padded row/column that
    :class:`HaarDWT2d` zero-padded for odd sizes). Orthonormal inverse:
    each source pixel is the ±-signed half-sum of the four subbands.
    """
    a = 0.5 * (ll + lh + hl + hh)
    b = 0.5 * (ll + lh - hl - hh)
    c = 0.5 * (ll - lh + hl - hh)
    d = 0.5 * (ll - lh - hl + hh)
    batch, channels, height, width = ll.shape
    block = torch.stack([a, b, c, d], dim=-1)  # (B, C, H, W, 4)
    return (
        block.view(batch, channels, height, width, 2, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, channels, 2 * height, 2 * width)
    )


class FSDDown(nn.Module):
    """Frequency-Spatial Dynamic Downsampling (paper Eq.12-13, exact form).

    Both arms are computed at the ceil-halved resolution; the channel gate
    is a single 8C sigmoid vector (Eq.12) split so the first half modulates
    the DWT subband arm (scaled by the learnable scalar ``rho``) and the
    second half the grouped-conv arm; Eq.13 fuses them with one output
    projection.
    """

    def __init__(self, channels: int, freeze_wavelet: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT2d(channels, freeze_wavelet=freeze_wavelet)
        self.gconv = nn.Conv2d(channels, channels, 2, 2, groups=channels, bias=False)
        self.conv_path = nn.Conv2d(channels, 4 * channels, 1, bias=False)
        self.out_proj = nn.Conv2d(4 * channels, channels, 1)
        self.rho = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if height % 2 or width % 2:
            # Pad BEFORE both arms so the depthwise stride-2 conv (which
            # floors) and the DWT (which ceils) land on the same grid.
            x = F.pad(x, (0, width % 2, 0, height % 2))
        dwt = self.dwt(x)                     # (B, 4C, H/2, W/2)
        conv = self.conv_path(self.gconv(x))  # (B, 4C, H/2, W/2)
        pooled = torch.cat([dwt, conv], dim=1).mean(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(pooled)          # Eq.12: 8C shared gate
        gate_dwt, gate_conv = gate.chunk(2, dim=1)
        return self.out_proj(self.rho * gate_dwt * dwt + gate_conv * conv)  # Eq.13
