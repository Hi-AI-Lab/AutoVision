# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Spatial-Frequency Shunt Feature Fusion (FSDC-DETR paper Eq.10-11).

Design doc 22 §4.3 (spec source) / doc 21 §4.2. The block replaces the
PAN fusion pair in the FSC hybrid encoder: the fine-scale input is
aligned IN-BLOCK by an embedded :class:`~...wavelet_downsample.FSDDown`
(Eq.10 — the Down is part of the block definition, not an external
adapter), the concatenated features are shunted {3C/8, C/2, C/8}, and the
cross assignment is preserved: the shunt path's 3C/8 arm goes through the
spatial refiner while the C/2 arm goes through the frequency-domain
global filter (paper Fig.4 reading; first re-check item once official
code is released, G6).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autovision.tasks.detection.fsdc.wavelet_downsample import (
    FSDDown,
)
from autovision.tasks.detection.shared.hybrid_encoder import (
    ConvNormLayer,
)

__all__ = ["FrequencyGlobalBranch", "SFSFF", "SpatialRefineModule"]


class SpatialRefineModule(nn.Module):
    """SRM minimal port (G4): DW3×3 → BN → GELU → PW1×1 → BN.

    The DEIMv2/RT-DETR upstream referenced by the paper does not ship an
    SRM in the transformers conversion; this is the smallest module
    matching the paper's description until the official FSDC-DETR code is
    released (first alignment item, G4).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm2(self.pw(self.act(self.norm1(self.dw(x)))))


class FrequencyGlobalBranch(nn.Module):
    """rFFT2 → Conv3×3 on the (real‖imag) representation → irFFT2.

    The conv runs inside an fp32 autocast island (the repo engine wraps
    training steps in bf16 autocast; torch.fft output feeding a conv would
    otherwise be cast to bf16 mid-chain — F7). ``norm="ortho"`` keeps the
    forward/inverse energy-normalized so the conv operates on a scale
    comparable to the spatial branch.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(2 * channels, 2 * channels, 3, 1, 1)
        # Deploy fallback switch (doc 22 §4.7): set by the owning encoder's
        # convert_to_deploy when fsdc.fft_export_fallback is requested —
        # the branch itself never self-disables (deploy() recursion calls
        # convert_to_deploy unconditionally).
        self.export_identity = False

    def forward(self, x: Tensor) -> Tensor:
        if self.export_identity:
            return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            packed = torch.cat([spectrum.real, spectrum.imag], dim=1)
            packed = self.conv(packed)
            half = spectrum.real.shape[1]
            spectrum = torch.complex(packed[:, :half], packed[:, half:])
            out = torch.fft.irfft2(spectrum, s=x.shape[-2:], norm="ortho")
        return out.to(x.dtype)


class SFSFF(nn.Module):
    """Spatial-Frequency Shunt Feature Fusion (Eq.10-11, exact form).

    Signature: ``forward(f_in, f_hat_in)`` with ``f_in`` at the coarse
    (H, W) grid and ``f_hat_in`` one level finer at (2H, 2W) — the PAN
    junction contract. Channel split {3C/8 → SRM, C/2 → FFT, C/8 →
    identity} requires ``channels % 8 == 0``.
    """

    def __init__(self, channels: int, freeze_wavelet: bool = False) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError(
                f"SFSFF requires channels % 8 == 0 (split 3C/8, C/2, C/8); "
                f"got {channels}."
            )
        self.channels = channels
        self.down = FSDDown(channels, freeze_wavelet=freeze_wavelet)
        self.proj_in = ConvNormLayer(2 * channels, channels, 1, 1)
        self.fft = FrequencyGlobalBranch(channels // 2)
        self.srm = SpatialRefineModule(3 * channels // 8)
        self.proj_out = ConvNormLayer(channels, channels, 1, 1)

    def forward(self, f_in: Tensor, f_hat_in: Tensor) -> Tensor:
        x = self.proj_in(torch.cat([f_in, self.down(f_hat_in)], dim=1))  # Eq.10
        x_f, x_s, x_id = x.split(
            [3 * self.channels // 8, self.channels // 2, self.channels // 8], dim=1
        )
        return self.proj_out(torch.cat([self.srm(x_f), self.fft(x_s), x_id], dim=1))  # Eq.11
