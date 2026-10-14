# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DBFSAF — Dual-Branch Frequency-Spatial Adaptive Fusion (FSDC-DETR Eq.1-6).

The FSDC-DETR paper's replacement for DEIMv2's blind concat+1×1 dual-branch
fusion. Input contract: a single tensor that is the channel concatenation
of the ViT semantic tap and the CNN (STA) detail feature at one pyramid
level — the block does not take two separate inputs (design doc 22 §3.3:
"keep the concat-then-fuse wiring, swap only the fusion operator").

Components (spec source: design doc 22 §4.1; shared with doc 21 §4.5):

- :class:`MKSRU` — Eq.7-9: DW5×5, channel split {1/2C → DW7×7 dilation 3,
  3/8C → DW7×7 dilation 2, 1/8C → identity} (Eq.8), then the multiplicative
  SiLU gate ``SiLU(Conv1×1(X̂)) ⊙ SiLU(Conv1×1(Conv3×3(X)))`` (Eq.9; no
  additive residual).
- :class:`FPU` — frequency processing unit (SFS-Conv): fp32 rFFT2 → 1×1 →
  irFFT2 with a global channel gate; fp32 island (F7).
- :class:`DBFSAF` — Eq.1-6: FDConv projection, Partial-γ channel split,
  two per-branch 1×1 copies, MKSRU/FPU branches, the 2C-channel softmax
  gate with learnable scalars α/β (Eq.4-5), and the bypass + out projection
  (Eq.6).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from autovision.tasks.detection.components.fdconv import FDConv
from autovision.tasks.detection.components.hybrid_encoder import (
    ConvNormLayer,
)

__all__ = ["DBFSAF", "FPU", "MKSRU"]


class MKSRU(nn.Module):
    """Multi-Kernel Spatial Refinement Unit (Eq.7-9). Requires C % 8 == 0."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError(
                f"MKSRU requires channels % 8 == 0 (split C/2, 3C/8, C/8); "
                f"got {channels}."
            )
        self.channels = channels
        self.dw5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        # Eq.8 mapping: half → dilation 3, 3/8 → dilation 2, 1/8 → identity.
        # same-padding = dilation * (kernel - 1) / 2.
        self.dw7_d3 = nn.Conv2d(
            channels // 2, channels // 2, 7, 1, 9, dilation=3, groups=channels // 2
        )
        self.dw7_d2 = nn.Conv2d(
            3 * channels // 8,
            3 * channels // 8,
            7,
            1,
            6,
            dilation=2,
            groups=3 * channels // 8,
        )
        self.gate = nn.Conv2d(channels, channels, 1)
        self.short_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gate_s = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        c = self.channels
        dilated = self.dw5(x)  # Eq.7
        a, b, identity = torch.split(dilated, [c // 2, 3 * c // 8, c // 8], dim=1)
        x_hat = torch.cat([self.dw7_d3(a), self.dw7_d2(b), identity], dim=1)  # Eq.8
        return F.silu(self.gate(x_hat)) * F.silu(
            self.gate_s(self.short_conv(x))
        )  # Eq.9


class FPU(nn.Module):
    """Frequency processing unit: fp32 rFFT2 → 1×1 → irFFT2 + global gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.freq_proj = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        # Deploy fallback switch (doc 22 §4.7) — set by DBFSAF's
        # convert_to_deploy, never self-disabled.
        self.export_identity = False

    def forward(self, x: Tensor) -> Tensor:
        weight = torch.sigmoid(self.gate(x.mean(dim=(2, 3), keepdim=True)))
        if self.export_identity:
            return x * weight
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            packed = torch.cat([spectrum.real, spectrum.imag], dim=1)
            packed = self.freq_proj(packed)
            half = spectrum.real.shape[1]
            spectrum = torch.complex(packed[:, :half], packed[:, half:])
            out = torch.fft.irfft2(spectrum, s=x.shape[-2:], norm="ortho")
        return out.to(x.dtype) * weight


class DBFSAF(nn.Module):
    """Dual-Branch Frequency-Spatial Adaptive Fusion (Eq.1-6).

    ``out_channels % 8 == 0`` and ``gamma`` such that
    ``out_channels * gamma % 8 == 0`` (the MKSRU split needs the refined
    arm's width divisible by 8); γ = 0.5 is the paper-ablation optimum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: float = 0.5,
        use_ksm: bool = False,
        fdconv_lite: bool = False,
    ) -> None:
        super().__init__()
        if out_channels % 8:
            raise ValueError(
                f"DBFSAF requires out_channels % 8 == 0, got {out_channels}."
            )
        c_fs = int(out_channels * gamma)
        if c_fs % 8:
            raise ValueError(
                f"DBFSAF gamma {gamma} gives a refined-arm width {c_fs} that is "
                f"not divisible by 8 (MKSRU split constraint)."
            )
        self.c_fs = c_fs
        self.c_c = out_channels - c_fs
        self.fdconv = FDConv(
            in_channels, out_channels, use_ksm=use_ksm, lite=fdconv_lite
        )
        # Eq.2-3: two per-branch 1×1 copies of the refined arm ("two copies").
        self.fs_proj_s = nn.Conv2d(c_fs, c_fs, 1)
        self.fs_proj_f = nn.Conv2d(c_fs, c_fs, 1)
        self.mksru = MKSRU(c_fs)
        self.fpu = FPU(c_fs)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.c_branch = ConvNormLayer(self.c_c, self.c_c, 3, 1)
        self.out_proj = ConvNormLayer(out_channels, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        f_init = self.fdconv(x)  # Eq.1
        f_fs, f_c = torch.split(f_init, [self.c_fs, self.c_c], dim=1)
        f_s = self.mksru(self.fs_proj_s(f_fs))  # Eq.2
        f_f = self.fpu(self.fs_proj_f(f_fs))  # Eq.3
        # Eq.4: softmax over the concatenated 2·C_fs channel dimension.
        pooled = torch.cat([f_s, f_f], dim=1).mean(dim=(2, 3), keepdim=True)
        gate = torch.softmax(pooled, dim=1)
        f_hat_s = self.alpha * gate[:, : self.c_fs] * f_s
        f_hat_f = self.beta * gate[:, self.c_fs :] * f_f  # Eq.5
        return self.out_proj(
            torch.cat([f_hat_s + f_hat_f, self.c_branch(f_c)], dim=1)
        )  # Eq.6
