# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""FSC hybrid encoder: AIFI/FPN faithful + PAN → SFS-FF (doc 22 §2.3/§4.3).

Replaces the HF ``Deimv2HybridEncoder`` inside a fully-built
``Deimv2Model``. The AIFI layer and the top-down FPN are line-faithful
ports of the HF forward (the FSDC-DETR paper defines no FPN replacement,
G3); the bottom-up PAN junction is either kept verbatim (``use_sfs_ff``
off — the swap-face equivalence arm, with HF-named ``downsample_convs``
/``pan_blocks`` so state dicts interchange) or replaced whole by
:class:`~...sfs_ff.SFSFF` (Eq.10-11 — the block embeds its own FSD-Down
alignment, so the SCDown + RepNCSPELAN5 pair disappears from the graph).

Eq.10 direction mapping at the PAN junction: the paper's ``F_in`` is the
COARSE map (the FPN output one level deeper) and ``F̂_in`` is the FINE
map (the running PAN output); the block returns the coarse resolution,
matching the replaced fusion's output contract.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from transformers.models.deimv2.modeling_deimv2 import (
    Deimv2AIFILayer,
    Deimv2ConvNormLayer,
    Deimv2EncoderOutput,
    Deimv2RepNCSPELAN5,
    Deimv2SCDown,
    fuse_feature_maps,
)

from autovision.tasks.detection.components.sfs_ff import SFSFF
from autovision.tasks.detection.fsdc_object_detection.config import FsdcArgs

__all__ = ["FsdcHybridEncoder"]


class FsdcHybridEncoder(nn.Module):
    """HF-faithful hybrid encoder with an optional FSC PAN."""

    def __init__(self, config, fsdc: FsdcArgs) -> None:
        super().__init__()
        self.config = config
        self.fsdc_args = fsdc
        self.in_channels = list(config.encoder_in_channels)
        self.num_fpn_stages = len(self.in_channels) - 1
        self.fuse_op = config.encoder_fuse_op
        self.encode_proj_layers = list(config.encode_proj_layers)

        self.aifi = nn.ModuleList(
            [Deimv2AIFILayer(config) for _ in self.encode_proj_layers]
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        hidden_dim = config.encoder_hidden_dim
        num_blocks = round(3 * config.depth_mult)
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(Deimv2ConvNormLayer(config, hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks))

        if fsdc.use_sfs_ff:
            self.pan_sfsff = nn.ModuleList(
                [
                    SFSFF(hidden_dim, freeze_wavelet=fsdc.freeze_wavelet)
                    for _ in range(self.num_fpn_stages)
                ]
            )
            self.downsample_convs = None
            self.pan_blocks = None
        else:
            self.downsample_convs = nn.ModuleList(
                [Deimv2SCDown(config, 3, 2) for _ in range(self.num_fpn_stages)]
            )
            self.pan_blocks = nn.ModuleList(
                [
                    Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks)
                    for _ in range(self.num_fpn_stages)
                ]
            )
            self.pan_sfsff = None

    def convert_to_deploy(self) -> None:
        """Export hook (TaskNetwork.deploy() recursion, doc 22 §4.7): with
        ``fsdc.fft_export_fallback``, bypass the SFS-FF frequency filters
        for ONNX runtimes without DFT support (semantic loss quantified in
        the deploy-arm ablation). Without the flag the FFT ops export
        as-is (static shapes, torch.export-verified)."""
        if self.fsdc_args.fft_export_fallback and self.pan_sfsff is not None:
            for sfs in self.pan_sfsff:
                sfs.fft.export_identity = True

    def forward(self, feature_maps, **kwargs) -> Deimv2EncoderOutput:
        feature_maps = list(feature_maps)

        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind], **kwargs)

        # Top-down FPN — line-faithful to HF Deimv2HybridEncoder.forward.
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = lateral_conv(fpn_feature_maps[-1])
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(
                top_fpn_feature_map, scale_factor=2.0, mode="nearest"
            )
            fused_feature_map = fuse_feature_maps(
                top_fpn_feature_map, backbone_feature_map, self.fuse_op
            )
            fpn_feature_maps.append(fpn_block(fused_feature_map))
        fpn_feature_maps.reverse()

        # Bottom-up PAN — the FSDC junction.
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx in range(self.num_fpn_stages):
            top_pan_feature_map = pan_feature_maps[-1]  # fine (2H, 2W)
            fpn_feature_map = fpn_feature_maps[idx + 1]  # coarse (H, W)
            if self.pan_sfsff is not None:
                # Eq.10-11: F_in = coarse FPN map, F̂_in = fine PAN map.
                pan_feature_maps.append(
                    self.pan_sfsff[idx](fpn_feature_map, top_pan_feature_map)
                )
            else:
                downsampled_feature_map = self.downsample_convs[idx](
                    top_pan_feature_map
                )
                fused_feature_map = fuse_feature_maps(
                    downsampled_feature_map, fpn_feature_map, self.fuse_op
                )
                pan_feature_maps.append(self.pan_blocks[idx](fused_feature_map))

        return Deimv2EncoderOutput(feature_maps=pan_feature_maps)
