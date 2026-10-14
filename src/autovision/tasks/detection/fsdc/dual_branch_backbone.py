# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Native FSDC dual-branch encoder: DINOv3 ViT + STA + DBFSAF (doc 22 §2.2/§3.2).

Replaces the HF ``Deimv2DINOv3ConvEncoder`` inside a fully-built
``Deimv2Model`` while preserving every contract the surrounding stack
relies on:

- **forward contract** — verbatim HF semantics: the ViT feature maps are
  bilinearly resized onto the pyramid grid (parameter-free, HF
  ``modeling_deimv2.py`` resize formula), the STA detail pyramid is
  concatenated channel-wise (semantic first, detail second — HF order),
  and each level is fused to ``encoder_hidden_dim``.
- **parameter naming** — the DINOv3 ViT stays at ``backbone`` so the
  training wrapper's ``.conv_encoder.backbone.`` LR-bucket fragment
  (deimv2 train_model) keeps routing it to the backbone LR; the STA
  keeps HF's ``spatial_tuning_adapter`` attribute names for state-dict
  compatibility; the fusion modules are ``dbfsaf_blocks`` (native) or
  HF-named ``fusion_proj`` (DBFSAF off — the swap-face equivalence arm).
- **weight transfer** — the pretrained ViT moves over from the replaced
  HF encoder: reused directly at 3 levels, rebuilt with four taps and
  strict-reloaded at 4 (``out_indices`` does not affect the weights).

Extensions over HF: the STA exposes its stride-4 stem output for the
4-level pyramid (G2) and the per-level fusion is DBFSAF (Eq.1-6)
instead of the blind 1×1.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoBackbone
from transformers.models.deimv2.modeling_deimv2 import Deimv2ConvNormLayer

from autovision.tasks.detection.components.dbfsaf import DBFSAF
from autovision.tasks.detection.fsdc.config import FsdcArgs

__all__ = ["FsdcDinoDualBranchEncoder", "FsdcSpatialTuningAdapter"]


def _four_taps(num_layers: int) -> list[int]:
    """Four evenly spaced 1-based hidden-state taps ending at the last block."""
    taps = [max(1, round(num_layers * k / 4)) for k in (1, 2, 3)]
    taps.append(num_layers)
    return taps


class FsdcSpatialTuningAdapter(nn.Module):
    """HF ``Deimv2SpatialTuningAdapter`` replica + stride-4 exposure (G2).

    Attribute names match HF exactly (``stem_conv``/``stem_pool``/
    ``conv2``/``conv3``/``conv4``/``act_fn``) so a 3-level instance is
    state-dict interchangeable with the HF module (swap-face test).
    """

    def __init__(self, config, num_levels: int = 3) -> None:
        super().__init__()
        inplanes = config.spatial_tuning_adapter_inplanes
        self.stem_conv = Deimv2ConvNormLayer(config, 3, inplanes, 3, 2, activation="gelu")
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Deimv2ConvNormLayer(config, inplanes, 2 * inplanes, 3, 2)
        self.conv3 = Deimv2ConvNormLayer(config, 2 * inplanes, 4 * inplanes, 3, 2)
        self.conv4 = Deimv2ConvNormLayer(config, 4 * inplanes, 4 * inplanes, 3, 2)
        self.act_fn = nn.GELU()
        self.num_levels = num_levels

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        hidden_1 = self.stem_pool(self.stem_conv(pixel_values))  # stride 4
        hidden_2 = self.conv2(hidden_1)  # stride 8
        hidden_3 = self.conv3(self.act_fn(hidden_2))  # stride 16
        hidden_4 = self.conv4(self.act_fn(hidden_3))  # stride 32
        if self.num_levels == 4:
            return hidden_1, hidden_2, hidden_3, hidden_4
        return hidden_2, hidden_3, hidden_4


class FsdcDinoDualBranchEncoder(nn.Module):
    """DINOv3 ViT + STA detail pyramid + per-level DBFSAF fusion."""

    def __init__(self, config, fsdc: FsdcArgs, source_backbone: nn.Module) -> None:
        super().__init__()
        if source_backbone is None:
            raise ValueError(
                "FsdcDinoDualBranchEncoder needs the replaced HF conv "
                "encoder's backbone (the DINOv3 ViT is transferred, not "
                "rebuilt, so pretrained weights survive the swap)."
            )
        self.config = config
        self.fsdc_args = fsdc
        self.num_levels = fsdc.sta_levels

        if self.num_levels == 3:
            self.backbone = source_backbone
        else:
            # Rebuild with four taps; the ViT weights are index-independent,
            # so a strict reload transfers everything (doc 22 §3.2). The
            # transfer is done at the WRAPPER level on purpose: the
            # DINOv3ViTBackbone keeps embeddings/rope/norm outside
            # ``.model``, so a ``.model``-scoped load would silently skip
            # the patch embeddings and RoPE tables.
            backbone_config = copy.deepcopy(source_backbone.config)
            taps = _four_taps(backbone_config.num_hidden_layers)
            backbone_config.out_indices = list(taps)
            backbone_config.out_features = [f"stage{t}" for t in taps]
            backbone = AutoBackbone.from_config(config=backbone_config)
            transfer = backbone.load_state_dict(source_backbone.state_dict())
            if transfer.missing_keys or transfer.unexpected_keys:
                raise ValueError(
                    "DINOv3 weight transfer to the 4-tap backbone failed: "
                    f"missing={transfer.missing_keys[:5]} "
                    f"unexpected={transfer.unexpected_keys[:5]}"
                )
            self.backbone = backbone

        if fsdc.backbone_weights:
            # Faithful fsdc-l: finetune the LVD-1689M pretrained ViT-S
            # (the official yml's weights_path), not a random ViT.
            self.load_dinov3_pretrained(fsdc.backbone_weights)

        inplanes = config.spatial_tuning_adapter_inplanes
        self.spatial_tuning_adapter = FsdcSpatialTuningAdapter(
            config, num_levels=self.num_levels
        )
        # Channel tables mirror HF's fusion_proj construction: [2i, 4i, 4i]
        # at 3 levels; the stride-4 level prepends the bare stem width.
        sta_channels = [inplanes, 2 * inplanes, 4 * inplanes, 4 * inplanes][
            -self.num_levels :
        ]
        embed_dim = self.backbone.config.hidden_size
        fusion_in = [embed_dim + c for c in sta_channels]
        hidden_dim = config.encoder_hidden_dim
        if fsdc.use_dbfsaf:
            self.dbfsaf_blocks = nn.ModuleList(
                [
                    DBFSAF(
                        in_channels,
                        hidden_dim,
                        gamma=fsdc.dbfsaf_gamma,
                        use_ksm=fsdc.fdconv_use_ksm,
                        fdconv_lite=fsdc.fdconv_lite,
                    )
                    for in_channels in fusion_in
                ]
            )
            self.fusion_proj = None
        else:
            # HF-named blind fusion: the swap-face equivalence arm (doc 22
            # §3.2) — a state-dict copy of the replaced HF fusion_proj
            # makes this module bit-identical to Deimv2DINOv3ConvEncoder.
            self.fusion_proj = nn.ModuleList(
                [
                    Deimv2ConvNormLayer(config, in_channels, hidden_dim, 1, 1)
                    for in_channels in fusion_in
                ]
            )
            self.dbfsaf_blocks = None

    def convert_to_deploy(self) -> None:
        """Export hook (TaskNetwork.deploy() recursion, doc 22 §4.7): with
        ``fsdc.fft_export_fallback``, bypass the DBFSAF FPU frequency
        filters (ONNX DFT fallback arm)."""
        if self.fsdc_args.fft_export_fallback and self.dbfsaf_blocks is not None:
            for block in self.dbfsaf_blocks:
                block.fpu.export_identity = True

    def load_dinov3_pretrained(self, weights_path: str) -> None:
        """Load the facebook DINOv3 ViT conversion into the backbone.

        The facebook safetensors spell the encoder submodule ``layer.*``
        while ``DINOv3ViTBackbone`` nests it under ``model.layer.*`` —
        that prefix rename is the ONLY mapping (211 keys, verified 1:1;
        embeddings/rope/norm share their names). Strict load, so a
        foreign file fails loud instead of partially initializing.
        """
        path = str(weights_path)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        renamed = {
            f"model.{key}" if key.startswith("layer.") else key: value
            for key, value in state.items()
        }
        try:
            self.backbone.load_state_dict(renamed, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"fsdc backbone_weights '{path}' does not match the "
                f"DINOv3 ViT backbone (expected the facebook conversion "
                f"'dinov3-vits16-pretrain-lvd1689m' layout): {error}"
            ) from error

    def forward(self, pixel_values: Tensor) -> list[Tensor]:
        backbone_output = self.backbone(pixel_values)
        feature_maps = backbone_output.feature_maps

        patch_size = self.backbone.config.patch_size
        height_patches = pixel_values.shape[2] // patch_size
        width_patches = pixel_values.shape[3] // patch_size

        # HF resize formula: level i lands at patches * 2^(num_scales-2-i).
        semantic_features = []
        num_scales = len(feature_maps)
        for i, feat in enumerate(feature_maps):
            resize_height = int(height_patches * 2 ** (num_scales - 2 - i))
            resize_width = int(width_patches * 2 ** (num_scales - 2 - i))
            semantic_features.append(
                F.interpolate(
                    feat,
                    size=[resize_height, resize_width],
                    mode="bilinear",
                    align_corners=False,
                )
            )

        detail_features = self.spatial_tuning_adapter(pixel_values)

        outputs = []
        for i, (semantic_feature, detail_feature) in enumerate(
            zip(semantic_features, detail_features)
        ):
            fused = torch.cat(
                [semantic_feature.to(detail_feature.device), detail_feature], dim=1
            )
            if self.dbfsaf_blocks is not None:
                outputs.append(self.dbfsaf_blocks[i](fused))
            else:
                outputs.append(self.fusion_proj[i](fused))
        return outputs
