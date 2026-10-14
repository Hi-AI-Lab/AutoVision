# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EdgeCrafter (ECViT): compact ViTs for edge dense prediction.

Port of the EdgeCrafter ECViT backbone (axial-RoPE ViT with a convolutional
pyramid patch embed) in the shape used by the AV-DETR "v2" family. Source
lineage: EdgeCrafter (Apache-2.0, Copyright (c) 2026 The EdgeCrafter Authors,
``Intellindust-AI-Lab/EdgeCrafter`` @ ``d7fdcb5``,
``ecdetseg/engine/edgecrafter/ecvit.py``), modified by Autovision; ledger
``docs/provenance/LEDGER.md`` (P1-6). Adapted with the framework's own
wrapper conventions:

- no upstream registry/distributed dependencies; presets are local,
- weights load from an explicit ``weights_path`` (strict); without one the
  backbone is randomly initialized (offline-friendly),
- the wrapper exposes the ``EncoderProtocol`` protocol duck-typed by
  :mod:`autovision.encoders.model_wrapper` and returns an ``(P3, P4, P5)``
  stride-8/16/32 pyramid directly.

Provenance: EdgeCrafter (Apache-2.0, @ ``d7fdcb5``); the axial RoPE is
re-ported from the Apache EdgeCrafter/DEIMv2 source — see the
``RopePositionEmbedding`` docstring, ``docs/provenance/LEDGER.md`` (P0-4,
P1-6) and ``docs/provenance/specs/S4-ecvit-rope.md``.
"""

from __future__ import annotations

import logging
import math
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from autovision.encoders._model_helpers import adapt_conv_input_channels
from autovision.encoders.model_wrapper import (
    ArchitectureInfo,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from autovision.errors import CheckpointError

__all__ = ["ECVIT_PRESETS", "ECViTModelWrapper", "VisionTransformer"]

_PRESET = object()  # sentinel: take the value from ECVIT_PRESETS

logger = logging.getLogger("autovision")


ECVIT_PRETRAINED_URLS: dict[str, str] = {
    "ecvitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth",
    "ecvittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvittplus.pth",
    "ecvits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvits.pth",
    "ecvitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitsplus.pth",
}

ECVIT_PRESETS: dict[str, dict[str, int | None | float]] = {
    "ecvitt": {"embed_dim": 192, "num_heads": 3, "proj_dim": None, "ffn_ratio": 4.0},
    "ecvittplus": {
        "embed_dim": 256,
        "num_heads": 4,
        "proj_dim": None,
        "ffn_ratio": 4.0,
    },
    "ecvits": {"embed_dim": 384, "num_heads": 6, "proj_dim": 256, "ffn_ratio": 4.0},
    "ecvitsplus": {"embed_dim": 384, "num_heads": 6, "proj_dim": 256, "ffn_ratio": 6.0},
}


class ConvNormLayer(nn.Module):
    """Conv2d + BN (+ activation); identical param layout to the shared
    detection component of the same name so checkpoints stay compatible."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        act: str | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the trailing dimension by half a period: ``(-x2, x1)``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Standard rotary-embedding application: ``x*cos + rotate_half(x)*sin``."""
    return x * cos + rotate_half(x) * sin


class RopePositionEmbedding(nn.Module):
    """Axial 2D rotary position embedding for patch tokens.

    Re-port of the EdgeCrafter axial RoPE (Apache-2.0, Copyright (c) 2026
    The EdgeCrafter Authors, ``Intellindust-AI-Lab/EdgeCrafter`` @
    ``d7fdcb5``; upstream states "Modified from DINOv3"; the DEIMv2 mirror
    carries the same code under its Apache header). Modifications per
    Apache-2.0 section 4b: dropped the min/max-period, coordinate
    shift/jitter/rescale training options and the 4-D output shaping —
    this port exposes the plain ``(HW, D)`` tables with the caller adding
    head/batch dims, and coordinates follow the buffer's dtype.

    The ``periods`` buffer is persistent ON PURPOSE: published checkpoints
    store periods that drift ~1e-4 from the init formula, so the buffer
    must round-trip through checkpoints rather than be recomputed.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = "separate",
    ) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "2D axial RoPE needs head_dim divisible by 4"
        self.base = base
        self.head_dim = head_dim
        self.normalize_coords = normalize_coords
        self.register_buffer("periods", torch.empty(head_dim // 4), persistent=True)
        self._init_weights()

    def _axial_coordinates(self, height: int, width: int) -> Tensor:
        """Patch-cell centers, per-axis normalized to ``[-1, +1]``: (HW, 2)."""
        device = self.periods.device
        dtype = self.periods.dtype
        if self.normalize_coords == "max":
            longest = max(height, width)
            rows = torch.arange(0.5, height, device=device, dtype=dtype) / longest
            cols = torch.arange(0.5, width, device=device, dtype=dtype) / longest
        elif self.normalize_coords == "min":
            shortest = min(height, width)
            rows = torch.arange(0.5, height, device=device, dtype=dtype) / shortest
            cols = torch.arange(0.5, width, device=device, dtype=dtype) / shortest
        elif self.normalize_coords == "separate":
            rows = torch.arange(0.5, height, device=device, dtype=dtype) / height
            cols = torch.arange(0.5, width, device=device, dtype=dtype) / width
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
        return grid.reshape(-1, 2) * 2.0 - 1.0

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        # Per-axis angular frequencies (D//4 per axis), interleaved over the
        # two coordinate axes and duplicated for the rotate-half pairing:
        # (HW, 2, D//4) -> (HW, D//2) -> (HW, D).
        coords = self._axial_coordinates(H, W)
        angular = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angular = angular.flatten(1, 2).repeat(1, 2)
        return angular.sin(), angular.cos()

    def _init_weights(self) -> None:
        exponents = 2 * torch.arange(
            self.head_dim // 4, dtype=self.periods.dtype, device=self.periods.device
        )
        self.periods.data.copy_(self.base ** (exponents / (self.head_dim // 2)))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPyramidPatchEmbed(nn.Module):
    """Conv stem downsampling 16x in four stride-2 stages (patch 16 only)."""

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, act: str = "relu"):
        super().__init__()
        if patch_size != 16:
            raise NotImplementedError(
                "Only support patch_size=16 for ConvPyramidPatchEmbed."
            )
        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2**i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]
        self.convs = nn.ModuleList(
            [
                ConvNormLayer(in_ch, out_ch, 3, 2, act=act)
                for in_ch, out_ch in zip([3] + channels[:-1], channels)
            ]
        )
        self.proj = nn.Conv2d(
            channels[-1], embed_dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(
    tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope_sincos is not None:
            sin, cos = rope_sincos
            # RoPE applies to patch tokens only; the register token keeps
            # its raw embedding (baseline behavior).
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]

            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)

            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop
        )
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: Any = nn.LayerNorm,
        ffn_layer: type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ECViT backbone: conv-pyramid patch embed + RoPE ViT blocks.

    ``return_layers`` selects the block outputs (patch tokens only) handed
    to the detection wrapper; the baseline default averages the last two.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        return_layers: list[int] | None = None,
        embed_layer: type[nn.Module] = ConvPyramidPatchEmbed,
        norm_layer: Any | None = None,
        act_layer: type[nn.Module] | None = None,
        ffn_layer: type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers or [3, 7, 11]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # Preserve EdgeCrafter behavior: GELU is used for transformer MLPs.
        act_layer = nn.GELU

        patch_embed: nn.Module
        if embed_layer is PatchEmbed:
            patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            patch_embed = embed_layer(embed_dim=embed_dim, patch_size=patch_size)
        self.patch_embed: nn.Module = patch_embed
        self.patch_size = patch_size

        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                )
                for i in range(depth)
            ]
        )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()
        trunc_normal_(self.register_token, std=0.02)

    def _init_vit_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_with_grid(self, x: Tensor) -> tuple[list[Tensor], tuple[int, int]]:
        outs: list[Tensor] = []
        x_embed = self.patch_embed(x)
        _, _, H, W = x_embed.shape

        x_embed = x_embed.flatten(2).transpose(1, 2)
        register_token = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((register_token, x_embed), dim=1)
        sin, cos = self.rope_embed(H=H, W=W)
        sin = sin.to(device=x_embed.device, dtype=x_embed.dtype)
        cos = cos.to(device=x_embed.device, dtype=x_embed.dtype)
        rope_sincos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])
        return outs, (H, W)

    def forward(self, x: Tensor) -> list[Tensor]:
        outs, _ = self.forward_with_grid(x)
        return outs


class ECViTModelWrapper(nn.Module):
    """EdgeCrafter ECViT backbone wrapper emitting a (P3, P4, P5) pyramid.

    Selected block outputs are averaged, reshaped to a spatial map,
    interpolated to strides 8/16/32, projected through per-level 1x1
    ConvNormLayers, and returned as ``(P3, P4, P5)`` — ready to feed the
    shared HybridEncoder. Without ``weights_path`` the backbone stays
    randomly initialized (offline builds); with one, weights load strictly.
    """

    def __init__(
        self,
        name: str,
        weights_path: str | Path | None = None,
        interaction_indexes: list[int] | None = None,
        embed_dim: int | object = _PRESET,
        num_heads: int | object = _PRESET,
        patch_size: int = 16,
        proj_dim: int | None | object = _PRESET,
        num_levels: int = 3,
        ffn_ratio: float | object = _PRESET,
        num_input_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if name not in ECVIT_PRESETS:
            raise ValueError(
                f"Unknown ECViT model name: {name}. Available: {list(ECVIT_PRESETS)}"
            )
        if num_levels != 3:
            raise NotImplementedError(
                "Only support num_levels=3 for ECViTModelWrapper."
            )

        preset = ECVIT_PRESETS[name]
        resolved_embed_dim = int(preset["embed_dim"] if embed_dim is _PRESET else embed_dim)  # type: ignore[arg-type]
        resolved_num_heads = int(preset["num_heads"] if num_heads is _PRESET else num_heads)  # type: ignore[arg-type]
        resolved_proj_dim = (
            preset["proj_dim"] if proj_dim is _PRESET else proj_dim
        )  # type: ignore[assignment]
        resolved_ffn_ratio = float(preset["ffn_ratio"] if ffn_ratio is _PRESET else ffn_ratio)  # type: ignore[arg-type]
        resolved_interaction_indexes = interaction_indexes or [10, 11]

        self.name = name
        self.interaction_indexes = resolved_interaction_indexes
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.embed_dim = resolved_embed_dim
        self.num_heads = resolved_num_heads
        self.proj_dim: list[int] = (
            [resolved_proj_dim] * num_levels
            if resolved_proj_dim is not None
            else [resolved_embed_dim]
        )

        self.backbone = VisionTransformer(
            embed_dim=resolved_embed_dim,
            num_heads=resolved_num_heads,
            return_layers=resolved_interaction_indexes,
            patch_size=patch_size,
            ffn_ratio=resolved_ffn_ratio,
            **kwargs,
        )

        if weights_path is not None:
            self._load_backbone_weights(weights_path)

        if num_input_channels != 3:
            # Adapt AFTER the weight load: the published checkpoints are
            # 3-channel, so the strict load must see the original stem.
            first_conv = self.backbone.patch_embed.convs[0].conv
            self.backbone.patch_embed.convs[0].conv = nn.Conv2d(
                num_input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                self.backbone.patch_embed.convs[0].conv.weight.copy_(
                    adapt_conv_input_channels(
                        first_conv.weight.data, num_input_channels
                    )
                )

        self.projector = nn.ModuleList(
            [
                ConvNormLayer(resolved_embed_dim, dim, kernel_size=1, stride=1)
                for dim in self.proj_dim
            ]
        )
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def backbone_model(self) -> nn.Module:
        return self.backbone

    def _load_backbone_weights(self, weights_path: str | Path) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"ECViT weights_path does not exist: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise CheckpointError(
                f"Failed to load ECViT weights '{path}' under the safe "
                f"loader (weights_only=True): {error}. Convert the file "
                f"once in a trusted context and re-save the plain "
                f"state_dict with torch.save."
            ) from error
        state_dict = dict(_unwrap_state_dict(state))
        # rope_embed.periods is persistent (upstream parity): it must ride
        # along — the published checkpoints' values drift ~1e-4 from the
        # init formula, so recomputing is NOT equivalent. Legacy self-trained
        # weights saved before the buffer became persistent lack the key;
        # those load with the freshly initialized periods (every other key
        # is still matched strictly).
        if "rope_embed.periods" not in state_dict:
            missing = set(self.backbone.state_dict()) - set(state_dict)
            missing.discard("rope_embed.periods")
            if missing:
                raise CheckpointError(
                    f"ECViT weights '{path}' are missing keys other than the "
                    f"legacy rope_embed.periods: {sorted(missing)[:8]}"
                )
            logger.info(
                "ECViT weights '%s' predate the persistent rope_embed.periods "
                "buffer; using the freshly initialized periods.", path,
            )
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone.load_state_dict(state_dict, strict=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bs = x.shape[0]

        return_layers, (H_c, W_c) = self.backbone.forward_with_grid(x)
        if len(return_layers) == 0:
            raise RuntimeError(
                "ECViT backbone returned no layers. Check interaction_indexes."
            )

        fused_feats = torch.mean(torch.stack(return_layers), dim=0)
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)

        # Pyramid sizes chain through the HybridEncoder's stride-2 convs
        # (2x, 1x, ceil-halve); truncating level 2 instead would disagree
        # by one pixel for odd patch grids and break the PAN fusion.
        sizes = (
            (2 * H_c, 2 * W_c),
            (H_c, W_c),
            (max(1, (H_c + 1) // 2), max(1, (W_c + 1) // 2)),
        )
        proj_feats: list[Tensor] = []
        for resize_H, resize_W in sizes:
            feature = F.interpolate(
                fused_feats,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            proj_feats.append(feature)

        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [
                layer(feat) for layer, feat in zip(self.projector, proj_feats)
            ]

        if len(proj_feats) != 3:
            raise RuntimeError(
                f"Expected 3 ECViT feature levels, got {len(proj_feats)}."
            )
        return proj_feats[0], proj_feats[1], proj_feats[2]

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self.forward(x)[-1]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def feature_dim(self) -> int:
        return self.proj_dim[-1]

    def get_model(self) -> "ECViTModelWrapper":
        return self

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}


def _unwrap_state_dict(state: object) -> Mapping[str, Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict mapping, got {type(state)!r}.")

    for key in ("state_dict", "model", "backbone"):
        maybe_state_dict = state.get(key)  # type: ignore[union-attr]
        if isinstance(maybe_state_dict, Mapping) and all(
            isinstance(k, str) and isinstance(v, Tensor)
            for k, v in maybe_state_dict.items()
        ):
            return maybe_state_dict

    if all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
        return state  # type: ignore[return-value]

    raise TypeError("Expected checkpoint to contain a tensor state dict.")
