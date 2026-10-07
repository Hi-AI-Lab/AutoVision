# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Detection backbone wrappers for the LT-DETR family.

Ports the baseline ``ltdetr_object_detection`` backbone adapters:

- :class:`ViTDetectionBackbone` (baseline ``DINOSTAs``): semantic features
  from intermediate ViT blocks, bilinearly resized into a stride-8/16/32
  pyramid, optionally fused channel-wise with a CNN spatial prior (STA)
  and projected to a shared ``hidden_dim`` via 1x1 convs. Only the
  pretrained ViT counts as "backbone" for LR grouping — the STA and the
  projection layers are randomly initialized and train at detector LR.
- :class:`ConvNeXtDetectionBackbone`: taps the last three ConvNeXt
  stages (strides 8/16/32); the whole backbone counts as "backbone".
- :class:`ECViTDetectionBackbone` (baseline ``ECViTBackboneWrapper``):
  pass-through around the ECViT wrapper, which already emits the
  stride-8/16/32 pyramid; the pretrained VisionTransformer counts as
  "backbone", its randomly initialized projector as connector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    GELU,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
    SyncBatchNorm,
)

from autovision._models.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision._models.dinov3.dinov3_wrapper import TransformersDinov3ConvNextWrapper

__all__ = [
    "CONVNEXT_DETECTION_PATCH_SIZE",
    "ConvNeXtDetectionBackbone",
    "ECViTDetectionBackbone",
    "SpatialPriorModule",
    "ViTDetectionBackbone",
]

#: DINOv3 ConvNeXt has no patch embedding; the detection pipeline still
#: treats the family as patch-16 for image-size arithmetic. Default only —
#: :class:`ConvNeXtDetectionBackbone` receives the authoritative value
#: from the variant spec.
CONVNEXT_DETECTION_PATCH_SIZE = 16


def _ceil_half(value: int) -> int:
    """Output size of a stride-2, kernel-3 pad-1 conv on ``value``."""
    return -(-value // 2)


class SpatialPriorModule(Module):
    """CNN detail pyramid at strides 8/16/32 (baseline SpatialPriorModulev2)."""

    def __init__(self, inplanes: int = 16) -> None:
        super().__init__()
        # 1/4
        self.stem = Sequential(
            *[
                Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                SyncBatchNorm(inplanes),
                GELU(),
                MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = Sequential(
            *[
                Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = Sequential(
            *[
                GELU(),
                Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = Sequential(
            *[
                GELU(),
                Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32
        return c2, c3, c4


class ViTDetectionBackbone(Module):
    """ViT -> stride-8/16/32 detection features (baseline ``DINOSTAs``)."""

    def __init__(
        self,
        model_wrapper: TransformersDinov2Wrapper,
        interaction_indexes: tuple[int, int, int] = (5, 8, 11),
        finetune: bool = True,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
        project_features: bool = True,
        resize_features: bool = True,
    ) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        embed_dim = model_wrapper.feature_dim()

        assert len(interaction_indexes) == 3
        self.interaction_indexes = tuple(interaction_indexes)
        self.patch_size = model_wrapper.patch_size()
        self.resize_features = resize_features

        if not finetune:
            model_wrapper.eval()
            model_wrapper.requires_grad_(False)

        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModule(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        self.project_features = project_features
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        if not self.project_features and (use_sta or hidden_dim != embed_dim):
            raise ValueError(
                "project_features=False requires use_sta=False and hidden_dim to "
                f"match the backbone embed_dim ({embed_dim}), but got "
                f"hidden_dim={hidden_dim}."
            )
        if not self.project_features:
            return

        self.convs = ModuleList(
            [
                Conv2d(
                    embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, bias=False
                ),
                Conv2d(
                    embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, bias=False
                ),
                Conv2d(
                    embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, bias=False
                ),
            ]
        )
        self.norms = ModuleList(
            [
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
            ]
        )

    @property
    def backbone_model(self) -> Module:
        """The pretrained ViT (only this part trains at backbone LR)."""
        return self._model_wrapper.get_model()

    @property
    def in_channels(self) -> list[int]:
        hidden_dim = (
            self.convs[0].out_channels
            if self.project_features
            else self._model_wrapper.feature_dim()
        )
        return [hidden_dim] * 3

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h_c, w_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        sem_feats = self._model_wrapper.forward_multiscale_features(
            x, self.interaction_indexes
        )

        # Target pyramid: shapes that chain through the encoder's stride-2
        # convolutions. The semantic taps (patch-grid based, e.g. grid 46 at
        # 644) are resized here rather than driving the shapes themselves.
        # Level 0 is input//8; each next level ceil-halves the PREVIOUS
        # level, exactly what a stride-2 3x3-pad-1 conv produces. Sizing
        # the lower levels independently from the input (e.g. ceil(x/16))
        # disagrees with the chain by one pixel at sizes like 644
        # (80 -> 40 vs ceil(644/16) = 41) and breaks the FPN/PAN fusion.
        if self.resize_features:
            h8, w8 = x.shape[2] // 8, x.shape[3] // 8
            sizes = [
                (h8, w8),
                (_ceil_half(h8), _ceil_half(w8)),
                (_ceil_half(_ceil_half(h8)), _ceil_half(_ceil_half(w8))),
            ]
        else:
            # All taps already share the patch-grid resolution.
            sizes = [(h_c, w_c)] * 3

        resized_feats = [
            F.interpolate(
                sem_feat["features"],
                size=list(size),
                mode="bilinear",
                align_corners=False,
            )
            for sem_feat, size in zip(sem_feats, sizes)
        ]

        if self.use_sta:
            detail_feats = self.sta(x)
            fused_feats = []
            for semantic_feat, detail_feat, size in zip(
                resized_feats, detail_feats, sizes
            ):
                detail_feat_interpolated = F.interpolate(
                    detail_feat,
                    size=list(size),
                    mode="bilinear",
                    align_corners=False,
                )
                fused_feats.append(
                    torch.cat([semantic_feat, detail_feat_interpolated], dim=1)
                )
        else:
            fused_feats = resized_feats

        if not self.project_features:
            return fused_feats[0], fused_feats[1], fused_feats[2]

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        return c2, c3, c4


class ConvNeXtDetectionBackbone(Module):
    """DINOv3 ConvNeXt -> detection pyramid from consecutive stages.

    ``num_stages=3`` (default) taps stages 1/2/3 (strides 8/16/32);
    ``num_stages=4`` taps all four stages (strides 4/8/16/32) — the
    Dome-faithful pyramid for tiny-object workloads (design doc 13 §5/D1).
    """

    #: 0-based stage indices for the default 3-level pyramid.
    STAGE_INDICES = (1, 2, 3)

    def __init__(
        self,
        model_wrapper: TransformersDinov3ConvNextWrapper,
        finetune: bool = True,
        patch_size: int = CONVNEXT_DETECTION_PATCH_SIZE,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = patch_size
        hidden_sizes = list(model_wrapper.config.hidden_sizes)
        if num_stages not in (3, 4):
            raise ValueError(
                f"ConvNeXtDetectionBackbone num_stages must be 3 or 4, "
                f"got {num_stages}."
            )
        if num_stages > len(hidden_sizes):
            raise ValueError(
                f"Backbone has {len(hidden_sizes)} stages; {num_stages} requested."
            )
        self.num_stages = num_stages
        self.stage_indices = tuple(
            range(len(hidden_sizes) - num_stages, len(hidden_sizes))
        )
        if not finetune:
            model_wrapper.eval()
            model_wrapper.requires_grad_(False)

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper.get_model()

    @property
    def in_channels(self) -> list[int]:
        hidden_sizes = list(self._model_wrapper.config.hidden_sizes)
        return [hidden_sizes[i] for i in self.stage_indices]

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outputs = self._model_wrapper.get_model()(
            pixel_values=x, output_hidden_states=True
        )
        hidden_states = outputs.hidden_states  # (embeddings, stage_0, ...)
        feats = tuple(hidden_states[i + 1] for i in self.stage_indices)
        if len(feats) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} ConvNeXt stage features, got {len(feats)}."
            )
        return feats


class ECViTDetectionBackbone(Module):
    """ECViT -> stride-8/16/32 detection features (pass-through).

    The ECViT wrapper already returns ``(P3, P4, P5)`` at the shared
    ``hidden_dim`` (its per-level projector), so this adapter only exposes
    the LR-grouping surface: ``backbone_model`` is the pretrained
    VisionTransformer, everything else (the 1x1 projector) is connector.
    """

    def __init__(self, model_wrapper: Module) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = 16

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper.backbone

    @property
    def in_channels(self) -> list[int]:
        return list(self._model_wrapper.proj_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = self._model_wrapper(x)
        return p3, p4, p5
