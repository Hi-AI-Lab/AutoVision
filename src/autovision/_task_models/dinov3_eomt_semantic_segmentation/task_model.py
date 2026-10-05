# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT semantic segmentation task model, DINOv3 variant (inference/export
facing)."""

from __future__ import annotations

from typing import Any

from autovision._task_models.dinov2_eomt_semantic_segmentation.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision._task_models.eomt.modeling import build_eomt_model
from autovision._task_models.task_model import TaskModel
from autovision._transforms.transform_args import NormalizeArgs

__all__ = ["Dinov3EomtSemanticSegmentation"]


class Dinov3EomtSemanticSegmentation(Dinov2EomtSemanticSegmentation):
    """DINOv3 variant: same encoder-only mask transformer contract, backed
    by the HF ``EomtDinov3ForUniversalSegmentation`` (RoPE, storage tokens).
    Only construction differs; preprocessing, forward, decode, checkpoint
    loading and export behavior are inherited from the DINOv2 variant.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        num_queries: int = 100,
        num_blocks: int = 4,
        num_upscale_blocks: int = 2,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: int = 4,
        use_gated_mlp: bool = False,
        drop_path_rate: float = 0.0,
        load_weights: bool = False,
        weight_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        # The DINOv2 parent __init__ pins the DINOv2 HF model class, so
        # skip it and replicate its TaskModel bookkeeping here.
        TaskModel.__init__(
            self,
            init_args=locals(),
            ignore_args={"load_weights", "kwargs", "self"},
        )
        self.num_classes = num_classes
        self.image_size = int(image_size)
        self.normalize = NormalizeArgs()
        self.model = build_eomt_model(
            num_classes,
            variant="dinov3",
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_queries=num_queries,
            num_blocks=num_blocks,
            num_upscale_blocks=num_upscale_blocks,
            image_size=self.image_size,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            use_gated_mlp=use_gated_mlp,
            drop_path_rate=drop_path_rate,
            load_weights=load_weights,
            weight_source=weight_source,
        )
