# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training arguments for the GTR pose family (roadmap doc 31 §11 S5b)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.train_model import TrainModelArgs

__all__ = ["GTRPoseEstimationTrainArgs"]


class GTRPoseEstimationTrainArgs(TrainModelArgs):
    """Official pose finetune defaults (``gtrpose_base`` + S config).

    Losses ``['vfl', 'keypoints']``; weights ``{loss_vfl: 2.0,
    loss_keypoints: 10.0, loss_oks: 4.0}``; matcher costs class 2 /
    keypoints 10 / OKS 4; lr 5e-4 with the backbone at 0.05x (official
    2.5e-5 / 5e-4). CDN hyperparameters (20 groups, label noise 0.5) are
    the upstream constants baked into the decoder.
    """

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 50_000

    num_classes: int = 2
    variant: Literal["s", "m", "l", "x"] = "s"
    image_size: tuple[int, int] = (640, 640)
    num_queries: int = 60
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    num_body_points: int = 17
    checkpoint: str | None = None
    load_checkpoint: bool = True
    losses: list[str] = Field(default_factory=lambda: ["vfl", "keypoints"])
    loss_weight_dict: dict[str, float] = Field(
        default_factory=lambda: {"loss_vfl": 2.0, "loss_keypoints": 10.0, "loss_oks": 4.0}
    )
    cost_class: float = 2.0
    cost_keypoints: float = 10.0
    cost_oks: float = 4.0
    lr: float = 5e-4
    backbone_lr_factor: float = 0.05
    optimizer_betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    weight_decay: float = 0.0001
    scheduler: Literal["cosine"] = "cosine"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    horizontal_flip_prob: float = 0.5
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    fused_optimizer: bool = False

    model_config = {"extra": "forbid"}

    def resolve_auto_fields(self, **kwargs) -> None:
        total_steps = kwargs.get("total_steps", 0)
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = (
                max(10, int(total_steps * 0.02)) if total_steps else 10
            )
