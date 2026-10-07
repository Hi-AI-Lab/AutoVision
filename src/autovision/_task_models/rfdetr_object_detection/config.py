# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RF-DETR object detection: model configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from autovision._configs.base import PydanticConfig
from autovision._task_models.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "RFDETR_MODEL_REGISTRY",
    "RFDETRObjectDetectionTrainArgs",
    "resolve_rfdetr_variant",
]

# Per-variant metadata from the official rf-detr configs: native input
# resolution and the input-size unit (patch_size * num_windows) that any
# custom resolution must be divisible by.
RFDETR_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rf-detr-base": {"resolution": 560, "size_divisor": 56},  # patch 14 x 4 windows
    "rf-detr-nano": {"resolution": 384, "size_divisor": 32},  # patch 16 x 2 windows
    "rf-detr-small": {"resolution": 512, "size_divisor": 32},
    "rf-detr-medium": {"resolution": 576, "size_divisor": 32},
    "rf-detr-large": {"resolution": 704, "size_divisor": 32},
}


def resolve_rfdetr_variant(model: str) -> dict[str, Any]:
    """Registry entry for a variant name (``rf-detr-nano`` etc.)."""
    if model not in RFDETR_MODEL_REGISTRY:
        supported = ", ".join(sorted(RFDETR_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown rfdetr detection model '{model}'. Supported: {supported}."
        )
    return RFDETR_MODEL_REGISTRY[model]


class RFDETRObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training args for the RF-DETR detection family (HF transformers).

    The optimizer/schedule/EMA defaults follow the official rf-detr recipe
    (AdamW lr 1e-4, encoder lr 1.5e-4 with ViT layer-wise decay 0.8,
    decoder component decay 0.7, clip 0.1, EMA 0.993 over ~100 updates).
    The loss (IA-BCE + L1 + GIoU, Hungarian matcher costs 2/5/2) is the one
    inside HF ``RfDetrForObjectDetection`` — a faithful port of the official
    criterion — so it is consumed, not re-implemented.
    """

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 12_000

    model: str = "rf-detr-nano"
    num_classes: int = 80
    # "auto" resolves to the variant's native resolution (base 560, nano
    # 384, small 512, medium 576, large 704); custom sizes must be divisible
    # by patch_size * num_windows.
    image_size: Literal["auto"] | ImageSizeTuple = "auto"
    load_weights: bool = True  # pretrained COCO checkpoint via the verified loader
    num_queries: int = 300
    num_select: int = 300  # eval decode width (official num_select)

    # optimizer / schedule — official rf-detr recipe
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    lr_component_decay: float = 0.7
    lr_vit_layer_decay: float = 0.8
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: Literal["auto", "step", "cosine"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    lr_milestones: tuple[float, ...] = (0.75,)  # fractions of total steps
    lr_gamma: float = 0.1

    # EMA — official: decay 0.993 with tau=100 warmup (same exponential
    # warmup formula as AutoVision's ModelEMA)
    use_ema_model: bool = True
    ema_momentum: float = 0.993
    ema_warmup_steps: int = 100

    eval_max_dets: int = 100

    def resolve_variant(self) -> dict[str, Any]:
        return resolve_rfdetr_variant(self.model)

    def resolve_image_size(self) -> None:
        """Pin ``image_size`` to a concrete (h, w) tuple ("auto" -> variant
        native resolution) and validate the patch-window divisibility."""
        info = self.resolve_variant()
        if self.image_size == "auto":
            resolution = int(info["resolution"])
            self.image_size = (resolution, resolution)
            return
        height, width = self.image_size
        divisor = int(info["size_divisor"])
        for name, value in (("height", height), ("width", width)):
            if value % divisor:
                suggestion = max(divisor, round(value / divisor) * divisor)
                raise ValueError(
                    f"rfdetr image_size {self.image_size} is invalid: {name} "
                    f"{value} is not divisible by patch_size * num_windows "
                    f"({divisor} for '{self.model}'). Try {suggestion}."
                )

    def resolve_auto(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        self.resolve_image_size()
        if self.lr_scheduler == "auto":
            self.lr_scheduler = "cosine"
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = max(100, min(500, total_steps // 10))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
