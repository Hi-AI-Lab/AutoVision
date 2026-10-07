# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DEIMv2 object detection training wrapper.

Consumes the HF ``Deimv2ForObjectDetection`` criterion (the official
matcher + VFL/MAL + L1 + GIoU + FGL/DDF port) instead of an AutoVision-side
criterion, and replicates the official DEIMv2 optimizer recipes: AdamW with
a variant-specific backbone LR multiplier (HGNetv2 0.5x, DINOv3 0.05x),
weight-decay exemptions for norm/bias parameters, gradient clip 0.1, and
ModelEMA 0.9999 over a 1000-update warmup — all at the official total
batch of 32.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from autovision._metrics.map_evaluator import COCOMAPEvaluator
from autovision._task_models.deimv2_object_detection.config import (
    Deimv2ObjectDetectionTrainArgs,
)
from autovision._task_models.deimv2_object_detection.task_model import (
    Deimv2ObjectDetection,
)
from autovision._task_models.object_detection_components.ema import ModelEMA
from autovision._task_models.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")

__all__ = ["Deimv2ObjectDetectionTrain", "_deimv2_param_buckets"]

# Official optimizer.yml regex groups, expressed as param-name fragments:
# the backbone ViT/CNN gets its own LR bucket; weight decay is exempt
# exactly where the official recipes exempt it — norm/bn/bias parameters
# ("norm" covers LayerNorm and the HGNetv2 "normalization" modules).
# Position embeddings / query tokens keep weight decay, matching the
# official regexes.
_DINOV3_FRAGMENT = ".conv_encoder.backbone."  # DINOv3 ViT (deimv2-dinov3-s)
_BACKBONE_FRAGMENT = ".conv_encoder.model."  # HGNetv2 CNN (N / PICO variants)
_NO_WD_FRAGMENTS = ("bias", "norm", "bn")


def _deimv2_param_buckets(
    task_model: Deimv2ObjectDetection, args: Deimv2ObjectDetectionTrainArgs
) -> dict[tuple[str, bool], list[torch.nn.Parameter]]:
    """Split HF parameters into (lr-bucket, weight-decay) groups.

    Buckets: ``dinov3`` for the DINOv3 ViT, ``backbone`` for the HGNetv2
    CNN (exactly one of the two exists per variant), ``decoder`` for the
    detection transformer, and ``other`` for the encoder, STA, queries,
    and heads.
    """
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in task_model._model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_wd = any(fragment in name for fragment in _NO_WD_FRAGMENTS)
        if _DINOV3_FRAGMENT in name:
            bucket = "dinov3"
        elif _BACKBONE_FRAGMENT in name:
            bucket = "backbone"
        elif ".decoder." in name:
            bucket = "decoder"
        else:
            bucket = "other"
        buckets.setdefault((bucket, no_wd), []).append(parameter)
    return buckets


def _bucket_lr(bucket: str, args: Deimv2ObjectDetectionTrainArgs, lr: float) -> float:
    if bucket in ("dinov3", "backbone"):
        return float(args.lr_backbone)
    return lr


class Deimv2ObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: Deimv2ObjectDetectionTrainArgs) -> None:
        model_args.resolve_image_size()
        task_model = Deimv2ObjectDetection(
            num_classes=model_args.num_classes,
            model=model_args.model,
            image_size=model_args.image_size,
            num_select=model_args.num_select,
            load_weights=model_args.load_weights,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.ema_model: ModelEMA | None = (
            ModelEMA(
                task_model,
                decay=model_args.ema_momentum,
                warmup_steps=model_args.ema_warmup_steps,
            )
            if model_args.use_ema_model
            else None
        )
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, COCOMAPEvaluator] = {}

    # ── steps ───────────────────────────────────────────────────────────
    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        out = self.model._forward_train(x, targets=targets)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError(
                "Deimv2ForObjectDetection returned no loss; the HF forward "
                "requires labels to compute the criterion."
            )
        # The engine aggregates the log window into a top-level
        # "train_loss" from the plain "loss" key (criterion convention).
        # Values stay as 0-dim tensors: the engine's log window sums them
        # on-device and only syncs at log points.
        log_dict: dict[str, Any] = {"loss": loss.detach()}
        for key, value in out["loss_dict"].items():
            log_dict[f"train_loss/{key}"] = (
                value.detach() if torch.is_tensor(value) else value
            )
        return TaskStepResult(loss=loss, log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        model = self.ema_model.model if self.ema_model is not None else self.model
        # Eval mode drops the denoising query group, so the same forward
        # serves both the decode and the monitored val loss.
        outputs = model._forward_train(x, targets=targets)
        decoded = model.postprocess(
            (outputs["pred_logits"], outputs["pred_boxes"]),
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=0.0,  # COCO protocol: cap by top-k / maxDets, not scores
        )
        evaluator = self._evaluators.setdefault(
            "val",
            COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets),
        )
        evaluator.update(decoded)
        evaluator.update_targets(targets, original_sizes=batch["original_size"])
        loss = outputs["loss"]
        log_dict: dict[str, float] = {}
        if loss is not None:
            log_dict = {
                f"val_loss/{key}": float(value)
                for key, value in outputs["loss_dict"].items()
            }
            log_dict["val_loss"] = float(loss)
        return TaskStepResult(
            loss=loss if loss is not None else x.new_zeros(()), log_dict=log_dict
        )

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    def compute_metrics(self, split: str) -> dict[str, float]:
        evaluator = self._evaluators.get(split)
        if evaluator is None:
            return {}
        metrics = evaluator.compute()
        evaluator.reset()
        return {f"{split}_metric/{name}": value for name, value in metrics.items()}

    def reset_metrics(self, split: str) -> None:
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            evaluator.reset()

    def export_metric_state(self, split: str) -> dict | None:
        evaluator = self._evaluators.get(split)
        return evaluator.export_state() if evaluator is not None else None

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator = self._evaluators.setdefault(
            "val",
            COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets),
        )
        evaluator.merge_state(state)

    # ── optimization ────────────────────────────────────────────────────
    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> tuple[torch.optim.Optimizer, Any]:
        args = self.model_args
        args.resolve_auto(
            total_steps=total_steps,
            train_num_batches=train_num_batches or total_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        scale = math.sqrt(global_batch_size / args.default_batch_size)
        lr = args.lr * scale

        param_groups = []
        for (bucket, no_wd), params in sorted(
            _deimv2_param_buckets(self.model, args).items()
        ):
            group: dict[str, Any] = {
                "name": f"{bucket}_no_wd" if no_wd else bucket,
                "params": params,
                "lr": _bucket_lr(bucket, args, lr),
            }
            if no_wd:
                group["weight_decay"] = 0.0
            param_groups.append(group)

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
        )

        warmup = int(args.lr_warmup_steps)
        milestones = [fraction * total_steps for fraction in args.lr_milestones]

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            if args.lr_scheduler == "step":
                factor = 1.0
                for milestone in milestones:
                    if step >= milestone:
                        factor *= args.lr_gamma
                return factor
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def clip_gradients(self, optimizer) -> float | None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return float(
            torch.nn.utils.clip_grad_norm_(params, self.model_args.gradient_clip_val)
        )

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    # ── export state ────────────────────────────────────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state)

    def get_export_state_dict(self) -> dict[str, Any]:
        if self.ema_model is not None:
            return {
                f"ema_model.{key}": value
                for key, value in self.ema_model.model.state_dict().items()
            }
        return self.state_dict()
