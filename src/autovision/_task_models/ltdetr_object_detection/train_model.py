# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""LT-DETR object detection training wrapper.

Mirrors the RT-DETR training wrapper (criterion, EMA, flat-cosine,
metrics) with baseline LT-DETR LR grouping:

- ViT variants: only the pretrained ViT trains at the backbone LR — the
  STA prior and the 1x1 projection layers are randomly initialized and
  merged into the detector group.
- ConvNeXt variants: the whole backbone trains at the backbone LR.
- ECViT variants: the pretrained VisionTransformer trains at the backbone
  LR; its randomly initialized 1x1 projector joins the detector group.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from autovision._metrics.map_evaluator import COCOMAPEvaluator
from autovision._optim import get_weight_decay_parameters
from autovision._task_models.ltdetr_object_detection.backbone_wrappers import (
    ECViTDetectionBackbone,
    ViTDetectionBackbone,
)
from autovision._task_models.ltdetr_object_detection.config import (
    LTDETRObjectDetectionTrainArgs,
    resolve_ltdetr_variant,
)
from autovision._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from autovision._task_models.object_detection_components.dfine_criterion import (
    DFINECriterionV2,
)
from autovision._task_models.object_detection_components.ema import ModelEMA
from autovision._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)
from autovision._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)
from autovision._task_models.object_detection_components.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)
from autovision._task_models.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")

__all__ = ["LTDETRObjectDetectionTrain"]


class LTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: LTDETRObjectDetectionTrainArgs) -> None:
        task_model = LTDETRObjectDetection(
            num_classes=model_args.num_classes,
            model_name=model_args.backbone,
            image_size=model_args.image_size,
            decoder=model_args.decoder,
            reg_max=model_args.reg_max,
            hidden_dim=model_args.hidden_dim
            if model_args.hidden_dim != "auto"
            else None,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers
            if model_args.dec_layers != "auto"
            else None,
            num_heads=model_args.num_heads
            if model_args.num_heads != "auto"
            else None,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            label_noise_ratio=model_args.label_noise_ratio,
            box_noise_scale=model_args.box_noise_scale,
            backbone_pretrained=model_args.backbone_pretrained,
            backbone_weights=model_args.backbone_weights,
            load_weights=model_args.backbone_pretrained,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.matcher = HungarianMatcher()
        decoder = model_args.decoder
        if decoder == "auto":
            _, variant = resolve_ltdetr_variant(f"ltdetr/{model_args.backbone}")
            decoder = variant.decoder_default
        if decoder == "dfine":
            self.criterion: RTDETRCriterionV2 | DFINECriterionV2 = DFINECriterionV2(
                self.matcher,
                reg_max=model_args.reg_max,
            )
        else:
            self.criterion = RTDETRCriterionV2(self.matcher)
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
        outputs_list = list(out.get("aux_outputs", [])) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        if isinstance(self.criterion, DFINECriterionV2):
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                corners_list=out.get("pred_corners_list"),
                ref_boxes_list=out.get("ref_boxes_list"),
                dn_corners_list=out.get("dn_pred_corners_list"),
                dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
            )
        else:
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
            )
        log_dict = {key: float(value.detach()) for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        # NOTE: the D-FINE criterion receives no corners here, so the val
        # loss covers the base VFL/bbox/GIoU terms only (FGL/DDF are
        # train-time distributional losses). Val and train losses are
        # therefore not directly comparable for decoder="dfine".
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        outputs = model(x)
        decoded = model.postprocess(
            outputs,
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=0.05,
        )
        evaluator = self._evaluators.setdefault("val", COCOMAPEvaluator(self.categories))
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {"boxes": boxes, "labels": labels}
                for boxes, labels in zip(batch["bboxes"], batch["classes"])
            ],
            original_sizes=batch["original_size"],
        )
        if isinstance(outputs, dict):
            outputs_list = [outputs]
        elif isinstance(outputs, tuple):
            outputs_list = [{"pred_logits": outputs[0], "pred_boxes": outputs[1]}]
        else:  # ObjectDetectionOutput dataclass
            outputs_list = [
                {"pred_logits": outputs.logits, "pred_boxes": outputs.boxes}
            ]
        losses = self.criterion(
            outputs_list,
            [
                {"boxes": b.to(x.device), "labels": c.to(x.device)}
                for b, c in zip(batch["bboxes"], batch["classes"])
            ],
            world_size=1,
        )
        return TaskStepResult(
            loss=losses["loss"],
            log_dict={key: float(value) for key, value in losses.items()},
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
            split, COCOMAPEvaluator(self.categories)
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
        if (
            args.lr_warmup_steps == "auto"
            or args.scheduler_flat_steps == "auto"
            or args.scheduler_no_aug_steps == "auto"
            or args.backbone_lr_factor == "auto"
            or args.hidden_dim == "auto"
            or args.dec_layers == "auto"
            or args.num_heads == "auto"
        ):
            args.resolve_auto(
                total_steps=total_steps,
                train_num_batches=train_num_batches or total_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        lr = args.lr * math.sqrt(global_batch_size / args.default_batch_size)
        backbone_lr = lr * args.backbone_lr_factor

        # Baseline LR grouping: no-weight-decay params across the whole
        # training model, then backbone/detector partition by membership.
        _, params_no_wd_list = get_weight_decay_parameters([self.model])
        params_no_wd = {id(p) for p in params_no_wd_list}

        backbone = self.model.backbone
        if isinstance(backbone, (ViTDetectionBackbone, ECViTDetectionBackbone)):
            backbone_params = list(backbone.backbone_model.parameters())
            backbone_param_ids = {id(p) for p in backbone_params}
            # Randomly initialized connector modules (STA prior +
            # projection convs/norms, or the ECViT projector) train at
            # detector LR.
            connector_params = [
                p for p in backbone.parameters() if id(p) not in backbone_param_ids
            ]
        else:
            backbone_params = list(backbone.parameters())
            connector_params = []

        param_groups: list[dict[str, Any]] = []
        backbone_wd = [p for p in backbone_params if id(p) not in params_no_wd]
        backbone_no_wd = [p for p in backbone_params if id(p) in params_no_wd]
        if backbone_wd:
            param_groups.append(
                {"name": "backbone", "params": backbone_wd, "lr": backbone_lr}
            )
        if backbone_no_wd:
            param_groups.append(
                {
                    "name": "backbone_no_wd",
                    "params": backbone_no_wd,
                    "lr": backbone_lr,
                    "weight_decay": 0.0,
                }
            )

        detector_params = (
            connector_params
            + list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters())
        )
        detector_wd = [p for p in detector_params if id(p) not in params_no_wd]
        detector_no_wd = [p for p in detector_params if id(p) in params_no_wd]
        if detector_wd:
            param_groups.append({"name": "detector", "params": detector_wd, "lr": lr})
        if detector_no_wd:
            param_groups.append(
                {
                    "name": "detector_no_wd",
                    "params": detector_no_wd,
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=args.optimizer_betas, weight_decay=args.weight_decay
        )
        scheduler = FlatCosineLRScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(args.lr_warmup_steps),
            flat_steps=int(args.scheduler_flat_steps),
            no_aug_steps=int(args.scheduler_no_aug_steps),
        )
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
