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
from autovision._task_models.object_detection_components.multistep_schedule import (
    MultiStepWarmupScheduler,
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
            reg_scale=model_args.reg_scale,
            enc_dim_feedforward=model_args.enc_dim_feedforward,
            enc_expansion=model_args.enc_expansion,
            enc_depth_mult=model_args.enc_depth_mult,
            hidden_dim=model_args.hidden_dim
            if model_args.hidden_dim != "auto"
            else None,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers
            if model_args.dec_layers != "auto"
            else None,
            num_heads=model_args.num_heads if model_args.num_heads != "auto" else None,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            label_noise_ratio=model_args.label_noise_ratio,
            box_noise_scale=model_args.box_noise_scale,
            backbone_pretrained=model_args.backbone_pretrained,
            backbone_weights=model_args.backbone_weights,
            load_weights=model_args.backbone_pretrained,
            pyramid_levels=model_args.pyramid_levels,
            density=model_args.density,
            ssl=model_args.ssl,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.matcher = HungarianMatcher()
        decoder = model_args.decoder
        if decoder == "auto":
            _, variant = resolve_ltdetr_variant(f"ltdetr/{model_args.backbone}")
            decoder = variant.decoder_default
        density = model_args.density
        ssl_args = model_args.ssl
        # The SSL criterion wiring only exists on the D-FINE criterion (v1,
        # design doc 14 §3.3); the batch-matched baseline (mode="off") needs
        # no wiring anywhere.
        ssl_params = (
            ssl_args.to_ssl_params()
            if ssl_args is not None and ssl_args.mode != "off"
            else None
        )
        if ssl_args is not None and decoder != "dfine":
            raise ValueError(
                "ssl (per-object contrastive head) requires the D-FINE "
                "decoder in v1; pass decoder='dfine' (design doc 14 §3.3)."
            )
        # The SIoU swap (design doc 15) lives in the same D-FINE criterion;
        # a non-dfine decoder with the flag set would silently train giou.
        if model_args.box_regression_iou == "siou" and decoder != "dfine":
            raise ValueError(
                "box_regression_iou='siou' requires the D-FINE decoder "
                "in v1; pass decoder='dfine' (design doc 15 §4.5)."
            )
        if decoder == "dfine":
            self.criterion: RTDETRCriterionV2 | DFINECriterionV2 = DFINECriterionV2(
                self.matcher,
                reg_max=model_args.reg_max,
                reg_scale=model_args.reg_scale,
                dn_fgl_loss_weight=model_args.dn_fgl_loss_weight,
                defe_density_map_weight=(
                    density.density_map_weight if density else 1.0
                ),
                defe_count_loss_weight=density.count_loss_weight if density else 1.0,
                defe_underestimation_penalty=(
                    density.density_underestimation_penalty if density else 0.3
                ),
                defe_count_underestimation_weight=(
                    density.count_underestimation_weight if density else 2.0
                ),
                ssl_params=ssl_params,
                box_regression_iou=model_args.box_regression_iou,
                siou_loss_coef=model_args.siou_loss_coef,
                siou_theta=model_args.siou_theta,
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
        self._max_dets = model_args.eval_max_dets
        self._evaluators: dict[str, COCOMAPEvaluator] = {}
        # Route B (SSL-only pretraining) flag, resolved once (design doc 14 §4.10).
        self._ssl_only = (
            ssl_args is not None and ssl_args.instance_source == "encoder_candidates"
        )
        if self._ssl_only:
            # Best-checkpoint signal for stage 1: "val_ssl_loss" is a key the
            # val metrics never carry, so the engine's
            # ``get(watch_key, -val_loss)`` fallback watches the NEGATED
            # world-averaged validation loss (the held-out contrastive
            # loss) — exactly the lower-is-better signal needed. The class
            # attribute stays "val_metric/map" for stage 2 / joint runs.
            self.watch_metric = "val_ssl_loss"
        # Density-guided query diagnostics accumulated over the val split
        # (per-image (query_count, gt_count)); consumed by compute_metrics
        # so the engine persists them (validation log_dict is dropped).
        self._val_query_stats: list[tuple[int, int]] = []
        # Fraction of MWAS windows the adaptive mask selected, per val
        # image — the gate-tightness readout: 1.0 means the density gate
        # passes every window (random-init DeFE head), shrinking toward
        # the dense-region fraction as the head converges (doc 13 §4.2).
        self._val_window_ratios: list[float] = []

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
        # Density-guided extras (design doc 13 §4.6): the query-validity
        # mask keeps padded queries out of the classification loss; the
        # defe dict carries the density map / GT heatmap / count estimate.
        query_mask = out.get("query_valid_mask")
        defe_info = out.get("defe_info")
        if isinstance(self.criterion, DFINECriterionV2):
            # SSL stride state lives on the criterion, written per step
            # (design doc 14 §4.4); the features come from the model's
            # train-only projection head.
            self.criterion.ssl_step = step
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                dn_pre_outputs=out.get("dn_pre_outputs"),
                corners_list=out.get("pred_corners_list"),
                ref_boxes_list=out.get("ref_boxes_list"),
                dn_corners_list=out.get("dn_pred_corners_list"),
                dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
                query_mask=query_mask,
                defe_info=defe_info,
                pre_outputs=out.get("pre_outputs"),
                enc_aux_outputs=out.get("enc_aux_outputs"),
                enc_meta=out.get("enc_meta"),
                ssl_z1=out.get("ssl_z1"),
                ssl_z2=out.get("ssl_z2"),
                ssl_s1=out.get("ssl_s1"),
                ssl_s2=out.get("ssl_s2"),
            )
        else:
            loss_dict = self.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                query_mask=query_mask,
                enc_aux_outputs=out.get("enc_aux_outputs"),
                enc_meta=out.get("enc_meta"),
            )
        # Detached GPU tensors: the engine floats them only when logging
        # (every log_every_n_steps); per-item float() here would sync the
        # GPU once per loss term on every step.
        log_dict = {key: value.detach() for key, value in loss_dict.items()}
        # SSL pool diagnostics (design doc 14 D10): paired_ratio is the
        # fraction of view-1 instances that found a validated cross-view
        # partner — a two_crop pair consumes two pool entries, so the
        # denominator is pool/2 (1.0 = perfect pairing); under the dropout
        # view every instance pairs trivially (denominator = pool).
        ssl_stats = getattr(self.criterion, "last_ssl_stats", None)
        if ssl_stats:
            pool = float(ssl_stats.get("pool_size", 0.0))
            paired = float(ssl_stats.get("paired_size", 0.0))
            ssl_view = (
                self.criterion.ssl_params.view
                if self.criterion.ssl_params
                else "two_crop"
            )
            denominator = max(pool / 2.0 if ssl_view == "two_crop" else pool, 1.0)
            log_dict["ssl_pool_size"] = pool
            log_dict["ssl_paired_ratio"] = paired / denominator
            if "candidate_size" in ssl_stats:
                # D14: paired + unpaired-reclaimed candidate columns.
                log_dict["ssl_candidate_size"] = float(ssl_stats["candidate_size"])
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        # NOTE: the D-FINE criterion receives no corners here, so the val
        # loss covers the base VFL/bbox/GIoU terms only (FGL/DDF are
        # train-time distributional losses). Val and train losses are
        # therefore not directly comparable for decoder="dfine".
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        query_mask = None
        queries_num = None
        ssl_z1 = ssl_z2 = None
        if self._ssl_only:
            # Route B: the decoder emits candidate features in eval too, so
            # the validation loss is the held-out contrastive loss (design
            # doc 14 §4.10). The mAP eval below still runs — its curve is
            # expected to be noise until stage-2 finetuning.
            raw = model._forward_train(x)
            outputs = (raw["pred_logits"], raw["pred_boxes"])
            ssl_z1 = raw.get("ssl_z1")
            ssl_z2 = raw.get("ssl_z2")
        elif getattr(model, "_density_enabled", False):
            # Dynamic-query models: the raw forward dict carries the
            # per-image query counts (dome metrics) and the padded-tail
            # query mask (design doc 13 §4.6).
            raw = model._forward_train(x)
            outputs = (raw["pred_logits"], raw["pred_boxes"])
            query_mask = raw.get("query_valid_mask")
            queries_num = raw.get("batch_queries_num")
            defe_info = raw.get("defe_info")
            if isinstance(defe_info, dict) and "window_mask" in defe_info:
                window_mask = defe_info["window_mask"]
                self._val_window_ratios.extend(
                    window_mask.float().mean(dim=(1, 2)).detach().tolist()
                )
        else:
            outputs = model(x)
        decoded = model.postprocess(
            outputs,
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            # Baseline/COCO protocol: NO score gate before the mAP eval —
            # pycocotools ranks everything up to maxDets. A 0.05 gate drops
            # the low-confidence detections that early-epoch recall lives
            # on and understates mAP.
            threshold=0.0,
        )
        evaluator = self._evaluators.setdefault(
            "val", COCOMAPEvaluator(self.categories, max_dets=self._max_dets)
        )
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
        criterion_kwargs: dict[str, Any] = {}
        if isinstance(self.criterion, DFINECriterionV2):
            # Route B: the SSL-only branch consumes the candidate features;
            # the auxiliary path ignores them (absent in eval).
            criterion_kwargs = {"ssl_z1": ssl_z1, "ssl_z2": ssl_z2}
        losses = self.criterion(
            outputs_list,
            [
                {"boxes": b.to(x.device), "labels": c.to(x.device)}
                for b, c in zip(batch["bboxes"], batch["classes"])
            ],
            world_size=1,
            query_mask=query_mask,
            **criterion_kwargs,
        )
        log_dict = {key: float(value) for key, value in losses.items()}
        if queries_num is not None:
            # Dome diagnostics (baseline "defe Ample Rate" / "Average
            # Anchor Number", design doc 13 §1.3): how often the dynamic
            # selection keeps at least one query per GT, and the mean
            # query count (efficiency readout against the fixed baseline).
            # Accumulated per image and surfaced through compute_metrics.
            gt_counts = [int(boxes.shape[0]) for boxes in batch["bboxes"]]
            self._val_query_stats.extend(zip(queries_num, gt_counts))
            log_dict["dome_avg_queries"] = float(
                sum(queries_num) / max(len(queries_num), 1)
            )
            log_dict["dome_ample_rate"] = float(
                sum(n >= g for n, g in zip(queries_num, gt_counts))
                / max(len(queries_num), 1)
            )
        return TaskStepResult(loss=losses["loss"], log_dict=log_dict)

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    def validation_report(self, metrics: dict[str, float]) -> str | None:
        """Box tables (overall + per-class) for the engine's val log line.

        ``metrics`` is the prefixed ``val_metric/*`` dict the engine just
        computed; the per-class rows come from the evaluator's last
        ``compute()``. Returning ``None`` is never expected here, but the
        engine treats any exception as "no report" and falls back to the
        one-line summary.
        """
        from autovision._metrics.report import format_detection_validation_report

        scalars = {
            key.split("/", 1)[1]: value
            for key, value in metrics.items()
            if key.startswith("val_metric/")
        }
        evaluator = self._evaluators.get("val")
        per_class = getattr(evaluator, "last_per_class", []) if evaluator else []
        footer = None
        if isinstance(self.criterion, DFINECriterionV2):
            # Mirrors the note on validation_step: the val loss skips the
            # train-time distributional terms, so the two totals in the
            # metrics file are different scopes by construction. The IoU
            # term name follows the criterion's box-loss swap (doc 15) so
            # the note never goes stale again.
            iou_name = (
                "SIoU"
                if getattr(self.criterion, "box_regression_iou", "giou") == "siou"
                else "GIoU"
            )
            footer = (
                f"note: val_loss covers VFL/bbox/{iou_name} only (FGL/DDF and "
                "the denoising/aux branches are train-time) — not comparable "
                "to train_loss"
            )
        return format_detection_validation_report(
            "Validation (EMA weights)" if self.ema_model is not None else "Validation",
            scalars,
            per_class,
            self._max_dets,
            footer=footer,
        )

    def compute_metrics(self, split: str) -> dict[str, float]:
        evaluator = self._evaluators.get(split)
        if evaluator is None:
            return {}
        metrics = evaluator.compute()
        evaluator.reset()
        if split == "val" and self._val_query_stats:
            # Density-guided diagnostics (design doc 13 §1.3), consumed
            # here so the engine's val logging / metrics.jsonl carry them.
            stats = self._val_query_stats
            metrics["dome_avg_queries"] = float(
                sum(n for n, _ in stats) / max(len(stats), 1)
            )
            metrics["dome_ample_rate"] = float(
                sum(n >= g for n, g in stats) / max(len(stats), 1)
            )
            self._val_query_stats.clear()
        if split == "val" and self._val_window_ratios:
            metrics["dome_window_ratio"] = float(
                sum(self._val_window_ratios) / max(len(self._val_window_ratios), 1)
            )
            self._val_window_ratios.clear()
        return {f"{split}_metric/{name}": value for name, value in metrics.items()}

    def reset_metrics(self, split: str) -> None:
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            evaluator.reset()
        if split == "val":
            self._val_query_stats.clear()
            self._val_window_ratios.clear()

    def export_metric_state(self, split: str) -> dict | None:
        evaluator = self._evaluators.get(split)
        evaluator_state = evaluator.export_state() if evaluator is not None else None
        if split == "val":
            # Fold the dome query stats into the gathered payload so the
            # rank-zero merge sees every rank's shard (DDP path).
            return {
                "evaluator": evaluator_state,
                "dome_query_stats": list(self._val_query_stats),
                "dome_window_ratios": list(self._val_window_ratios),
            }
        return evaluator_state

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator_state = state
        if split == "val" and isinstance(state, dict) and "evaluator" in state:
            evaluator_state = state.get("evaluator")
            self._val_query_stats.extend(state.get("dome_query_stats", []))
            self._val_window_ratios.extend(state.get("dome_window_ratios", []))
        evaluator = self._evaluators.setdefault(
            split, COCOMAPEvaluator(self.categories, max_dets=self._max_dets)
        )
        if evaluator_state is not None:
            evaluator.merge_state(evaluator_state)

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
            or args.lr_milestones == "auto"
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

        # The SSL projection head lives on the task model root (outside
        # encoder/decoder), so it needs its own explicit group at the base
        # LR — mirroring RF-DETR-SSL, which assigns the head the base LR.
        if getattr(self.model, "ssl_head", None) is not None:
            ssl_all = [p for p in self.model.ssl_head.parameters() if p.requires_grad]
            ssl_wd = [p for p in ssl_all if id(p) not in params_no_wd]
            ssl_no_wd = [p for p in ssl_all if id(p) in params_no_wd]
            if ssl_wd:
                param_groups.append({"name": "ssl", "params": ssl_wd, "lr": lr})
            if ssl_no_wd:
                param_groups.append(
                    {
                        "name": "ssl_no_wd",
                        "params": ssl_no_wd,
                        "lr": lr,
                        "weight_decay": 0.0,
                    }
                )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
        )
        if args.lr_scheduler == "linear":
            # Baseline DINOv2 recipe: warmup then constant.
            scheduler = MultiStepWarmupScheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=int(args.lr_warmup_steps),
                milestones=(),
                gamma=args.lr_gamma,
            )
        elif args.lr_scheduler == "multistep":
            scheduler = MultiStepWarmupScheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=int(args.lr_warmup_steps),
                milestones=tuple(int(f * total_steps) for f in args.lr_milestones),
                gamma=args.lr_gamma,
            )
        else:
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

    # ── checkpoint hooks ────────────────────────────────────────────────
    @staticmethod
    def _is_ssl_head_key(key: str) -> bool:
        """Match SSL-head keys in every state-dict layout: bare
        (``ssl_head.*`` inside an EMA copy), train-wrapper
        (``model.ssl_head.*``) or export (``ema_model.ssl_head.*``)."""
        return key.startswith("ssl_head.") or ".ssl_head." in key

    def _fill_missing_ssl_keys(self, incoming: dict[str, Any]) -> dict[str, Any]:
        """Fill SSL-head keys absent from *incoming* with their current values.

        Adding the SSL branch on top of a detection-pretrained checkpoint is
        a supported workflow: the head keeps its fresh init instead of
        failing the strict load. Only keys that belong to the SSL head are
        filled — any other missing key, and any unexpected key, stays loud.
        """
        own = self.state_dict()
        missing_ssl = {
            key: value
            for key, value in own.items()
            if self._is_ssl_head_key(key) and key not in incoming
        }
        if missing_ssl:
            incoming = {**incoming, **missing_ssl}
        return incoming

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs: Any):
        incoming = dict(state_dict)
        own_keys = set(self.state_dict().keys())
        # Stage-2 workflow (design doc 14 §4.10): a stage-1 SSL-only
        # checkpoint carries the pretraining-only projection head — drop it
        # when the new config has no head, instead of failing the strict
        # load. (The engine's ``checkpoint=`` init path lands here; resume
        # with a mismatched ssl config stays loud for every non-SSL key.)
        incoming = {
            key: value
            for key, value in incoming.items()
            if not (self._is_ssl_head_key(key) and key not in own_keys)
        }
        return super().load_state_dict(
            self._fill_missing_ssl_keys(incoming), strict=strict, **kwargs
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.ema_model is not None:
            ema_weights = dict(ema_state.get("model", {}))
            own_ema = self.ema_model.model.state_dict()
            missing_ssl = {
                key: value
                for key, value in own_ema.items()
                if self._is_ssl_head_key(key) and key not in ema_weights
            }
            if missing_ssl:
                ema_state = {**ema_state, "model": {**ema_weights, **missing_ssl}}
            self.ema_model.load_state_dict(ema_state)

    # ── export state ────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict[str, Any]:
        if self.ema_model is not None:
            return {
                f"ema_model.{key}": value
                for key, value in self.ema_model.model.state_dict().items()
            }
        return self.state_dict()
