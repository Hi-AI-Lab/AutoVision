# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Object detection finetuning (COCO / YOLO formats)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from autovision._accelerator import accelerator_supports_bf16
from autovision._commands import common_helpers
from autovision._commands.train import _is_rank_zero
from autovision._commands.train_task.task_helpers import (
    TaskTrainResult,
    _dataset_size,
    _SaveCheckpointArgs,
)
from autovision._configs.validate import pydantic_model_validate
from autovision._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision._data.mmap_helpers import dataset_mmap_context
from autovision._data.object_detection_dataset import ObjectDetectionDataset
from autovision._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from autovision._engine import TaskEngine
from autovision._logging import set_up_file_logging
from autovision._task_models.rtdetr_object_detection import (
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision._transforms.detection_transforms import (
    RTDETRDetectionCollateFunction,
    RTDETRDetectionTransform,
    RTDETRDetectionTransformArgs,
    TwoViewDetectionCollateFunction,
    build_val_transform_args,
)
from autovision.types import PathLike

logger = logging.getLogger("autovision")

_DATA_FORMATS = {"coco": COCOObjectDetectionDataArgs, "yolo": YOLOObjectDetectionDataArgs}


def _load_data_config(data: PathLike | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        payload = dict(data)
    else:
        with open(data) as file:
            payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("'data' must be a dict or a YAML file with a mapping.")
    payload.setdefault("format", "yolo")
    if "format" in payload and payload["format"] not in _DATA_FORMATS:
        supported = ", ".join(sorted(_DATA_FORMATS))
        raise ValueError(
            f"Unknown detection data format: '{payload['format']}'. Supported: {supported}."
        )
    return payload


def train_object_detection(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "rtdetr/resnet18",
    *,
    steps: int | str = "auto",
    batch_size: int | str = "auto",
    gradient_accumulation_steps: int = 1,
    num_workers: int | str = "auto",
    devices: int | str = "auto",
    precision: str = "auto",
    seed: int = 0,
    checkpoint: PathLike | None = None,
    resume_interrupted: bool = False,
    overwrite: bool = False,
    model_args: dict[str, Any] | None = None,
    transform_args: dict[str, Any] | None = None,
    save_checkpoint_args: dict[str, Any] | None = None,
    trainer_args: dict[str, Any] | None = None,
    debug_args: dict[str, Any] | None = None,
    loggers: dict[str, dict[str, Any] | None] | None = None,
) -> TaskTrainResult:
    """Object detection finetuning (COCO/YOLO formats).

    See the design doc (§8) for the contract; returns a
    :class:`TaskTrainResult` with checkpoint/export paths and best metrics.
    """
    out_dir = common_helpers.get_out_dir(out, resume_interrupted, overwrite)
    set_up_file_logging(out_dir / "train.log")

    # 1. data
    data_config_cls = _DATA_FORMATS[_load_data_config(data)["format"]]
    data_payload = _load_data_config(data)
    if isinstance(data, (str, Path)) and "data_config_file" in {
        f for f in data_config_cls.model_fields
    }:
        data_payload["data_config_file"] = str(Path(data).resolve())
    data_args = pydantic_model_validate(data_config_cls, data_payload)
    base_dir = Path(data).parent if isinstance(data, (str, Path)) else None
    data_args.resolve_data_paths(base_dir)
    classes = data_args.classes()
    num_classes = len(classes)

    # 2. model args (num_classes inferred from the data). The model string
    # dispatches the detection family: "rtdetr/*" (default), "ltdetr/*"
    # (SSL-backbone LT-DETR), "dfine/*" (distributional refinement) or
    # "picodet/*" (dense anchor-point PicoDet).
    is_ltdetr = isinstance(model, str) and model.startswith("ltdetr/")
    is_dfine = isinstance(model, str) and model.startswith("dfine/")
    is_picodet = isinstance(model, str) and model.startswith("picodet/")
    if is_ltdetr:
        from autovision._task_models.ltdetr_object_detection import (
            LTDETRObjectDetectionTrain,
            LTDETRObjectDetectionTrainArgs,
        )

        args_cls = LTDETRObjectDetectionTrainArgs
        train_cls: type = LTDETRObjectDetectionTrain
        family_model_args = {"backbone": model.split("/", 1)[1], **(model_args or {})}
    elif is_dfine:
        from autovision._task_models.dfine_object_detection import (
            DFINEObjectDetectionTrain,
            DFINEObjectDetectionTrainArgs,
        )

        args_cls = DFINEObjectDetectionTrainArgs
        train_cls = DFINEObjectDetectionTrain
        family_model_args = dict(model_args or {})
    elif is_picodet:
        from autovision._task_models.picodet_object_detection import (
            PicoDetObjectDetectionTrain,
            PicoDetObjectDetectionTrainArgs,
        )

        args_cls = PicoDetObjectDetectionTrainArgs
        train_cls = PicoDetObjectDetectionTrain
        family_model_args = {
            "backbone": model.split("/", 1)[1], **(model_args or {})
        }
    else:
        args_cls = RTDETRObjectDetectionTrainArgs
        train_cls = RTDETRObjectDetectionTrain
        family_model_args = dict(model_args or {})
    resolved_model_args = pydantic_model_validate(
        args_cls, {"num_classes": num_classes, **family_model_args}
    )
    if is_ltdetr:
        # Snap the image size to the backbone's patch grid (640 -> 644 for
        # patch-14 backbones) BEFORE the transform pipeline reads it.
        resolved_model_args.resolve_image_size()
    default_image_size = resolved_model_args.image_size

    # SSL two_crop pipeline (design doc 14 §4.5): every loader item becomes
    # an (view1, view2) pair — the batch doubles to 2B view-major rows.
    # The batch-matched baseline (ssl.mode="off") runs the same doubled
    # pipeline, so both runs see identical data throughput.
    ssl_args = getattr(resolved_model_args, "ssl", None)
    ssl_two_crop = ssl_args is not None and ssl_args.view == "two_crop"

    # 3. save args
    save_args = pydantic_model_validate(
        _SaveCheckpointArgs, dict(save_checkpoint_args or {})
    )

    # 4. batch size / steps
    default_bs = args_cls.default_batch_size
    batch_size_resolved = (
        default_bs if batch_size == "auto" else int(batch_size)
    )
    default_steps = args_cls.default_steps
    steps_resolved = default_steps if steps == "auto" else int(steps)

    # 5. datasets (mmap-managed) — built without transforms first so the
    # loader length is known for the step-window resolution.
    num_devices = (
        int(os.getenv("WORLD_SIZE", "1"))
        if devices == "auto"
        else int(devices)
    )
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, _dataset_size(data_args, "train")), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = global_batch_size // num_devices

    with dataset_mmap_context(
        lambda: data_args.list_image_info("train"),
        data=str(
            data_args.train.annotations
            if isinstance(data_args, COCOObjectDetectionDataArgs)
            else data_args.train
        ),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
    ) as train_rows, dataset_mmap_context(
        lambda: data_args.list_image_info("val"),
        data=str(
            data_args.val.annotations
            if isinstance(data_args, COCOObjectDetectionDataArgs)
            else data_args.val
        ),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
    ) as val_rows:
        train_dataset = ObjectDetectionDataset(data_args, train_rows)
        if ssl_two_crop:
            from autovision._data.two_view import TwoViewObjectDetectionDataset

            train_dataset = TwoViewObjectDetectionDataset(train_dataset)
        val_dataset = ObjectDetectionDataset(data_args, val_rows)
        train_loader = common_helpers.build_task_dataloader(
            train_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=True,
        )
        val_loader = common_helpers.build_task_dataloader(
            val_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=False,
        )
        if len(train_dataset) == 0:
            raise ValueError("The training dataset is empty.")

        # 6. transforms (resolved after the loader length is known).
        # LT-DETR gets the family recipe: patch-aligned scale jitter and
        # mosaic on (overrides still win).
        if is_ltdetr:
            from autovision._task_models.ltdetr_object_detection.transforms import (
                build_ltdetr_train_transform_args,
            )

            train_transform_args = build_ltdetr_train_transform_args(
                default_image_size, resolved_model_args.backbone, transform_args
            )
        else:
            train_transform_args = pydantic_model_validate(
                RTDETRDetectionTransformArgs,
                {"image_size": default_image_size, **(transform_args or {})},
            )
        train_transform_args.resolve_auto(
            num_channels=3, default_image_size=default_image_size
        )
        train_transform_args.resolve_step_schedule(
            total_steps=steps_resolved,
            train_num_batches=len(train_loader),
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        train_transform_args.resolve_incompatible()
        val_transform_args = build_val_transform_args(train_transform_args)

        train_transform = RTDETRDetectionTransform(train_transform_args)
        val_transform = RTDETRDetectionTransform(val_transform_args)
        train_dataset.set_transform(train_transform)
        val_dataset.set_transform(val_transform)
        train_collate = RTDETRDetectionCollateFunction(train_transform_args, train=True)
        if ssl_two_crop:
            train_collate = TwoViewDetectionCollateFunction(train_collate)
        val_collate = RTDETRDetectionCollateFunction(val_transform_args, train=False)
        train_loader.collate_fn = train_collate
        val_loader.collate_fn = val_collate

        # 7. model
        train_model = train_cls(resolved_model_args)
        train_model.set_categories(classes)

        # 8. fit
        precision_resolved = precision
        if precision_resolved == "auto":
            precision_resolved = (
                "bf16-mixed"
                if accelerator_supports_bf16()
                else "32-true"
            )
        from autovision._debug.debug_args import DebugArgs
        from autovision._events import track_training_started
        from autovision._loggers import (
            finalize_loggers,
            get_loggers,
            log_hyperparams_to_loggers,
        )

        resolved_debug_args = pydantic_model_validate(
            DebugArgs, dict(debug_args or {})
        )
        logger_instances = get_loggers(loggers, out_dir)
        track_training_started(
            task_type="object_detection", model_name=model,
            batch_size=global_batch_size, devices=num_devices,
            steps=steps_resolved,
        )
        engine = TaskEngine()
        hyperparams = {
            "task": "object_detection",
            "steps": steps_resolved,
            "batch_size": global_batch_size,
            "num_classes": num_classes,
            "classes": list(classes.values()),
        }
        if _is_rank_zero():
            log_hyperparams_to_loggers(logger_instances, hyperparams)
        fit_result = engine.fit_task(
            train_model,
            train_loader,
            val_loader,
            out_dir=out_dir,
            steps=steps_resolved,
            gradient_accumulation_steps=gradient_accumulation_steps,
            step_hooks=[train_transform, train_collate],
            val_every_num_steps=(trainer_args or {}).get("val_every_num_steps"),
            save_every_num_steps=save_args.save_every_num_steps,
            save_last=save_args.save_last,
            save_best=save_args.save_best,
            devices=devices,
            precision=precision_resolved,
            seed=seed,
            checkpoint=Path(checkpoint) if checkpoint else None,
            resume_interrupted=resume_interrupted,
            logger_fn=lambda metrics, step: [
                lg.log_metrics(metrics, step) for lg in logger_instances
            ] if _is_rank_zero() else None,
            log_every_n_steps=(trainer_args or {}).get(
                "log_every_n_steps", min(50, max(1, len(train_loader)))
            ),
            normalize_args=train_transform_args.normalize.to_dict(),
            model_class_path=train_model.get_task_model().class_path,
            model_init_args={
                **train_model.get_task_model().init_args,
                "model_name": model,
            },
            debug_args=resolved_debug_args,
        )
        finalize_loggers(logger_instances)

    if _is_rank_zero():
        logger.info("Task training finished: %d steps.", fit_result.steps_run)
    return TaskTrainResult(
        out_dir=out_dir,
        steps_run=fit_result.steps_run,
        checkpoint_path=fit_result.checkpoint_path,
        best_checkpoint_path=fit_result.best_checkpoint_path,
        exported_model_path=fit_result.exported_model_path,
        best_exported_model_path=fit_result.best_exported_model_path,
        best_metrics=fit_result.best_metrics,
        classes=classes,
    )
