# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Quantify the EMA drag on validation mAP (design doc 13 follow-up).

The engine's validation evaluates the EMA model; with momentum 0.999 the
EMA window (~1k steps) still averages early weights, so early-run val
mAP understates the online model. This diagnostic evaluates BOTH on the
same val split with the exact training-side validation pipeline:

- **online** model: restored from the run checkpoint's ``state_dict``
  (``model.*`` keys);
- **EMA** model: restored from the run's ``exported_last.pt`` (exports
  carry the EMA weights via ``get_export_state_dict``).

Usage (full val split on GPU, after training frees the card):

    .venv/bin/python scripts/diag_ema_vs_online.py

Smoke on CPU with a subset:

    .venv/bin/python scripts/diag_ema_vs_online.py --device cpu --max-batches 2
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from autovision.cli import common_helpers
from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.object_detection_dataset import ObjectDetectionDataset
from autovision.tasks.detection.avdetr import (
    AVDETRObjectDetectionTrain,
    AVDETRObjectDetectionTrainArgs,
)
from autovision.augment.detection import build_val_transform_args
from autovision.augment.detection.args import (
    RTDETRDetectionTransformArgs,
)
from autovision.augment.detection.collate import (
    RTDETRDetectionCollateFunction,
)
from autovision.augment.detection.transform import (
    RTDETRDetectionTransform,
)

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3], REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_device(batch: dict, device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="runs/visdrone_ltdetr_convnextt_base_800",
        help="Training run dir containing checkpoints/last.ckpt and "
        "exported_models/exported_last.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Evaluate only the first N val batches (0 = all images)",
    )
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir)
    base = _load_script("visdrone_ltdetr_800.py")

    # The real run derives num_classes from the annotations (10 for this
    # VisDrone export) — the TrainArgs default (80) would mismatch heads.
    categories = json.load(
        open(base.VISDRONE / "annotations" / "visdrone_val.json")
    )["categories"]
    num_classes = len(categories)

    args = AVDETRObjectDetectionTrainArgs(
        num_classes=num_classes, **base.MODEL_ARGS
    )
    args.resolve_image_size()
    train_model = AVDETRObjectDetectionTrain(args).to(arguments.device)

    # Online weights: the checkpoint's TrainModel state_dict (`model.*`).
    checkpoint = torch.load(
        run_dir / "checkpoints" / "last.ckpt", map_location="cpu",
        weights_only=False,
    )
    online_state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = train_model.model.load_state_dict(
        online_state, strict=False
    )
    print(f"online weights: missing {len(missing)}, unexpected {len(unexpected)}")

    # EMA weights: the export's `train_model` dict carries the EMA copy of
    # the task model (`ema_model.*` keys). Manual trusted load: exports
    # written before the str(backbone_weights) fix embed a PosixPath in
    # model_init_args, which the pickle-free safe loader rejects.
    export = torch.load(
        run_dir / "exported_models" / "exported_last.pt",
        map_location="cpu", weights_only=False,
    )
    ema_state = {
        key[len("ema_model."):]: value
        for key, value in export["train_model"].items()
        if key.startswith("ema_model.")
    }
    train_model.ema_model.model.load_state_dict(ema_state, strict=True)
    print(f"ema weights: loaded from exported_last.pt ({len(ema_state)} tensors)")

    # Val split through the exact training-side validation pipeline.
    data_args = COCOObjectDetectionDataArgs(**base.DATA_ARGS)
    rows = list(data_args.list_image_info("val"))
    dataset = ObjectDetectionDataset(data_args, rows)
    train_transform_args = RTDETRDetectionTransformArgs.model_validate(
        {**base.TRANSFORM_ARGS, "image_size": tuple(args.image_size)}
    )
    val_args = build_val_transform_args(train_transform_args)
    dataset.set_transform(RTDETRDetectionTransform(val_args))
    loader = common_helpers.build_task_dataloader(
        dataset, batch_size=8, num_workers=4, train=False
    )
    loader.collate_fn = RTDETRDetectionCollateFunction(val_args, train=False)

    train_model.set_categories(
        {i: c["name"] for i, c in enumerate(sorted(categories, key=lambda c: c["id"]))}
    )

    def evaluate(use_ema: bool) -> dict[str, float]:
        backup = train_model.ema_model
        train_model.ema_model = backup if use_ema else None
        train_model.reset_metrics("val")
        done, start = 0, time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                train_model.validation_step(
                    _to_device(batch, arguments.device), 0
                )
                done += 1
                if arguments.max_batches and done >= arguments.max_batches:
                    break
        metrics = {
            key: value
            for key, value in train_model.compute_metrics("val").items()
            if "loss" not in key
        }
        metrics["wall_s"] = round(time.perf_counter() - start, 1)
        train_model.ema_model = backup
        return metrics

    online = evaluate(use_ema=False)
    ema = evaluate(use_ema=True)
    print("\n=== EMA drag diagnostic ===")
    for name, metrics in (("online", online), ("ema   ", ema)):
        print(
            f"{name}: map {metrics.get('val_metric/map', -1):.4f}  "
            f"map50 {metrics.get('val_metric/map_50', -1):.4f}  "
            f"({metrics['wall_s']}s)"
        )
    online_map = online.get("val_metric/map", 0)
    if online_map > 0:
        drag = (online_map - ema.get("val_metric/map", 0)) / online_map * 100
        print(f"EMA drag: {drag:+.1f}% of online mAP")


if __name__ == "__main__":
    main()
