# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Ablation drivers for the RF-DETR vs RT-DETR VisDrone gap analysis.

run_ablation("rfdetr-scratch")  — RF-DETR nano WITHOUT the COCO pretrained
    detector (load_weights=False): isolates the pretrained-detector factor.
run_ablation("rtdetr-corrected") — RT-DETR with pilot-appropriate EMA
    (0.995/200 instead of 0.9999/2000) and eval_max_dets=500: removes the
    twin-setup asymmetries.
"""

from __future__ import annotations

import sys
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")

DATA_ARGS = {
    "format": "coco",
    "train": {
        "annotations": VISDRONE / "annotations" / "visdrone_train.json",
        "images": VISDRONE / "VisDrone2019-DET-train" / "images",
    },
    "val": {
        "annotations": VISDRONE / "annotations" / "visdrone_val.json",
        "images": VISDRONE / "VisDrone2019-DET-val" / "images",
    },
}

EPOCHS = 10
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
STEPS = TRAIN_IMAGES * EPOCHS // GLOBAL_BATCH
TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "scale_jitter": {"sizes": [800]},
}


def run_ablation(name: str) -> None:
    if name == "rfdetr-scratch":
        model = "rfdetr/rf-detr-nano"
        out = "runs/visdrone_rfdetr_nano_800_scratch"
        model_args = {
            "image_size": (800, 800),
            "num_queries": 300,
            "num_select": 500,
            "load_weights": False,  # the ablation: no pretrained detector
            "lr": 1e-4,
            "lr_encoder": 1.5e-4,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 500,
            "ema_momentum": 0.993,
            "ema_warmup_steps": 200,
            "eval_max_dets": 500,
        }
        micro_batch, accum = 4, 2
    elif name == "rtdetr-corrected":
        model = "rtdetr/resnet18"
        out = "runs/visdrone_rtdetr_resnet18_800_corrected"
        model_args = {
            "image_size": (800, 800),
            "backbone_pretrained": True,
            "num_queries": 300,
            "num_top_queries": 500,
            "ema_momentum": 0.995,  # pilot-appropriate EMA (was 0.9999)
            "ema_warmup_steps": 200,  # (was 2000 > total steps)
            "eval_max_dets": 500,  # new family field (was fixed 100)
        }
        micro_batch, accum = 8, 1
    else:
        raise ValueError(f"Unknown ablation '{name}'")

    result = train_object_detection(
        out=out,
        data=DATA_ARGS,
        model=model,
        steps=STEPS,
        batch_size=micro_batch,
        gradient_accumulation_steps=accum,
        num_workers=8,
        model_args=model_args,
        transform_args=TRANSFORM_ARGS,
        overwrite=True,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print(f"=== {name} ===")
    print("best export:", result.best_exported_model_path)
    print("best metrics:", result.best_metrics)


if __name__ == "__main__":
    run_ablation(sys.argv[1])
