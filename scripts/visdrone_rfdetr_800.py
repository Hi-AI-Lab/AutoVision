# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone RF-DETR baseline at 800x800.

Mirrors ``scripts/visdrone_ltdetr_800.py`` (same data paths, 10-epoch
pilot, global batch 8, scale jitter locked at 800, no mosaic/mixup) with
the RF-DETR family: HF ``RfDetrForObjectDetection`` fine-tuned from the
verified Roboflow checkpoint with the official recipe (AdamW lr 1e-4 /
encoder 1.5e-4 layer-decayed 0.8, cosine schedule, EMA 0.993).

Requires transformers >= 5.9 (RfDetr classes). Run with the overlay env:

    PYTHONPATH=/tmp/torchshim:src AUTOVISION_WEIGHT_SOURCE=auto \
        /tmp/rf590env/bin/python scripts/visdrone_rfdetr_800.py [--smoke]
"""

from __future__ import annotations

import argparse
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
MICRO_BATCH = 4  # 800px + group queries peak ~20GB at batch 8; batch 4 x accum 2
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
STEPS = TRAIN_IMAGES * EPOCHS // GLOBAL_BATCH  # 8088
AUG_STOP_STEP = round(STEPS * 108 / 160)

MODEL_ARGS = {
    "image_size": (800, 800),
    "num_queries": 300,
    "num_select": 500,  # dense scenes: widen eval decode (official 300)
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "weight_decay": 1e-4,
    "gradient_clip_val": 0.1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "use_ema_model": True,
    "ema_momentum": 0.993,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "scale_jitter": {"sizes": [800], "step_stop": AUG_STOP_STEP},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny validation run")
    parser.add_argument(
        "--erase",
        action="store_true",
        help="box-aware background erasure (V2 constrained mode: every "
        "object keeps >= 2 patches; rf-detr-ssl-v2 design doc 8)",
    )
    namespace = parser.parse_args()

    if namespace.smoke:
        steps, val_every, save_every, out_name = 60, 30, 60, "smoke"
    else:
        steps, val_every, save_every, out_name = STEPS, 1000, 2000, "full"

    result = train_object_detection(
        out=f"runs/visdrone_rfdetr_nano_800_{out_name}",
        data=DATA_ARGS,
        model="rfdetr/rf-detr-nano",
        steps=steps,
        batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=(
            {
                **TRANSFORM_ARGS,
                "random_erase_prob": 0.5,
                "random_erase_ratio": 0.4,
                "random_erase_interior": True,
                "random_erase_min_keep": 2,
            }
            if namespace.erase
            else TRANSFORM_ARGS
        ),
        overwrite=True,
        trainer_args={
            "val_every_num_steps": val_every,
            "log_every_n_steps": max(1, val_every // 2),
        },
        save_checkpoint_args={"save_every_num_steps": save_every},
    )
    print("=== run summary ===")
    print("steps_run:", result.steps_run)
    print("best checkpoint:", result.best_checkpoint_path)
    print("best export:", result.best_exported_model_path)
    if result.best_metrics:
        print("best metrics:", result.best_metrics)


if __name__ == "__main__":
    main()
