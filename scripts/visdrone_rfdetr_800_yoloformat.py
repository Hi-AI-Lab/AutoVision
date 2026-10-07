# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone RF-DETR at 800x800 trained on YOLO-format data.

Equivalence twin of ``scripts/visdrone_rfdetr_800.py`` (COCO): identical
model, recipe, resolution and steps — only the data format differs. The
result should match the COCO run's val mAP (0.2499) up to seed noise,
proving the YOLO data path is on par with the COCO path.

Requires transformers >= 5.9 overlay env:
    PYTHONPATH=/tmp/torchshim:src /tmp/rf590env/bin/python scripts/visdrone_rfdetr_800_yoloformat.py
"""

from __future__ import annotations

from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")

DATA_ARGS = {
    "format": "yolo",
    "path": VISDRONE / "yolo",
    "train": "train/images",
    "val": "val/images",
    "names": {
        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
        5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
    },
}

EPOCHS = 10
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
MICRO_BATCH = 4
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
STEPS = TRAIN_IMAGES * EPOCHS // GLOBAL_BATCH  # 8088

MODEL_ARGS = {
    "image_size": (800, 800),
    "num_queries": 300,
    "num_select": 500,
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "ema_momentum": 0.993,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "scale_jitter": {"sizes": [800]},
}

result = train_object_detection(
    out="runs/visdrone_rfdetr_nano_800_yoloformat",
    data=DATA_ARGS,
    model="rfdetr/rf-detr-nano",
    steps=STEPS,
    batch_size=MICRO_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_workers=8,
    model_args=MODEL_ARGS,
    transform_args=TRANSFORM_ARGS,
    overwrite=True,
    trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
    save_checkpoint_args={"save_every_num_steps": 2000},
)
print("=== run summary ===")
print("steps_run:", result.steps_run)
print("best checkpoint:", result.best_checkpoint_path)
print("best export:", result.best_exported_model_path)
print("best metrics:", result.best_metrics)
