# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR finetuning: DINOv3 ConvNeXt backbone, local weights.

Launch on the GPU (after the current job frees it):

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr.py

Recipe:
- ``ltdetr/dinov3-convnext-tiny``: SSL backbone -> stride 8/16/32 pyramid
  -> HybridEncoder -> RT-DETRv2 decoder (registry recipe: hidden 384,
  encoder FFN 2048, 3 decoder layers, sampling points 4/4/4).
- Backbone weights come from the local DINOv3 ConvNeXt-tiny checkpoint
  (``backbone_weights``); the whole backbone trains at ``backbone_lr``
  = lr x 0.05 (the convnext family default), the encoder/decoder at lr.
- VisDrone is dense (~54 GT/image on average, 500+ in crowded frames):
  num_queries stays at the LT-DETR default of 300 so one-to-one matching
  does not cap recall.
- steps 29000 ~= 72 epochs at global batch 16 (8 per device x 2 grad
  accumulation), the RT-DETR/D-FINE baseline schedule length; EMA warmup
  2000 keeps early best-checkpoint selection responsive.
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny/backbone_state_dict.pt"
)

start = time.perf_counter()
result = train_object_detection(
    out="runs/visdrone_ltdetr_convnextt",
    data={
        "format": "coco",
        "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
                  "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
        "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
                "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
    },
    model="detr/dinov3-convnext-tiny",
    steps=29_000,
    batch_size=8,
    gradient_accumulation_steps=2,
    num_workers=8,
    model_args={
        "backbone_weights": BACKBONE_WEIGHTS,
        "image_size": (640, 640),
        "num_queries": 300,
        "backbone_lr_factor": 0.05,
        "ema_warmup_steps": 2000,
        "ema_momentum": 0.999,
    },
    trainer_args={"val_every_num_steps": 2000, "log_every_n_steps": 50},
    save_checkpoint_args={"save_every_num_steps": 4000},
)
print("\n" + "=" * 60)
print("VisDrone LT-DETR (dinov3-convnext-tiny) finished")
print("=" * 60)
print(f"steps:        {result.steps_run}")
print(f"classes:      {list(result.classes.values())}")
print(f"best ckpt:    {result.best_checkpoint_path}")
print(f"best export:  {result.best_exported_model_path}")
print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
print(f"wall time:    {time.perf_counter() - start:.0f}s")
