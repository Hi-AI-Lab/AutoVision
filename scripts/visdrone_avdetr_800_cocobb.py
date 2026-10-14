# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR 800px, COCO-finetuned backbone (Dome-sibling起点对齐对照).

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_800.py

800x800 derivation (from the ltdetr module's size rules):
- Divisibility: image_size must be square and divisible by 2 x patch_size
  = 32 for the ConvNeXt family; 800 = 32 x 25.
- Scale-jitter ladder (family recipe, auto): multiples of 32 covering
  +/-25% of 800 -> 14 distinct sizes 576..1024, then the 800 base fills
  the list to 30 entries (base keeps ~53% sampling weight). Worst-case
  batch is 1024px.
- Memory: the 640 run peaked at 17.3GB with batch 8 (ladder to 800).
  Scaling activations per image, batch 4 at the 1024 ladder tail lands
  around 21GB — fits the 24GB card; batch 6 would need ~29GB and OOMs.
  Hence batch 4 x grad accumulation 4 = global 16, which also keeps the
  LR scaling factor at exactly 1.0 (same optimizer trajectory as 640).
- Steps: 404 steps/epoch (6471 images / 16) x 72 epochs ~= 29000, the
  same schedule as the 640 run — resolution and microbatch shape are the
  only changes, giving a clean comparison.
- This run also picks up the engine optimizations (lazy loss logging,
  foreach EMA, TF32) merged after the 640 run launched.
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/backbone_state_dict.pt"
)

start = time.perf_counter()
result = train_object_detection(
    out="runs/visdrone_ltdetr_convnextt_800_cocobb",
    data={
        "format": "coco",
        "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
                  "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
        "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
                "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
    },
    model="avdetr/dinov3-convnext-tiny",
    steps=29_000,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_workers=8,
    model_args={
        "backbone_weights": BACKBONE_WEIGHTS,
        "image_size": (800, 800),
        "num_queries": 300,
        "backbone_lr_factor": 0.05,
        "ema_warmup_steps": 2000,
        "ema_momentum": 0.999,
        # VisDrone official eval protocol (matches the Dome-family logs'
        # maxDets=500); dense frames hold >100 objects.
        "eval_max_dets": 500,
    },
    trainer_args={"val_every_num_steps": 2000, "log_every_n_steps": 500},
    save_checkpoint_args={"save_every_num_steps": 4000},
)
print("\n" + "=" * 60)
print("VisDrone LT-DETR 800px (dinov3-convnext-tiny) finished")
print("=" * 60)
print(f"steps:        {result.steps_run}")
print(f"best ckpt:    {result.best_checkpoint_path}")
print(f"best export:  {result.best_exported_model_path}")
print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
print(f"wall time:    {time.perf_counter() - start:.0f}s")
