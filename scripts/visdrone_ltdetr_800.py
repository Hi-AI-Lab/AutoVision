# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR baseline at 800x800 — density-OFF twin of the Dome run.

Exact ablation twin of ``scripts/visdrone_ltdetr_dome_800.py``: identical
backbone (COCO-detection ConvNeXt-tiny, safetensors), decoder (dfine),
Dome-S optimizer/EMA/warmup recipe, 64,710-step schedule, and augmented
windows. The ONLY difference is the absence of ``density`` — fixed
top-``num_queries`` proposals, no DeFE / MWAS / dynamic NMS (design doc
13 §8 ablation arm a).

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_800.py
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection
from autovision._task_models.ltdetr_object_detection.transforms import (
    ltdetr_scale_jitter_sizes,
)

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
# COCO-detection-finetuned ConvNeXt-tiny backbone (LT-DETR/COCO extract),
# loaded directly from the safetensors source of truth.
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

# Same schedule as the Dome run: 160 epochs x 6471 images at global
# batch 16; heavy augs stop at Dome's epoch 108/160 boundary.
EPOCHS = 10  # 10-epoch pilot (full Dome recipe: 160)
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 16
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 4,044 @ 10 epochs
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 43,679

# Module-level so the twin-consistency test can diff the two scripts
# programmatically (the ONLY allowed difference vs the Dome script is
# the absence of "density").
MODEL_ARGS = {
    "backbone": "dinov3-convnext-tiny",
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    # Dome twin: dfine decoder, 250 fixed queries (= Dome
    # min_num_select), same denoising/reg_max recipe.
    "decoder": "dfine",
    # Dome-faithful 4-level pyramid (strides 4/8/16/32) — design doc 13 D1.
    "pyramid_levels": 4,
    "num_queries": 250,
    "num_top_queries": 500,
    "reg_max": 32,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    # NO "density" key — the ablation variable.
    # Dome optimizer recipe (identical to the Dome run).
    "lr": 2e-4,
    "backbone_lr_factor": 0.1,
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,
    "ema_momentum": 0.999,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}
# Identical augmentations to the Dome run: no mosaic / no mixup,
# photometric 0.5 / zoom-out 0.5 / IoU-crop 0.8 / hflip 0.5,
# multiscale on base 800, heavy augs stop at epoch 108/160.
TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "scale_jitter": {
        # Capped at 960: the 4-level pyramid's 1024px ladder tail peaks
        # ~20GB and brushes the allocator limit on the 24GB card.
        "sizes": [s for s in ltdetr_scale_jitter_sizes((800, 800), 32) if s <= 960],
        "step_stop": AUG_STOP_STEP,
    },
}


def main() -> None:
    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_base_800",
        data={
            "format": "coco",
            "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
                      "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
            "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
                    "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
        },
        model="ltdetr/dinov3-convnext-tiny",
        steps=STEPS,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR base 800px (dinov3-convnext-tiny, density OFF)")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
