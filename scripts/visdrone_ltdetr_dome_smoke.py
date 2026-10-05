# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Short end-to-end smoke for the Dome 800px run (real data, full engine).

Same model/config as ``visdrone_ltdetr_dome_800.py`` (4-level pyramid,
COCO backbone, density branch) but compressed: 60 steps, one validation
at step 30, checkpoint at step 30. Validates the full engine loop —
safetensors load, dataloader/augmentation windows, defe losses in the
metrics stream, dome query diagnostics on val, checkpoint round-trip.

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_dome_smoke.py
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection
from autovision._task_models.ltdetr_object_detection.transforms import (
    ltdetr_scale_jitter_sizes,
)

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

STEPS = 60
AUG_STOP_STEP = 45  # exercise the aug-stop switch inside the smoke window
SCALE_JITTER_CAP = 960


def main() -> None:
    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_dome_800_smoke",
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
        gradient_accumulation_steps=1,
        num_workers=8,
        model_args={
            "backbone_weights": BACKBONE_WEIGHTS,
            "image_size": (800, 800),
            "decoder": "dfine",
            "num_queries": 250,
            "num_top_queries": 500,
            "reg_max": 32,
            "num_denoising": 100,
            "pyramid_levels": 4,
            "density": {"max_num_select": 500},
            "lr": 2e-4,
            "backbone_lr_factor": 0.1,
            "weight_decay": 6.5e-5,
            "lr_warmup_steps": 10,
            "ema_momentum": 0.9999,
            "ema_warmup_steps": 10,
            "eval_max_dets": 500,
        },
        transform_args={
            "mosaic_prob": None,
            "mixup_prob": None,
            "photometric_step_stop": AUG_STOP_STEP,
            "zoom_out_step_stop": AUG_STOP_STEP,
            "iou_crop_step_stop": AUG_STOP_STEP,
            "scale_jitter": {
                "sizes": [
                    s for s in ltdetr_scale_jitter_sizes((800, 800), 32)
                    if s <= SCALE_JITTER_CAP
                ],
                "step_stop": AUG_STOP_STEP,
            },
        },
        trainer_args={"val_every_num_steps": 30, "log_every_n_steps": 10},
        save_checkpoint_args={"save_every_num_steps": 30},
    )
    print("\n" + "=" * 60)
    print("Dome 800px smoke finished")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
