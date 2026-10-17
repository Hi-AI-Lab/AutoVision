# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""VisDrone LT-DETR at 1280x1280 — ConvNeXt-LARGE, COCO-full warm start.

Backbone-upgrade twin of ``visdrone_ltdetr_1280_convnextb_baseline_coco.py``
(dinov3-convnext-base -> dinov3-convnext-large, NO box-aware eraser).

Checkpoint ``dinov3_convnext_large_ltdetr_coco_251218_03fe6750.pt`` was
verified to mandate the SAME head architecture as the base sibling:
encoder width 384 / decoder width 256 / 6 rtdetrv2 decoder layers /
300 queries (read off the checkpoint tensors: encoder input_proj 1x1 convs
(384, 384/768/1536) — the latter are exactly ConvNeXt-Large's stage-1..3
widths — and dec self_attn in_proj (768, 256), dec layers 0..5). So
MODEL_ARGS keeps the base script's shapes verbatim.

Resolution / batch handling:

- ``image_size`` (1280, 1280); ``scale_jitter`` pinned to ``[1280]`` (the
  collate samples train resolution from ``sizes``, not ``image_size``).
- ConvNeXt-large is ~2.2x the base backbone (198M vs 89M params). Base at
  micro-batch 2 already leans on the 24 GB card, so large runs micro-batch
  1 x accum 8 — the GLOBAL batch stays at 8 (LR recipe, 40,443 steps,
  AUG_STOP_STEP 27299 and LR milestones unchanged).

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup .venv/bin/python scripts/visdrone_ltdetr_1280_convnextl_baseline_coco.py \
        > runs/visdrone_ltdetr_convnextl_1280_cocofull_console.log 2>&1 &
"""

import time
from pathlib import Path

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
DATA_ARGS = {
    "format": "coco",
    "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
              "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
    "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
            "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
}
#: The SHA-verified published ConvNeXt-LARGE detector checkpoint
#: (backbone + encoder + decoder, 80 COCO classes).
DETECTOR_CHECKPOINT = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/dinov3_convnext_large_ltdetr_coco_251218_03fe6750.pt"
)

IMAGE_SIZE = (1280, 1280)

# 50-epoch horizon — micro-batch 1 x accum 8 keeps the global batch at 8,
# so STEPS / AUG_STOP_STEP / LR milestones match the tiny/base runs exactly.
EPOCHS = 50
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
MICRO_BATCH = 1
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 40,443
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 27,299

MODEL_ARGS = {
    "backbone": "dinov3-convnext-large",
    "backbone_pretrained": False,       # backbone comes from the checkpoint
    "detector_checkpoint": DETECTOR_CHECKPOINT,
    "image_size": IMAGE_SIZE,
    # Checkpoint-mandated architecture — verified identical to the base
    # baseline shapes (see module docstring).
    "decoder": "rtdetrv2",
    "dec_layers": 6,
    "pyramid_levels": 3,
    "hidden_dim": 384,                  # encoder width (checkpoint)
    "decoder_hidden_dim": 256,          # decoder width (checkpoint)
    "num_queries": 300,
    "num_top_queries": 300,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    "reg_max": 32,
    "reg_scale": 4.0,
    # Official eval protocol for the dense benchmark.
    "eval_max_dets": 500,
    # Pilot LR recipe, identical to the tiny/base 1280 runs (1.4e-4 ->
    # 9.9e-5 effective at global batch 8, backbone x0.1).
    "lr": 1.4e-4,
    "lr_reference_batch": 16,
    "backbone_lr_factor": 0.1,
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,
    "lr_scheduler": "multistep",
    "lr_milestones": (0.5, 0.75),
    "lr_gamma": 0.8,
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "photometric_step_start": 0,
    "zoom_out_step_start": 0,
    "iou_crop_step_start": 0,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "random_erase_step_stop": AUG_STOP_STEP,
    # Train resolution comes from THIS list (collate samples it), not from
    # image_size — keep it in sync with IMAGE_SIZE.
    "scale_jitter": {"sizes": [1280], "step_stop": AUG_STOP_STEP},
}


def main() -> None:
    from autovision import train_object_detection

    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextl_1280_cocofull",
        data=DATA_ARGS,
        model="detr/dinov3-convnext-large",
        steps=STEPS,
        batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print(f"VisDrone LT-DETR 1280px ConvNeXt-LARGE {EPOCHS} epochs "
          f"(COCO FULL warm start, global batch {GLOBAL_BATCH})")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
