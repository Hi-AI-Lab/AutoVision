# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR at 1280x1280 — COCO-full warm start + box-aware eraser.

1280 twin of ``visdrone_ltdetr_800_baseline_coco_erase.py`` (which is itself a
bit-identical copy of ``visdrone_ltdetr_800_baseline_coco.py`` with the
box-aware eraser, design doc 16, switched ON in TRANSFORM_ARGS):

    random_erase_prob 0.5 / ratio 0.4 / interior=True / min_keep=2,
    step_stop AUG_STOP_STEP (same Dome aug-stop cadence as every other aug).

Resolution changes vs the 800 script (everything else untouched):

- ``image_size`` (800, 800) -> (1280, 1280). Square and divisible by the
  2 x patch divisor (8 for ConvNeXt), so the patch grid stays integral.
- ``scale_jitter`` pinned to ``[1280]`` (single-size ladder, same shape as
  the 800 script's ``[800]``). NOTE: the collate samples the train
  resolution from ``sizes`` (collate.py), not from ``image_size`` — leaving
  ``[800]`` here would silently train at 800 until AUG_STOP_STEP and only
  then switch to 1280.
- ``batch_size`` 8 -> 4 with ``gradient_accumulation_steps`` 1 -> 2 so the
  GLOBAL batch stays 8: LR recipe (1.4e-4 @ ref batch 16), step math
  (40,443), AUG_STOP_STEP and the LR milestones are all unchanged.

Everything else — COCO FULL detector checkpoint (backbone + encoder +
decoder), checkpoint-mandated baseline architecture (rtdetrv2 6 layers,
encoder 384 / decoder 256, 300 queries), LR/EMA/aug/schedule recipe, steps —
is untouched. Val stays erase-free (the loader clears random_erase_prob).

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup .venv/bin/python scripts/visdrone_ltdetr_1280_baseline_coco_erase.py \
        > runs/visdrone_ltdetr_convnextt_1280_cocofull_erase_console.log 2>&1 &
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
#: The SHA256-verified full published checkpoint (backbone + encoder +
#: decoder). Same file as /tmp/lightly_ckpts/... copied into the weight dir.
DETECTOR_CHECKPOINT = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/dinov3_convnext_tiny_ltdetr_coco_full.pt"
)

IMAGE_SIZE = (1280, 1280)

# 50-epoch horizon — micro-batch 4 x accum 2 keeps the global batch at 8,
# so STEPS / AUG_STOP_STEP / LR milestones match the 800 runs exactly.
EPOCHS = 50
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
MICRO_BATCH = 4
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 40,443
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 27,299

MODEL_ARGS = {
    "backbone": "dinov3-convnext-tiny",
    "backbone_pretrained": False,       # backbone comes from the checkpoint
    "detector_checkpoint": DETECTOR_CHECKPOINT,
    "image_size": IMAGE_SIZE,
    # Checkpoint-mandated architecture (baseline shapes).
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
    # Pilot LR recipe, identical to the 800 runs (1.4e-4 -> 9.9e-5 effective
    # at global batch 8).
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
    # Box-aware background eraser, doc 16 (design doc 16 / rf-detr-ssl-v2
    # doc 8): prob 0.5 / ratio 0.4, constrained-interior mode (every object
    # keeps >= 2 patches). Val stays erase-free (the loader clears
    # random_erase_prob). The stop follows the Dome aug-stop cadence
    # (doc 16 P1.5).
    "random_erase_prob": 0.5,
    "random_erase_ratio": 0.4,
    "random_erase_interior": True,
    "random_erase_min_keep": 2,
    "random_erase_step_stop": AUG_STOP_STEP,
    # Train resolution comes from THIS list (collate samples it), not from
    # image_size — keep it in sync with IMAGE_SIZE. Single size, same shape
    # as the 800 script.
    "scale_jitter": {"sizes": [1280], "step_stop": AUG_STOP_STEP},
}


def main() -> None:
    from autovision import train_object_detection

    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_1280_cocofull_erase",
        data=DATA_ARGS,
        model="detr/dinov3-convnext-tiny",
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
    print(f"VisDrone LT-DETR 1280px {EPOCHS} epochs "
          f"(COCO FULL warm start + box-aware eraser, global batch {GLOBAL_BATCH})")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
