# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""VisDrone LT-DETR at 1280x1280 — DINOv3 ViT-S/16, COCO-full warm start.

ViT-family twin of ``visdrone_ltdetr_1280_convnextb_baseline_coco.py`` (NO
box-aware eraser). Same resolution / batch / schedule / aug recipe; the
BACKBONE FAMILY is the variable (ConvNeXt-base -> DINOv3 ViT-S/16).

Checkpoint ``dinov3_vits16_ltdetr_coco_251218_4812416b.pt`` mandates a
lighter head than the ConvNeXt checkpoints (verified from its tensors):

- encoder width 224 (enc linear1 (896, 224)) AND decoder width 224
  (self_attn in_proj (672, 224), ffn (1792, 224)) — the ViT recipe keeps
  the two SHARED, unlike the ConvNeXt baseline's 384/256 split;
- 4 decoder layers (0..3) instead of ConvNeXt's 6;
- ViT backbone goes through the STA wrapper (windowed attention +
  conv_inplane=32 channel budget): 42 extras tensors (backbone.convs /
  backbone.norms) load beside the 188 raw ViT tensors;
- 3-level pyramid from interaction_indexes (5, 8, 11) — the ViT layers
  are grouped into thirds, one per stride (8/16/32);
- image_size 1280 is divisible by the 2 x patch divisor (2 x 16 = 32).

Optional flags:

- ``--token-drop``  in-backbone background token dropping (design docs
  17/20: window 32px = 2x2 patches @16, keep_ratio 0.4, step_stop at the
  Dome aug-stop cadence). OFF by default — with it, this run is the A/B
  twin of the plain 1280 vits16 baseline.

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup .venv/bin/python scripts/visdrone_ltdetr_1280_vits16_baseline_coco.py \
        > runs/visdrone_ltdetr_vits16_1280_cocofull_console.log 2>&1 &

    # token-drop twin (A/B):
    .venv/bin/python scripts/visdrone_ltdetr_1280_vits16_baseline_coco.py --token-drop
"""

import argparse
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
#: The SHA256-verified published DINOv3 ViT-S/16 detector checkpoint
#: (backbone + STA extras + encoder + decoder, 80 COCO classes).
DETECTOR_CHECKPOINT = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/dinov3_vits16_ltdetr_coco_251218_4812416b.pt"
)

IMAGE_SIZE = (1280, 1280)

# 50-epoch horizon — micro-batch 4 x accum 2 keeps the global batch at 8.
# ViT-S/16 (~22M backbone) is far lighter than ConvNeXt-base, so batch 4
# should fit; if it OOMs fall back to batch 2 x accum 4.
EPOCHS = 50
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
MICRO_BATCH = 4
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 40,443
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 27,299

MODEL_ARGS = {
    "backbone": "dinov3-vits16",
    "backbone_pretrained": False,       # backbone comes from the checkpoint
    "detector_checkpoint": DETECTOR_CHECKPOINT,
    "image_size": IMAGE_SIZE,
    # Checkpoint-mandated architecture — verified from the checkpoint
    # tensors (see module docstring): shared 224 width, 4 decoder layers.
    "decoder": "rtdetrv2",
    "dec_layers": 4,
    "pyramid_levels": 3,
    "hidden_dim": 224,                  # encoder width (checkpoint)
    "decoder_hidden_dim": 224,          # decoder width (checkpoint, shared)
    "num_queries": 300,
    "num_top_queries": 300,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    "reg_max": 32,
    "reg_scale": 4.0,
    # Official eval protocol for the dense benchmark.
    "eval_max_dets": 500,
    # Pilot LR recipe, identical to the ConvNeXt 1280/800 runs (1.4e-4 ->
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-drop",
        action="store_true",
        help="in-backbone background token dropping (design docs 17/20: "
        "window-granularity keep mask from the GT boxes; ViT-v3 masked "
        "forward per doc 20). OFF by default.",
    )
    args = parser.parse_args()
    model_args = dict(MODEL_ARGS)
    if args.token_drop:
        # Window 32px = 2x2 patches at patch 16; 1280 % 32 == 0. The stop
        # follows the Dome aug-stop cadence (docs 16 P1.5 / 17 §4.4).
        model_args["token_drop"] = {
            "window_size": 32,
            "keep_ratio": 0.4,
            "step_stop": AUG_STOP_STEP,
        }

    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_vits16_1280_cocofull",
        data=DATA_ARGS,
        model="detr/dinov3-vits16",
        steps=STEPS,
        batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_workers=8,
        model_args=model_args,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print(f"VisDrone LT-DETR 1280px ViT-S/16 {EPOCHS} epochs "
          f"(COCO FULL warm start, global batch {GLOBAL_BATCH}, "
          f"token_drop={args.token_drop})")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
