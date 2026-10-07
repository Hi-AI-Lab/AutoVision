# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone DEIMv2 baseline at 800x800.

Mirrors ``scripts/visdrone_rfdetr_800.py`` (same data paths, 10-epoch
pilot, effective global batch 16 = per-device 8 x accum 2 — the LR is
auto-scaled by sqrt(16/32), scale jitter locked at 800, no mosaic/mixup)
with the DEIMv2 family: HF ``Deimv2ForObjectDetection`` fine-tuned from the
verified transformers-format conversion of the official Intellindust COCO
checkpoint, with the official per-variant recipe (AdamW, backbone LR
multiplier 0.5x for HGNetv2 / 0.05x for DINOv3, wd exemptions for
norm/bias, trainable backbone BN per the official freeze_norm: False,
cosine schedule, pilot-scale ModelEMA 0.993).

Requires transformers >= 5.9 (Deimv2 classes). Run:

    .venv/bin/python scripts/visdrone_deimv2_800.py [--smoke]
    .venv/bin/python scripts/visdrone_deimv2_800.py --model deimv2-dinov3-s
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
MICRO_BATCH = 4  # batch_size below is the per-device batch: 8 x accum 2
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH  # -> effective global batch 16
STEPS = TRAIN_IMAGES * EPOCHS // GLOBAL_BATCH  # 8088
AUG_STOP_STEP = round(STEPS * 108 / 160)

MODEL_ARGS = {
    "image_size": (800, 800),
    "num_select": 500,  # dense scenes: widen eval decode (official 300)
    "gradient_clip_val": 0.1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "use_ema_model": True,
    # Pilot-scale EMA: the official 0.9999 targets 60k+ step COCO runs —
    # at 8k steps its averaging window spans the whole run and the EMA
    # weights lag training badly (0.993 ~ 143-step window, same as the
    # rf-detr pilot).
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
    parser.add_argument(
        "--model",
        default="deimv2-hgnetv2-n",
        help="DEIMv2 variant (deimv2-hgnetv2-n | deimv2-dinov3-s | "
        "deimv2-hgnetv2-pico)",
    )
    parser.add_argument(
        "--erase",
        action="store_true",
        help="box-aware background erasure (V2 constrained mode: every "
        "object keeps >= 2 patches; rf-detr-ssl-v2 design doc 8)",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny validation run")
    namespace = parser.parse_args()

    if namespace.smoke:
        steps, val_every, save_every, out_name = 60, 30, 60, "smoke"
    else:
        steps, val_every, save_every, out_name = STEPS, 1000, 2000, "full"

    transform_args = dict(TRANSFORM_ARGS)
    if namespace.erase:
        transform_args.update(
            {
                "random_erase_prob": 0.5,
                "random_erase_ratio": 0.4,
                "random_erase_interior": True,
                "random_erase_min_keep": 2,
            }
        )

    variant_tag = namespace.model.removeprefix("deimv2-")
    suffix = "-erase" if namespace.erase else ""
    result = train_object_detection(
        out=f"runs/visdrone_deimv2_{variant_tag}_800_{out_name}{suffix}",
        data=DATA_ARGS,
        model=f"deimv2/{namespace.model}",
        steps=steps,
        batch_size=GLOBAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        model_args={**MODEL_ARGS},
        transform_args=transform_args,
        trainer_args={"val_every_num_steps": val_every},
        save_checkpoint_args={"save_every_num_steps": save_every, "save_last": True},
    )
    print("best metrics:", result.best_metrics)


if __name__ == "__main__":
    main()
