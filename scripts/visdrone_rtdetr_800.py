# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone RT-DETR baseline at 800x800.

Twin of ``scripts/visdrone_rfdetr_800.py`` for family comparison: same
data paths, 10-epoch pilot, global batch 8, scale jitter locked at 800,
no mosaic/mixup — with the RT-DETR family's own recipe (AdamW 5e-4
sqrt-scaled, torchvision resnet18 pretrained backbone, flat-cosine
windows, EMA 0.9999).

Run:  .venv/bin/python scripts/visdrone_rtdetr_800.py [--smoke]
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
STEPS = TRAIN_IMAGES * EPOCHS // GLOBAL_BATCH  # 8088

MODEL_ARGS = {
    "image_size": (800, 800),
    "backbone_pretrained": True,
    "num_queries": 300,  # match the rfdetr twin's query width
    "num_top_queries": 500,  # dense scenes: widen eval decode
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "scale_jitter": {"sizes": [800]},
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
        out=f"runs/visdrone_rtdetr_resnet18_800_{out_name}",
        data=DATA_ARGS,
        model="rtdetr/resnet18",
        steps=steps,
        batch_size=GLOBAL_BATCH,
        gradient_accumulation_steps=1,
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
