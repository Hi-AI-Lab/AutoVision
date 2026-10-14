# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR base pilot on the OFFICIAL 12-class COCO export.

Identical model/recipe to ``scripts/visdrone_ltdetr_800.py`` (imported
from it) — the ONLY difference is the data: the 12-class COCO export
produced by ``scripts/export_visdrone_coco12.py`` (ten objects +
``ignored regions`` + ``others``, matching the official Dome-S
``annotations_coco`` layout). ``num_classes`` is inferred from the
annotations (12), making headline mAP comparable with the official log.

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_800_coco12.py
"""

import importlib.util
import json
import time
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone"
)

_spec = importlib.util.spec_from_file_location(
    "visdrone_ltdetr_800",
    Path(__file__).resolve().parent / "visdrone_ltdetr_800.py",
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

DATA_ARGS = {
    "format": "coco",
    "train": {
        "annotations": str(VISDRONE / "annotations_coco12/visdrone_train.json"),
        "images": str(VISDRONE / "VisDrone2019-DET-train/images"),
    },
    "val": {
        "annotations": str(VISDRONE / "annotations_coco12/visdrone_val.json"),
        "images": str(VISDRONE / "VisDrone2019-DET-val/images"),
    },
}


def main() -> None:
    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_base_800_coco12",
        data=DATA_ARGS,
        model="avdetr/dinov3-convnext-tiny",
        steps=base.STEPS,
        batch_size=8,
        gradient_accumulation_steps=1,
        num_workers=8,
        model_args=base.MODEL_ARGS,
        transform_args=base.TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 4000},
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR base 800px — OFFICIAL 12-class eval arm")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(
        "best metrics: "
        + json.dumps(
            {k: round(v, 4) for k, v in (result.best_metrics or {}).items()}
        )
    )
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
