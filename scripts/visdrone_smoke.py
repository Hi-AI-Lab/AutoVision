# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone smoke training: real dataset, small RT-DETR, short run.

Exercises the full M1 pipeline on real data: YOLO dataset loading ->
mmap row cache -> step-scheduled detection transforms -> RT-DETR training
with EMA -> COCO mAP validation -> best/last checkpoints + exports.
"""

import json
import sys
import time
from pathlib import Path

VISDRONE = Path("/home/hdjs/Research/Gitlab/HiDet/datasets/VisDrone/VisDrone_dataset")

data_config = {
    "format": "yolo",
    "path": str(VISDRONE),
    "train": str(VISDRONE / "images" / "train"),
    "val": str(VISDRONE / "images" / "val"),
    "names": {
        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
        5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
    },
}

model_args = {
    "backbone": "torchvision/resnet18",
    "image_size": [320, 320],
    "hidden_dim": 128,
    "num_queries": 100,
    "dec_layers": 2,
    "num_heads": 8,
    "num_top_queries": 100,
}

transform_args = {
    "scale_jitter": {"sizes": [288, 320, 352]},
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": 0.5,
    "random_iou_crop_prob": 0.5,
}

from autovision import train_object_detection

start = time.perf_counter()
result = train_object_detection(
    out="runs/visdrone_smoke",
    data=data_config,
    steps=40,
    batch_size=16,
    num_workers=4,
    model_args=model_args,
    transform_args=transform_args,
    trainer_args={"val_every_num_steps": 20},
    save_checkpoint_args={"save_every_num_steps": 20},
    overwrite=True,
)
elapsed = time.perf_counter() - start

print("\n" + "=" * 60)
print("VisDrone smoke training finished")
print("=" * 60)
print(f"steps:          {result.steps_run}")
print(f"classes:        {len(result.classes)} -> {list(result.classes.values())}")
print(f"checkpoint:     {result.checkpoint_path}")
print(f"best ckpt:      {result.best_checkpoint_path}")
print(f"exported:       {result.exported_model_path}")
print(f"best export:    {result.best_exported_model_path}")
print(f"best metrics:   {json.dumps(result.best_metrics, indent=2)}")
print(f"wall time:      {elapsed:.1f}s")

# Post-training: load the exported model and predict on a real VisDrone image.
import torch

from autovision import load_task_model

model = load_task_model(result.exported_model_path, device="cpu")
sample_image = sorted((VISDRONE / "images" / "val").glob("*.jpg"))[0]
prediction = model.predict(sample_image, threshold=0.2)
print("\npost-training inference on:", sample_image.name)
print(f"detections (score>0.2): {len(prediction['bboxes'])}")
print("SMOKE TEST PASSED")
