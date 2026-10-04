# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone convergence v6: RT-DETR-R18 baseline shape, 72-epoch schedule.

Launch with the GPU-enabled environment (the base miniconda python ships a
CPU-only torch build and runs ~25x slower):

    conda activate hidet && cd /home/hdjs/Research/Gitlab/autovision && \
    PYTHONPATH=src python scripts/visdrone_convergence_v6.py

Recipe rationale (review of runs/visdrone_convergence_v5, whose 30-epoch /
hidden-128 / 100-query config plateaued at mAP ~0.02 by epoch 17):
- steps 29000  = 72 epochs (RT-DETRv2/D-FINE baseline schedule length).
- num_queries 300: VisDrone averages ~54 GT/image (500+ in dense frames);
  100 queries one-to-one match caps recall at a fraction of the objects.
- hidden_dim 256 + 3 decoder layers: RT-DETR-R18 baseline capacity.
- backbone_lr_factor 0.1: baseline value (v5 ran 0.05).
- ema_warmup_steps 2000: early validations track the raw model instead of a
  history-dominated average, so best-checkpoint selection reacts faster.
"""

import json
import time
from pathlib import Path

VISDRONE = Path("/home/hdjs/Research/Gitlab/HiDet/datasets/VisDrone/VisDrone_dataset")

from autovision import train_object_detection

start = time.perf_counter()
result = train_object_detection(
    out="runs/visdrone_convergence_v6",
    data={"format": "yolo", "path": str(VISDRONE),
          "train": str(VISDRONE / "images" / "train"),
          "val": str(VISDRONE / "images" / "val"),
          "names": {i: n for i, n in enumerate(
              ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"])}},
    steps=29_000,
    batch_size=8,
    gradient_accumulation_steps=2,
    num_workers=8,
    model_args={"backbone": "torchvision/resnet18", "backbone_pretrained": True,
                "image_size": [640, 640], "hidden_dim": 256,
                "num_queries": 300, "dec_layers": 3, "num_heads": 8,
                "backbone_lr_factor": 0.1,
                "ema_warmup_steps": 2000, "ema_momentum": 0.999},
    transform_args={"scale_jitter": {"sizes": [576, 608, 640, 672, 704]}},
    trainer_args={"val_every_num_steps": 2000, "log_every_n_steps": 50},
    save_checkpoint_args={"save_every_num_steps": 4000},
)
print("\n" + "=" * 60)
print("finished")
print("=" * 60)
print(f"steps:        {result.steps_run}")
print(f"best ckpt:    {result.best_checkpoint_path}")
print(f"best export:  {result.best_exported_model_path}")
print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
print(f"wall time:    {time.perf_counter() - start:.0f}s")
