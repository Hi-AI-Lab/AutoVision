# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone 10-epoch smoke pretrain for the RF-DETR package (run on GPU 0).

Run with the transformers>=5.9 overlay env:
    PYTHONPATH=/tmp/torchshim:src /tmp/rf590env/bin/python scripts/visdrone_rfdetr_pretrain_10ep.py
"""

from pathlib import Path

from autovision import pretrain

VISDRONE_TRAIN = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone/"
    "VisDrone2019-DET-train/images"
)

result = pretrain(
    out=Path("output/rfdetr_nano_visdrone_10ep"),
    data=VISDRONE_TRAIN,
    model="rfdetr/rf-detr-nano",  # pretrained Roboflow checkpoint, verified download
    method="distillation",  # DistillationV3, teacher: torchvision/resnet18 (cached)
    epochs=10,
    batch_size=128,
    devices=1,
    num_workers=8,
    overwrite=True,
)

print("=== run summary ===")
print("out_dir:", result.out_dir)
print("epochs_run:", result.epochs_run)
print("global_step:", result.global_step)
print("checkpoint:", result.checkpoint_path)
print("exported:", result.exported_model_path)
if result.metrics:
    print("metrics:", result.metrics)
