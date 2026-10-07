# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone 10-epoch pretrain for RF-DETR at image_size=800 (run on GPU 0).

Blur sigmas are fractions of the image size, so they are rescaled by
224/800 to keep the same physical blur strength as the 224px recipe
(sigma cap 0.1 x 224 = ~22px); at native 800 fractions the blur kernel
would reach ~480 taps and cost seconds per image on CPU.

Run with the transformers>=5.9 overlay env:
    PYTHONPATH=/tmp/torchshim:src /tmp/rf590env/bin/python scripts/visdrone_rfdetr_pretrain_10ep_800.py
"""

from pathlib import Path

from autovision import pretrain

VISDRONE_TRAIN = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone/"
    "VisDrone2019-DET-train/images"
)

result = pretrain(
    out=Path("output/rfdetr_nano_visdrone_10ep_800"),
    data=VISDRONE_TRAIN,
    model="rfdetr/rf-detr-nano",  # pretrained Roboflow checkpoint, verified download
    method="distillation",  # DistillationV3, teacher: torchvision/resnet18 (cached)
    epochs=10,
    batch_size=32,  # 800^2 = 2500 tokens/image through the ViT backbone
    devices=1,
    num_workers=10,
    transform_args={
        "image_size": (800, 800),  # pydantic tuple (list would fail validation)
        "gaussian_blur": {"sigmas": (0.0, 0.028)},  # same ~22px cap as 224px run
    },
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
