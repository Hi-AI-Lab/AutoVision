# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR smoke test (CPU-friendly, real data).

Two stages:

1. Full pipeline on a tiny VisDrone COCO subset with the offline
   ``dinov2-vittest-notpretrained`` variant — exercises COCO loading,
   mmap cache, LT-DETR transforms, engine, EMA, mAP, exports.
2. The real DINOv3 ConvNeXt path: build ``ltdetr/dinov3-convnext-tiny``
   offline and load the locally converted backbone weights, then run one
   training step + one inference forward on real VisDrone images.

    python scripts/visdrone_ltdetr_smoke.py
"""

import time
from pathlib import Path

import torch

from autovision import load_task_model, train_object_detection
from autovision._task_models.ltdetr_object_detection import (
    LTDETRObjectDetectionTrain,
    LTDETRObjectDetectionTrainArgs,
)

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
SMOKE_TRAIN = Path(__file__).parent.parent / "data" / "visdrone_smoke_train.json"
SMOKE_VAL = Path(__file__).parent.parent / "data" / "visdrone_smoke_val.json"
CONVNEXT_TINY_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny/backbone_state_dict.pt"
)

# ── Stage 1: full pipeline on the smoke subset ───────────────────────────

start = time.perf_counter()
result = train_object_detection(
    out="runs/visdrone_ltdetr_smoke",
    data={
        "format": "coco",
        "train": {"annotations": str(SMOKE_TRAIN),
                  "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
        "val": {"annotations": str(SMOKE_VAL),
                "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
    },
    model="ltdetr/dinov2-vittest-notpretrained",
    steps=4,
    batch_size=2,
    num_workers=0,
    model_args={
        "image_size": (56, 56),
        "num_queries": 10,
        "num_top_queries": 10,
        "hidden_dim": 64,
        "dec_layers": 2,
    },
    transform_args={"scale_jitter": None},
    trainer_args={"val_every_num_steps": 2},
    overwrite=True,
)
print(f"\n[stage 1] steps={result.steps_run} "
      f"map={result.best_metrics.get('val_metric/map')} "
      f"export={result.exported_model_path}")
assert result.steps_run == 4
assert result.best_checkpoint_path and result.best_checkpoint_path.exists()
assert "val_metric/map" in result.best_metrics

val_images = sorted((VISDRONE / "VisDrone2019-DET-val/images").glob("*.jpg"))
model = load_task_model(result.exported_model_path, device="cpu")
prediction = model.predict(val_images[0], threshold=0.05)
print(f"[stage 1] inference on {val_images[0].name}: "
      f"{len(prediction['bboxes'])} detections")
print("STAGE 1 PASSED (pipeline end-to-end)")

# ── Stage 2: real DINOv3 ConvNeXt backbone + one training step ───────────
args = LTDETRObjectDetectionTrainArgs(
    num_classes=10,
    backbone="dinov3-convnext-tiny",
    image_size=(320, 320),
    backbone_weights=CONVNEXT_TINY_WEIGHTS,
)
train_model = LTDETRObjectDetectionTrain(args)
task_model = train_model.model
print(f"\n[stage 2] decoder={task_model.decoder_name} "
      f"hidden_dim={task_model.hidden_dim} image_size={task_model.image_size}")

# The local checkpoint must load with every key consumed.
missing_before = set(task_model.backbone.backbone_model.state_dict().keys())
loaded = task_model.backbone.backbone_model.load_state_dict(
    torch.load(CONVNEXT_TINY_WEIGHTS, map_location="cpu", weights_only=True),
    strict=True,
)
assert not loaded.missing_keys and not loaded.unexpected_keys
print("[stage 2] backbone weights loaded strict "
      f"({len(missing_before)} tensors)")

batch = {
    "image": torch.rand(2, 3, 320, 320),
    "bboxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
               torch.tensor([[0.3, 0.6, 0.1, 0.1], [0.7, 0.2, 0.15, 0.15]])],
    "classes": [torch.tensor([3]), torch.tensor([0, 9])],
}
train_model.train()
step_result = train_model.training_step(batch, step=0)
step_result.loss.backward()
print(f"[stage 2] training loss {float(step_result.loss):.4f} "
      f"(backward OK)")

task_model.eval()
with torch.no_grad():
    logits, boxes = task_model(batch["image"][:1])
decoded = task_model.postprocess(
    (logits, boxes), [{"orig_w": 1920, "orig_h": 1080}], threshold=0.05
)
print(f"[stage 2] inference: {len(decoded[0]['scores'])} queries decoded")
print("STAGE 2 PASSED (dinov3-convnext-tiny + local weights)")
print(f"\nwall time: {time.perf_counter() - start:.0f}s")
