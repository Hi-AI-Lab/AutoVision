# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone YOLO11n training via the official ultralytics API (standalone).

NOTE (license): ultralytics is AGPL-3.0 and is deliberately NOT an
AutoVision dependency — this script is a standalone comparison entry
point. Run it in the dedicated env, never inside the package:

    PYTHONPATH=/tmp/yoloshim /tmp/yoloenv/bin/python scripts/visdrone_yolo11n_800.py

Mirrors the other VisDrone twins: 10 epochs, imgsz 800, same YOLO-format
dataset layout.
"""

from __future__ import annotations

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/tmp/yolo11n.pt")  # COCO-pretrained YOLO11n
    model.train(
        data="/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone/yolo/visdrone.yaml",
        epochs=10,
        imgsz=800,
        batch=4,       # co-scheduled with another GPU job; raise to 8 when solo
        workers=4,
        device=0,
        project="runs",
        name="visdrone_yolo11n_800_full",
        exist_ok=True,
        seed=0,
    )
    metrics = model.val()
    print("=== run summary ===")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
