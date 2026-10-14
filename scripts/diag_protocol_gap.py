# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""One-off: measure the evaluation-protocol gap on the 50ep rfdetr export.

The rf-detr-ssl-v2 anchor (0.2974 @39ep) was measured under a protocol that
DELETES predictions falling inside VisDrone ignored regions; our pipeline
evaluates every prediction against the full 10-class GT. This script
quantifies that difference three ways on the same predictions:

  A. current protocol          - all 10-class GT, no ignore handling
  B. COCO crowd protocol       - ignored regions added as iscrowd=1 GT
                                 (detections matching a crowd are ignored)
  C. hard-delete protocol      - predictions whose center falls inside an
                                 ignored region are dropped (Dome style)

Plus a finer size breakdown (AP for <16px / 16-32px / 32-64px / >=64px
GT side length) on protocol A.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from autovision.tasks.detection.rfdetr.task_model import (
    RFDETRObjectDetection,
)
from autovision.tasks.task_model_helpers import _safe_load_export

VISDRONE = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone"
)
EXPORT = Path(
    "/home/hdjs/Research/Project/AutoVision/runs/"
    "visdrone_rfdetr_nano_800_full-erase-siou-50ep/exported_models/exported_best.pt"
)
#: VisDrone DET annotation categories outside the 10 evaluated classes.
IGNORE_CATEGORIES = {0, 11}


def load_best_model() -> RFDETRObjectDetection:
    model = RFDETRObjectDetection(
        num_classes=10,
        model="rf-detr-nano",
        image_size=(800, 800),
        num_queries=300,
        num_select=500,
        load_weights=True,
        box_regression_iou="siou",
    )
    export = _safe_load_export(EXPORT)
    state_dict = {
        key.removeprefix("ema_model."): value
        for key, value in export["train_model"].items()
    }
    renames = (
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
    )
    remapped = {}
    for key, value in state_dict.items():
        for old, new in renames:
            if old in key:
                key = key.replace(old, new)
                break
        remapped[key] = value
    model.load_state_dict(remapped, strict=True)
    return model.to("cuda").eval()


def load_ignore_regions() -> dict[str, list[tuple[int, int, int, int]]]:
    """Original VisDrone DET annotations -> per-image ignored-region boxes."""
    regions: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    ann_dir = VISDRONE / "VisDrone2019-DET-val" / "annotations"
    for txt in sorted(ann_dir.glob("*.txt")):
        stem = txt.stem
        for line in txt.read_text().splitlines():
            parts = line.split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, _score, category = parts[:6]
            if int(category) in IGNORE_CATEGORIES:
                regions[stem].append(
                    (int(x), int(y), int(w), int(h))
                )
    return regions


def main() -> None:
    ann_path = VISDRONE / "annotations" / "visdrone_val.json"
    coco_gt = COCO(str(ann_path))
    images = coco_gt.dataset["images"]
    stem_to_id = {Path(img["file_name"]).stem: img["id"] for img in images}

    model = load_best_model()
    ignore_regions = load_ignore_regions()
    print(f"images with ignored regions: {len(ignore_regions)}")

    img_dir = VISDRONE / "VisDrone2019-DET-val" / "images"
    results = []
    with torch.no_grad():
        for info in tqdm(sorted(images, key=lambda i: i["id"]), desc="val"):
            tensor, meta = model.preprocess_image(img_dir / info["file_name"])
            raw = model.forward_backend(
                model.preprocess_batch(tensor.unsqueeze(0).to("cuda"))
            )
            pred = model.postprocess(raw, [meta], threshold=0.0)[0]
            for box, score, label in zip(
                pred["bboxes"].tolist(),
                pred["scores"].tolist(),
                pred["labels"].tolist(),
            ):
                x1, y1, x2, y2 = box
                results.append(
                    {
                        "image_id": info["id"],
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                    }
                )
    print(f"predictions: {len(results)}")

    def summarize(evaluator: COCOeval, tag: str) -> None:
        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        aind = params.areaRngLbl.index("all")
        mind = len(params.maxDets) - 1
        precision = evaluator.eval["precision"]

        def mean_of(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        ap = mean_of(precision[tind:, :, :, aind, mind])
        ap50 = mean_of(precision[tind, :, :, aind, mind])
        print(
            f"{tag:28s} AP {ap:.4f}  AP50 {ap50:.4f}"
        )
        return ap

    # Custom 6-band areaRng so one evaluate() yields the fine size
    # breakdown alongside the headline; protocol B/C reuse the default.
    size_bands = [
        ("all", 0.0, float(1e10)),
        ("tiny<16px", 0.0, 16.0**2),
        ("xs 16-32", 16.0**2, 32.0**2),
        ("s  32-64", 32.0**2, 64.0**2),
        ("m  64-96", 64.0**2, 96.0**2),
        ("l  >=96px", 96.0**2, float(1e10)),
    ]

    def run_eval(gt_dataset, dt_list, tag: str) -> float:
        gt = COCO()
        gt.dataset = gt_dataset
        gt.createIndex()
        dt = gt.loadRes(dt_list)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, 500]
        evaluator.params.areaRng = [[lo, hi] for _, lo, hi in size_bands]
        evaluator.params.areaRngLbl = [label for label, _, _ in size_bands]
        evaluator.evaluate()
        evaluator.accumulate()

        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        mind = len(params.maxDets) - 1
        precision = evaluator.eval["precision"]

        def mean_of(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        for label, _, _ in size_bands:
            aind = params.areaRngLbl.index(label)
            ap = mean_of(precision[tind:, :, :, aind, mind])
            print(f"  [{tag}] {label:9s} AP {ap:.4f}")
        aind = params.areaRngLbl.index("all")
        ap50 = mean_of(precision[tind, :, :, aind, mind])
        print(f"  [{tag}] headline  AP50 {ap50:.4f}")
        return mean_of(precision[tind:, :, :, aind, mind])

    print("A. current protocol (all 10-class GT):")
    ap_a = run_eval(coco_gt.dataset, [
        {k: v for k, v in r.items() if not k.startswith(("cx", "cy"))}
        for r in results
    ], "A")

    # ── B. COCO crowd protocol: ignored regions as iscrowd GT ──────────
    crowd_payload = json.loads(json.dumps(coco_gt.dataset))
    max_ann = max(a["id"] for a in crowd_payload["annotations"])
    next_id = max_ann + 1
    added = 0
    for stem, regions in ignore_regions.items():
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        for x, y, w, h in regions:
            crowd_payload["annotations"].append(
                {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": 0,  # any valid cat; iscrowd exempts it
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 1,
                    "ignore": 1,
                }
            )
            next_id += 1
            added += 1
    print(f"B. crowd protocol (+{added} ignored-region GT):")
    ap_b = run_eval(crowd_payload, [
        {k: v for k, v in r.items() if not k.startswith(("cx", "cy"))}
        for r in results
    ], "B")

    # ── C. hard-delete: drop predictions with center inside a region ───
    id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in images}
    kept = []
    for r in results:
        dropped = any(
            x <= r["cx"] <= x + w and y <= r["cy"] <= y + h
            for x, y, w, h in ignore_regions.get(id_to_stem[r["image_id"]], ())
        )
        if not dropped:
            kept.append(
                {k: v for k, v in r.items() if not k.startswith(("cx", "cy"))}
            )
    print(f"C. hard-delete (kept {len(kept)}/{len(results)}):")
    ap_c = run_eval(coco_gt.dataset, kept, "C")

    print("\n=== protocol gap vs rf-detr-ssl-v2 anchor (0.2974) ===")
    print(f"current {ap_a:.4f} | crowd {ap_b:+.4f} | hard-delete {ap_c:+.4f}")


if __name__ == "__main__":
    main()
