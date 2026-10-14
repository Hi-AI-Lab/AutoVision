# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Offline diagnostic eval for the 50ep RF-DETR VisDrone run (one-off).

Loads the exported best checkpoint, runs the val split through the standard
decode (threshold 0, top-k 500, maxDets 500) and reports COCO AP with the
size breakdown (APs/APm/APl) plus per-class AP and GT/prediction size
distributions — the data needed to locate where mAP is lost.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

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


def load_best_model() -> RFDETRObjectDetection:
    """Build at the training config and load the exported EMA weights.

    ``load_task_model`` cannot rebuild rfdetr exports yet: it drops
    ``load_weights``, lands in the hermetic branch (224px backbone, 100
    queries) and the shapes mismatch the 800px/300-query export. Build
    with pretrained config geometry instead, then swap in the weights.
    """
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
    # The run trained under transformers 5.9 (overlay env), whose DINOv2
    # attention uses the nested naming; 5.19 renamed it. Remap the four
    # renamed tensors per layer (everything else is layout-stable).
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


def main() -> None:
    ann = VISDRONE / "annotations" / "visdrone_val.json"
    coco_gt = COCO(str(ann))
    images = {img["id"]: img for img in coco_gt.dataset["images"]}
    img_dir = VISDRONE / "VisDrone2019-DET-val" / "images"

    model = load_best_model()
    model.eval()

    results = []
    with torch.no_grad():
        for image_id, info in tqdm(sorted(images.items()), desc="val"):
            tensor, meta = model.preprocess_image(img_dir / info["file_name"])
            raw = model.forward_backend(
                model.preprocess_batch(tensor.unsqueeze(0).to("cuda"))
            )
            pred = model.postprocess(raw, [meta], threshold=0.0)[0]
            boxes = pred["bboxes"]
            if len(boxes) == 0:
                continue
            scores = pred["scores"]
            labels = pred["labels"]
            for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                x1, y1, x2, y2 = box
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    print(f"predictions: {len(results)} across {len(images)} images")

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.evaluate()
    evaluator.accumulate()
    # summarize() is skipped on purpose: its headline stats are pinned to
    # maxDets=100 and return -1 under any other ladder. Headline numbers
    # come straight from the precision/recall arrays, exactly like the
    # repo's COCOMAPEvaluator (identical formula => comparable numbers).
    import numpy as np

    params = evaluator.params
    tind = int(np.where(params.iouThrs == 0.5)[0][0])
    tind75 = int(np.where(params.iouThrs == 0.75)[0][0])
    aind = params.areaRngLbl.index("all")
    mind = len(params.maxDets) - 1
    precision = evaluator.eval["precision"]  # [T, R, K, A, M]
    recall = evaluator.eval["recall"]  # [T, K, A, M]

    def _mean(values) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    print("=== headline (COCOMAPEvaluator formula) ===")
    print(f"map    = {_mean(precision[tind:, :, :, aind, mind]):.4f}")
    print(f"map_50 = {_mean(precision[tind, :, :, aind, mind]):.4f}")
    print(f"map_75 = {_mean(precision[tind75, :, :, aind, mind]):.4f}")
    print(f"mar    = {_mean(recall[:, :, aind, mind]):.4f}")
    print(
        f"AP small  = {_mean(precision[tind:, :, :, params.areaRngLbl.index('small'), mind]):.4f}"
    )
    print(
        f"AP medium = {_mean(precision[tind:, :, :, params.areaRngLbl.index('medium'), mind]):.4f}"
    )
    print(
        f"AP large  = {_mean(precision[tind:, :, :, params.areaRngLbl.index('large'), mind]):.4f}"
    )

    # Per-class AP@[.5:.95] (area=all, maxDets=500).
    print("\nper-class AP@[.5:.95] (maxDets 500):")
    for index, cat_id in enumerate(params.catIds):
        name = coco_gt.loadCats([cat_id])[0]["name"]
        ap = _mean(precision[tind:, :, index, aind, mind])
        ap50 = _mean(precision[tind, :, index, aind, mind])
        print(f"  {name:17s} AP {ap:.4f}  AP50 {ap50:.4f}")

    # Size distribution of val GT vs predictions (sqrt-area buckets).
    def bucket(area: float) -> str:
        side = area**0.5
        if side < 16:
            return "<16px"
        if side < 32:
            return "16-32px"
        if side < 64:
            return "32-64px"
        return ">=64px"

    gt_areas = [a["area"] for a in coco_gt.dataset["annotations"]]
    pred_areas = [r["bbox"][2] * r["bbox"][3] for r in results]
    gt_hist = Counter(bucket(a) for a in gt_areas)
    pred_hist = Counter(bucket(a) for a in pred_areas)
    print("\nGT side-length distribution:", dict(gt_hist))
    print("prediction distribution    :", dict(pred_hist))


if __name__ == "__main__":
    main()
