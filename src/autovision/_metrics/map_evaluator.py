# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""COCO-style evaluation for object detection (pycocotools backend)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

__all__ = ["COCOMAPEvaluator", "predictions_to_coco"]


def predictions_to_coco(
    predictions: Sequence[dict],
    categories: dict[int, str],
    original_sizes: Sequence[tuple[int, int]],
) -> tuple[dict, list[dict]]:
    """Build (coco_gt_dict, coco_dt_list) from in-memory results.

    ``predictions[i]``: {"boxes": (N,4) xyxy absolute pixels,
    "scores": (N,), "labels": (N,) internal ids}. Internal ids map back to
    user categories via insertion order in ``categories``.
    """
    internal_to_user = {index: class_id for index, class_id in enumerate(sorted(categories))}
    gt = {
        "images": [
            {"id": index, "width": size[0], "height": size[1]}
            for index, size in enumerate(original_sizes)
        ],
        "annotations": [],  # filled by compute() from real targets
        "categories": [
            {"id": internal_to_user[index], "name": name}
            for index, name in enumerate(sorted(categories))
        ],
    }
    dt = []
    for index, prediction in enumerate(predictions):
        boxes = prediction.get("bboxes", prediction.get("boxes"))
        scores = prediction["scores"]
        labels = prediction["labels"]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            dt.append(
                {
                    "image_id": index,
                    "category_id": internal_to_user[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return gt, dt


class COCOMAPEvaluator:
    """COCO mAP over accumulated predictions (single class or multi-class).

    Ground truth is provided once via :meth:`set_ground_truth`; predictions
    accumulate per validation round and ``compute()`` runs ``COCOeval``.
    """

    def __init__(self, categories: dict[int, str], max_dets: int = 100) -> None:
        """``max_dets`` caps detections per image in the COCO eval.

        Default 100 matches pycocotools. Dense benchmarks evaluate higher
        (the VisDrone official protocol uses 500: crowded frames hold
        hundreds of objects, and the cap directly bounds recall).
        """
        self.categories = categories  # internal id -> name
        self.max_dets = max_dets
        self._predictions: list[dict] = []
        self._ground_truth: list[dict] = []  # {"boxes","labels","size"} per image

    def set_ground_truth(self, targets: Sequence[dict]) -> None:
        self._ground_truth = [
            {
                "boxes": t["boxes"].detach().cpu() if torch.is_tensor(t["boxes"]) else torch.as_tensor(t["boxes"]),
                "labels": t["labels"].detach().cpu() if torch.is_tensor(t["labels"]) else torch.as_tensor(t["labels"]),
            }
            for t in targets
        ]

    def update(self, predictions: Sequence[dict]) -> None:
        self._predictions.extend(predictions)

    def update_targets(
        self,
        targets: Sequence[dict],
        original_sizes: Sequence[tuple[int, int]],
    ) -> None:
        """Accumulate ground truth for the current validation round.

        Target boxes arrive as normalized cxcywh (invariant under the val
        resize); ``original_sizes`` (w, h) per item maps them to absolute
        xyxy on the original frame (matching predictions).
        """
        for index, target in enumerate(targets):
            boxes = (
                target["boxes"].detach().cpu()
                if torch.is_tensor(target["boxes"]) else torch.as_tensor(target["boxes"])
            )
            labels = (
                target["labels"].detach().cpu()
                if torch.is_tensor(target["labels"]) else torch.as_tensor(target["labels"])
            )
            orig_w, orig_h = original_sizes[index]
            if boxes.numel():
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack(
                    [(cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                     (cx + w / 2) * orig_w, (cy + h / 2) * orig_h], dim=-1,
                )
            self._ground_truth.append(
                {"boxes": boxes, "labels": labels, "size": (int(orig_w), int(orig_h))}
            )

    def reset(self) -> None:
        self._predictions = []
        self._ground_truth = []

    def export_state(self) -> dict:
        """Snapshot accumulated predictions/targets (CPU tensors) for
        cross-rank gathering in distributed validation."""
        return {
            "predictions": list(self._predictions),
            "ground_truth": list(self._ground_truth),
        }

    def merge_state(self, state: dict) -> None:
        """Merge a snapshot produced by :meth:`export_state` (e.g. gathered
        from other ranks under DDP)."""
        self._predictions.extend(state["predictions"])
        self._ground_truth.extend(state["ground_truth"])

    def compute(self) -> dict[str, float]:
        from pycocotools.cocoeval import COCOeval

        if not self._ground_truth or not self._predictions:
            return {"map": 0.0, "map_50": 0.0}
        # True original sizes when recorded by update_targets; the fallback
        # derives a lower bound from the GT extent (set_ground_truth does not
        # carry image sizes).
        original_sizes: list[tuple[int, int]] = []
        for target in self._ground_truth:
            size = target.get("size")
            if size is not None:
                original_sizes.append((int(size[0]), int(size[1])))
                continue
            boxes = target["boxes"]
            if boxes.numel() > 0:
                width = float(boxes[:, 2].max())
                height = float(boxes[:, 3].max())
            else:
                width, height = 1.0, 1.0
            original_sizes.append((int(width) + 1, int(height) + 1))
        gt_dict, dt_list = predictions_to_coco(
            self._predictions, self.categories, original_sizes
        )
        # Replace dummy full-image GT annotations with real boxes.
        gt_annotations = []
        annotation_id = 1
        internal_to_user = {
            index: class_id for index, class_id in enumerate(sorted(self.categories))
        }
        for index, target in enumerate(self._ground_truth):
            for box, label in zip(target["boxes"], target["labels"]):
                x1, y1, x2, y2 = [float(v) for v in box]
                gt_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": index,
                        "category_id": internal_to_user[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": max((x2 - x1) * (y2 - y1), 1.0),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        gt_dict["annotations"] = gt_annotations

        with tempfile.TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.json"
            gt_file.write_text(json.dumps(gt_dict))
            from pycocotools.coco import COCO

            coco_gt = COCO(str(gt_file))
        if not dt_list:
            return {"map": 0.0, "map_50": 0.0}
        coco_dt = coco_gt.loadRes(dt_list)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        # Three-element ladder per pycocotools convention; [1, 10, 100]
        # equals the library default, so the default path is numerically
        # unchanged.
        evaluator.params.maxDets = [1, 10, self.max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        if self.max_dets == 100:
            # Official path: stats[0]/stats[1] are AP/AP50 at maxDets=100.
            evaluator.summarize()
            return {
                "map": float(evaluator.stats[0]),
                "map_50": float(evaluator.stats[1]),
            }
        # For other caps, summarize() is unusable as-is: its headline
        # stats[0] is pinned to maxDets=100 in pycocotools (the bare
        # ``_summarize(1)`` call), returning -1 under any other ladder.
        # Compute AP/AP50 at the configured cap directly from the precision
        # array — numerically identical to the official stats when the cap
        # is 100.
        import numpy as np

        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        aind = params.areaRngLbl.index("all")
        mind = len(params.maxDets) - 1
        precision = evaluator.eval["precision"]  # [T, R, K, A, M]

        def _mean(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        return {
            "map": _mean(precision[tind:, :, :, aind, mind]),
            "map_50": _mean(precision[tind, :, :, aind, mind]),
        }
