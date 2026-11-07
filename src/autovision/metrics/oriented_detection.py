# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented detection mAP (self-contained rotated-box COCO-style metric).

pycocotools cannot represent rotated boxes, so this evaluator runs the
matching itself: per class and per IoU threshold (0.50:0.05:0.95), a greedy
score-descending assignment on the exact rotated IoU
(:func:`rotated_iou_matrix`), then all-point-interpolated AP. ``map`` is
the macro average over classes that have ground truth across all ten
thresholds; ``map_50`` is the same average at IoU 0.50.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from autovision.errors import AutoVisionError
from autovision.geometry.obb import (
    rotated_iou_matrix,
    xyxyxyxy_to_xywhr,
)

__all__ = ["OrientedDetectionMetricCollector"]

_IOU_THRESHOLDS = tuple(0.5 + 0.05 * i for i in range(10))


def _average_precision(
    scores_tps: list[tuple[float, int]],
    num_gt: int,
    interpolation: str = "all-point",
) -> float:
    """VOC all-point (or COCO 101-point) AP from score-ordered TP flags.

    METRIC-02: the envelope must be computed over the WHOLE precision
    sequence BEFORE integration. The old implementation interleaved the
    monotone-precision envelope with the (reverse-order) integration, so a
    higher-precision entry earlier in the sequence could not retroactively
    lift a recall segment already integrated with a lower precision — a
    systematic AP underestimate (audit reproduction: [(0.9 TP), (0.8 FP)]
    on one GT scored 0.5 instead of 1.0).
    """
    if num_gt == 0:
        return 0.0
    if interpolation not in ("all-point", "coco101"):
        raise ValueError(
            f"interpolation must be 'all-point' or 'coco101', got {interpolation!r}."
        )
    scores_tps = sorted(scores_tps, key=lambda entry: -entry[0])
    tp_cum = 0
    fp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, tp in scores_tps:
        tp_cum += tp
        fp_cum += 1 - tp
        recalls.append(tp_cum / num_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))
    # Monotone precision envelope from the right over the FULL sequence.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    if interpolation == "coco101":
        # COCO protocol: sample the envelope on the fixed 101-point recall
        # grid, taking the envelope precision at recall >= the point.
        ap = 0.0
        for step in range(101):
            point = step / 100.0
            precision_at = 0.0
            for recall, precision in zip(recalls, precisions):
                if recall >= point and precision > precision_at:
                    precision_at = precision
            ap += precision_at
        return ap / 101.0

    # All-point: integrate over the strictly increasing recall steps in
    # SCORE order, using the pre-computed envelope precision at each step.
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in zip(recalls, precisions):
        if recall > prev_recall:
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return float(ap)


class OrientedDetectionMetricCollector:
    """Accumulates oriented detections + ground truth and computes mAP.

    Predictions arrive as per-image ``{"labels", "polygons", "scores"}``
    with polygons in absolute original-frame pixels (the OBB decode
    output). Targets arrive per image as normalized corner arrays
    (``corners``: (n, 8)) that are rescaled to original-frame pixels here
    — the oriented analogue of the COCO evaluator's normalized-cxcywh
    rescaling.
    """

    def __init__(self, categories: dict[int, str]) -> None:
        self.categories = categories  # internal id -> name
        self._predictions: list[dict] = []
        self._ground_truth: list[dict] = []

    def update(self, predictions: list[dict[str, torch.Tensor]]) -> None:
        self._predictions.extend(
            {
                "labels": p["labels"].detach().cpu(),
                "polygons": p["polygons"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
            }
            for p in predictions
        )

    def update_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        original_sizes: list[tuple[int, int]],
    ) -> None:
        for index, target in enumerate(targets):
            corners = (
                target["corners"].detach().cpu()
                if torch.is_tensor(target["corners"])
                else torch.as_tensor(target["corners"])
            ).reshape(-1, 8)
            labels = (
                target["labels"].detach().cpu()
                if torch.is_tensor(target["labels"])
                else torch.as_tensor(target["labels"])
            )
            orig_w, orig_h = original_sizes[index]
            if corners.numel():
                corners = corners.clone()
                corners[:, 0::2] *= orig_w
                corners[:, 1::2] *= orig_h
            self._ground_truth.append({"corners": corners, "labels": labels})

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
        if not self._ground_truth:
            return {"map": 0.0, "map_50": 0.0}
        if len(self._predictions) != len(self._ground_truth):
            # A silent zip truncation here would DROP every unmatched
            # prediction — exactly the false positives the PR stream needs
            # (the audit reproduction only worked through this hole).
            raise AutoVisionError(
                f"{len(self._predictions)} prediction record(s) for "
                f"{len(self._ground_truth)} ground-truth record(s); update "
                "and update_targets must be called for the same images."
            )

        # Per threshold -> per class -> score-ordered (score, tp) stream.
        streams: dict[float, dict[int, list[tuple[float, int]]]] = {
            threshold: defaultdict(list) for threshold in _IOU_THRESHOLDS
        }
        num_gt_per_class: dict[int, int] = defaultdict(int)

        for gt, pred in zip(self._ground_truth, self._predictions):
            gt_xywhr = (
                xyxyxyxy_to_xywhr(gt["corners"])
                if gt["corners"].numel()
                else torch.zeros(0, 5)
            )
            gt_labels = gt["labels"]
            pred_xywhr = (
                xyxyxyxy_to_xywhr(pred["polygons"].reshape(-1, 8))
                if pred["polygons"].numel()
                else torch.zeros(0, 5)
            )
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            # METRIC-01: iterate the UNION of this image's GT and predicted
            # classes. The old gt_labels.unique() loop silently DROPPED
            # predictions for every class absent from that image's ground
            # truth — background false positives never entered the PR
            # stream and AP inflated systematically (audit reproduction:
            # an official-AP-0.5 scenario scored 1.0). Pred-only classes
            # here still accumulate FPs; classes with no GT anywhere in
            # the dataset stay out of the macro mean (pycocotools
            # semantics) via the num_gt > 0 gate below.
            class_ids: set[int] = set(
                int(v) for v in gt_labels.unique().tolist()
            ) | set(int(v) for v in pred_labels.unique().tolist())
            for class_id in sorted(class_ids):
                gt_class = gt_xywhr[gt_labels == class_id]
                num_gt_per_class[class_id] += int(gt_class.shape[0])
                class_mask = pred_labels == class_id
                pred_class = pred_xywhr[class_mask]
                scores_class = pred_scores[class_mask]
                if pred_class.shape[0] == 0:
                    continue
                order = torch.argsort(scores_class, descending=True)
                pred_class = pred_class[order]
                scores_class = scores_class[order]
                iou = rotated_iou_matrix(pred_class, gt_class)
                for threshold in _IOU_THRESHOLDS:
                    matched: set[int] = set()
                    stream = streams[threshold][class_id]
                    for k in range(pred_class.shape[0]):
                        best_j = -1
                        best_iou = -1.0
                        for j in range(gt_class.shape[0]):
                            if j in matched:
                                continue
                            value = float(iou[k, j])
                            if value >= threshold and value > best_iou:
                                best_iou = value
                                best_j = j
                        if best_j >= 0:
                            matched.add(best_j)
                            stream.append((float(scores_class[k]), 1))
                        else:
                            stream.append((float(scores_class[k]), 0))

        classes_with_gt = sorted(
            class_id for class_id, count in num_gt_per_class.items() if count > 0
        )
        if not classes_with_gt:
            return {"map": 0.0, "map_50": 0.0}

        def class_mean_at(threshold: float) -> float:
            aps = [
                _average_precision(
                    streams[threshold].get(class_id, []), num_gt_per_class[class_id]
                )
                for class_id in classes_with_gt
            ]
            return sum(aps) / len(aps)

        return {
            "map": sum(class_mean_at(t) for t in _IOU_THRESHOLDS) / len(_IOU_THRESHOLDS),
            "map_50": class_mean_at(0.5),
        }
