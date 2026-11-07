# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO-style evaluation for object detection (pycocotools backend)."""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("autovision")

__all__ = ["COCOMAPEvaluator", "HEADLINE_PROTOCOL", "predictions_to_coco"]

#: H3-6c output-side protocol declaration. The headline precision/recall/F1
#: are read at the F1-optimal operating point (see ``compute()``), a number
#: that is routinely mistaken for the COCO summary protocol. This string is
#: the single source of the wording surfaced by the ``val`` CLI footer.
#: (It deliberately travels as a constant rather than a key inside the
#: ``compute()`` dict: every value there flows into the engine's
#: ``val_metric/*`` logging, which rounds each entry — a str value would
#: crash training.)
HEADLINE_PROTOCOL: str = (
    "note: headline precision/recall/F1 are readings at ONE operating "
    "point (IoU 0.5, headline_score_threshold, macro-averaged over classes "
    "with ground truth; crowd GT is ignored) — AP/AR remain the COCO "
    "protocol. The pre-0.6.1 readings swept pycocotools' recall-threshold "
    "axis against a fixed recall, which is not an operating point (audit "
    "METRIC-04)."
)


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
            # METRIC audit P3-10: names are the REAL class names (looked up
            # through the user id), not the numeric ids the old enumerate
            # wrote — the temp gt.json is human-readable.
            {
                "id": internal_to_user[index],
                "name": categories[internal_to_user[index]],
            }
            for index in range(len(categories))
        ],
    }
    dt = []
    for index, prediction in enumerate(predictions):
        boxes = prediction.get("bboxes", prediction.get("boxes"))
        if boxes is None:
            boxes = []
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

    def __init__(
        self, categories: dict[int, str], max_dets: int = 100,
        *, auto_raise: bool = False,
        headline_score_threshold: float = 0.50,
    ) -> None:
        """``max_dets`` caps detections per image in the COCO eval.

        Default 100 matches pycocotools. Dense benchmarks evaluate higher
        (the VisDrone official protocol uses 500: crowded frames hold
        hundreds of objects, and the cap directly bounds recall).
        ``auto_raise=True`` (design doc D29b) tracks the densest image
        seen and lifts the cap for the ``compute()`` evaluation with a
        warning — ``params.maxDets`` is only consumed there, so no
        retroactive work is needed. The attribute itself is never mutated
        (H3-6c): a permanently lifted cap leaked one round's densest-image
        count into every later round's protocol, so two ``compute()``
        readings were not comparable. Leave auto-raise off when the caller
        pinned an explicit protocol cap (e.g. VisDrone's 500).
        ``headline_score_threshold`` (METRIC-04, design A) is the SINGLE
        score operating point the headline precision/recall/F1 are read
        at — the old sweep over pycocotools' recall-threshold axis paired
        precisions from different thresholds with one fixed recall, which
        is not an operating point and made the F1 systematically exceed
        the recall it was paired with.
        """
        self.categories = categories  # internal id -> name
        self.max_dets = max_dets
        self.auto_raise = auto_raise
        self.headline_score_threshold = float(headline_score_threshold)
        self._max_instances_seen = 0
        self._matches_for_round: list | None = None
        self._predictions: list[dict] = []
        self._ground_truth: list[dict] = []  # {"boxes","labels","size"} per image
        #: Per-class rows from the last :meth:`compute` (name, ap, ar, f1,
        #: precision, recall) — consumed by the validation report renderer.
        # Detail rows mix a str name with float metrics.
        self.last_per_class: list[dict[str, Any]] = []

    def set_ground_truth(self, targets: Sequence[dict]) -> None:
        """Deprecated: size/crowd metadata cannot ride this shape.

        ``update_targets`` is the wired entry point (it carries
        ``original_sizes`` and the crowd flags the protocol needs); this
        method has no production caller left. Deprecated in 0.6.1,
        removal in 0.7.0.
        """
        import warnings

        warnings.warn(
            "COCOMAPEvaluator.set_ground_truth is deprecated in 0.6.1 and "
            "will be removed in 0.7.0 — use update_targets (it carries the "
            "original sizes and crowd flags the protocol needs).",
            DeprecationWarning,
            stacklevel=2,
        )
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
        xyxy on the original frame (matching predictions). Optional
        ``"iscrowds"`` per target carries the COCO crowd flag (1 = ignore
        region under the official protocol; missing = all zeros).
        """
        for index, target in enumerate(targets):
            boxes = (
                target["boxes"].detach().cpu()
                if torch.is_tensor(target["boxes"]) else torch.as_tensor(target["boxes"])
            )
            self._max_instances_seen = max(
                self._max_instances_seen, int(boxes.shape[0])
            )
            labels = (
                target["labels"].detach().cpu()
                if torch.is_tensor(target["labels"]) else torch.as_tensor(target["labels"])
            )
            crowds = target.get("iscrowds")
            if crowds is None:
                crowds = torch.zeros(labels.shape[0], dtype=torch.long)
            else:
                crowds = (
                    crowds.detach().cpu()
                    if torch.is_tensor(crowds) else torch.as_tensor(crowds)
                )
            orig_w, orig_h = original_sizes[index]
            if boxes.numel():
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack(
                    [(cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                     (cx + w / 2) * orig_w, (cy + h / 2) * orig_h], dim=-1,
                )
            self._ground_truth.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "iscrowds": crowds,
                    "size": (int(orig_w), int(orig_h)),
                }
            )

    def reset(self) -> None:
        self._predictions = []
        self._ground_truth = []
        self._matches_for_round = None
        # METRIC audit P3-11: the auto-raise high-water mark leaked across
        # rounds INTO other datasets when one process evaluated several
        # datasets back to back; reset() starts every new evaluation
        # clean (auto_raise re-derives it from the fresh round).
        self._max_instances_seen = 0

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
        # Re-derive from the merged GT so auto_raise sees the DENSEST image
        # across ranks, not just this rank's shard.
        self._max_instances_seen = max(
            [self._max_instances_seen]
            + [
                int(t["boxes"].shape[0])
                for t in self._ground_truth
                if torch.is_tensor(t.get("boxes"))
            ]
        )

    def _matches_cache(self) -> list:
        """Matches for the current round, computed once per compute()."""
        if self._matches_for_round is None:
            from autovision.metrics.error_analysis import match_detections

            def _empty_like(record: dict) -> dict:
                # GT-only rounds (a validation pass with no decodable
                # predictions) must stay per-image aligned with EMPTY
                # records — dropping them would misalign every later image.
                return {
                    "boxes": torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                }

            def _as_match_record(record: dict) -> dict:
                # P0-1: the production decode contract returns "bboxes"
                # (decode_detection_outputs / picodet postprocessor); the
                # matcher consumes "boxes". Normalize both — a bboxes-keyed
                # record must never fall through to the empty fallback (it
                # turned EVERY production headline into 0/0/0 while the
                # tests, all using "boxes", stayed green).
                boxes = record.get("boxes", record.get("bboxes"))
                if boxes is None:
                    return _empty_like(record)
                return {
                    "boxes": boxes,
                    "scores": record.get("scores"),
                    "labels": record.get("labels"),
                }

            predictions = [
                _as_match_record(record) for record in self._predictions
            ]
            targets = [
                {
                    "boxes": record["boxes"],
                    "labels": record["labels"],
                    "iscrowds": record.get("iscrowds"),
                }
                for record in self._ground_truth
            ]
            self._matches_for_round = match_detections(
                predictions,
                targets,
                iou_threshold=0.5,
                score_threshold=self.headline_score_threshold,
            )
        return self._matches_for_round

    def _compute_matches(self) -> list:
        """Deprecated alias (kept one MINOR for external callers)."""
        return self._matches_cache()

    def _single_point_headline(self) -> tuple[float, float, float]:
        """METRIC-04 (design A): P/R/F1 at ONE score operating point.

        Greedy per-class matching at IoU 0.5 over the accumulated round,
        score-gated at ``headline_score_threshold``; crowd GT rows are
        ignored (neither FP-for-the-prediction nor FN-for-the-count,
        mirroring pycocotools' ``dtIg``). Precision/recall are macro
        averages over classes with ground truth; F1 combines those two
        macro values — all three describe the SAME operating point.
        """
        matches = self._matches_cache()
        per_class: dict[int, dict[str, int]] = {}
        for match in matches:
            row = per_class.setdefault(match.class_id, {"tp": 0, "fp": 0, "fn": 0})
            if match.kind in ("tp", "fp", "fn"):
                row[match.kind] += 1
            # "ignored" (crowd hit) counts toward neither P nor R.
        precisions: list[float] = []
        recalls: list[float] = []
        for class_id, counts in per_class.items():
            gt_total = counts["tp"] + counts["fn"]
            if gt_total == 0:
                continue  # classes without (non-crowd) GT: not in the mean
            # A class whose GT survives but which has no surviving
            # prediction has UNDEFINED precision — 0.0, mirroring "nothing
            # was predicted correctly" (excluding it would inflate the
            # macro precision; recall is unambiguous at 0).
            predicted = counts["tp"] + counts["fp"]
            precisions.append(
                counts["tp"] / predicted if predicted > 0 else 0.0
            )
            recalls.append(counts["tp"] / gt_total)
        if not precisions:
            return 0.0, 0.0, 0.0
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def compute(self) -> dict[str, float]:
        # H3-6c: the auto-raise is a per-call adjustment, not a state
        # mutation. Lifting ``self.max_dets`` permanently made the first
        # dense round rewrite the protocol for every later round; the cap
        # below is a local value, so the attribute keeps the caller's
        # pinned number and repeated compute() calls stay comparable.
        cap = self.max_dets
        if self.auto_raise and self._max_instances_seen > self.max_dets:
            cap = self._max_instances_seen
            logger.warning(
                "Dataset holds up to %d instances on one image but maxDets "
                "caps per-image detections at %d — evaluating with the cap "
                "raised to %d (pass max_dets explicitly to pin the "
                "protocol).",
                self._max_instances_seen, self.max_dets, cap,
            )
        from pycocotools.cocoeval import COCOeval

        self.last_per_class = []
        if not self._ground_truth or not self._predictions:
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
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
            crowds = target.get("iscrowds")
            for position, (box, label) in enumerate(
                zip(target["boxes"], target["labels"])
            ):
                x1, y1, x2, y2 = [float(v) for v in box]
                gt_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": index,
                        "category_id": internal_to_user[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": max((x2 - x1) * (y2 - y1), 1.0),
                        # Crowd regions are ignore regions in the official
                        # COCO protocol (no recall requirement, overlapping
                        # detections not penalized); earlier revisions wrote
                        # 0 unconditionally, which counted the 446 val2017
                        # crowd boxes as regular GT.
                        "iscrowd": int(crowds[position]) if crowds is not None else 0,
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
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        coco_dt = coco_gt.loadRes(dt_list)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        # Three-element ladder per pycocotools convention; [1, 10, 100]
        # equals the library default, so the default path is numerically
        # unchanged.
        evaluator.params.maxDets = [1, 10, cap]
        evaluator.evaluate()
        evaluator.accumulate()
        # summarize() is deliberately skipped: its headline stats[0]/stats[1]
        # are pinned to maxDets=100 in pycocotools (the bare ``_summarize(1)``
        # call), returning -1 under any other ladder — and printing a table
        # of the wrong cap next to the right one invites misreading. Every
        # headline number is computed directly from the precision/recall
        # arrays, numerically identical to the official stats when the cap
        # is 100.
        import numpy as np

        params = evaluator.params
        tind = int(np.where(params.iouThrs == 0.5)[0][0])
        tind75 = int(np.where(params.iouThrs == 0.75)[0][0])
        aind = params.areaRngLbl.index("all")
        mind = len(params.maxDets) - 1
        precision = evaluator.eval["precision"]  # [T, R, K, A, M]
        recall = evaluator.eval["recall"]  # [T, K, A, M] — no score axis

        def _mean(values) -> float:
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        metrics = {
            "map": _mean(precision[tind:, :, :, aind, mind]),
            "map_50": _mean(precision[tind, :, :, aind, mind]),
            "map_75": _mean(precision[tind75, :, :, aind, mind]),
            "mar": _mean(recall[:, :, aind, mind]),
        }
        # Per-area slices (COCO protocol: small < 32², medium 32²–96²,
        # large > 96² px²) — the areaRng ladder pycocotools always builds,
        # previously pinned to "all"; additive keys, headline numbers are
        # numerically unchanged.
        for area_label in ("small", "medium", "large"):
            area_index = params.areaRngLbl.index(area_label)
            metrics[f"map_{area_label}"] = _mean(
                precision[tind:, :, :, area_index, mind]
            )

        # METRIC-04 (design A): headline P/R/F1 at ONE score operating
        # point (IoU 0.5, score-gated, crowd-ignored, macro over classes
        # with GT) — the old R-axis sweep paired precisions from different
        # recall thresholds with one fixed recall and overstated F1.
        p50 = precision[tind, :, :, aind, mind]  # [R, K]
        headline_p, headline_r, headline_f1 = self._single_point_headline()
        metrics["f1"] = headline_f1
        metrics["precision"] = headline_p
        metrics["recall"] = headline_r

        # Per-class detail for the validation report (category order);
        # classes without ground truth are skipped. METRIC-04: the P/R/F1
        # columns are the same single operating point as the headline.
        class_names = [name for _, name in sorted(self.categories.items())]
        per_class_point: dict[int, dict[str, float]] = {}
        for match in self._matches_cache():
            row = per_class_point.setdefault(match.class_id, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
            if match.kind in ("tp", "fp", "fn"):
                row[match.kind] += 1.0
        self.last_per_class = []
        for k, name in enumerate(class_names):
            if k >= p50.shape[1] or not (p50[:, k] > -1).any():
                continue
            counts = per_class_point.get(k, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
            gt_total = counts["tp"] + counts["fn"]
            if gt_total > 0:
                predicted = counts["tp"] + counts["fp"]
                p_point = counts["tp"] / predicted if predicted > 0 else 0.0
                r_point = counts["tp"] / gt_total
                f1_point = (
                    2 * p_point * r_point / (p_point + r_point)
                    if (p_point + r_point) > 0 else 0.0
                )
            else:
                p_point = r_point = f1_point = 0.0
            self.last_per_class.append(
                {
                    "name": name,
                    "ap": _mean(precision[tind:, :, k, aind, mind]),
                    "ar": _mean(recall[:, k, aind, mind]),
                    "f1": float(f1_point),
                    "precision": float(p_point),
                    "recall": float(r_point),
                }
            )
        return metrics
