# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""B4 pre-measurement: false-negative pressure of the per-object SSL objective.

Instance discrimination pushes every pair of distinct objects apart —
including two objects of the same class in one image (the classic
false-negative problem, sharper in detection than in NLP). This tool
quantifies that pressure on a dataset BEFORE training so the ``ssl.mode``
default (``unsup`` vs ``sup``) can be chosen from data instead of
guesswork (design doc 14 §4.5; upstream RF-DETR-SSL ``datasets/ssl_stats.py``).

Usage::

    python scripts/ssl_cooccurrence_stats.py --annotations data/train/_annotations.coco.json

Interpretation: ``instance_share_with_same_class_partner`` > 0.5 marks a
crowded dataset (each such pair becomes a forced negative under the
unsupervised objective) — try ``mode="sup"`` with
``same_image_positive_only=True`` there. Upstream measured 98.6% on
VisDrone-mini, yet unsup still won the ablation — treat the threshold as
a prior, not a rule.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def same_class_cooccurrence_stats(ann_file: str | Path) -> dict[str, float]:
    """Measure same-class co-occurrence over a COCO annotation file.

    Crowd annotations are skipped (they are not individual instances in the
    contrastive pool).

    Returns:
        Dict with:

        - ``num_images_annotated``: images holding at least one non-crowd
          annotation;
        - ``num_instances``: total non-crowd instances;
        - ``instance_share_with_same_class_partner``: fraction of instances
          that have at least one same-class instance in the same image;
        - ``mean_same_class_ratio_per_instance``: mean over instances of
          ``same_class_others / (instances_in_image - 1)``;
        - ``image_share_with_same_class_pair``: fraction of annotated images
          containing at least one same-class pair.
    """
    data: dict[str, Any] = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    per_image: dict[int, list[int]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        per_image[ann["image_id"]].append(int(ann["category_id"]))

    total_instances = 0
    instances_with_partner = 0
    ratio_sum = 0.0
    images_with_pair = 0
    for categories in per_image.values():
        count = len(categories)
        if count == 0:
            continue
        has_pair = False
        for index, category in enumerate(categories):
            same_others = sum(
                1
                for other_index, other in enumerate(categories)
                if other_index != index and other == category
            )
            total_instances += 1
            if same_others > 0:
                instances_with_partner += 1
                has_pair = True
            ratio_sum += same_others / (count - 1) if count > 1 else 0.0
        if has_pair:
            images_with_pair += 1

    annotated_images = len(per_image)
    return {
        "num_images_annotated": float(annotated_images),
        "num_instances": float(total_instances),
        "instance_share_with_same_class_partner": (
            instances_with_partner / total_instances if total_instances else 0.0
        ),
        "mean_same_class_ratio_per_instance": ratio_sum / total_instances if total_instances else 0.0,
        "image_share_with_same_class_pair": images_with_pair / annotated_images if annotated_images else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", required=True, help="COCO instances_*.json path")
    args = parser.parse_args()
    stats = same_class_cooccurrence_stats(args.annotations)
    for name, value in stats.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
