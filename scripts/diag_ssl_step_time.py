# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""One-off: per-section step-time breakdown for the rfdetr SSL run.

Times, per training step on the real 800px two_crop pipeline:
  forward   - model._forward_train (HF model + HF loss incl. its internal
              matcher invocations, + SSL projection taps)
  matcher   - our group-0 pool re-match (LwDetrHungarianMatcher)
  ssl_loss  - ssl_contrastive_loss + SIGReg statistic
  backward  - loss.backward()

Run against the shared GPU while training executes — absolute numbers are
inflated by contention, but the SECTION PROPORTIONS are the answer.
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.object_detection_dataset import ObjectDetectionDataset
from autovision.data.two_view import TwoViewObjectDetectionDataset
from autovision.tasks.detection.rfdetr_object_detection.config import (
    RFDETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.rfdetr_object_detection.train_model import (
    RFDETRObjectDetectionTrain,
)
from autovision.augment.detection.collate import (
    RTDETRDetectionCollateFunction,
    TwoViewDetectionCollateFunction,
)
from autovision.augment.detection.args import (
    RTDETRDetectionTransformArgs,
)
from autovision.augment.detection.transform import (
    RTDETRDetectionTransform,
)

VISDRONE = "/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone"
STEPS = 12
WARMUP = 3


def make_ssl_train_model(with_ssl: bool) -> RFDETRObjectDetectionTrain:
    args = RFDETRObjectDetectionTrainArgs(
        num_classes=10,
        model="rf-detr-nano",
        image_size=(800, 800),
        num_queries=300,
        num_select=500,
        load_weights=True,
        ssl=(
            {"mode": "unsup", "view": "two_crop", "loss_coef": 0.1}
            if with_ssl
            else None
        ),
    )
    tm = RFDETRObjectDetectionTrain(args)
    tm.to('cuda').train()
    return tm


def make_batch() -> dict:
    data = COCOObjectDetectionDataArgs(
        format="coco",
        train={
            "annotations": "/home/hdjs/Research/Project/AutoVision/data/visdrone_smoke_train.json",
            "images": f"{VISDRONE}/VisDrone2019-DET-train/images",
        },
        val={
            "annotations": f"{VISDRONE}/annotations/visdrone_val.json",
            "images": f"{VISDRONE}/VisDrone2019-DET-val/images",
        },
    )
    ds = ObjectDetectionDataset(data, list(data.list_image_info("train"))[:8])
    tvs = TwoViewObjectDetectionDataset(ds)
    targs = RTDETRDetectionTransformArgs(
        image_size=(800, 800),
        random_erase_prob=0.5,
        random_erase_ratio=0.4,
        random_erase_interior=True,
        scale_jitter={"sizes": [800]},
    )
    targs.resolve_auto(num_channels=3, default_image_size=(800, 800))
    targs.resolve_incompatible()
    ds.set_transform(RTDETRDetectionTransform(targs))
    collate = TwoViewDetectionCollateFunction(
        RTDETRDetectionCollateFunction(targs, train=True)
    )
    batch = collate([tvs[i] for i in range(4)])
    batch["image"] = batch["image"].to("cuda")
    return batch


def run(with_ssl: bool) -> None:
    tag = "ssl" if with_ssl else "base"
    train_model = make_ssl_train_model(with_ssl)
    if with_ssl:
        assert train_model.ssl_params is not None
    batch = make_batch()
    params = train_model.ssl_params
    matcher = train_model._ssl_matcher
    timings: dict[str, list[float]] = defaultdict(list)
    for step in range(STEPS):
        marks = {"start": time.perf_counter()}
        x = batch["image"]
        targets = [
            {"boxes": b.to(x.device), "labels": c.to(x.device)}
            for b, c in zip(batch["bboxes"], batch["classes"])
        ]
        out = train_model.model._forward_train(x, targets=targets)
        marks["forward"] = time.perf_counter()
        if params is not None:
            indices = matcher(
                {
                    "logits": out["pred_logits"],
                    "pred_boxes": out["pred_boxes"],
                },
                [
                    {"class_labels": t["labels"], "boxes": t["boxes"]}
                    for t in targets
                ],
                train_model.model._model.config.group_detr,
            )
            marks["matcher"] = time.perf_counter()
            from autovision.tasks.detection.components.ssl_contrastive import (
                ssl_contrastive_loss,
            )

            loss_ssl = ssl_contrastive_loss(
                out["ssl_z1"],
                out["ssl_z2"],
                {"pred_logits": out["pred_logits"]},
                targets,
                indices,
                params,
            )
            marks["ssl_loss"] = time.perf_counter()
            loss = out["loss"] + loss_ssl * params.loss_coef
        else:
            loss = out["loss"]
        loss.backward()
        marks["backward"] = time.perf_counter()
        train_model.zero_grad(set_to_none=True)
        if step >= WARMUP:
            keys = list(marks)
            for a, b in zip(keys, keys[1:]):
                timings[f"{a}->{b}"].append(marks[b] - marks[a])
    print(f"=== {tag} ({STEPS - WARMUP} timed steps, contended GPU) ===")
    total = sum(sum(v) / len(v) for v in timings.values())
    for key, values in timings.items():
        mean = sum(values) / len(values)
        print(f"  {key:16s} {mean * 1000:7.1f} ms  ({mean / total:.0%})")
    print(f"  {'TOTAL':16s} {total * 1000:7.1f} ms")


if __name__ == "__main__":
    run(with_ssl=False)
    run(with_ssl=True)
