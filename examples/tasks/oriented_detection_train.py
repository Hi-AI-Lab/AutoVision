#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented object detection: YOLO-OBB -> predict.

The default model is the GTR ``gtr/s`` family (the released OBB checkpoint is
DOTA-trained). Training the GTR family needs ``pip install -e '.[gtr]'`` and
a CUDA machine (Triton kernels); use ``--dry-run`` to inspect the data
contract offline. The default demo writes normalized 8-point labels for two
classes. Real data is a YOLO-OBB ``data.yaml``:

    format: yolo-obb
    train: images/train
    val: images/val
    names: {0: plane, 1: ship}
"""

from __future__ import annotations

from _common import (
    OBB_TRANSFORM_ARGS,
    build_parser,
    demo_root,
    dry_run,
    ensure_out_available,
    finish,
    gtr_requirement,
    obb_demo,
    reject_num_classes,
)

import autovision


def main() -> None:
    args = build_parser(
        "Oriented object detection example (YOLO-OBB by default)",
        default_out="runs/examples/oriented_detection_train",
    ).parse_args()
    reject_num_classes(args, "oriented object detection")

    data = args.data or obb_demo(demo_root(args, "oriented_detection"))
    if args.dry_run:
        dry_run("oriented_object_detection", data, args)
        return
    ensure_out_available(args.out, overwrite=args.overwrite)
    gtr_requirement()

    result = autovision.train_oriented_object_detection(
        out=args.out,
        data=data,
        model="gtr/s",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={
            "image_size": (128, 128),
            "num_queries": 16,
            "num_denoising": 0,
            "load_checkpoint": False,
            "lr_warmup_steps": 2,
        },
        transform_args=OBB_TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
