#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection training: train -> validate -> predict.

The default model is the GTR ``detr/gtrvit-s`` family on the bundled
``coco8`` smoke set. Training the GTR family needs ``pip install -e '.[gtr]'``
and a CUDA machine (Triton kernels); use ``--dry-run`` to inspect the data
contract offline:

    python examples/tasks/detection_train.py --steps 2
    python examples/tasks/detection_train.py --steps 2 \
        --predict-image examples/tasks/assets/airport_aerial.jpg

The demo runs a tiny 64px / 10-query configuration from random init. For a
real finetune pass the official MIT weights:
``model_args={"checkpoint": "gtr_s_coco"}`` requires editing this script.
Real data: pass a COCO/YOLO ``data.yaml`` (see docs/index.md).
"""

from __future__ import annotations

from _common import build_parser, dry_run, finish, gtr_requirement

import autovision


def main() -> None:
    args = build_parser(
        "Object detection training example (coco8 by default)",
        default_out="runs/examples/detection_train",
    ).parse_args()

    data = args.data or autovision.ensure_coco8()
    if args.dry_run:
        dry_run("detection", data)
        return
    gtr_requirement()

    result = autovision.train_object_detection(
        out=args.out,
        data=data,
        model="detr/gtrvit-s",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={
            "image_size": (64, 64),
            "num_queries": 10,
            "num_top_queries": 10,
            "num_denoising": 0,
            "lr_warmup_steps": 2,
        },
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
