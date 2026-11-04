#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation: paired images/masks -> predict.

Masks are PNG class-index maps; label 255 is ignored. The default model is
the GTR ``gtr/s`` family (the released checkpoint is Cityscapes-trained).
Training the GTR family needs ``pip install -e '.[gtr]'`` and a CUDA machine
(Triton kernels); use ``--dry-run`` to inspect the data contract offline.
The default demo writes three 64x64 images and masks under
``<out>/_demo_data/semantic``.
"""

from __future__ import annotations

from _common import (
    build_parser,
    demo_root,
    dry_run,
    ensure_out_available,
    finish,
    gtr_requirement,
    semantic_demo,
)

import autovision


def main() -> None:
    args = build_parser(
        "Semantic segmentation example (paired images/masks by default)",
        default_out="runs/examples/semantic_segmentation_train",
    ).parse_args()

    data = args.data or semantic_demo(demo_root(args, "semantic_segmentation"))
    if args.dry_run:
        dry_run("semantic_segmentation", data)
        return
    ensure_out_available(args.out, overwrite=args.overwrite)
    gtr_requirement()

    result = autovision.train_semantic_segmentation(
        out=args.out,
        data=data,
        model="gtr/s",
        num_classes=args.num_classes or 3,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        overwrite=args.overwrite,
        model_args={
            "image_size": (128, 128),
            "load_checkpoint": False,
            "lr_warmup_steps": 2,
        },
        trainer_args={"val_every_num_steps": max(1, args.steps)},
    )
    finish(result, args)


if __name__ == "__main__":
    main()
