#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Self-supervised pretraining + embedding extraction demo.

Covers two under-exercised subsystems: ``pretext/`` (via
``autovision.pretrain``) and ``features/`` (via ``autovision.features.embed``
and ``autovision.load_embedding_model``). A small SimCLR backbone is trained
on an image folder, then embeddings are extracted from it and the exported
model is reloaded as a reusable embedding model.

    python examples/tasks/pretrain_demo.py
    python examples/tasks/pretrain_demo.py --method dino --data <img_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import autovision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="image folder (default: coco8 images dir)",
    )
    parser.add_argument(
        "--method",
        default="simclr",
        help="pretext method, e.g. simclr / dino / detconb (see "
        "autovision.pretext.teachers.list_methods())",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/examples/pretrain_demo")
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.data is None:
        coco8 = autovision.ensure_coco8()
        args.data = Path(coco8["train"]["images"])
    if not Path(args.data).is_dir():
        raise SystemExit(f"Image dir not found: {args.data}")

    print(f"pretrain: method={args.method} data={args.data} epochs={args.epochs}")
    result = autovision.pretrain(
        out=args.out,
        data=str(args.data),
        model="torchvision/resnet18",
        method=args.method,
        epochs=args.epochs,
        overwrite=True,
    )
    checkpoint = result.exported_model_path
    print("exported embedding model:", checkpoint)

    embeddings = autovision.features.embed(
        out=args.out / "embeddings",
        data=str(args.data),
        checkpoint=checkpoint,
        format="csv_features",
        overwrite=True,
    )
    print("embeddings:", embeddings)

    model = autovision.load_embedding_model(checkpoint)
    print("reloaded embedding model:", type(model).__name__)


if __name__ == "__main__":
    main()
