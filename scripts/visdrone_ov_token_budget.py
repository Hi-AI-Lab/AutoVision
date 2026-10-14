# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""G2 gate tool for design doc 27: open-vocabulary prompt budget statistics.

Encodes a vocabulary through the selected frozen text tower and reports
the numbers the G2 gate needs (doc 27 §6): per-encoding wall time, cache
size, and the per-class prompt token footprint. Pure statistics — no
training, no dataset required.

Examples::

    # hermetic default (identity embedder, no download):
    python scripts/visdrone_ov_token_budget.py

    # real towers (downloads through the verified weight path on first
    # use — the G1 pinning step precedes this):
    python scripts/visdrone_ov_token_budget.py --encoder siglip-base
    python scripts/visdrone_ov_token_budget.py --vocab coco80 --encoder clip-b32
    python scripts/visdrone_ov_token_budget.py --vocab file:path/to/names.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autovision.tasks.detection.components.open_vocab import (  # noqa: E402
    VISDRONE_CLASS_NAMES,
    encode_class_names,
    resolve_text_embed_dim,
    text_cache_path,
)

_COCO80: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _load_vocab(name: str) -> tuple[str, ...]:
    if name == "visdrone":
        return VISDRONE_CLASS_NAMES
    if name == "coco80":
        return _COCO80
    if name.startswith("file:"):
        lines = Path(name[5:]).read_text(encoding="utf-8").splitlines()
        names = tuple(line.strip() for line in lines if line.strip())
        if not names:
            raise SystemExit(f"empty vocabulary file: {name[5:]}")
        return names
    raise SystemExit(
        f"unknown vocab {name!r}: use 'visdrone', 'coco80' or 'file:<path>'."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        default="identity",
        choices=("identity", "siglip-base", "clip-b32", "clip-b16"),
        help="frozen text tower ('identity' is the hermetic hash embedder)",
    )
    parser.add_argument(
        "--vocab",
        default="visdrone",
        help="'visdrone' | 'coco80' | 'file:<path>' (one name per line)",
    )
    parser.add_argument(
        "--template",
        default="a photo of a {}.",
        help="prompt template with the '{}' class-name slot",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="encode this many times to average the timing (first run pays "
        "the tower download/cache miss)",
    )
    args = parser.parse_args()

    class_names = _load_vocab(args.vocab)
    print(f"vocab: {len(class_names)} classes ({args.vocab})")
    print(f"encoder: {args.encoder} (embed_dim={resolve_text_embed_dim(args.encoder)})")
    print(f"template: {args.template!r}")

    timings = []
    embeds = None
    for run in range(max(args.repeat, 1)):
        started = time.perf_counter()
        embeds = encode_class_names(
            class_names=class_names,
            text_encoder=args.encoder,
            prompt_template=args.template,
            cache_embeddings=True,
        )
        timings.append(time.perf_counter() - started)
    print(f"embeds: shape={tuple(embeds.shape)} dtype={embeds.dtype} "
          f"norm range=[{embeds.norm(dim=-1).min():.4f}, {embeds.norm(dim=-1).max():.4f}]")
    print(f"encode time: first+avg = {timings[0]:.1f}s / "
          f"{sum(timings) / len(timings):.1f}s over {len(timings)} run(s)")

    cache_file = text_cache_path(
        args.encoder, class_names, args.template, None, None
    )
    if cache_file.exists():
        size_kb = cache_file.stat().st_size / 1024
        print(f"cache: {cache_file} ({size_kb:.1f} KiB)")
    else:
        print("cache: MISS (cache_embeddings disabled or write failed)")
    if args.encoder == "identity":
        print("note: the identity embedder is the hermetic hash stand-in — "
              "run with a real encoder for the G2 gate numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
