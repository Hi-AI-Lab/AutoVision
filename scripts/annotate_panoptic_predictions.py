#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotate an image with the user-held DINOv3 EoMT panoptic models.

The six checkpoints live in ``~/Research/Model/AutoVision/panoptic_segmentation``
and are **not redistributable** (DINOv3 License + AGPL-3.0 training pipeline).
This script only produces local prediction images/JSON for inspection.

Usage::

    python scripts/annotate_panoptic_predictions.py \
        --image /path/to/baidu.jpg \
        --out ~/Research/Model/AutoVision/test_predictions/panoptic_baidu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from annotate_dfine_predictions import _font, _montage  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from autovision import load_task_model  # noqa: E402

VARIANTS = {
    "dinov3-vitt16-eomt-panoptic-coco": "VITT16",
    "dinov3-vitt16plus-eomt-panoptic-coco": "VITT16PLUS",
    "dinov3-vits16-eomt-panoptic-coco": "VITS16",
    "dinov3-vitb16-eomt-panoptic-coco": "VITB16",
    "dinov3-vitl16-eomt-panoptic-coco": "VITL16",
    "dinov3-vitl16-eomt-panoptic-coco-1280": "VITL16-1280",
}

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (0, 200, 150), (200, 80, 0), (80, 80, 255), (255, 120, 180),
    (120, 255, 80), (180, 120, 0), (0, 90, 200), (200, 0, 120),
]


def _load_names(asset_path: Path) -> tuple[list[str], int]:
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    things = [entry["name"] for entry in asset["thing_classes"]]
    stuffs = [entry["name"] for entry in asset["stuff_classes"]]
    return things + stuffs, len(things)


def _segments(
    masks: np.ndarray,
    segment_ids: torch.Tensor,
    scores: torch.Tensor,
    names: list[str],
    num_thing: int,
) -> list[dict]:
    segments = []
    total = masks.shape[0] * masks.shape[1]
    for sid, score in zip(segment_ids.tolist(), scores.tolist()):
        selected = masks[..., 1] == sid
        if not selected.any():
            continue
        classes = np.unique(masks[selected][..., 0])
        class_id = int(classes[0])
        ys, xs = np.nonzero(selected)
        area = int(selected.sum())
        segments.append(
            {
                "segment_id": int(sid),
                "class_id": class_id,
                "name": names[class_id] if 0 <= class_id < len(names) else str(class_id),
                "score": float(score),
                "area": area,
                "area_fraction": area / total,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "is_thing": class_id < num_thing,
            }
        )
    return segments


def _anchor(mask: np.ndarray, bbox: list[int]) -> tuple[int, int] | None:
    x0, y0, x1, y1 = bbox
    candidates = [((x0 + x1) // 2, (y0 + y1) // 2)]
    ys, xs = np.nonzero(mask)
    if len(xs):
        candidates.append((int(xs.mean()), int(ys.mean())))
        candidates.append((int(xs[0]), int(ys[0])))
    for x, y in candidates:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            return x, y
    return None


def _render(
    image: Image.Image,
    masks: np.ndarray,
    segments: list[dict],
    title: str,
    with_legend: bool,
) -> Image.Image:
    overlay = np.asarray(image).copy().astype(np.float32)
    alpha = 0.45
    for segment in segments:
        selected = masks[..., 1] == segment["segment_id"]
        color = np.array(PALETTE[segment["segment_id"] % len(PALETTE)], dtype=np.float32)
        overlay[selected] = overlay[selected] * (1.0 - alpha) + color * alpha
    base = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    header = 42
    panel_w = 330 if with_legend else 0
    canvas = Image.new(
        "RGB", (base.width + panel_w, base.height + header), "white"
    )
    canvas.paste(base, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black", font=_font(18))

    # On-image labels: all things and stuff covering >= 0.2% of the frame.
    label_font = _font(14)
    for segment in segments:
        if not segment["is_thing"] and segment["area_fraction"] < 0.002:
            continue
        selected = masks[..., 1] == segment["segment_id"]
        point = _anchor(selected, segment["bbox"])
        if point is None:
            continue
        x, y = point
        text = f"{segment['name']} {segment['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        tw, th = right - left, bottom - top
        tx = max(0, min(x, base.width - tw - 8))
        ty = max(header, min(y + header, canvas.height - th - 8))
        color = PALETTE[segment["segment_id"] % len(PALETTE)]
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
        draw.text((tx + 3, ty + 3), text, fill="white", font=label_font)

    if with_legend:
        legend_font = _font(13)
        x0 = base.width + 10
        y0 = header + 4
        draw.text(
            (x0, y0),
            f"Segments: {len(segments)}",
            fill="black",
            font=_font(15),
        )
        y = y0 + 26
        ordered = sorted(segments, key=lambda item: -item["area"])
        for segment in ordered:
            color = PALETTE[segment["segment_id"] % len(PALETTE)]
            draw.rectangle([x0, y + 2, x0 + 14, y + 16], fill=color)
            area_pct = 100.0 * segment["area_fraction"]
            text = f"{segment['name']} {segment['score']:.2f} {area_pct:.1f}%"
            draw.text((x0 + 20, y), text, fill="black", font=legend_font)
            y += 19
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True, help="input image")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/test_predictions/panoptic_baidu",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/panoptic_segmentation",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    source.save(args.out / "source.jpg", quality=92)
    all_segments: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    panels: list[tuple[str, Image.Image]] = [("source", source)]

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant}")
        asset_dir = args.store / variant
        weights = asset_dir / "weights.pt"
        if not weights.is_file():
            raise SystemExit(f"missing weights: {weights}")
        names, num_thing = _load_names(asset_dir / "ASSET.json")
        model = load_task_model(weights, device=args.device)
        output = model.predict(source)
        masks = output["masks"].detach().cpu().numpy()
        segments = _segments(
            masks, output["segment_ids"].detach().cpu(), output["scores"].detach().cpu(),
            names, num_thing,
        )
        key = VARIANTS[variant]
        ignore = int((masks[..., 1] == -1).sum())
        input_size = model.image_size
        input_side = int(input_size) if isinstance(input_size, int) else int(input_size[0])
        all_segments[key] = {
            "image_size": [source.width, source.height],
            "model_input": [input_side, input_side],
            "threshold": model.threshold,
            "ignore_area": ignore,
            "segments": segments,
        }
        stuff = [s for s in segments if not s["is_thing"]]
        things = [s for s in segments if s["is_thing"]]
        top_stuff = sorted(stuff, key=lambda item: -item["area"])[:5]
        top_things = sorted(things, key=lambda item: -item["score"])[:5]
        summary[key] = {
            "segments": len(segments),
            "things": len(things),
            "stuff": len(stuff),
            "top_stuff": [
                (s["name"], round(100 * s["area_fraction"], 1)) for s in top_stuff
            ],
            "top_things": [(s["name"], round(s["score"], 3)) for s in top_things],
            "ignore_fraction": round(ignore / (source.width * source.height), 4),
        }
        title = (
            f"{key}  input {input_side}  threshold {model.threshold:.1f}  "
            f"{len(segments)} segments"
        )
        overlay = _render(source, masks, segments, title, with_legend=False)
        overlay_path = args.out / f"{key}.png"
        overlay.save(overlay_path)
        legend = _render(source, masks, segments, title, with_legend=True)
        legend_path = args.out / f"{key}-legend.jpg"
        legend.save(legend_path, quality=92)
        panels.append((key, overlay))
        print(f"{variant}: {len(segments)} segments -> {legend_path}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _montage(panels, args.out / "comparison.jpg", columns=2)
    (args.out / "segments.json").write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Panoptic predictions on the Baidu image",
        "",
        f"Image: `{args.image}` ({source.width}x{source.height})",
        "",
        "| Model | Segments (thing/stuff) | Top stuff (area %) | Top things (score) |",
        "|---|---:|---|---|",
    ]
    for key, entry in summary.items():
        stuff_text = ", ".join(f"{n} {p}%" for n, p in entry["top_stuff"])
        thing_text = ", ".join(f"{n} {s}" for n, s in entry["top_things"])
        lines.append(
            f"| {key} | {entry['segments']} "
            f"({entry['things']}/{entry['stuff']}) | {stuff_text} | {thing_text} |"
        )
    lines += [
        "",
        "> These weights are user-held: DINOv3 License + AGPL-3.0 training",
        "> pipeline. Do not redistribute or use commercially.",
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
