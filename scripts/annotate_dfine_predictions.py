#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotate images with the official Apache-2.0 D-FINE checkpoints.

The D-FINE store holds upstream complete detectors. This script runs the
upstream D-FINE deploy path (same preprocessing/postprocessing as
``tools/inference/torch_inf.py``), maps COCO class ids to names, and writes
annotated images plus a JSON detection dump.

Usage::

    python scripts/annotate_dfine_predictions.py \
        --image /path/to/baidu.jpg \
        --out ~/Research/Model/AutoVision/test_predictions/dfine_baidu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DFINE_ROOT = Path.home() / "Research/Software/upstream/D-FINE"
DEFAULT_STORE = Path.home() / "Research/Model/AutoVision/apache_detection/dfine"

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

#: Distinguishable palette (RGB); cycled by class id.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _load_model(dfine_root: Path, config: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(dfine_root))
    from src.core import YAMLConfig  # type: ignore[import-not-found]

    cfg = YAMLConfig(str(config), resume=str(checkpoint))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "ema" in state and state["ema"]:
        state = state["ema"].get("module", state["model"])
    else:
        state = state["model"]
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        # The official L checkpoint was trained without LAB (no ``lab.*``
        # keys); building LAB layers and leaving them at scale=1/bias=0 is
        # equivalent, but disabling them matches the trained graph exactly.
        if not any(".lab." in key for key in state):
            cfg.yaml_cfg["HGNetv2"]["use_lab"] = False
    cfg.model.load_state_dict(state)

    class DeployModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return DeployModel().to(device).eval()


def _predict(model, image: Image.Image, device: str, threshold: float):
    width, height = image.size
    tensor = T.Compose([T.Resize((640, 640)), T.ToTensor()])(image).unsqueeze(0)
    tensor = tensor.to(device)
    orig_size = torch.tensor([[width, height]], device=device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    labels = labels.detach().cpu()[0]
    boxes = boxes.detach().cpu()[0]
    scores = scores.detach().cpu()[0]
    keep = scores >= threshold
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    order = torch.argsort(scores, descending=True)
    labels, boxes, scores = labels[order], boxes[order], scores[order]
    detections = []
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        name = COCO_NAMES[int(label)] if 0 <= int(label) < len(COCO_NAMES) else str(label)
        detections.append(
            {
                "label": int(label),
                "name": name,
                "score": float(score),
                "box": [float(value) for value in box],
            }
        )
    return detections


def _annotate(image: Image.Image, detections: list[dict], title: str) -> Image.Image:
    font = _font(18)
    header = 42
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black", font=font)

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color = PALETTE[det["label"] % len(PALETTE)]
        x0, y0 = int(x0), int(y0) + header
        x1, y1 = int(x1), int(y1) + header
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = f"{det['name']} {det['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        tx = max(0, min(x0, canvas.width - tw - 6))
        ty = max(header, y0 - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
        draw.text((tx + 3, ty + 3), text, fill="white", font=font)
    return canvas


def _montage(panels: list[tuple[str, Image.Image]], out: Path, columns: int = 2) -> None:
    cell_w = 520
    resized = [
        (name, ImageOps.contain(img, (cell_w, 900))) for name, img in panels
    ]
    cell_h = max(img.height for _, img in resized) + 34
    rows = (len(resized) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    for index, (name, img) in enumerate(resized):
        row, col = divmod(index, columns)
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 10, y + 6), name, fill="black", font=font)
        canvas.paste(img, (x + (cell_w - img.width) // 2, y + 34))
    canvas.save(out, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True, help="input image")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/test_predictions/dfine_baidu",
    )
    parser.add_argument("--variants", nargs="+", default=["s", "m", "l"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dfine-root", type=Path, default=DEFAULT_DFINE_ROOT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    source.save(args.out / "source.jpg", quality=92)
    all_detections: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    panels: list[tuple[str, Image.Image]] = [("source", source)]

    for variant in args.variants:
        config = args.dfine_root / "configs/dfine" / f"dfine_hgnetv2_{variant}_coco.yml"
        checkpoint = args.store / f"dfine_{variant}_coco" / "weights.pth"
        if not config.is_file() or not checkpoint.is_file():
            raise SystemExit(f"missing config/checkpoint for {variant}: {config}, {checkpoint}")
        model = _load_model(args.dfine_root, config, checkpoint, args.device)
        detections = _predict(model, source, args.device, args.threshold)
        all_detections[f"DFINE-{variant.upper()}"] = detections
        top = sorted(
            {d["name"] for d in detections},
            key=lambda name: -sum(1 for d in detections if d["name"] == name),
        )[:8]
        summary[f"DFINE-{variant.upper()}"] = {
            "detections": len(detections),
            "top_classes": top,
            "top_scores": [round(d["score"], 3) for d in detections[:10]],
        }
        title = (
            f"D-FINE-{variant.upper()}  640x640  "
            f"threshold {args.threshold:.2f}  {len(detections)} detections"
        )
        annotated = _annotate(source, detections, title)
        path = args.out / f"DFINE-{variant.upper()}.jpg"
        annotated.save(path, quality=92)
        panels.append((f"D-FINE-{variant.upper()}", annotated))
        print(f"{variant}: {len(detections)} detections -> {path}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _montage(panels, args.out / "comparison.jpg")
    (args.out / "detections.json").write_text(
        json.dumps(all_detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# D-FINE Baidu image predictions",
        "",
        f"Image: `{args.image}` ({source.width}x{source.height})",
        f"Threshold: {args.threshold}",
        "",
        "| Model | Detections | Top classes |",
        "|---|---:|---|",
    ]
    for key, entry in summary.items():
        lines.append(
            f"| {key} | {entry['detections']} | {', '.join(entry['top_classes'])} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
