# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Validation-batch visualization for object detection (GT + predictions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw

from autovision._transforms.transform_args import NormalizeArgs

__all__ = ["save_detection_examples"]


def _cxcywh_norm_to_xyxy(boxes, width: int, height: int):
    """Normalized cxcywh -> absolute xyxy (skips degenerate rows)."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.reshape(-1, 4).unbind(-1)
    return torch.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height,
         (cx + w / 2) * width, (cy + h / 2) * height], dim=-1,
    )


def save_detection_examples(
    out_dir: Path | str,
    batch: dict,
    step: int,
    predictions: Sequence[dict] | None = None,
    max_examples: int = 3,
    normalize_args: Any = None,
) -> list[Path]:
    """Render the first N images of a batch with GT boxes (green, normalized
    cxcywh on the input grid) and, when provided, prediction boxes (red, xyxy
    absolute pixels); mirrors the baseline image_examples."""
    out_dir = Path(out_dir) / "image_examples" / f"step_{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    written = []
    num = min(max_examples, images.shape[0])
    for index in range(num):
        image = images[index]
        _, height, width = image.shape
        display = image.float()
        if display.min() < 0:
            # Normalized input (negative values): un-normalize for display.
            # This check must come first — normalized bright pixels reach
            # ~2.6 and would trip a ">1.5 means raw 0-255" heuristic,
            # flattening the whole image to the mean color.
            args = normalize_args or {}
            mean = args.get("mean", NormalizeArgs().mean)
            std = args.get("std", NormalizeArgs().std)
            mean = torch.tensor(mean).view(-1, 1, 1)
            std = torch.tensor(std).view(-1, 1, 1)
            display = display * std + mean
        elif display.max() > 1.5:  # raw 0-255 input
            display = display / 255.0
        grid = Image.fromarray(
            (display.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        draw = ImageDraw.Draw(grid)
        gt_boxes = batch["bboxes"][index] if isinstance(batch["bboxes"], list) else batch["bboxes"]
        gt_labels = batch["classes"][index] if isinstance(batch["classes"], list) else batch["classes"]
        gt_boxes = gt_boxes.detach().cpu()
        for box, label in zip(
            _cxcywh_norm_to_xyxy(gt_boxes, width, height),
            gt_labels.reshape(-1),
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 < x1 or y2 < y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
            draw.text((x1 + 2, y1 + 2), str(int(label)), fill=(0, 200, 0))
        if predictions is not None and index < len(predictions):
            # Predictions must be in the SAME frame as the drawn image
            # (input-grid xyxy) — original-frame boxes need rescaling first.
            original_size = batch.get("original_size")
            if original_size is not None and index < len(original_size):
                orig = original_size[index]
            else:
                orig = (width, height)  # already input-frame
            scale_x = width / float(orig[0])
            scale_y = height / float(orig[1])
            for box in predictions[index]["bboxes"]:
                x1, y1, x2, y2 = [float(v) for v in box]
                x2, y2 = x1 + (x2 - x1) * scale_x, y1 + (y2 - y1) * scale_y
                x1, y1 = x1 * scale_x, y1 * scale_y
                if x2 < x1 or y2 < y1:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        path = out_dir / f"example_{index}.jpg"
        grid.save(path)
        written.append(path)
    return written
