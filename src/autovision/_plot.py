# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Debug visualization for pretraining augmentations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

__all__ = ["plot_example_augmentations"]


def plot_example_augmentations(
    train_batch: dict,
    max_examples: int = 10,
) -> Image.Image:
    """Grid of the first examples across all views of a pretraining batch.

    ``train_batch["views"]`` holds one (B, C, H, W) tensor per view.
    """
    views = train_batch["views"]
    num_views = len(views)
    num_examples = min(max_examples, views[0].shape[0])
    cell = 128
    grid = Image.new(
        "RGB",
        (num_views * cell, num_examples * cell),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    for view_index, view in enumerate(views):
        for example_index in range(num_examples):
            tensor = view[example_index, :3].float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)
            array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            tile = Image.fromarray(array).resize((cell, cell))
            grid.paste(tile, (view_index * cell, example_index * cell))
    for view_index in range(num_views):
        draw.text((view_index * cell + 4, 2), f"view {view_index}", fill=(200, 0, 0))
    for example_index in range(num_examples):
        draw.text((2, example_index * cell + 2), f"#{example_index}", fill=(200, 0, 0))
    return grid


def save_example_augmentations(
    train_batch: dict, out_path: Path | str, max_examples: int = 10
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = plot_example_augmentations(train_batch, max_examples=max_examples)
    grid.save(out_path)
    return out_path
