# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Command-line interface (typer thin shell over the public API).

Commands share the exact validation of the Python API: keyword-style
``key=value`` arguments are parsed by the API functions themselves via the
config validators.

    autovision-cli pretrain out=runs/e1 data=images/ model=torchvision/resnet18 epochs=2
    autovision-cli train-object-detection out=runs/d1 data=data.yaml model_args='{"image_size":[64,64]}'
    autovision-cli embed out=embeddings.pt checkpoint=runs/e1/exported_models/exported_last.pt
    autovision-cli list-models | list-methods | extract-video-frames ...
"""

from __future__ import annotations

import json
from typing import Any

import typer

app = typer.Typer(
    name="autovision-cli",
    help="Autovision: enterprise visual model training framework.",
    no_args_is_help=True,
    add_completion=False,
)

_KEY_VALUE_ERROR = (
    "Arguments must be key=value pairs, e.g. out=runs/e1 model=torchvision/resnet18"
)


def _parse_kwargs(arguments: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` CLI arguments into API kwargs.

    Values are JSON-decoded when possible (numbers, booleans, lists, nested
    dicts); otherwise kept as strings (paths stay strings — the configs
    coerce them).
    """
    kwargs: dict[str, Any] = {}
    for argument in arguments:
        if "=" not in argument:
            raise typer.BadParameter(_KEY_VALUE_ERROR)
        key, _, value = argument.partition("=")
        key = key.lstrip("-")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


def _run(fn, arguments: list[str]) -> None:
    kwargs = _parse_kwargs(arguments)
    fn(**kwargs)


@app.command("pretrain")
def pretrain_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Self-supervised pretraining / distillation."""
    from autovision import pretrain

    _run(pretrain, list(arguments or []))


@app.command("train-object-detection")
def train_object_detection_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Object detection finetuning (COCO/YOLO)."""
    from autovision import train_object_detection

    _run(train_object_detection, list(arguments or []))


@app.command("train-image-classification")
def train_image_classification_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Image classification finetuning (ImageFolder layout)."""
    from autovision import train_image_classification

    _run(train_image_classification, list(arguments or []))


@app.command("train-semantic-segmentation")
def train_semantic_segmentation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Semantic segmentation finetuning (linear head, image/mask dirs)."""
    from autovision import train_semantic_segmentation

    _run(train_semantic_segmentation, list(arguments or []))


@app.command("embed")
def embed_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Extract embeddings from a pretrained checkpoint."""
    from autovision import embed

    _run(embed, list(arguments or []))


@app.command("benchmark")
def benchmark_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Benchmark inference latency across torch/ONNX/TensorRT backends."""
    from autovision import benchmark

    _run(benchmark, list(arguments or []))


@app.command("extract-video-frames")
def extract_video_frames_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Extract JPEG frames from videos with ffmpeg."""
    from autovision._commands.extract_video_frames import extract_video_frames

    _run(extract_video_frames, list(arguments or []))


@app.command("list-models")
def list_models_command() -> None:
    """Print all available model names."""
    from autovision import list_models

    for name in list_models():
        typer.echo(name)


@app.command("list-methods")
def list_methods_command() -> None:
    """Print all registered pretraining methods."""
    from autovision import list_methods

    for name in list_methods():
        typer.echo(name)


def cli_entrypoint() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    cli_entrypoint()
