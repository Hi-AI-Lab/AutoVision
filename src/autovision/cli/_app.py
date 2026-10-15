# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
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
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(
    name="autovision-cli",
    help="AutoVision: enterprise visual model training framework.",
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
    from autovision.cli.extract_video_frames import extract_video_frames

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


@app.command("val")
def val_command(arguments: list[str] = typer.Argument(None)) -> None:
    """COCO evaluation of an exported detection model.

        autovision-cli val model=runs/d1/exported_models/exported_best.pt data=coco8 [out=metrics.json]
    """
    from autovision.cli.model_handle import evaluate_object_detection

    kwargs = _parse_kwargs(list(arguments or []))
    out = kwargs.pop("out", None)
    metrics = evaluate_object_detection(**kwargs)
    if out is not None:
        Path(str(out)).write_text(json.dumps(metrics, indent=2))
    for key, value in sorted(metrics.items()):
        typer.echo(f"{key}\t{value:.4f}" if isinstance(value, float) else f"{key}\t{value}")


@app.command("predict")
def predict_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Run prediction over an image or directory.

        autovision-cli predict model=model.pt images=dir/ out=preds/ threshold=0.5
    """
    from autovision.cli.model_handle import predict_to_files

    kwargs = _parse_kwargs(list(arguments or []))
    if "out" in kwargs:
        kwargs["out_dir"] = kwargs.pop("out")
    typer.echo(str(predict_to_files(**kwargs)))


@app.command("download")
def download_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Download + extract a registered dataset.

        autovision-cli download dataset=coco2017-val out=datasets/
    """
    from autovision.data.downloads import download_dataset

    kwargs = _parse_kwargs(list(arguments or []))
    dataset = kwargs.pop("dataset", None)
    if dataset is None:
        raise typer.BadParameter("dataset=<name> is required")
    typer.echo(str(download_dataset(str(dataset), Path(kwargs.get("out", "datasets")))))


@app.command("convert-data")
def convert_data_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Convert detection annotations between YOLO and COCO.

        autovision-cli convert-data from=yolo images=img/ names=a.yaml out=train.json
        autovision-cli convert-data from=coco annotations=train.json out-labels=labels/
    """
    from autovision.data.convert import coco_to_yolo, yolo_to_coco

    kwargs = _parse_kwargs(list(arguments or []))
    source = str(kwargs.pop("from", ""))
    if source == "yolo":
        names = kwargs.pop("names", None)
        if names is None:
            raise typer.BadParameter("from=yolo requires names=<yaml|json list|inline list>")
        if isinstance(names, str):
            loaded = yaml.safe_load(Path(names).read_text())
            names = (
                loaded.get("names", loaded) if isinstance(loaded, dict) else loaded
            )
        typer.echo(str(yolo_to_coco(names=names, **kwargs)))
    elif source == "coco":
        typer.echo(str(coco_to_yolo(**kwargs)))
    else:
        raise typer.BadParameter("from= must be 'yolo' or 'coco'")


@app.command("plot-curves")
def plot_curves_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Plot training curves from a run's metrics.jsonl (needs matplotlib).

        autovision-cli plot-curves run=runs/d1
    """
    from autovision.cli.plot import plot_training_curves

    kwargs = _parse_kwargs(list(arguments or []))
    run = Path(str(kwargs.pop("run")))
    metrics = run / "metrics.jsonl" if run.is_dir() else run
    typer.echo(str(plot_training_curves(metrics, out_path=kwargs.get("out"))))


def cli_entrypoint() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    cli_entrypoint()
