# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object-detection model handle: ``train / val / predict / export``.

A thin stateful wrapper over the functional API — every call goes through
the same configs and validators, the handle only carries the plumbing
(which exported artifact to load, which classes the model was trained
with). Two entry points:

- :func:`detection` — handle for a registry model key, chainable into
  training::

      model = autovision.detection("rtdetr/resnet18").train(
          out="runs/d1", data=autovision.ensure_coco8(), steps=200,
      )
      model.val(data=...)
      model.predict("image.jpg")
      model.export("model.onnx")

- :func:`load_model` — handle for an exported ``*.pt`` from a run's
  ``exported_models/`` directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from autovision.config.validate import pydantic_model_validate
from autovision.errors import AutoVisionError
from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.tasks.task_model_helpers import load_task_model
from autovision.types import PathLike

__all__ = [
    "DetectionModel",
    "detection",
    "evaluate_object_detection",
    "load_model",
    "predict_to_files",
]


def _resolve_data_args(data: PathLike | dict[str, Any]) -> Any:
    """Build the pydantic data-args for a config dict / YAML / ``coco8``."""
    from autovision.cli.train_task.object_detection import (
        _DATA_FORMATS,
        _load_data_config,
    )

    payload = _load_data_config(data)
    data_args = pydantic_model_validate(_DATA_FORMATS[payload["format"]], payload)
    data_args.resolve_data_paths(None)
    return data_args


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        return handle.size


def _record_to_jsonable(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        output[key] = (
            value.detach().cpu().tolist() if torch.is_tensor(value) else value
        )
    return output


def evaluate_object_detection(
    model: PathLike | torch.nn.Module,
    data: PathLike | dict[str, Any],
    split: str = "val",
    threshold: float = 0.0,
    max_images: int | None = None,
) -> dict[str, float]:
    """COCO-style evaluation of a detection model on a data config.

    ``model`` is an exported ``*.pt`` path or a loaded task network.
    Predictions come from the network's own preprocess/forward/postprocess
    (boxes land in original-frame pixels) and ground truth from the data
    config's normalized cxcywh annotations — exactly the contract
    :class:`~autovision.metrics.map_evaluator.COCOMAPEvaluator` expects.
    """
    from autovision.data.file_helpers import open_image

    if isinstance(model, (str, Path)):
        network = load_task_model(model, device="cpu")
    else:
        network = model
    data_args = _resolve_data_args(data)

    evaluator = COCOMAPEvaluator(categories=dict(data_args.classes()))
    for index, item in enumerate(data_args.list_image_info(split)):
        if max_images is not None and index >= max_images:
            break
        image_path = item["image_path"]
        width, height = _image_size(image_path)
        open_image(image_path)  # fail fast on unreadable inputs
        prediction = network.predict(image_path, threshold=threshold)
        evaluator.update([prediction])
        target = {
            "boxes": torch.tensor(
                json.loads(item["bboxes"]), dtype=torch.float32
            ),
            "labels": torch.tensor(
                json.loads(item["class_labels"]), dtype=torch.long
            ),
        }
        # COCO crowd flags (ignore regions under the official protocol);
        # YOLO-format rows carry none and evaluate as all-regular GT.
        if "iscrowds" in item:
            target["iscrowds"] = torch.tensor(
                json.loads(item["iscrowds"]), dtype=torch.long
            )
        evaluator.update_targets([target], [(width, height)])
    return evaluator.compute()


class DetectionModel:
    """Stateful handle around one detection model (see module docstring)."""

    def __init__(self, model: str | PathLike | None = None) -> None:
        self.model_key = str(model) if model is not None else None
        self.train_result: Any | None = None
        self._network: Any | None = None
        self._exported_path: Path | None = None
        self._classes: dict[int, str] | None = None

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    def from_export(
        cls, model: PathLike, device: str | None = "cpu"
    ) -> "DetectionModel":
        """Handle for an exported ``*.pt`` (run's ``exported_models/``).

        Defaults to CPU: :meth:`predict` feeds path/PIL inputs through the
        network's CPU-side preprocess, so a CPU network is the only choice
        that works for every input kind. Pass ``device="cuda"`` together
        with tensor inputs moved to that device if you need GPU inference.
        """
        handle = cls(model=None)
        handle._exported_path = Path(model)
        handle._network = load_task_model(handle._exported_path, device=device)
        return handle

    def _require_network(self) -> Any:
        if self._network is None:
            raise AutoVisionError(
                "No trained/exported model on this handle yet — call "
                ".train(...) first, or construct via "
                "autovision.load_model(<exported.pt>)."
            )
        return self._network

    # ── train ────────────────────────────────────────────────────────
    def train(
        self, out: PathLike, data: PathLike | dict[str, Any], **kwargs: Any
    ) -> "DetectionModel":
        """Run ``train_object_detection`` for this handle's model key.

        All keyword arguments pass through unchanged (``steps``,
        ``batch_size="auto"``, ``model_args``, ``transform_args``,
        ``trainer_args``, ...). Returns ``self`` so calls chain:
        ``detection("rtdetr/resnet18").train(...).val(...)``. The raw
        ``TaskTrainResult`` is kept on ``.train_result``.
        """
        if self.model_key is None:
            raise AutoVisionError(
                ".train() requires a registry model key — construct via "
                "autovision.detection(<model>), not load_model(<exported.pt>)."
            )
        from autovision.cli.train_task.object_detection import train_object_detection

        kwargs.setdefault("model", self.model_key)
        self.train_result = train_object_detection(out=out, data=data, **kwargs)
        result = self.train_result
        self._exported_path = Path(
            result.best_exported_model_path or result.exported_model_path
        )
        self._classes = dict(result.classes)
        self._network = load_task_model(self._exported_path, device="cpu")
        return self

    def to(self, device: str) -> "DetectionModel":
        """Move the underlying network to ``device`` (returns ``self``).

        ``predict`` moves its preprocessed inputs onto the network's device,
        so path/PIL/tensor inputs work on any device after the move.
        """
        self._require_network().to(device=device)
        return self

    # ── val / predict ────────────────────────────────────────────────
    def val(
        self,
        data: PathLike | dict[str, Any],
        split: str = "val",
        threshold: float = 0.0,
        max_images: int | None = None,
    ) -> dict[str, float]:
        """COCO-style metrics (``map``/``map_50``/...) on a data config."""
        return evaluate_object_detection(
            self._require_network(),
            data,
            split=split,
            threshold=threshold,
            max_images=max_images,
        )

    def predict(self, image: Any, **kwargs: Any) -> Any:
        """Single-image inference (path / PIL / tensor) -> detection dict."""
        return self._require_network().predict(image, **kwargs)

    def predict_sahi(self, image: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tiled inference for large images (delegates to the network)."""
        return self._require_network().predict_sahi(image, **kwargs)

    # ── export ───────────────────────────────────────────────────────
    def export(self, path: PathLike, **kwargs: Any) -> Path:
        """Export by suffix: ``.onnx``, ``.engine``/``.trt``, or ``.pt``.

        ``.pt`` copies the exported payload produced by training (the
        format :func:`load_model` reads back); ``.onnx``/``.engine``
        delegate to the network's export mixins and accept their kwargs
        (``precision=...``, ...).
        """
        path = Path(path)
        network = self._require_network()
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            return Path(network.export_onnx(str(path), **kwargs))
        if suffix in {".engine", ".trt"}:
            return Path(network.export_tensorrt(str(path), **kwargs))
        if suffix == ".pt":
            if self._exported_path is None:
                raise AutoVisionError(
                    ".export('.pt') needs a training-produced export; run "
                    ".train(...) first."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._exported_path, path)
            return path
        raise AutoVisionError(
            f"Unsupported export target '{path.name}' — use .onnx, "
            ".engine/.trt, or .pt."
        )


def detection(model: str) -> DetectionModel:
    """Handle for a registry model key (see :class:`DetectionModel`)."""
    return DetectionModel(model)


def load_model(model: PathLike, device: str | None = "cpu") -> DetectionModel:
    """Handle for an exported ``*.pt`` (see :class:`DetectionModel`)."""
    return DetectionModel.from_export(model, device=device)


def predict_to_files(
    model: PathLike | torch.nn.Module,
    images: PathLike,
    out_dir: PathLike,
    threshold: float = 0.5,
    save_visuals: bool = True,
    class_names: dict[int, str] | None = None,
) -> Path:
    """Run prediction over a file/directory and persist JSON (+ overlays).

    Writes ``out_dir/predictions.json`` (one record per image with
    ``bboxes``/``scores``/``labels`` as plain lists) and, when
    ``save_visuals``, overlay PNGs under ``out_dir/visuals/``.
    """
    from autovision.visualize.detection import draw_prediction

    network = (
        load_task_model(model, device="cpu")
        if isinstance(model, (str, Path))
        else model
    )
    image_path = Path(images)
    if image_path.is_dir():
        paths = sorted(
            p
            for p in image_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        paths = [image_path]
    if not paths:
        raise AutoVisionError(f"No images found at '{image_path}'.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "visuals"
    records: list[dict[str, Any]] = []
    for path in paths:
        result = network.predict(path, threshold=threshold)
        records.append({"image": str(path), **_record_to_jsonable(result)})
        if save_visuals:
            visuals_dir.mkdir(exist_ok=True)
            rendered = draw_prediction(path, result, class_names=class_names)
            rendered.save(visuals_dir / f"{path.stem}_pred.png")
    (out_dir / "predictions.json").write_text(json.dumps(records, indent=2))
    return out_dir
