# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared helpers for the per-task examples.

The examples are intentionally small and offline: each task gets a synthetic
dataset with two or three images, a tiny model configuration, and two or three
training steps. Pass ``--data`` to point at a real YAML/JSON config instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autovision.errors import AutoVisionError

#: CC0 demo image (San Diego airport aerial). All inference examples default
#: to it; pass ``--image`` for task-specific imagery.
AIRPORT_ASSET = Path(__file__).resolve().parent / "assets" / "airport_aerial.jpg"
ASSET = AIRPORT_ASSET

#: DOTA-v1.0 classes used by the GTR OBB checkpoints.
DOTA_CLASSES = (
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
)

#: Cityscapes 19-class order used by the GTR semseg checkpoints.
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)


def _coco_names() -> dict[int, str]:
    from autovision.tasks.detection.pretrained import COCO_CLASSES

    return {index: name for index, name in enumerate(COCO_CLASSES)}


def _dota_names() -> dict[int, str]:
    return {index: name for index, name in enumerate(DOTA_CLASSES)}


def _cityscapes_names() -> dict[int, str]:
    return {index: name for index, name in enumerate(CITYSCAPES_CLASSES)}

#: Tiny EOMT configuration used by the instance/panoptic examples.
EOMT_MODEL_ARGS = {
    "image_size": 64,
    "num_queries": 6,
    "num_blocks": 2,
    "num_upscale_blocks": 1,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}

#: OBB has its own transform schema (no scale-jitter/crop keys).
OBB_TRANSFORM_ARGS = {
    "photometric_distort_prob": None,
    "horizontal_flip_prob": 0.0,
}


def build_parser(description: str, *, default_out: str, default_steps: int = 2) -> argparse.ArgumentParser:
    """Common CLI for every task example."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="real data YAML/JSON config; omitted = generate a tiny demo set",
    )
    parser.add_argument("--out", type=Path, default=Path(default_out))
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --out directory",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="class count for classification/instance/semantic examples",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write/print the demo data config and exit (no training)",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        default=None,
        help="after training, run the exported model and save prediction.png",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="copy the exported .pt to this path (ONNX/TRT: use autovision-cli export)",
    )
    parser.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        help=(
            "also export the trained model to ONNX at this path "
            "(parity-gated; pair with <task>.py --onnx to serve it)"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    return parser


def build_inference_parser(
    description: str, *, default_out: str, default_image: Path = ASSET
) -> argparse.ArgumentParser:
    """CLI shared by the inference-only task examples."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", type=Path, default=default_image, help="input image"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="exported .pt from <task>_train.py (task-specific default below)",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="optional JSON mapping {class_id: name} for exported checkpoints",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = CUDA when available, else CPU",
    )
    parser.add_argument("--out", type=Path, default=Path(default_out))
    return parser


def resolve_device(choice: str) -> str:
    """Resolve ``auto``/``cpu``/``cuda`` against the installed torch build."""
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def load_class_names(path: Path | None) -> dict[int, str] | None:
    """Read an optional ``{id: name}`` JSON map for exported checkpoints."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return {int(key): str(value) for key, value in payload.items()}


def _tolist(value: Any) -> list[Any]:
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _shape_str(value: Any) -> str:
    """Human-readable shape of a tensor/array-like record value."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return "x".join(str(int(dim)) for dim in shape)
    return "unknown shape"


def summarize_prediction(
    prediction: dict[str, Any],
    class_names: dict[int, str] | None,
    *,
    label_width: int = 12,
    zero_note: str | None = None,
) -> list[str]:
    """Task-aware summary lines for one ``network.predict`` record.

    The record keys decide the wording: instance outputs (bboxes /
    polygons / keypoints) count detections, classification names the
    top-1, panoptic counts segments, and semantic / depth describe the
    dense map. ``zero_note`` replaces the default hint when an instance
    output is empty (the train entry adds its "only 1-3 steps" context).
    """
    def entry(label: str, value: str) -> str:
        return f"{label:<{label_width}}= {value}"

    def named(value: int) -> str:
        return class_names.get(value, str(value)) if class_names else str(value)

    instance_output = any(
        key in prediction for key in ("bboxes", "polygons", "keypoints")
    )
    if instance_output and prediction.get("labels") is not None:
        values = [int(value) for value in _tolist(prediction["labels"])]
        counts = dict(Counter(named(value) for value in values).most_common())
        lines = [entry("detections", str(len(values))), entry("classes", str(counts))]
        if not values:
            lines.append(
                "note: "
                + (zero_note or "no detections above the threshold; "
                   "lower --threshold to inspect.")
            )
        return lines
    if "labels" in prediction and "scores" in prediction:
        # A key present with a None value (a task surface omitting the
        # field) must degrade to the fallback summary, not TypeError in
        # ``list(None)``.
        labels_value = prediction.get("labels")
        scores_value = prediction.get("scores")
        if labels_value is None or scores_value is None:
            return [
                entry("outputs", ", ".join(sorted(prediction))),
                "note: image-level output; no per-instance summary.",
            ]
        labels = _tolist(labels_value)
        scores = _tolist(scores_value)
        label = named(int(labels[0])) if labels else "?"
        score = f"{float(scores[0]):.2f}" if scores else "?"
        return [entry("top1", f"{label} ({score})")]
    if "segment_ids" in prediction:
        segments = prediction["segment_ids"]
        count = len(segments) if hasattr(segments, "__len__") else 0
        lines = [entry("segments", str(count))]
        if count == 0:
            lines.append(
                "note: no confident segments; lower --threshold to inspect."
            )
        return lines
    if "mask" in prediction:
        return [entry("mask", f"{_shape_str(prediction['mask'])} class map")]
    if "depth" in prediction:
        return [entry("depth", f"{_shape_str(prediction['depth'])} metres")]
    return [
        entry("outputs", ", ".join(sorted(prediction))),
        "note: image-level output; no per-instance summary.",
    ]


def reject_num_classes(args: argparse.Namespace, task: str) -> None:
    """Exit cleanly when ``--num-classes`` cannot apply to ``task``.

    The shared parser exposes the flag for every train example, but only
    classification / instance / semantic consume it — the other tasks
    infer the class set from the data config, and a silently ignored
    value reads like a configured one.
    """
    if getattr(args, "num_classes", None) is not None:
        raise SystemExit(
            f"--num-classes does not apply to the {task} example: its class "
            "set comes from the data config. Only classification / instance "
            "/ semantic examples accept the flag."
        )


def load_exported_model(path: Path, device: str) -> Any:
    """Load an exported ``.pt`` with clean one-line exits for common failures.

    A missing file or corrupt payload raised raw ``CheckpointError``
    tracebacks from the inference scripts; the underlying loader keeps its
    semantics — this helper only converts the error surface to the one-line
    SystemExit every other example failure uses.
    """
    if not Path(path).is_file():
        raise SystemExit(f"Weights file does not exist: '{path}'.")
    try:
        import autovision

        return autovision.load_model(path, device=device)
    except AutoVisionError as error:
        raise SystemExit(str(error)) from None


def run_prediction(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run one image eagerly, or through a compiled artifact when set.

    The compiled backends (``--onnx`` / ``--engine`` / ``--openvino``) swap
    only the forward: the network keeps its own pre/post-processing, so the
    returned record is identical in shape to ``model.predict`` regardless of
    the backend. ``model`` may be a ``DetectionModel`` handle or a bare task
    network — the examples build both shapes depending on the ``--weights``
    flag. TensorRT runs on CUDA; ONNX/OpenVINO run CPU-resident sessions.
    """
    artifact = (
        getattr(args, "engine", None)
        or getattr(args, "openvino", None)
        or getattr(args, "onnx", None)
    )
    if artifact is not None:
        from autovision.inference import InferenceEngine

        backend = (
            "tensorrt"
            if getattr(args, "engine", None) is not None
            else "openvino"
            if getattr(args, "openvino", None) is not None
            else "onnx"
        )
        network = model.network if hasattr(model, "network") else model
        device = "cuda" if backend == "tensorrt" else "cpu"
        try:
            engine = InferenceEngine(
                network, backend=backend, artifact=artifact, device=device
            )
        except AutoVisionError as error:
            # Geometry/task identity mismatches between the artifact and
            # the constructed network (fail-closed metadata check) read
            # better as a one-line exit than a traceback.
            raise SystemExit(str(error)) from None
        return engine.run(args.image, threshold=args.threshold)
    return model.predict(args.image, threshold=args.threshold)


def export_task_to_onnx(pt_path: Path, out_path: Path) -> Path:
    """Export a trained ``.pt`` to ONNX through the parity-gated pipeline.

    Same entry as ``autovision-cli export``: every task family with ONNX
    export support works, and the run is verified against the torch
    reference by default (GTR routes trace through the export-safe GLA
    lowering, so CPU export works without CUDA).
    """
    from autovision.cli.export import export_model

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return export_model(pt_path, out_path, device="cpu")
    except AutoVisionError as error:
        # Parity-gate refusals and unsupported families read better as a
        # one-line exit; training itself has already succeeded by now.
        raise SystemExit(f"ONNX export failed: {error}") from None


def predict_and_annotate(
    model: Any,
    args: argparse.Namespace,
    class_names: dict[int, str] | None,
    *,
    title: str = "inference",
) -> None:
    """Run one image, save the overlay + JSON, and print a compact summary."""
    from autovision.results import InferenceResult
    from autovision.visualize import draw_prediction

    prediction = run_prediction(model, args)
    args.out.mkdir(parents=True, exist_ok=True)
    rendered = draw_prediction(
        args.image,
        prediction,
        class_names=class_names,
        min_score=args.threshold,
    )
    rendered.save(args.out / "prediction.jpg")
    payload = InferenceResult.from_record(prediction).to_json(masks="rle")
    (args.out / "detections.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    for line in summarize_prediction(prediction, class_names):
        print(line)
    print(f"title       = {title}")
    print(f"prediction  = {args.out / 'prediction.jpg'}")
    print(f"json        = {args.out / 'detections.json'}")



def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _image(path: Path, seed: int, size: int = 64) -> None:
    """A deterministic noisy RGB image with a filled square."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    array[size // 8 : size // 2, size // 8 : size // 2] = (30, 200, 30)
    Image.fromarray(array).save(path)


def classification_demo(root: Path) -> dict[str, Any]:
    """ImageFolder layout: ``train/<class>/*.png`` and ``val/<class>/*.png``."""
    root.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", 4), ("val", 2)):
        for class_index, class_name in enumerate(("cat", "dog")):
            for index in range(count):
                _image(
                    root / split / class_name / f"{class_name}_{index}.png",
                    seed=1000 * class_index + index + (0 if split == "train" else 100),
                )
    return {"train": root / "train", "val": root / "val"}


def semantic_demo(root: Path) -> dict[str, Any]:
    """Paired images/masks; masks hold class indices 0..2."""
    data: dict[str, Any] = {}
    for split, count in (("train", 3), ("val", 1)):
        images = root / split / "images"
        masks = root / split / "masks"
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_{index}"
            _image(images / f"{stem}.png", seed=2000 + index)
            mask = _rng(2000 + index).integers(0, 3, (64, 64), dtype=np.uint8)
            Image.fromarray(mask).save(masks / f"{stem}.png")
        data[split] = {"images": images, "masks": masks}
    return data


def instance_demo(root: Path) -> dict[str, Any]:
    """COCO instances: two polygon instances per image (two categories)."""
    data: dict[str, Any] = {}
    for split, count in (("train", 3), ("val", 1)):
        images = root / split / "images"
        images.mkdir(parents=True, exist_ok=True)
        records = []
        annotations: list[dict[str, Any]] = []
        for index in range(count):
            image_id = index + 1
            stem = f"{split}_{index}"
            _image(images / f"{stem}.png", seed=3000 + index)
            records.append(
                {
                    "id": image_id,
                    "file_name": f"{stem}.png",
                    "width": 64,
                    "height": 64,
                }
            )
            for category_id, (x, y) in ((1, (4, 4)), (2, (36, 36))):
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, 24, 24],
                        "segmentation": [[x, y, x + 24, y, x + 24, y + 24, x, y + 24]],
                        "area": 24 * 24,
                        "iscrowd": 0,
                    }
                )
        annotations_path = root / split / "instances.json"
        annotations_path.write_text(
            json.dumps(
                {
                    "images": records,
                    "annotations": annotations,
                    "categories": [
                        {"id": 1, "name": "square_a"},
                        {"id": 2, "name": "square_b"},
                    ],
                }
            )
        )
        data[split] = {"annotations": annotations_path, "images": images}
    return data


def panoptic_demo(root: Path) -> dict[str, Any]:
    """Panoptic masks (1=thing, 2=stuff) plus a segments JSON per split."""
    data: dict[str, Any] = {}
    for split, count in (("train", 2), ("val", 1)):
        images = root / split / "images"
        masks = root / split / "masks"
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        annotations: dict[str, list[dict[str, Any]]] = {}
        for index in range(count):
            stem = f"{split}_{index}"
            _image(images / f"{stem}.png", seed=4000 + index)
            mask: np.ndarray = np.zeros((64, 64), dtype=np.uint8)
            mask[:32, :32] = 1
            mask[32:, 32:] = 2
            Image.fromarray(mask).save(masks / f"{stem}.png")
            annotations[stem] = [
                {"id": 1, "category_id": 1, "iscrowd": 0},
                {"id": 2, "category_id": 2, "iscrowd": 0},
            ]
        segments = root / split / "segments.json"
        segments.write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": 1, "name": "car", "isthing": 1},
                        {"id": 2, "name": "sky", "isthing": 0},
                    ],
                    "annotations": annotations,
                }
            )
        )
        data[split] = {"images": images, "masks": masks, "segments": segments}
    return data


def obb_demo(root: Path) -> dict[str, Any]:
    """YOLO-OBB layout: normalized 8-point labels, two classes."""
    names = {"0": "plane", "1": "ship"}
    square = [0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]
    for split, count in (("train", 4), ("val", 2)):
        images = root / "images" / split
        labels = root / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _image(images / f"{split}_{index}.png", seed=5000 + index)
            row = "0 " + " ".join(f"{value:g}" for value in square)
            (labels / f"{split}_{index}.txt").write_text(row + "\n")
    return {
        "format": "yolo-obb",
        "path": str(root),
        "train": str(root / "images" / "train"),
        "val": str(root / "images" / "val"),
        "names": names,
    }


def pose_demo(root: Path, size: int = 128) -> dict[str, Any]:
    """COCO person_keypoints JSON with 17 visible points per person."""
    data: dict[str, Any] = {}
    for split, count in (("train", 2), ("val", 1)):
        images = root / split / "images"
        images.mkdir(parents=True, exist_ok=True)
        records = []
        annotations: list[dict[str, Any]] = []
        for index in range(count):
            image_id = index + 1
            stem = f"{split}_{index}"
            _image(images / f"{stem}.png", seed=6000 + index, size=size)
            records.append(
                {
                    "id": image_id,
                    "file_name": f"{stem}.png",
                    "width": size,
                    "height": size,
                }
            )
            keypoints = []
            for point in range(17):
                keypoints.extend(
                    [
                        24 + point * 4,
                        24 + (point % 5) * 8,
                        2,
                    ]
                )
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [16, 16, size - 32, size - 32],
                    "area": float((size - 32) ** 2),
                    "iscrowd": 0,
                    "num_keypoints": 17,
                    "keypoints": keypoints,
                }
            )
        annotations_path = root / split / "person_keypoints.json"
        annotations_path.write_text(
            json.dumps(
                {
                    "images": records,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "person"}],
                }
            )
        )
        data[split] = {"annotations": annotations_path, "images": images}
    return data


def depth_demo(root: Path, size: int = 128) -> dict[str, Any]:
    """Paired images and float32 ``.npy`` depth maps (metres)."""
    data: dict[str, Any] = {}
    for split, count in (("train", 2), ("val", 1)):
        images = root / split / "images"
        depths = root / split / "depths"
        images.mkdir(parents=True, exist_ok=True)
        depths.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_{index}"
            _image(images / f"{stem}.png", seed=7000 + index, size=size)
            depth = _rng(7000 + index).uniform(1.0, 6.0, (size, size)).astype(np.float32)
            np.save(depths / f"{stem}.npy", depth)
        data[split] = {"images": images, "depths": depths}
    return data


def demo_root(args: argparse.Namespace, task: str) -> Path:
    # Keep demo data outside --out: get_out_dir() requires --out to be
    # empty, and --overwrite would delete the data we just generated.
    return args.out.parent / "_demo_data" / task


def dry_run(name: str, data: Any, args: argparse.Namespace | None = None) -> None:
    """Print the data contract without starting a training run.

    ``args`` (when passed) surfaces the follow-up options that a dry-run
    silently skips — ``--predict-image`` / ``--export`` / ``--export-onnx``
    all consume a trained model, so a user passing them under ``--dry-run``
    deserves a note instead of silent no-ops.
    """
    if args is not None:
        ignored = [
            flag
            for flag, attr in (
                ("--predict-image", "predict_image"),
                ("--export", "export"),
                ("--export-onnx", "export_onnx"),
            )
            if getattr(args, attr, None) is not None
        ]
        if ignored:
            if len(ignored) == 1:
                joined = ignored[0]
                need = "it needs"
            else:
                joined = ", ".join(ignored[:-1]) + f" and {ignored[-1]}"
                need = "they need"
            print(
                f"note: {joined} ignored under --dry-run "
                f"({need} a trained model; re-run without --dry-run)."
            )
    print(f"[{name}] demo data config:")
    print(json.dumps(data, default=str, indent=2))
    print("\nPass --data <yaml|json> to train on a real dataset.")


def _sample_image(args: argparse.Namespace) -> Path | None:
    return args.predict_image


def finish(result: Any, args: argparse.Namespace) -> None:
    """Print the run summary, optionally copy the export and render a sample."""
    print(f"steps_run       = {result.steps_run}")
    print(f"checkpoint      = {result.checkpoint_path}")
    print(f"exported_model  = {result.exported_model_path}")
    print(f"best_metrics    = {result.best_metrics}")
    if result.classes:
        print(f"classes         = {result.classes}")

    if args.export is not None:
        source = result.exported_model_path
        if source is None:
            print("No exported model was produced; skipping --export.")
        elif args.export.suffix.lower() == ".pt":
            args.export.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, args.export)
            print(f"copied          = {args.export}")
        else:
            print(
                "This example only copies .pt files. Use "
                f"`autovision-cli export model={source} out={args.export}` "
                "for ONNX / OpenVINO / TensorRT."
            )

    if getattr(args, "export_onnx", None) is not None:
        source = result.exported_model_path
        if source is None:
            print("No exported model was produced; skipping --export-onnx.")
        else:
            artifact = export_task_to_onnx(source, args.export_onnx)
            print(f"onnx            = {artifact}")

    image_path = _sample_image(args)
    if image_path is None:
        return
    if result.exported_model_path is None:
        print("No exported model was produced; skipping prediction.")
        return
    import torch

    from autovision.tasks.task_model_helpers import load_task_model
    from autovision.visualize import draw_prediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = load_task_model(result.exported_model_path, device=device)
    record = network.predict(image_path, threshold=args.threshold)
    for line in summarize_prediction(
        record,
        result.classes or None,
        label_width=16,
        zero_note=(
            "no detections above the threshold. The training examples run "
            "only 1-3 steps for speed; use `inference_demo.py` for a released "
            "checkpoint, or lower --threshold just to inspect the render path."
        ),
    ):
        print(line)
    rendered = draw_prediction(
        image_path, record, class_names=result.classes or None, min_score=args.threshold
    )
    output = args.out / "prediction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output)
    print(f"prediction      = {output}")


def gtr_requirement(
    *, need_cuda: bool = True, device: str | None = None
) -> None:
    """Fail early with an actionable message when GTR's stack is missing.

    ``need_cuda=False`` only checks the ``gtr`` extra: constructing a GTR
    network and running it through ``backend='onnx'`` never executes the
    CUDA-only GLA kernels, so a CPU-only machine can serve an exported
    artifact that way (the eager path still needs CUDA).

    ``device`` is the resolved ``--device`` choice: when given, the CUDA
    check targets that selection instead of machine capability, so an
    explicit ``--device cpu`` on a CUDA machine exits with guidance rather
    than a deep Triton traceback from the GLA kernels.
    """
    import torch

    if importlib.util.find_spec("fla") is None:
        raise SystemExit(
            "This example runs the GTR family and needs the `gtr` extra:\n"
            "  pip install -e '.[gtr]'\n"
            "Training examples: re-run with --dry-run to inspect the data "
            "contract. Inference examples: pass --weights exported from a "
            "non-GTR training run."
        )
    cuda_ok = torch.cuda.is_available() if device is None else device == "cuda"
    if need_cuda and not cuda_ok:
        reason = (
            "no CUDA device is available" if device is None
            else f"requested device='{device}'"
        )
        raise SystemExit(
            "This example runs the GTR eager forward, which needs "
            f"device='cuda' ({reason}; the GLA kernels are CUDA-only). "
            "CPU alternative: export ONNX on a CUDA machine and re-run "
            "with --onnx <artifact> (inference examples that support it)."
        )


def gtr_eager_device_guard(network: Any, *, device: str) -> None:
    """Exit cleanly when a GTR-family network would run eager forward on CPU.

    Exported ``.pt`` metadata cannot answer "is this GTR": a GTR detection
    checkpoint resolves to the same generic ``DETRObjectDetection`` class
    and plain-numeric ``model_init_args`` as any other DETR export. The
    family is therefore detected from the live module tree — GTR task
    models live in ``tasks/<task>/gtr/`` packages and the detection GTR
    backbone in ``tasks/detection/shared/gtr/``. Construction and ONNX
    execution are CPU-safe; only the eager forward runs fla's CUDA-only
    Triton GLA kernels, so a CPU-loaded checkpoint would otherwise crash
    deep in Triton at the first ``predict``.
    """
    if device != "cpu":
        return
    if not any(
        "gtr" in type(module).__module__.split(".")
        for module in network.modules()
    ):
        return
    raise SystemExit(
        "This checkpoint is a GTR-family network: its eager forward needs "
        "device='cuda' (requested device='cpu'; the GLA kernels are "
        "CUDA-only). CPU alternative: export ONNX on a CUDA machine and "
        "re-run with --onnx <artifact> (detection.py)."
    )


def ensure_out_available(out: Path, *, overwrite: bool) -> None:
    """Exit cleanly when ``--out`` collides with an existing run directory.

    Mirrors ``cli.common_helpers.get_out_dir`` but speaks CLI flags: a
    second example run without ``--overwrite`` prints one actionable line
    instead of the Python API's ``ValueError`` traceback.
    """
    if out.is_file():
        raise SystemExit(f"Output path '{out}' is a file; pass a directory.")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory '{out}' exists and is not empty. "
            "Re-run with --overwrite to replace it."
        )
