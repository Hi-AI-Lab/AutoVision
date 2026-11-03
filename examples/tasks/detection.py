#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection inference: load weights -> predict -> annotate.

No training. By default this loads the official ``detr/gtrvit-s`` GTR
checkpoint (MIT, COCO AP 53.6) and runs it on the bundled CC0 airport image:

    python examples/tasks/detection.py
    python examples/tasks/detection.py --image path/to/image.jpg --threshold 0.5
    python examples/tasks/detection.py --effect blur

The GTR forward needs the ``gtr`` extra on a CUDA machine; pass
``--model rfdetr/rf-detr-nano`` for a CPU-friendly released checkpoint.

``--onnx`` runs the forward through an exported ONNX artifact instead of
the eager model — this is the GTR CPU deployment path (the network keeps
its own pre/post-processing; only the forward is swapped):

    python -c "import autovision; autovision.detection( \
        'detr/gtrvit-s', pretrained=True).export('gtrvit_s.onnx', precision='fp32')"
    python examples/tasks/detection.py --onnx gtrvit_s.onnx

``--effect`` renders through the annotators migrated from supervision
(``autovision.visualize.effects``): ``corner`` / ``round`` replace the plain
box with corner brackets or rounded boxes, while ``blur`` / ``pixelate``
apply privacy filtering to the detections before boxes and labels are drawn.

Pass ``--weights path/to/exported.pt`` to use a model produced by
``train_object_detection`` instead of the released checkpoint. For a full
training walkthrough, see ``examples/tasks/detection_train.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from _common import gtr_requirement
from PIL import Image

import autovision
from autovision.results import InferenceResult
from autovision.tasks.detection.pretrained import pretrained_class_names
from autovision.visualize import (
    BlurAnnotator,
    BoxCornerAnnotator,
    LabelAnnotator,
    PixelateAnnotator,
    RoundBoxAnnotator,
    annotate_result,
    draw_prediction,
)

ASSET = Path(__file__).resolve().parent / "assets" / "airport_aerial.jpg"

#: Rendering styles backed by ``autovision.visualize.effects``.
EFFECTS = ("none", "corner", "round", "blur", "pixelate")


def _tolist(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _load_rgb(path: Path) -> Image.Image:
    """Decode ``path`` to an RGB Pillow canvas (same reader as draw_prediction)."""
    from autovision.data.file_helpers import open_image

    return Image.fromarray(open_image(path).permute(1, 2, 0).numpy()).convert("RGB")


def _render_effect(
    image: Path,
    result: InferenceResult,
    effect: str,
    class_names: dict[int, str] | None,
) -> Image.Image:
    """Render ``result`` with one supervision-derived effect."""
    canvas = _load_rgb(image)
    if effect == "corner":
        canvas = BoxCornerAnnotator().annotate(canvas, result)
        return LabelAnnotator().annotate(canvas, result, class_names=class_names)
    if effect == "round":
        canvas = RoundBoxAnnotator().annotate(canvas, result)
        return LabelAnnotator().annotate(canvas, result, class_names=class_names)
    privacy = BlurAnnotator() if effect == "blur" else PixelateAnnotator()
    canvas = privacy.annotate(canvas, result)
    return annotate_result(canvas, result, class_names=class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ASSET)
    parser.add_argument(
        "--model",
        default="detr/gtrvit-s",
        help=(
            "released registry key (ignored when --weights is given); "
            "GTR keys need the `gtr` extra on CUDA"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="exported .pt from training; overrides --model",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help=(
            "exported .onnx artifact: run the forward through it "
            "(backend='onnx') instead of the eager model; works on "
            "CPU-only machines and overrides --weights"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--effect",
        choices=EFFECTS,
        default="none",
        help=(
            "supervision-derived rendering style: corner/round boxes or "
            "blur/pixelate privacy (default: none)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = CUDA when available, else CPU",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/examples/detection")
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    if args.onnx is not None:
        # ONNX forward: construction never runs the GLA kernels, so the
        # CUDA requirement does not apply here (the `gtr` extra still is —
        # the GLA module builds fla primitives at construction time). The
        # network stays CPU-resident: the ONNX session runs on the CPU EP
        # and the network only supplies pre/post-processing.
        if not args.onnx.is_file():
            raise SystemExit(f"ONNX artifact not found: {args.onnx}")
        if args.model.startswith("detr/gtrvit"):
            gtr_requirement(need_cuda=False)
        names = pretrained_class_names(args.model)
        model = autovision.detection(args.model, pretrained=True)
    elif args.weights is not None:
        model = autovision.load_model(args.weights, device=device)
        names = None  # custom checkpoints carry their own class ids
    else:
        if args.model.startswith("detr/gtrvit"):
            gtr_requirement()
        names = pretrained_class_names(args.model)
        model = autovision.detection(args.model, pretrained=True)
        model.to(device)

    if args.onnx is not None:
        from autovision.inference import InferenceEngine

        engine = InferenceEngine(model.network, backend="onnx", artifact=args.onnx)
        prediction = engine.run(args.image, threshold=args.threshold)
    else:
        prediction = model.predict(args.image, threshold=args.threshold)
    labels = _tolist(prediction["labels"])
    scores = _tolist(prediction["scores"])
    boxes = _tolist(prediction["bboxes"])
    class_names = [
        names[int(label)] if names is not None else str(int(label))
        for label in labels
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.effect == "none":
        rendered = draw_prediction(
            args.image,
            prediction,
            class_names=names,
            min_score=args.threshold,
        )
    else:
        result = InferenceResult.from_record(prediction)
        if args.threshold > 0:
            result = result.filter_scores(args.threshold)
        rendered = _render_effect(args.image, result, args.effect, names)
    rendered.save(args.out / "prediction.jpg")
    records = [
        {"name": name, "score": float(score), "box": box}
        for name, score, box in zip(class_names, scores, boxes)
    ]
    (args.out / "detections.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    print(f"image       = {args.image} ({rendered.width}x{rendered.height})")
    print(f"weights     = {args.weights or args.model} @ threshold {args.threshold}")
    backend = f"onnx ({args.onnx})" if args.onnx is not None else "eager"
    print(f"backend     = {backend}")
    device_shown = "cpu (onnx session)" if args.onnx is not None else device
    print(f"device      = {device_shown}")
    print(f"effect      = {args.effect}")
    print(f"detections  = {len(records)}")
    print(f"classes     = {dict(Counter(class_names).most_common())}")
    if not records:
        print("note: no detections above the threshold; lower --threshold to inspect.")
    print(f"prediction  = {args.out / 'prediction.jpg'}")
    print(f"json        = {args.out / 'detections.json'}")


if __name__ == "__main__":
    main()
