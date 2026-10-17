# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference runtime: torch / ONNX / TensorRT with one pre/post.

The engine keeps the task network's own pre/post-processing and swaps only
the forward pass (the export parity gate's exact contract), so file, video,
and stream inference produce the same coordinates as ``val``/``predict``.
``run_stream`` drives a video file or RTSP source through the engine and
persists per-frame predictions (JSONL) plus an annotated MP4.

    from autovision.inference import InferenceEngine, run_stream

    run_stream(model="runs/d1/exported_models/exported_best.pt",
               source="rtsp://camera/live", out_dir="preds/",
               backend="onnx", artifact="model.onnx", batch_size=4)

Video/stream I/O needs the ``video`` extra (OpenCV); the ``onnx`` and
``tensorrt`` backends additionally need ``export`` / the ``tensorrt``
package respectively.
"""

from __future__ import annotations

from autovision.inference.engine import InferenceEngine
from autovision.inference.stream import StreamRunResult, run_stream
from autovision.inference.video import AnnotatedVideoWriter, VideoFrameStream, VideoInfo

__all__ = [
    "AnnotatedVideoWriter",
    "InferenceEngine",
    "StreamRunResult",
    "VideoFrameStream",
    "VideoInfo",
    "run_stream",
]
