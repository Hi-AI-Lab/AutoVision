# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Stream inference run: video/RTSP -> predictions JSONL (+ annotated MP4).

The loop is deliberately thin: :class:`~autovision.inference.engine.InferenceEngine`
runs the network's own pre/post-processing per batch, records land in
``predictions.jsonl`` (one line per frame, boxes in original-frame pixels),
and rendered frames go to ``annotated.mp4`` when ``render=True`` and the
task emits detection boxes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from autovision.errors import AutoVisionError
from autovision.inference.engine import InferenceEngine
from autovision.inference.video import AnnotatedVideoWriter, VideoFrameStream
from autovision.types import PathLike

__all__ = ["StreamRunResult", "run_stream"]

logger = logging.getLogger("autovision")

#: Tensors above this element count are summarized instead of serialized
#: (dense outputs — depth maps, masks — would bloat the JSONL).
_JSON_MAX_ELEMENTS = 10_000


@dataclass(frozen=True)
class StreamRunResult:
    """Artifacts + counters of one :func:`run_stream` call."""

    out_dir: Path
    predictions_path: Path
    video_path: Path | None
    frames: int
    elapsed_s: float
    backend: str
    #: Number of distinct track ids over the run (0 when tracking is off).
    tracks: int = 0

    def summary(self) -> str:
        video = f", video={self.video_path}" if self.video_path else ""
        tracked = f", tracks={self.tracks}" if self.tracks else ""
        return (
            f"frames={self.frames} backend={self.backend} "
            f"elapsed={self.elapsed_s:.2f}s jsonl={self.predictions_path}{video}{tracked}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufc":
            return _jsonable(torch.from_numpy(np.ascontiguousarray(value)))
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if not torch.is_tensor(value):
        return value
    if value.numel() > _JSON_MAX_ELEMENTS:
        return {
            "shape": list(value.shape),
            "summarized": True,
            "mean": float(value.float().mean()),
            "min": float(value.float().min()),
            "max": float(value.float().max()),
        }
    return value.detach().cpu().tolist()


def _render_frame(
    frame: Any,
    record: dict[str, Any],
    class_names: dict[int, str] | None,
    tracks: list[Any] | None = None,
    track_class_names: dict[int, str] | None = None,
) -> Any:
    """Overlay detection boxes on one RGB frame; pass-through otherwise.

    With ``tracks`` the track id joins the label (``#7 cat 0.91``); the id
    is matched to its detection by the tracker's bbox. ``class_names``
    drives the DETECTION labels; ``track_class_names`` the track labels
    (``None`` keeps bare ids — e.g. frames with no class ids at all).
    """
    if "bboxes" not in record:
        return frame
    from PIL import ImageDraw

    from autovision.visualize.detection import draw_prediction

    tensor = torch.from_numpy(frame).permute(2, 0, 1)
    rendered = draw_prediction(tensor, record, class_names=class_names)
    if tracks:
        draw = ImageDraw.Draw(rendered)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            label_id = track.class_id
            name = (track_class_names or {}).get(label_id, str(label_id))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=2)
            draw.text(
                (x1 + 2, max(0.0, y2 - 12)),
                f"#{track.track_id} {name} {track.score:.2f}",
                fill=(255, 200, 0),
            )
    return np.asarray(rendered)


def run_stream(
    source: PathLike,
    out_dir: PathLike,
    *,
    engine: InferenceEngine | None = None,
    model: PathLike | None = None,
    backend: str = "torch",
    artifact: PathLike | None = None,
    device: str | None = None,
    providers: Sequence[str] | None = None,
    batch_size: int = 1,
    threshold: float = 0.5,
    render: bool = True,
    max_frames: int | None = None,
    class_names: dict[int, str] | None = None,
    track: bool = False,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
) -> StreamRunResult:
    """Run inference over a video file / RTSP stream and persist artifacts.

    Pass either a ready ``engine`` or ``model`` (+ backend selection, where
    ``onnx``/``tensorrt`` also need ``artifact``). Writes
    ``out_dir/predictions.jsonl`` always, and ``out_dir/annotated.mp4``
    when ``render`` and the task's predictions carry boxes.

    ``track=True`` (roadmap A3) runs a :class:`~autovision.tracking.
    BYTETracker` over the detections: each frame's JSONL entry gains
    ``tracks`` (``track_id`` / ``bbox`` / ``score`` / ``class_id`` /
    ``age``) and the rendered overlays carry the id — ids are stable within
    one run, in stream order (batch aggregation does not reorder frames).
    The tracker receives the source's real frame rate, so ``track_buffer``
    keeps its documented meaning (frames at that rate).
    """
    if batch_size < 1:
        raise AutoVisionError(f"batch_size must be >= 1, got {batch_size}.")
    if engine is None:
        if model is None:
            raise AutoVisionError("run_stream needs either engine= or model=.")
        engine = InferenceEngine(
            model,
            backend=backend,
            artifact=artifact,
            device=device,
            providers=providers,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    started = time.perf_counter()
    stream = VideoFrameStream(source, max_frames=max_frames)

    tracker = None
    if track:
        from autovision.tracking import BYTETracker

        # ``track_buffer`` counts frames AT the source frame rate.
        fps = stream.info.fps if stream.info.fps and stream.info.fps > 0 else 30.0
        tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            frame_rate=max(1, int(round(fps))),
        )

    writer: AnnotatedVideoWriter | None = None

    frames_done = 0
    track_ids: set[int] = set()
    buffer: list[Any] = []
    buffer_indices: list[int] = []

    def flush() -> None:
        nonlocal frames_done, writer
        if not buffer:
            return
        tensors = [
            torch.from_numpy(frame).permute(2, 0, 1) for frame in buffer
        ]
        records = engine.run_batch(tensors, threshold=threshold)
        for frame, index, record in zip(buffer, buffer_indices, records):
            entry: dict[str, Any] = {"frame": index}
            if stream.info.fps and stream.info.fps > 0:
                entry["time_s"] = round(index / stream.info.fps, 4)
            entry["predictions"] = {
                key: _jsonable(value) for key, value in record.items()
            }
            tracks = None
            track_names = class_names
            if tracker is not None and "bboxes" in record:
                import numpy as _np

                boxes = record["bboxes"].detach().cpu().numpy() if torch.is_tensor(
                    record["bboxes"]
                ) else _np.asarray(record["bboxes"])
                scores = record.get("scores")
                labels = record.get("labels")
                tracks = tracker.update(
                    boxes,
                    _np.asarray(scores.detach().cpu() if torch.is_tensor(scores) else scores),
                    _np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels)
                    if labels is not None
                    else None,
                )
                if labels is None:
                    # No class ids in this record — label the tracks with the
                    # bare id instead of a fabricated class-0 name.
                    track_names = None
                track_ids.update(t.track_id for t in tracks)
                entry["tracks"] = [
                    {
                        "track_id": t.track_id,
                        "bbox": list(t.bbox),
                        "score": round(float(t.score), 6),
                        "class_id": t.class_id,
                        "age": t.age,
                    }
                    for t in tracks
                ]
            handle.write(json.dumps(entry) + "\n")
            if render:
                if writer is None:
                    # Lazy: live streams report a 0x0 probe result.
                    height, width = frame.shape[:2]
                    writer = AnnotatedVideoWriter(
                        out_dir / "annotated.mp4",
                        fps=stream.info.fps,
                        frame_size=(width, height),
                    )
                writer.write(_render_frame(frame, record, class_names, tracks, track_names))
        frames_done += len(buffer)
        buffer.clear()
        buffer_indices.clear()

    try:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for index, frame in stream:
                buffer.append(frame)
                buffer_indices.append(index)
                if len(buffer) >= batch_size:
                    flush()
            flush()
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    video_path: Path | None = None
    if writer is not None:
        if frames_done:
            video_path = writer.path
        else:
            writer.path.unlink(missing_ok=True)
    logger.info(
        "Stream inference finished: %d frames in %.2fs (%s).",
        frames_done, elapsed, predictions_path,
    )
    return StreamRunResult(
        out_dir=out_dir,
        predictions_path=predictions_path,
        video_path=video_path,
        frames=frames_done,
        elapsed_s=elapsed,
        backend=engine.backend,
        tracks=len(track_ids),
    )
