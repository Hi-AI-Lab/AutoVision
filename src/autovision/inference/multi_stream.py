# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-source video/stream inference (roadmap 90d, phase 2).

One reader thread per source feeds a bounded per-source queue; a single
batcher thread builds mixed-source batches and is the only caller of the
inference engine (TensorRT/OpenVINO runners are not thread-safe). Each
source gets its own ``predictions.jsonl`` (+ ``annotated.mp4`` when
rendering); a dropping policy bounds memory when a source outruns the GPU.

Reconnect is for live sources only (``rtsp://`` / ``http(s)://``); a file
source ends at EOF. Backoff is exponential (0.5s, 1s, 2s, ... capped at
10s) with a per-source reconnect budget that RECOVERS (H3-5d): once a
reconnected source stays healthy for ~30 s of frames (300 when its fps is
unknown), the give-up budget and backoff restart from zero — the
cumulative ``reconnects`` count in the stats keeps the run's history and
is never reset.

A source whose inference keeps failing is aborted by a per-source circuit
breaker (H3-5f) instead of silently dragging the run into a partial
"success"; the other sources keep running.

``reader_factory`` is injectable so tests can exercise scheduling,
reconnect and backpressure without cameras or OpenCV; the default reader
reuses :class:`~autovision.inference.video.VideoFrameStream`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlparse

import numpy as np

from autovision.errors import AutoVisionError
from autovision.inference.engine import InferenceEngine
from autovision.inference.video import (
    AnnotatedVideoWriter,
    VideoFrameStream,
    VideoInfo,
    redact_source,
)
from autovision.results import InferenceResult
from autovision.types import PathLike
from autovision.visualize.detection import draw_prediction

__all__ = [
    "MultiStreamResult",
    "MultiStreamRunner",
    "StreamSource",
    "StreamStats",
    "run_multi_stream",
]

_LIVE_SCHEMES = {"rtsp", "rtmp", "http", "https"}

# H3-5f: per-source circuit breaker. A source whose last
# ``_FAILURE_WINDOW`` processed frames contain at least
# ``_FAILURE_MIN_FAILURES`` failures at a rate above ``_FAILURE_RATE`` is
# aborted (marked error, artifacts closed, removed from the rotation);
# persistent inference faults must not masquerade as a successful run.
_FAILURE_WINDOW = 20
_FAILURE_MIN_FAILURES = 10
_FAILURE_RATE = 0.5

logger = logging.getLogger("autovision.inference.multi_stream")


@dataclass(frozen=True)
class StreamSource:
    """One input; ``reconnect=None`` derives it from the URL scheme."""

    name: str
    url: str
    reconnect: bool | None = None
    max_reconnects: int = 10

    @property
    def is_live(self) -> bool:
        if self.reconnect is not None:
            return bool(self.reconnect)
        return urlparse(self.url).scheme.lower() in _LIVE_SCHEMES


@dataclass
class StreamStats:
    name: str
    url: str
    fps: float = 25.0
    frames_read: int = 0
    frames_written: int = 0
    frames_dropped: int = 0
    frames_failed: int = 0
    reconnects: int = 0
    #: H3-5f: set when the failure circuit breaker aborted this source
    #: mid-run — the source's artifacts are partial, not a clean run.
    aborted: bool = False
    errors: list[str] = field(default_factory=list)
    jsonl: Path | None = None
    video: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": redact_source(self.url),
            "fps": self.fps,
            "frames_read": self.frames_read,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "frames_failed": self.frames_failed,
            "reconnects": self.reconnects,
            "aborted": self.aborted,
            "errors": list(self.errors[-5:]),
            "jsonl": str(self.jsonl) if self.jsonl else None,
            "video": str(self.video) if self.video else None,
        }


@dataclass
class MultiStreamResult:
    out_dir: Path
    sources: dict[str, StreamStats]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "elapsed_s": self.elapsed_s,
            "sources": {
                name: stats.to_dict() for name, stats in sorted(self.sources.items())
            },
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.out_dir / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target

    def summary(self) -> str:
        parts = [
            f"{stats.name}: frames={stats.frames_written} "
            f"failed={stats.frames_failed} dropped={stats.frames_dropped} "
            f"reconnects={stats.reconnects}"
            for stats in sorted(self.sources.values(), key=lambda item: item.name)
        ]
        return (
            f"sources={len(self.sources)} elapsed={self.elapsed_s:.2f}s | "
            + " | ".join(parts)
        )


@dataclass(frozen=True)
class _FramePacket:
    source_index: int
    frame_index: int
    frame: np.ndarray
    timestamp: float


class _ReaderProtocol:
    info: VideoInfo

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _OpenCVReader(_ReaderProtocol):
    def __init__(self, url: str) -> None:
        self._stream = VideoFrameStream(url)
        self.info = self._stream.info

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        return iter(self._stream)

    def close(self) -> None:
        # VideoFrameStream releases in its iterator's finally block.
        return None


class MultiStreamRunner:
    """Run one engine over N sources with per-source artifacts and metrics."""

    def __init__(
        self,
        sources: Sequence[str | StreamSource],
        out_dir: PathLike,
        *,
        engine: Any,
        batch_size: int = 4,
        queue_size: int = 8,
        drop_policy: str = "drop_oldest",
        threshold: float = 0.5,
        render: bool = True,
        class_names: dict[int, str] | None = None,
        mask_format: str = "rle",
        max_json_elements: int = 10_000,
        max_frames_per_source: int | None = None,
        reader_factory: Callable[[str], _ReaderProtocol] | None = None,
    ) -> None:
        if not sources:
            raise AutoVisionError("MultiStreamRunner needs at least one source.")
        if batch_size < 1:
            raise AutoVisionError(f"batch_size must be >= 1, got {batch_size}.")
        if queue_size < 1:
            raise AutoVisionError(f"queue_size must be >= 1, got {queue_size}.")
        if mask_format not in {"rle", "dense", "none"}:
            raise AutoVisionError(
                "mask_format must be 'rle', 'dense', or 'none', "
                f"got {mask_format!r}."
            )
        if max_json_elements < 1:
            raise AutoVisionError(
                f"max_json_elements must be >= 1, got {max_json_elements}."
            )
        if drop_policy not in {"drop_oldest", "block"}:
            raise AutoVisionError(
                f"drop_policy must be 'drop_oldest' or 'block', got {drop_policy!r}."
            )
        if not callable(getattr(engine, "run_batch", None)):
            raise AutoVisionError("engine must expose run_batch(images, threshold=...).")
        if max_frames_per_source is not None and max_frames_per_source < 1:
            raise AutoVisionError("max_frames_per_source must be >= 1.")
        resolved: list[StreamSource] = []
        seen: set[str] = set()
        for index, source in enumerate(sources):
            entry = source if isinstance(source, StreamSource) else StreamSource(
                name=Path(str(source)).stem or f"source{index}", url=str(source)
            )
            if not entry.name or entry.name in seen:
                raise AutoVisionError(
                    f"Stream names must be unique and non-empty; got "
                    f"{entry.name!r} at index {index}."
                )
            if entry.max_reconnects < 0:
                raise AutoVisionError(
                    f"Stream '{entry.name}' has negative max_reconnects."
                )
            seen.add(entry.name)
            resolved.append(entry)
        self.sources = tuple(resolved)
        self.out_dir = Path(out_dir)
        self.engine = engine
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.drop_policy = drop_policy
        self.threshold = threshold
        self.render = render
        self.class_names = dict(class_names or {})
        self.mask_format = mask_format
        self.max_json_elements = max_json_elements
        self.max_frames_per_source = max_frames_per_source
        self.reader_factory = reader_factory or _OpenCVReader

    def run(self, *, max_seconds: float | None = None) -> MultiStreamResult:
        """Run every source until EOF / budget and write per-source artifacts.

        Each round assembles a mixed-source batch of up to ``batch_size``
        frames per still-active source (bounded by the current queue depth),
        then runs one ``engine.run_batch`` call over the whole batch — so a
        single fast source fills batches while the others are idle, and no
        source gets more than ``batch_size`` frames per round.
        """
        if max_seconds is not None and max_seconds <= 0:
            raise AutoVisionError("max_seconds must be > 0.")
        started = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        queues: list[queue.Queue[Any]] = [
            queue.Queue(maxsize=self.queue_size) for _ in self.sources
        ]
        done = [False for _ in self.sources]
        stop = threading.Event()
        lock = threading.Lock()
        stats = {
            source.name: StreamStats(
                name=source.name, url=redact_source(source.url)
            )
            for source in self.sources
        }
        stats_list = [stats[source.name] for source in self.sources]

        def _inc(stat: StreamStats, field_name: str, amount: int = 1) -> None:
            with lock:
                setattr(stat, field_name, getattr(stat, field_name) + amount)

        def _record_error(stat: StreamStats, message: str) -> None:
            with lock:
                stat.errors.append(message)
                # A persistent failure must not grow the list without bound.
                del stat.errors[:-100]

        def _put(packet: _FramePacket, stat: StreamStats, source_index: int) -> None:
            target = queues[source_index]
            if self.drop_policy == "block":
                try:
                    target.put(packet, timeout=0.2)
                    return
                except queue.Full:
                    _inc(stat, "frames_dropped")
                    return
            try:
                target.put_nowait(packet)
                return
            except queue.Full:
                pass
            try:
                target.get_nowait()
                _inc(stat, "frames_dropped")
            except queue.Empty:
                pass
            try:
                target.put_nowait(packet)
            except queue.Full:
                _inc(stat, "frames_dropped")

        def _reader(source_index: int, source: StreamSource) -> None:
            stat = stats_list[source_index]
            # H3-5d: consecutive reconnects without a proven recovery. This
            # drives both the backoff delay and the give-up budget, and is
            # reset after a healthy stretch; ``stat.reconnects`` stays the
            # cumulative lifetime count (history — never reset).
            attempts = 0
            try:
                while not stop.is_set():
                    reader: _ReaderProtocol | None = None
                    healthy = 0
                    recovery_frames = 300
                    try:
                        reader = self.reader_factory(source.url)
                        with lock:
                            stat.fps = float(reader.info.fps or 25.0)
                        # H3-5d: ~30 s of healthy frames at the source's
                        # reported rate proves the reconnect stuck; a fixed
                        # 300 frames when the rate is unknown.
                        fps_value = float(reader.info.fps or 0.0)
                        recovery_frames = (
                            int(fps_value * 30) if fps_value > 0 else 300
                        )
                        for _frame_index, frame in reader.frames():
                            if stop.is_set():
                                break
                            if self.max_frames_per_source is not None:
                                with lock:
                                    if stat.frames_read >= self.max_frames_per_source:
                                        break
                            _inc(stat, "frames_read")
                            healthy += 1
                            if attempts and healthy >= recovery_frames:
                                # The source recovered, so it may spend its
                                # reconnect budget again (and the backoff
                                # restarts from the first step).
                                attempts = 0
                            _put(
                                _FramePacket(
                                    source_index=source_index,
                                    frame_index=int(stat.frames_read) - 1,
                                    frame=frame,
                                    timestamp=time.time(),
                                ),
                                stat,
                                source_index,
                            )
                        if self.max_frames_per_source is not None and stat.frames_read >= self.max_frames_per_source:
                            break
                        if not source.is_live:
                            break
                        raise AutoVisionError("live stream ended")
                    except Exception as error:  # noqa: BLE001 - source fault
                        _record_error(stat, f"{type(error).__name__}: {error}")
                        if stop.is_set() or not source.is_live:
                            break
                        if attempts >= source.max_reconnects:
                            # H3-5d: the give-up budget counts reconnects
                            # since the last recovery, not the run total.
                            break
                        _inc(stat, "reconnects")
                        attempts += 1
                        delay = min(10.0, 0.5 * (2 ** min(attempts - 1, 4)))
                        stop.wait(delay)
                    finally:
                        if reader is not None:
                            try:
                                reader.close()
                            except Exception:  # noqa: BLE001 - best effort
                                pass
            finally:
                done[source_index] = True

        threads = [
            threading.Thread(
                target=_reader, args=(index, source), name=f"av-{source.name}", daemon=True
            )
            for index, source in enumerate(self.sources)
        ]
        for thread in threads:
            thread.start()

        jsonl_handles: dict[int, Any] = {}
        videos: dict[int, AnnotatedVideoWriter] = {}
        active = set(range(len(self.sources)))
        # H3-5f: sliding failure window per source (1 = failed frame).
        failure_windows: dict[int, deque[int]] = {
            index: deque(maxlen=_FAILURE_WINDOW)
            for index in range(len(self.sources))
        }
        aborted: set[int] = set()
        # H3-5e: per-source rotation counter for resolution changes.
        rotations: dict[int, int] = {}

        def _abort_source(index: int, stat: StreamStats, reason: str) -> None:
            """H3-5f: trip the source's circuit breaker.

            The source stops being consumed, its artifacts are closed and
            it is marked in the summary; the other sources keep running.
            Its reader winds down through its own EOF/reconnect budget.
            """
            aborted.add(index)
            stat.aborted = True
            _record_error(stat, reason)
            active.discard(index)
            handle = jsonl_handles.get(index)
            if handle is not None:
                try:
                    handle.close()
                except OSError:  # noqa: BLE001 - cleanup must not outrun containment
                    logger.debug("Closing the aborted source's JSONL failed.")
            writer = videos.pop(index, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001 - MP4 finalize fails exactly
                    # when the disk is full / codec wedged — the very fault
                    # that tripped this abort. Escaping the except handler
                    # would re-kill the whole run (review P2-1).
                    logger.warning(
                        "Finalizing the aborted source's video failed; the "
                        "partial segment stays on disk."
                    )
            logger.warning(
                "Source '%s' aborted by the failure circuit breaker: %s",
                stat.name,
                reason,
            )

        def _note_frame_failure(index: int, stat: StreamStats, message: str) -> None:
            """Account one failed frame; trip the breaker past the threshold."""
            _inc(stat, "frames_failed")
            _record_error(stat, message)
            window = failure_windows[index]
            window.append(1)
            failures = sum(window)
            if (
                failures >= _FAILURE_MIN_FAILURES
                and failures / len(window) > _FAILURE_RATE
            ):
                _abort_source(
                    index,
                    stat,
                    f"{failures} failed frames in the last {len(window)} "
                    f"({failures / len(window):.0%} > "
                    f"{_FAILURE_RATE:.0%})",
                )

        def _note_frame_success(index: int) -> None:
            failure_windows[index].append(0)
        # INFER-01: last-resort net — see the except around the main loop.
        run_error: Exception | None = None
        try:
            for index, source in enumerate(self.sources):
                source_dir = self.out_dir / source.name
                source_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = source_dir / "predictions.jsonl"
                stats_list[index].jsonl = jsonl_path
                jsonl_handles[index] = open(
                    jsonl_path, "w", encoding="utf-8"
                )
            while not stop.is_set():
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    stop.set()
                    break
                # Each round takes up to ``batch_size`` frames per source
                # (H3-5a), bounded by what is queued right now — a fast source
                # fills batches while idle sources contribute nothing.
                batch: list[_FramePacket] = []
                for index in list(active):
                    for _ in range(self.batch_size):
                        try:
                            batch.append(queues[index].get_nowait())
                        except queue.Empty:
                            break
                if not batch:
                    if all(done[index] for index in active):
                        break
                    time.sleep(0.01)
                    continue
                frames = [packet.frame for packet in batch]
                try:
                    records = self.engine.run_batch(frames, threshold=self.threshold)
                except Exception as error:  # noqa: BLE001 - keep other sources alive
                    logger.warning(
                        "Inference batch failed (%d frames): %s", len(batch), error
                    )
                    for packet in batch:
                        if packet.source_index in aborted:
                            continue
                        _note_frame_failure(
                            packet.source_index,
                            stats_list[packet.source_index],
                            f"inference failed: {error}",
                        )
                    continue
                if len(records) != len(batch):
                    logger.warning(
                        "Engine returned %d records for %d frames",
                        len(records),
                        len(batch),
                    )
                    for packet in batch:
                        if packet.source_index in aborted:
                            continue
                        _note_frame_failure(
                            packet.source_index,
                            stats_list[packet.source_index],
                            "engine returned "
                            f"{len(records)} records for {len(batch)} frames",
                        )
                    continue
                for packet, record in zip(batch, records):
                    index = packet.source_index
                    stat = stats_list[index]
                    if index in aborted:
                        # H3-5f: the breaker tripped mid-batch — the source's
                        # artifacts are closed, so skip its stragglers.
                        continue
                    # Record validation/serialization is isolated per source
                    # (H3-1e): one malformed record must not kill the whole
                    # multi-source run, so the failure lands in that source's
                    # stats and the remaining packets proceed.
                    try:
                        payload = {
                            "source": stat.name,
                            "frame_index": packet.frame_index,
                            "timestamp": packet.timestamp,
                            "predictions": InferenceResult.from_record(
                                record
                            ).to_json(
                                masks=self.mask_format,
                                max_elements=self.max_json_elements,
                            ),
                        }
                        jsonl_handles[index].write(
                            json.dumps(payload, allow_nan=False) + "\n"
                        )
                        jsonl_handles[index].flush()
                    except Exception as error:  # noqa: BLE001 - keep other sources alive
                        _note_frame_failure(
                            index, stat, f"serialization failed: {error}"
                        )
                        continue
                    _inc(stat, "frames_written")
                    _note_frame_success(index)
                    if self.render:
                        # INFER-01: writer setup/writes are isolated per
                        # source like every other failure mode here — a
                        # single source's disk-full/codec failure used to
                        # propagate out of run() and take down the whole
                        # multi-source job with the summary lost. The
                        # failed source trips its breaker; the rest run on.
                        try:
                            writer = videos.get(index)
                            height, width = packet.frame.shape[:2]
                            if writer is None or writer.frame_size != (width, height):
                                if writer is not None:
                                    writer.close()
                                    # H3-5e: a resolution change rotates the
                                    # file name instead of overwriting the
                                    # earlier segment; JSONL is unaffected.
                                    rotations[index] = rotations.get(index, 0) + 1
                                    segment = f"annotated-{rotations[index]:04d}.mp4"
                                else:
                                    segment = "annotated.mp4"
                                source_dir = self.out_dir / stat.name
                                writer = AnnotatedVideoWriter(
                                    source_dir / segment,
                                    fps=stat.fps,
                                    frame_size=(width, height),
                                )
                                videos[index] = writer
                                stat.video = writer.path
                            from PIL import Image

                            try:
                                rendered = draw_prediction(
                                    Image.fromarray(packet.frame),
                                    record,
                                    class_names=self.class_names,
                                )
                            except Exception as error:  # noqa: BLE001
                                _record_error(stat, f"render failed: {error}")
                                rendered = packet.frame
                            writer.write(np.asarray(rendered))
                        except Exception as error:  # noqa: BLE001 - keep other sources alive
                            _abort_source(
                                index,
                                stat,
                                f"video writer failed: "
                                f"{type(error).__name__}: {error}",
                            )
                            continue
                for index in list(active):
                    if done[index] and queues[index].empty():
                        active.discard(index)
                if not active:
                    break
        except Exception as error:  # noqa: BLE001 - INFER-01: an uncontained
            # failure (e.g. the artifact setup loop above) must still leave
            # a summary on disk — per-source failures are contained at their
            # own level, this is the last-resort net.
            run_error = error
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=5.0)
            for handle in jsonl_handles.values():
                try:
                    handle.close()
                except OSError:  # noqa: BLE001 - review P2-1: best-effort
                    logger.debug("Closing a JSONL handle failed during cleanup.")
            for writer in videos.values():
                try:
                    writer.close()
                except Exception:  # noqa: BLE001 - review P2-1: best-effort
                    logger.warning(
                        "Finalizing a video writer failed during cleanup; the "
                        "partial segment stays on disk."
                    )
        elapsed = time.perf_counter() - started
        result = MultiStreamResult(
            out_dir=self.out_dir,
            sources=stats,
            elapsed_s=elapsed,
        )
        if run_error is not None:
            # INFER-01: the summary is saved even on an uncontained failure —
            # and a failing SAVE must not mask the original error (review
            # P2-1: disk exhaustion is exactly when both fail).
            try:
                result.save()
            except Exception as save_error:  # noqa: BLE001
                logger.error(
                    "Saving the multi-stream summary failed (%s); the "
                    "original run error follows.", save_error,
                )
                raise run_error from save_error
            raise run_error
        result.save()
        return result


def _sources(values: Sequence[str | StreamSource] | str) -> list[StreamSource]:
    parts: list[str | StreamSource]
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
    else:
        parts = list(values)
    result: list[StreamSource] = []
    for index, value in enumerate(parts):
        if isinstance(value, StreamSource):
            result.append(value)
        else:
            result.append(
                StreamSource(
                    name=Path(str(value)).stem or f"source{index}",
                    url=str(value),
                )
            )
    return result


def run_multi_stream(
    sources: Sequence[str | StreamSource] | str,
    out_dir: PathLike,
    *,
    model: PathLike | None = None,
    backend: str = "torch",
    artifact: PathLike | None = None,
    device: str | None = None,
    providers: Sequence[str] | None = None,
    openvino_device: str = "CPU",
    **kwargs: Any,
) -> MultiStreamResult:
    """Convenience wrapper mirroring :func:`run_stream` for N sources.

    ``sources`` is a comma-separated string or a sequence; each entry gets
    a name from its file stem (provide :class:`StreamSource` for explicit
    names). ``kwargs`` are the runner scheduling/render options.
    """
    if model is None:
        raise AutoVisionError("run_multi_stream needs model= (or build a runner).")
    engine = InferenceEngine(
        model,
        backend=backend,
        artifact=artifact,
        device=device,
        providers=providers,
        openvino_device=openvino_device,
    )
    runner = MultiStreamRunner(_sources(sources), out_dir, engine=engine, **kwargs)
    return runner.run()
