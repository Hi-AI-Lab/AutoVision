# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Video / stream I/O for the reference inference runtime.

OpenCV's FFmpeg backend reads local files and network streams (``rtsp://``,
``http://``) through one interface; frames are handed to the runtime as
RGB ``uint8`` ``(H, W, 3)`` arrays — the same channel order the training
data layer's image readers produce, so pre/post-processing stays identical
between file, camera, and stream sources.

OpenCV is the ``video`` extra (``pip install -e '.[video]'``); importing
this module without it succeeds, use fails with guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from autovision.errors import AutoVisionError
from autovision.types import PathLike

__all__ = ["AnnotatedVideoWriter", "VideoFrameStream", "VideoInfo"]


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise AutoVisionError(
            "Video/stream inference needs OpenCV: pip install -e '.[video]'"
        ) from error
    return cv2


@dataclass(frozen=True)
class VideoInfo:
    """Source properties; ``fps``/``frame_count`` are 0 when unknown (live streams)."""

    width: int
    height: int
    fps: float
    frame_count: int


class VideoFrameStream:
    """Iterate a video file or network stream as ``(index, RGB frame)``.

    ``source`` is anything OpenCV opens: a path, ``rtsp://``,
    ``http://``, or a camera index. Iteration ends at the source's last
    frame, at ``max_frames``, or on read failure (live streams: a dropped
    read ends the run — reconnect handling is out of scope). The
    ``*_timeout_ms`` knobs are best-effort FFmpeg-backend properties for
    network sources; local files ignore them.
    """

    def __init__(
        self,
        source: PathLike,
        *,
        max_frames: int | None = None,
        open_timeout_ms: int | None = 10_000,
        read_timeout_ms: int | None = 5_000,
    ) -> None:
        cv2 = _require_cv2()
        self.source = str(source)
        self.max_frames = max_frames
        self._capture = cv2.VideoCapture()
        # Timeouts must be set on the empty capture, before open().
        for prop, value in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout_ms),
            ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_ms),
        ):
            if value is not None and hasattr(cv2, prop):
                self._capture.set(getattr(cv2, prop), value)
        self._capture.open(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            raise AutoVisionError(
                f"Could not open video source '{self.source}' — check the path "
                "or stream URL (RTSP/HTTP sources need OpenCV's FFmpeg backend)."
            )
        self.info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        cv2 = _require_cv2()
        index = 0
        try:
            while self.max_frames is None or index < self.max_frames:
                ok, frame = self._capture.read()
                if not ok:
                    break
                yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            self._capture.release()


class AnnotatedVideoWriter:
    """Write RGB frames to an MP4 file (OpenCV ``mp4v`` codec)."""

    def __init__(
        self, path: PathLike, *, fps: float, frame_size: tuple[int, int]
    ) -> None:
        cv2 = _require_cv2()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps and fps > 0 else 25.0,
            frame_size,
        )
        if not self._writer.isOpened():
            raise AutoVisionError(
                f"Could not open an MP4 writer for '{self.path}' (codec mp4v)."
            )

    def write(self, frame: Any) -> None:
        cv2 = _require_cv2()
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()
