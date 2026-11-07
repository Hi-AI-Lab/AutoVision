# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference serving (roadmap F2): FastAPI + batch + metrics.

A deliberately MINIMAL, testable reference — the prerequisite for
inference monitoring, not a production gateway:

- ``POST /predict`` — JSON body ``{"images": ["<base64>", ...]}`` (or a
  single ``{"image": ...}``); results come back in input order. Pre/post
  processing is the network's own (the same source as evaluation and the
  stream runtime), so served predictions match offline readings.
- ``GET /healthz`` — liveness + the loaded backend/model identity.
- ``GET /readyz`` — readiness: 200 only when every configured variant's
  inference engine is callable, 503 otherwise.
- ``GET /drift`` — prediction-distribution drift (PSI) against an optional
  baseline; ``enabled: false`` when no baseline was configured.
- ``GET /metrics`` — Prometheus exposition text (requests, errors, images,
  latency summary) for scraping.

Production guards (``create_app`` is a library entry point where
authentication stays opt-in; ``autovision.serving.server.serve`` enforces the
fail-closed posture for non-loopback binds; size/concurrency limits default to
production-safe values):

- ``api_key=`` enables a constant-time bearer/API-key check; without it the
  app stays unauthenticated (documented boundary — ``serve()`` refuses
  non-loopback binds without a key, or requires an explicit
  ``AUTOVISION_SERVE_ALLOW_INSECURE=1``).
- ``max_body_bytes`` bounds the decoded request body while streaming.
- ``max_image_pixels`` rejects decompression bombs before decoding;
  ``max_request_pixels`` bounds the decoded pixel total of one request.
- ``rate_limit_per_minute`` enables a per-client token bucket (per process).
- ``variants=`` / ``variant_weights=`` provide A/B or canary routing across
  ready engines. The bucketing key is a server-issued id (H2-5a), and
  ``X-AB-Variant`` steering is only honored with ``allow_variant_override=True``
  — clients cannot pick their variant.
- Inference runs in a worker thread; a per-variant semaphore bounds
  concurrent engine calls (TensorRT/OpenVINO runners are not thread-safe, so
  the default serializes calls while keeping the event loop responsive).

Explicitly OUT of scope (documented boundary, README): TLS and multi-worker
orchestration live in ``autovision.serving.server.serve``; the monitoring
layers built on top of ``/metrics`` and ``/drift`` (alerting, full A/B
analysis) are deployment/product concerns.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import re
import secrets
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from autovision.errors import AutoVisionError
from autovision.results import InferenceResult
from autovision.runtime.env import Env
from autovision.serving.drift import (
    DriftBaseline,
    DriftMonitor,
    load_drift_baseline,
)
from autovision.serving.metrics import MetricsRegistry
from autovision.serving.ratelimit import TokenBucketRateLimiter

__all__ = ["create_app"]

logger = logging.getLogger("autovision.serving")

_REQUESTS_HELP = "Served requests by endpoint and status"
_VARIANTS_HELP = "Served requests by variant and status"
_ERRORS_HELP = "Failed predictions"
_DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
# 16M pixels covers a 4000x4000 industrial camera frame (H2-2c); the old 40M
# default decoded a ~156KB base64 payload into ~120MB of RGB raster.
_DEFAULT_MAX_IMAGE_PIXELS = 16_000_000
# Client-supplied request ids must match this shape before they reach the log
# stream or response headers (H2-5b): a raw header would let a client forge
# log lines via newlines or bloat every record.
_REQUEST_ID_PATTERN = re.compile(r"^[0-9a-zA-Z-]{1,64}$")


class _PayloadTooLarge(AutoVisionError):
    """A request body or decoded image exceeded its configured size limit."""


class _ClientInputError(AutoVisionError):
    """Client-input validation failure whose message is safe to echo back.

    Messages of this type are composed only of client-supplied values and
    configured limits — never server internals (paths, backend details) — so
    the 400 body may carry them verbatim. A bare ``AutoVisionError`` raised
    across the engine/serialization boundary is treated as potentially
    sensitive instead and is masked before it reaches the response (H2-5c).
    """


def _decode_image(payload: str, *, max_pixels: int) -> Any:
    """base64 image payload → HWC uint8 numpy array (RGB)."""
    import numpy as np
    from PIL import Image

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as error:  # noqa: BLE001 - malformed payload is a 400
        # binascii's message only describes the client's own bytes, so it is
        # safe to echo (H2-5c: client-input errors keep their original text).
        raise _ClientInputError(f"Invalid base64 image payload: {error}") from error
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise _PayloadTooLarge(
                    f"Image is {width}x{height} ({width * height} pixels), "
                    f"which exceeds max_image_pixels={max_pixels}."
                )
            frame = image.convert("RGB")
    except AutoVisionError:
        raise
    except Image.DecompressionBombError as error:
        raise _PayloadTooLarge(
            "Image exceeds Pillow's decompression-bomb limit."
        ) from error
    except Exception as error:  # noqa: BLE001 - undecodable payload is a 400
        # PIL's own message embeds an object address — keep the body stable.
        raise _ClientInputError(
            "Unsupported image payload — expected base64-encoded PNG/JPEG bytes."
        ) from error
    return np.array(frame, copy=True)


async def _read_body_limited(request: Any, limit: int) -> bytes:
    """Read the request body, aborting as soon as ``limit`` is exceeded.

    ``Content-Length`` is checked first (cheap rejection); the streaming
    loop also enforces the cap for chunked/lying clients.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError as error:
            raise AutoVisionError(f"Invalid Content-Length header: {declared!r}.") from error
        if declared_bytes > limit:
            raise _PayloadTooLarge(
                f"Request body of {declared_bytes} bytes exceeds "
                f"max_body_bytes={limit}."
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _PayloadTooLarge(
                f"Request body exceeds max_body_bytes={limit}."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _sanitize_request_id(value: str) -> str:
    """Clamp a client-supplied ``X-Request-ID`` before it hits logs/headers.

    An unsanitized header lets a client forge log lines (a newline starts a
    new record) or bloat every log entry. Conforming ids pass through
    unchanged; anything else is truncated to 64 characters with foreign
    characters replaced by ``?``.
    """
    if _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    # Truncate first, then neutralize the remainder: the bound holds the
    # length even for hostile input, and the replacement removes control
    # characters that could fake record boundaries.
    return re.sub(r"[^0-9a-zA-Z-]", "?", value[:64])


def _authorized(request: Any, api_key: str | None) -> bool:
    """Constant-time API-key check (no-op when ``api_key`` is not set)."""
    if api_key is None:
        return True
    header = request.headers.get("authorization", "")
    candidate = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not candidate:
        candidate = request.headers.get("x-api-key", "")
    # compare_digest rejects non-ASCII str inputs with TypeError; compare the
    # UTF-8 bytes so a malformed credential is a 401, not an uncaught 500.
    return secrets.compare_digest(
        candidate.encode("utf-8"), api_key.encode("utf-8")
    )


def _client_key(request: Any, *, trusted_proxy_hops: int = 0) -> str:
    """The rate-limit bucket key (SERVE-06).

    ``trusted_proxy_hops=0`` (default) trusts nobody: the socket peer is
    the key, which under a reverse proxy collapses EVERY client into one
    bucket. With ``hops=N`` the client is the N-th entry counting from the
    RIGHT of ``X-Forwarded-For`` — every trusted proxy APPENDS the peer it
    saw, so with one proxy the chain is ``"client"`` (or
    ``"spoofed, client"``) and the real client is the last entry; each
    extra trusted proxy shifts the selection one entry left. Spoofed
    entries ahead of the client shift the selection by one, so only the
    operator-configured proxy count is authoritative.
    """
    if trusted_proxy_hops > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            parts = [part.strip() for part in forwarded.split(",") if part.strip()]
            index = len(parts) - trusted_proxy_hops
            if index >= 0:
                return parts[index]
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return str(host or "unknown")


def _engine_info(engine: Any) -> dict[str, Any]:
    network = getattr(engine, "network", None)
    artifact = getattr(engine, "artifact", None)
    # Probes are unauthenticated by design; expose only the filename, never
    # the deployment's absolute path.
    artifact_name = Path(artifact).name if artifact else None
    return {
        "backend": getattr(engine, "backend", None),
        "model": type(network).__name__ if network is not None else None,
        "artifact": artifact_name,
        "openvino_device": getattr(engine, "openvino_device", None),
    }


def _serialize_records(
    records: list[dict[str, Any]],
    names: dict[int, str],
    mask_format: str,
    max_json_elements: int,
) -> list[dict[str, Any]]:
    """RLE/JSON conversion off the event loop (heavy for instance masks)."""
    results: list[dict[str, Any]] = []
    for record in records:
        entry = InferenceResult.from_record(record).to_json(
            masks=mask_format, max_elements=max_json_elements
        )
        depth = entry.get("depth")
        if isinstance(depth, dict) and depth.get("summarized"):
            # H2-5f: a summarized depth map is a shape/dtype digest, not
            # pixels — silently unusable downstream. Flag it so clients can
            # fall back to the file/object-store channel.
            entry["depth_truncated"] = True
        if names and isinstance(entry.get("labels"), list):
            entry["class_names"] = [
                names.get(int(label), str(label)) for label in entry["labels"]
            ]
        results.append(entry)
    return results


def validate_serve_config(
    *,
    max_batch: int = 16,
    api_key: str | None = None,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS,
    max_request_pixels: int | None = None,
    max_concurrency: int = 1,
    queue_timeout_s: float | None = 30.0,
    inference_timeout_s: float | None = 120.0,
    decode_timeout_s: float | None = None,
    warmup_timeout_s: float = 60.0,
    mask_format: str = "rle",
    max_json_elements: int = 10_000,
    rate_limit_per_minute: int | None = None,
    rate_limit_burst: int | None = None,
    trusted_proxy_hops: int = 0,
    trust_proxy_headers: bool = False,
    threshold: float = 0.5,
    allow_variant_override: bool = False,
    docs_enabled: bool = False,
    protect_observability_endpoints: bool = True,
    drift_window: int = 1000,
    drift_threshold: float = 0.2,
    drift_min_samples: int = 30,
    backend: str = "torch",
    artifact: Any = None,
    device: Any = None,
    providers: Any = None,
    openvino_device: Any = None,
    class_names: Any = None,
    drift_baseline: Any = None,
    engine: Any = None,
    variant_weights: Any = None,
) -> None:
    """SERVE-07: pure validation of the serving knobs — no I/O, no imports
    of heavy modules, callable from the workers>1 PARENT process.

    The workers>1 launch serializes this config into the environment and
    every worker rebuilds :func:`create_app`; without this check a bad
    value exploded inside each worker (uvicorn restart loop) with no clean
    error at the CLI. Keyword-only with the create_app defaults so a
    partial config still validates the caller's actual values. The
    signature covers every pure-validatable ``create_app`` knob (plus the
    identity pass-throughs) so an unknown key still raises the TypeError
    the parent turns into a named error; knobs whose only check happens
    against a live engine (engine/artifact identity) stay with create_app.
    """
    # Review P2-1: strict types — a string ``threshold="0.9"`` used to pass
    # validation while the request path operated on the raw string (every
    # request 500'd). Numeric knobs reject non-numeric types outright.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise AutoVisionError(
            f"threshold must be a number in [0, 1], got {threshold!r}."
        )
    if not 0.0 <= float(threshold) <= 1.0:
        raise AutoVisionError(f"threshold must be in [0, 1], got {threshold!r}.")
    if warmup_timeout_s is not None and warmup_timeout_s < 0:
        raise AutoVisionError(
            f"warmup_timeout_s must be >= 0 or None, got {warmup_timeout_s!r}."
        )
    for name, value in (
        ("allow_variant_override", allow_variant_override),
        ("docs_enabled", docs_enabled),
        ("protect_observability_endpoints", protect_observability_endpoints),
        ("trust_proxy_headers", trust_proxy_headers),
    ):
        if not isinstance(value, bool):
            raise AutoVisionError(f"{name} must be a bool, got {value!r}.")
    if isinstance(drift_window, bool) or not isinstance(drift_window, int):
        raise AutoVisionError(f"drift_window must be an integer, got {drift_window!r}.")
    if drift_window < 1:
        raise AutoVisionError(f"drift_window must be >= 1, got {drift_window}.")
    if drift_threshold < 0:
        raise AutoVisionError(
            f"drift_threshold must be >= 0, got {drift_threshold}."
        )
    if drift_min_samples < 0:
        raise AutoVisionError(
            f"drift_min_samples must be >= 0, got {drift_min_samples}."
        )
    # Review P2-1: the rate-limit and backend ranges the TokenBucket /
    # InferenceEngine constructors enforce were only reachable INSIDE the
    # worker before — a workers>1 launch with a bad backend restarted in a
    # loop. The lazy import keeps this function import-light.
    if rate_limit_per_minute is not None and rate_limit_per_minute < 1:
        raise AutoVisionError(
            f"rate_limit_per_minute must be >= 1, got {rate_limit_per_minute}."
        )
    if rate_limit_burst is not None and rate_limit_burst < 1:
        raise AutoVisionError(
            f"rate_limit_burst must be >= 1, got {rate_limit_burst}."
        )
    from autovision.inference.engine import _BACKENDS

    if backend not in _BACKENDS:
        raise AutoVisionError(
            f"Unknown inference backend '{backend}'; expected one of "
            f"{list(_BACKENDS)}."
        )
    if max_batch < 1:
        raise AutoVisionError(f"max_batch must be >= 1, got {max_batch}.")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise AutoVisionError("api_key must be a non-empty string when provided.")
    if max_body_bytes < 1:
        raise AutoVisionError(f"max_body_bytes must be >= 1, got {max_body_bytes}.")
    if max_image_pixels < 1:
        raise AutoVisionError(
            f"max_image_pixels must be >= 1, got {max_image_pixels}."
        )
    if (
        max_request_pixels is not None
        and max_request_pixels < max_image_pixels
    ):
        raise AutoVisionError(
            f"max_request_pixels must be >= max_image_pixels={max_image_pixels}, "
            f"got {max_request_pixels}."
        )
    if max_concurrency < 1:
        raise AutoVisionError(
            f"max_concurrency must be >= 1, got {max_concurrency}."
        )
    if queue_timeout_s is not None and queue_timeout_s <= 0:
        raise AutoVisionError(
            f"queue_timeout_s must be > 0 or None, got {queue_timeout_s}."
        )
    if inference_timeout_s is not None and inference_timeout_s <= 0:
        raise AutoVisionError(
            "inference_timeout_s must be > 0 or None, got "
            f"{inference_timeout_s}."
        )
    if decode_timeout_s is not None and decode_timeout_s <= 0:
        raise AutoVisionError(
            f"decode_timeout_s must be > 0 or None, got {decode_timeout_s}."
        )
    if warmup_timeout_s < 0:
        raise AutoVisionError(
            f"warmup_timeout_s must be >= 0, got {warmup_timeout_s}."
        )
    if mask_format not in {"rle", "dense", "none"}:
        raise AutoVisionError(
            "mask_format must be 'rle', 'dense', or 'none', "
            f"got {mask_format!r}."
        )
    if max_json_elements < 1:
        raise AutoVisionError(
            f"max_json_elements must be >= 1, got {max_json_elements}."
        )
    if rate_limit_burst is not None and rate_limit_per_minute is None:
        raise AutoVisionError(
            "rate_limit_burst requires rate_limit_per_minute."
        )
    if trusted_proxy_hops < 0:
        raise AutoVisionError(
            f"trusted_proxy_hops must be >= 0, got {trusted_proxy_hops}."
        )


def create_app(
    model: Any = None,
    *,
    engine: Any = None,
    variants: dict[str, Any] | None = None,
    variant_weights: dict[str, float] | None = None,
    backend: str = "torch",
    artifact: "str | Path | None" = None,
    device: str | None = None,
    providers: "list[str] | None" = None,
    openvino_device: str = "CPU",
    threshold: float = 0.5,
    max_batch: int = 16,
    allow_variant_override: bool = False,
    class_names: dict[int, str] | None = None,
    api_key: str | None = None,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS,
    max_request_pixels: int | None = None,
    max_concurrency: int = 1,
    queue_timeout_s: float | None = 30.0,
    inference_timeout_s: float | None = 120.0,
    mask_format: str = "rle",
    max_json_elements: int = 10_000,
    rate_limit_per_minute: int | None = None,
    rate_limit_burst: int | None = None,
    trust_proxy_headers: bool = False,
    trusted_proxy_hops: int = 0,
    decode_timeout_s: float | None = None,
    warmup_timeout_s: float = 60.0,
    docs_enabled: bool = False,
    protect_observability_endpoints: bool = True,
    drift_baseline: "str | Path | dict[str, Any] | DriftBaseline | None" = None,
    drift_window: int = 1000,
    drift_threshold: float = 0.2,
    drift_min_samples: int = 30,
) -> Any:
    """Build the FastAPI app around a task model, engine, or variant engines.

    ``model`` may be an exported ``.pt`` path or a ``TaskNetwork`` (it gets
    wrapped in an :class:`~autovision.inference.InferenceEngine`); pass a
    ready engine via ``engine=`` instead. ``variants`` maps variant names to
    ready engines for A/B/canary routing (mutually exclusive with
    ``model=``/``engine=``); ``variant_weights`` defaults to equal splits.
    The bucketing key is a server-issued per-request id, so clients cannot
    steer their assignment via ``X-Request-ID`` (H2-5a);
    ``allow_variant_override=True`` re-enables the ``X-AB-Variant`` header
    for testing (unknown names still get a 400).

    ``api_key`` enables an opt-in bearer/``X-API-Key`` check. The other
    guards default to production-safe values; ``max_concurrency`` stays at
    1 because the TensorRT/OpenVINO runners are not thread-safe.
    ``max_request_pixels`` bounds the decoded pixel total of one request's
    batch (413 when exceeded); it defaults to ``max_batch * max_image_pixels``
    and must stay >= ``max_image_pixels`` so the two 413 messages (per-image
    vs. aggregate) keep their distinct meanings.
    ``queue_timeout_s`` bounds the wait for an inference slot (503 on
    overload; ``None`` waits indefinitely). ``inference_timeout_s`` bounds
    the engine call itself (504 on timeout; the timed-out variant is then
    marked unresponsive and later requests for it return 503 until restart,
    because a thread stuck in a non-thread-safe backend cannot be cancelled
    safely — other variants keep serving, H2-3).
    ``docs_enabled`` exposes
    FastAPI's ``/docs``/``/openapi.json`` (off by default: they reveal the
    inference schema). ``protect_observability_endpoints`` puts ``/metrics``
    and ``/drift`` behind the same API key (probes stay public for Kubernetes).
    ``rate_limit_per_minute`` is per process (see the
    module docstring). ``drift_baseline`` accepts a baseline path/dict or a
    :class:`~autovision.serving.drift.DriftBaseline`.

    New in 0.6.1: ``decode_timeout_s`` (SERVE-04: the decode budget is
    separate from the inference budget; a decode timeout never poisons the
    variant; ``None`` computes ``min(10, inference_timeout_s / 2)`` and
    there is no explicit off switch), ``warmup_timeout_s`` (SERVE-05: the
    boot waits at most this long per engine before failing; ``0``/``None``
    disable warmup), ``trusted_proxy_hops`` (SERVE-06: the rate-limit key
    selects the client N entries from the right of ``X-Forwarded-For``;
    ``trust_proxy_headers`` is deprecated and maps to 1).
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as error:  # pragma: no cover - import guard path
        raise AutoVisionError(
            "The 'fastapi' package is required for the serving reference. "
            "Install it with: pip install 'autovision[serving]'."
        ) from error

    # SERVE-07: the pure knob validation lives in validate_serve_config so
    # the workers>1 launch path can reject a bad config in the PARENT
    # process — the old shape exploded inside every worker (uvicorn restart
    # loop, no clean CLI error). api_key is resolved from the environment
    # here first because the validation is type-only.
    if api_key is None:
        # Prefer the environment over a CLI argument: process listings and
        # shell history would otherwise expose the key.
        api_key = Env.SERVING_API_KEY.value
    validate_serve_config(
        max_batch=max_batch,
        api_key=api_key,
        max_body_bytes=max_body_bytes,
        max_image_pixels=max_image_pixels,
        max_request_pixels=max_request_pixels,
        max_concurrency=max_concurrency,
        queue_timeout_s=queue_timeout_s,
        inference_timeout_s=inference_timeout_s,
        decode_timeout_s=decode_timeout_s,
        warmup_timeout_s=warmup_timeout_s,
        mask_format=mask_format,
        max_json_elements=max_json_elements,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_burst=rate_limit_burst,
        trusted_proxy_hops=trusted_proxy_hops,
    )
    if max_request_pixels is None:
        # The whole decoded batch is held in memory until the engine returns,
        # so the aggregate budget scales with both knobs; with this default
        # the aggregate check cannot fire before the per-image cap, and an
        # explicit (smaller) value is the only way to tighten it.
        max_request_pixels = max_batch * max_image_pixels
    # SERVE-06: proxy trust is now a HOP COUNT. The old bool is deprecated
    # (AGENTS.md §3.4: warn this minor, remove next) — it mapped to
    # "exactly one trusted proxy", which is what the old first-entry
    # selection actually assumed.
    if trust_proxy_headers and trusted_proxy_hops == 0:
        logger.warning(
            "trust_proxy_headers=True is deprecated in 0.6.1 and will be "
            "removed in 0.7.0; pass trusted_proxy_hops=<number of trusted "
            "reverse proxies> instead (mapped to 1 now)."
        )
        trusted_proxy_hops = 1
    # SERVE-04: the decode budget defaults to a fraction of the inference
    # budget — a slow decode must never permanently poison a healthy
    # engine (the poison contract is reserved for stuck FORWARDS, where
    # the non-thread-safe backend cannot be safely reused).
    if decode_timeout_s is None and inference_timeout_s is not None:
        decode_timeout_s = min(10.0, inference_timeout_s / 2.0)

    import torch

    from autovision.inference import InferenceEngine

    if variants is not None:
        if engine is not None or model is not None:
            raise AutoVisionError(
                "pass either variants= or model=/engine=, not both."
            )
        if not isinstance(variants, dict) or not variants:
            raise AutoVisionError("variants must be a non-empty dict.")
        engines = dict(variants)
        for name in engines:
            if not isinstance(name, str) or not name:
                raise AutoVisionError("Variant names must be non-empty strings.")
    else:
        if engine is None:
            if model is None:
                raise AutoVisionError("create_app needs either engine= or model=.")
            engine = InferenceEngine(
                model, backend=backend, artifact=artifact, device=device,
                providers=providers, openvino_device=openvino_device,
                # INFER-04: the serving entry point refuses metadata-stripped
                # artifacts — the identity gate is vacuous for them and a
                # pose network silently driving a detection engine is the
                # exact failure the metadata exists to prevent.
                require_metadata=True,
            )
        engines = {"default": engine}

    default_variant = next(iter(engines))
    weights: dict[str, float]
    if variant_weights is None:
        weights = {name: 1.0 / len(engines) for name in engines}
    else:
        if set(variant_weights) != set(engines):
            raise AutoVisionError(
                "variant_weights keys must exactly match the variant names."
            )
        raw_weights = {name: float(value) for name, value in variant_weights.items()}
        if any(not math.isfinite(value) for value in raw_weights.values()):
            raise AutoVisionError("variant_weights must be finite.")
        if any(value < 0 for value in raw_weights.values()):
            raise AutoVisionError("variant_weights must be >= 0.")
        total = sum(raw_weights.values())
        if total <= 0:
            raise AutoVisionError("variant_weights must sum to a positive value.")
        weights = {name: value / total for name, value in raw_weights.items()}

    limiter: TokenBucketRateLimiter | None = None
    if rate_limit_per_minute is not None:
        try:
            limiter = TokenBucketRateLimiter(
                rate_limit_per_minute, rate_limit_burst
            )
        except ValueError as error:
            raise AutoVisionError(str(error)) from error

    drift_monitor: DriftMonitor | None = None
    if drift_baseline is not None:
        baseline = (
            load_drift_baseline(drift_baseline)
            if isinstance(drift_baseline, (str, Path))
            else drift_baseline
        )
        drift_monitor = DriftMonitor(
            baseline,
            window=drift_window,
            threshold=drift_threshold,
            min_samples=drift_min_samples,
        )

    names = dict(class_names or {})
    metrics = MetricsRegistry()
    metrics.set_gauge("autovision_model_loaded", 1.0, help_text="1 when a model is loaded")
    readiness = {
        name: callable(getattr(candidate, "run_batch", None))
        for name, candidate in engines.items()
    }
    metrics.set_gauge(
        "autovision_model_ready", 1.0 if all(readiness.values()) else 0.0,
        help_text="1 when every inference engine is callable",
    )
    for name, ready in readiness.items():
        metrics.set_gauge(
            "autovision_variant_ready", 1.0 if ready else 0.0,
            {"variant": name}, help_text="1 when this variant is callable",
        )
    # H2-3: poison flags and concurrency slots are per variant. A timed-out
    # engine wedges only its own variant (its slot stays held — the worker
    # thread may still be inside a non-thread-safe backend); other variants
    # keep serving, and /readyz only goes 503 when every variant is poisoned.
    inference_slots: dict[str, asyncio.Semaphore] = {
        name: asyncio.Semaphore(max_concurrency) for name in engines
    }
    inference_poisoned: dict[str, threading.Event] = {
        name: threading.Event() for name in engines
    }
    docs_kwargs = (
        {}
        if docs_enabled
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    def _warmup(engine: Any) -> None:
        """One full-pipeline call (decode + forward + postprocess) per
        engine (design D14a): the first real request otherwise pays the
        allocator/session warm-up AND the lazy postprocessing imports.
        Failures are warnings, never fatal: the readiness probe remains
        the source of truth for serving traffic."""
        try:
            import torch

            engine.run_batch([torch.zeros(3, 64, 64)], threshold=threshold)
            logger.info("Serving warmup complete (%s).", type(engine).__name__)
        except Exception as error:  # noqa: BLE001 - warmup must not kill boot
            logger.warning(
                "Serving warmup skipped (%s: %s); the first request will "
                "pay the warm-up cost instead.", type(error).__name__, error,
            )

    @asynccontextmanager
    async def _lifespan(_app: Any) -> AsyncIterator[None]:
        # SERVE-05: an engine that wedges during warmup (CUDA stall, NFS
        # artifact) used to hang boot forever with no probe to observe it.
        # The bounded wait turns that into a named startup failure; the
        # cancelled thread may linger (asyncio.to_thread cannot be killed),
        # but the process is exiting anyway. warmup_timeout_s=0 disables
        # warmup entirely.
        if warmup_timeout_s == 0 or warmup_timeout_s is None:
            # Review P3-4: ``None`` is the documented "no budget" escape —
            # it disables warmup exactly like 0.
            logger.info("Serving warmup disabled (warmup_timeout_s=0).")
            yield
            return
        for variant_name, warmup_engine in engines.items():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(_warmup, warmup_engine),
                    timeout=warmup_timeout_s,
                )
            except asyncio.TimeoutError:
                raise AutoVisionError(
                    f"Serving warmup for variant '{variant_name}' exceeded "
                    f"warmup_timeout_s={warmup_timeout_s}s — refusing to "
                    "boot an engine that never completed a single call. "
                    "Pass a larger warmup_timeout_s= or disable warmup with "
                    "warmup_timeout_s=0."
                ) from None
        yield

    app = FastAPI(
        title="AutoVision reference serving", version="0.3", **docs_kwargs,
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        raw_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        # H2-5b: sanitize before the id reaches the log stream, the response
        # header, or the A/B bucketing hash — not just the log.
        request_id = _sanitize_request_id(raw_id)
        request.state.request_id = request_id
        # H2-5a: the A/B bucketing key is always issued by the server, so a
        # client cannot pick its variant by choosing X-Request-ID values.
        request.state.routing_key = uuid.uuid4().hex
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Log in finally so unhandled 500s still carry the request id.
            logger.info(
                "request method=%s path=%s status=%d latency_ms=%.2f request_id=%s",
                request.method, request.url.path, status,
                (time.perf_counter() - started) * 1000.0, request_id,
            )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> Any:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.exception(
            "unhandled request error path=%s request_id=%s",
            request.url.path,
            request_id,
        )
        # /predict already counted its 500 before re-raising; avoid a double
        # count while still counting failures from other endpoints.
        if not getattr(request.state, "counted_500", False):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": request.url.path, "status": "500"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
        return JSONResponse(
            {"error": "Internal server error.", "request_id": request_id},
            status_code=500,
            headers={"X-Request-ID": request_id},
        )

    def _select_variant(request: Request) -> str | None:
        # H2-5a: client steering only with the explicit opt-in.
        requested = (
            request.headers.get("x-ab-variant") if allow_variant_override else None
        )
        if requested is not None:
            return requested if requested in engines else None
        if len(engines) == 1:
            return default_variant
        # H2-3: route within healthy variants only — a wedged variant stops
        # receiving traffic without taking the whole process down. (If every
        # variant is poisoned, the per-variant check below 503s anyway.)
        healthy = {
            name: weight
            for name, weight in weights.items()
            if not inference_poisoned[name].is_set()
        } or weights
        total = sum(healthy.values())
        # H2-5a: the bucketing key is the server-issued routing key, never
        # the client-controlled X-Request-ID.
        routing_key = getattr(request.state, "routing_key", "")
        digest = hashlib.sha256(routing_key.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
        cumulative = 0.0
        for name, weight in healthy.items():
            cumulative += weight / total
            if bucket <= cumulative:
                return name
        return next(iter(healthy))

    def _decode_request_images(images_raw: list[str]) -> list[Any]:
        """Decode one request's images under the pixel budget (H2-2b).

        SERVE-04: decoding is its own phase with its own timeout — a slow
        decode (huge valid images on a contended CPU) must not consume the
        engine's inference budget, because the timeout handler permanently
        poisons the variant. The decoded frames for the whole batch stay in
        memory until the engine returns, so the aggregate cap stays here.
        """
        decoded: list[Any] = []
        total_pixels = 0
        for payload in images_raw:
            frame = _decode_image(payload, max_pixels=max_image_pixels)
            total_pixels += int(frame.shape[0]) * int(frame.shape[1])
            if total_pixels > max_request_pixels:
                raise _PayloadTooLarge(
                    f"Request decodes to {total_pixels} pixels in total, which "
                    f"exceeds max_request_pixels={max_request_pixels} (each "
                    f"image is within max_image_pixels={max_image_pixels}; "
                    "lower max_batch= or raise max_request_pixels=)."
                )
            decoded.append(frame)
        return decoded

    def _forward_decoded(selected_engine: Any, decoded: list[Any]) -> list[dict[str, Any]]:
        tensors = [torch.from_numpy(frame).permute(2, 0, 1) for frame in decoded]
        return selected_engine.run_batch(tensors, threshold=threshold)

    def _count_variant(variant: str, status: str) -> None:
        metrics.inc(
            "autovision_variant_requests_total",
            {"variant": variant, "status": status},
            help_text=_VARIANTS_HELP,
        )

    def _observe_error_latency(started: float, error_class: str) -> None:
        """Record failed-request latency under an ``error_class`` label.

        Without this, error paths vanish from the latency histogram and a
        slow-failing deployment is indistinguishable from a fast one (H2-5d).
        Successful requests stay unlabeled so the historical series survives.
        """
        metrics.observe(
            "autovision_predict_latency_seconds",
            time.perf_counter() - started,
            {"error_class": error_class},
            help_text="Per-request predict latency (seconds)",
        )

    def _throttle(request: Request, endpoint: str) -> Any:
        """Apply the shared token bucket before auth; return a 429 or None."""
        throttle_started = time.perf_counter()
        if limiter is None:
            return None
        # SERVE-06: the bucket key is namespaced per endpoint — without it,
        # /metrics scraping and /predict traffic shared one per-client
        # bucket and squeezed each other out.
        key = f"{endpoint}:{_client_key(request, trusted_proxy_hops=trusted_proxy_hops)}"
        allowed, retry_after = limiter.allow(key)
        if allowed:
            return None
        metrics.inc(
            "autovision_requests_total",
            {"endpoint": endpoint, "status": "429"},
            help_text=_REQUESTS_HELP,
        )
        # SERVE-08 (review P3-1): the observability endpoints' 429s join the
        # latency histogram like /predict's — the throttle start is the
        # function entry, an upper bound of the request cost up to here.
        metrics.observe(
            "autovision_predict_latency_seconds",
            time.perf_counter() - throttle_started,
            {"error_class": "rate_limited"},
            help_text="Per-request predict latency (seconds)",
        )
        return JSONResponse(
            {
                "error": "Rate limit exceeded.",
                "request_id": getattr(request.state, "request_id", ""),
                "retry_after_s": retry_after,
            },
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )

    def _update_drift_metrics(report: dict[str, Any]) -> None:
        status = report.get("status")
        if status is not None:
            metrics.set_gauge(
                "autovision_drift_detected", 1.0 if status == "drift" else 0.0,
                help_text="1 when the drift monitor reports drift",
            )
        if report.get("psi_label") is not None:
            metrics.set_gauge(
                "autovision_drift_psi", float(report["psi_label"]),
                {"kind": "label"}, help_text="PSI of the served label distribution",
            )
        if report.get("psi_score") is not None:
            metrics.set_gauge(
                "autovision_drift_psi", float(report["psi_score"]),
                {"kind": "score"}, help_text="PSI of the served score distribution",
            )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        info = _engine_info(engines[default_variant])
        poisoned_variants = sorted(
            name for name, event in inference_poisoned.items() if event.is_set()
        )
        return {
            "status": "ok",
            **info,
            "threshold": threshold,
            "variant": default_variant,
            "variants": sorted(engines),
            # Liveness stays 200 (restarting the stuck engine thread does
            # not help), but the flag lets orchestrators trigger a restart.
            "poisoned": bool(poisoned_variants),
            "poisoned_variants": poisoned_variants,
        }

    @app.get("/readyz")
    def readyz() -> Any:
        poisoned_variants = [
            name for name, event in inference_poisoned.items() if event.is_set()
        ]
        # H2-3: a poisoned variant can no longer serve — report it not-ready
        # regardless of callability.
        for name in poisoned_variants:
            metrics.set_gauge(
                "autovision_variant_ready", 0.0,
                {"variant": name},
                help_text="1 when this variant is callable",
            )
        if len(poisoned_variants) == len(engines):
            # Only a fully poisoned process is not-ready (H2-3): the process
            # must be restarted; report not-ready so an orchestrator removes
            # this instance from load balancing.
            metrics.set_gauge(
                "autovision_model_ready", 0.0,
                help_text="1 when every inference engine is callable",
            )
            for name in engines:
                metrics.set_gauge(
                    "autovision_variant_ready", 0.0,
                    {"variant": name},
                    help_text="1 when this variant is callable",
                )
            return JSONResponse(
                {
                    "ready": False,
                    "poisoned": True,
                    "poisoned_variants": poisoned_variants,
                    "error": (
                        "inference engine timed out and can no longer be "
                        "used; restart the process"
                    ),
                },
                status_code=503,
            )
        current = {
            name: callable(getattr(candidate, "run_batch", None))
            for name, candidate in engines.items()
        }
        ready = all(current.values())
        metrics.set_gauge(
            "autovision_model_ready", 1.0 if ready else 0.0,
            help_text="1 when every inference engine is callable",
        )
        for name, variant_ready in current.items():
            if name not in poisoned_variants:
                metrics.set_gauge(
                    "autovision_variant_ready", 1.0 if variant_ready else 0.0,
                    {"variant": name}, help_text="1 when this variant is callable",
                )
        if not ready:
            return JSONResponse(
                {
                    "ready": False,
                    "variants": current,
                    "poisoned_variants": poisoned_variants,
                    "error": "one or more inference engines are not callable",
                },
                status_code=503,
            )
        return {
            "ready": True,
            "variants": current,
            "poisoned_variants": poisoned_variants,
            "default": default_variant,
            **_engine_info(engines[default_variant]),
        }

    @app.get("/drift")
    def drift(request: Request) -> Any:
        throttled = _throttle(request, "drift")
        if throttled is not None:
            return throttled
        if protect_observability_endpoints and not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "drift", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
        if drift_monitor is None:
            return {"enabled": False}
        report = drift_monitor.snapshot()
        _update_drift_metrics(report)
        return report

    @app.get("/metrics")
    def prometheus_metrics(request: Request) -> Any:
        throttled = _throttle(request, "metrics")
        if throttled is not None:
            return throttled
        if protect_observability_endpoints and not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "metrics", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
        if drift_monitor is not None:
            # A Prometheus scrape is the natural sampling point; computing PSI
            # here keeps the gauge fresh without doing O(window) work per request.
            _update_drift_metrics(drift_monitor.snapshot())
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict")
    async def predict(request: Request) -> Any:
        started = time.perf_counter()
        request_id = getattr(request.state, "request_id", "")
        if limiter is not None:
            # Rate-limit before auth so unauthenticated brute force is
            # throttled too.
            key = f"predict:{_client_key(request, trusted_proxy_hops=trusted_proxy_hops)}"
            allowed, retry_after = limiter.allow(key)
            if not allowed:
                # SERVE-08: the 429 joins the latency histogram like every
                # other error class (it used to vanish from /metrics).
                metrics.inc(
                    "autovision_requests_total",
                    {"endpoint": "predict", "status": "429"},
                    help_text=_REQUESTS_HELP,
                )
                _observe_error_latency(started, "rate_limited")
                return JSONResponse(
                    {
                        "error": "Rate limit exceeded.",
                        "request_id": request_id,
                        "retry_after_s": retry_after,
                    },
                    status_code=429,
                    headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
                )
            # Only the 429 branch observes error latency: an ALLOWED request
            # is not an error and must not pollute the rate_limited series
            # (review P1-1 — the misplaced line fired on every admitted
            # request and double-counted it into the histogram).
        if not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": "Unauthorized.", "request_id": request_id},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        variant = _select_variant(request)
        if variant is None:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            return JSONResponse(
                {
                    "error": "Unknown X-AB-Variant; expected one of "
                    f"{sorted(engines)}.",
                    "request_id": request_id,
                },
                status_code=400,
            )
        request.state.variant = variant
        if inference_poisoned[variant].is_set():
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "503"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "poisoned")
            _count_variant(variant, "503")
            return JSONResponse(
                {
                    "error": (
                        f"Inference engine for variant '{variant}' is "
                        "unresponsive after a previous timeout; restart "
                        "the service."
                    ),
                    "request_id": request_id,
                },
                status_code=503,
                headers={"Retry-After": "60"},
            )
        try:
            body = await _read_body_limited(request, max_body_bytes)
        except _PayloadTooLarge as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "413")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=413
            )
        except AutoVisionError as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "400")
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=400
            )
        try:
            payload = json.loads(body) if body else None
        except Exception as error:  # noqa: BLE001
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "400")
            return JSONResponse(
                {"error": f"Invalid JSON body: {error}", "request_id": request_id},
                status_code=400,
            )
        if isinstance(payload, dict) and "image" in payload and "images" not in payload:
            payload = {"images": [payload["image"]]}
        images_raw = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images_raw, list) or not images_raw:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "400")
            return JSONResponse(
                {
                    "error": "Body must be {\"images\": [<base64>, ...]} (or {\"image\": ...}).",
                    "request_id": request_id,
                },
                status_code=400,
            )
        if len(images_raw) > max_batch:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "413")
            return JSONResponse(
                {
                    "error": f"Batch of {len(images_raw)} exceeds max_batch={max_batch}.",
                    "request_id": request_id,
                },
                status_code=413,
            )
        slot_acquired = False
        variant_slots = inference_slots[variant]
        try:
            if queue_timeout_s is None:
                await variant_slots.acquire()
            else:
                try:
                    await asyncio.wait_for(
                        variant_slots.acquire(), timeout=queue_timeout_s
                    )
                except asyncio.TimeoutError:
                    metrics.inc(
                        "autovision_requests_total",
                        {"endpoint": "predict", "status": "503"},
                        help_text=_REQUESTS_HELP,
                    )
                    _observe_error_latency(started, "queue_timeout")
                    _count_variant(variant, "503")
                    metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
                    return JSONResponse(
                        {
                            "error": (
                                "Inference queue is saturated; retry later "
                                f"(queue_timeout_s={queue_timeout_s})."
                            ),
                            "request_id": request_id,
                        },
                        status_code=503,
                        headers={"Retry-After": "1"},
                    )
            slot_acquired = True
            # SERVE-04: decode and forward carry separate budgets. A decode
            # timeout/failure never poisons the variant — the engine is
            # healthy, the CLIENT input (or the CPU) was slow. Only a stuck
            # FORWARD trips the poison contract (H2-3), because reusing a
            # wedged non-thread-safe runner is unsafe.
            if decode_timeout_s is None:
                decoded = await run_in_threadpool(
                    _decode_request_images, images_raw
                )
            else:
                try:
                    decoded = await asyncio.wait_for(
                        run_in_threadpool(_decode_request_images, images_raw),
                        timeout=decode_timeout_s,
                    )
                except asyncio.TimeoutError:
                    metrics.inc(
                        "autovision_requests_total",
                        {"endpoint": "predict", "status": "504"},
                        help_text=_REQUESTS_HELP,
                    )
                    _observe_error_latency(started, "decode_timeout")
                    _count_variant(variant, "504")
                    metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
                    metrics.inc(
                        "autovision_decode_timeouts_total",
                        help_text="Requests whose decode exceeded decode_timeout_s",
                    )
                    logger.warning(
                        "Decoding exceeded %.3fs (request_id=%s); the engine "
                        "is NOT poisoned.",
                        decode_timeout_s,
                        request_id,
                    )
                    return JSONResponse(
                        {
                            "error": (
                                "Decoding the request images timed out after "
                                f"{decode_timeout_s}s; retry with fewer or "
                                "smaller images."
                            ),
                            "request_id": request_id,
                        },
                        status_code=504,
                        headers={"Retry-After": "1"},
                    )
            if inference_timeout_s is None:
                records = await run_in_threadpool(
                    _forward_decoded, engines[variant], decoded
                )
            else:
                try:
                    records = await asyncio.wait_for(
                        run_in_threadpool(
                            _forward_decoded, engines[variant], decoded
                        ),
                        timeout=inference_timeout_s,
                    )
                except asyncio.TimeoutError:
                    # The worker may still be inside the engine. Keep the
                    # variant's slot held and poison only that variant (H2-3):
                    # reusing a stuck TensorRT/OpenVINO runner would be unsafe,
                    # but the other variants are unaffected.
                    inference_poisoned[variant].set()
                    metrics.set_gauge(
                        "autovision_variant_ready", 0.0,
                        {"variant": variant},
                        help_text="1 when this variant is callable",
                    )
                    metrics.inc(
                        "autovision_requests_total",
                        {"endpoint": "predict", "status": "504"},
                        help_text=_REQUESTS_HELP,
                    )
                    _observe_error_latency(started, "timeout")
                    _count_variant(variant, "504")
                    metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
                    metrics.inc(
                        "autovision_inference_timeouts_total",
                        help_text="Engine calls that exceeded inference_timeout_s",
                    )
                    logger.error(
                        "Inference timed out after %.3fs (request_id=%s); "
                        "marking the engine unresponsive.",
                        inference_timeout_s,
                        request_id,
                    )
                    return JSONResponse(
                        {
                            "error": (
                                "Inference timed out after "
                                f"{inference_timeout_s}s; the engine for "
                                f"variant '{variant}' is now unresponsive "
                                "and must be restarted."
                            ),
                            "request_id": request_id,
                        },
                        status_code=504,
                        headers={"Retry-After": "60"},
                    )
            if len(records) != len(images_raw):
                # SERVE-02: a count mismatch is a server-side contract
                # failure — AutoVisionError routes it through the 500
                # handler (stable phrase + request_id) instead of FastAPI's
                # raw 500 page.
                raise AutoVisionError(
                    "Engine returned "
                    f"{len(records)} records for {len(images_raw)} images."
                )
            results = await run_in_threadpool(
                _serialize_records,
                records,
                names,
                mask_format,
                max_json_elements,
            )
        except _PayloadTooLarge as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "413")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=413
            )
        except _ClientInputError as error:
            # H2-5c: these messages are built from client input and configured
            # limits only, so the original text stays in the response.
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "validation")
            _count_variant(variant, "400")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=400
            )
        except AutoVisionError as error:
            # H2-5c: everything else crossing the engine/serialization boundary
            # may embed server internals (absolute paths, backend details) —
            # map it to a stable phrase + request_id and keep the original
            # text server-side. SERVE-02: the status is now 500 — a contract
            # violation inside the engine (ResultValidationError) or an
            # unresolved framework error is a SERVER failure, and reporting
            # it as 400 poisoned retry policies, SLO error splits and alert
            # triage. Client-input errors keep 400 via _ClientInputError.
            logger.error(
                "predict failed after a validated request (request_id=%s): %s",
                request_id,
                error,
            )
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "500"},
                help_text=_REQUESTS_HELP,
            )
            _observe_error_latency(started, "internal")
            _count_variant(variant, "500")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {
                    "error": (
                        "Prediction failed; inspect the server log with this "
                        "request_id."
                    ),
                    "request_id": request_id,
                },
                status_code=500,
            )
        except Exception:
            # Anything else becomes FastAPI's 500 — count it before re-raising
            # so a failing deployment stays visible on /metrics. The global
            # handler must not count the same failure twice.
            request.state.counted_500 = True
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "500"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "500")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            _observe_error_latency(started, "internal")
            raise
        finally:
            if slot_acquired and not inference_poisoned[variant].is_set():
                variant_slots.release()
        elapsed = time.perf_counter() - started
        metrics.inc(
            "autovision_requests_total",
            {"endpoint": "predict", "status": "200"},
            help_text=_REQUESTS_HELP,
        )
        _count_variant(variant, "200")
        metrics.inc(
            "autovision_images_total", value=float(len(records)),
            help_text="Images served",
        )
        metrics.observe(
            "autovision_predict_latency_seconds", elapsed,
            help_text="Per-request predict latency (seconds)",
        )
        metrics.observe(
            "autovision_variant_predict_latency_seconds", elapsed,
            {"variant": variant},
            help_text="Per-variant predict latency (seconds)",
        )
        if drift_monitor is not None:
            drift_monitor.record_many(records)
        return {
            "results": results,
            "count": len(results),
            "latency_s": elapsed,
            "request_id": request_id,
            "variant": variant,
        }

    return app
