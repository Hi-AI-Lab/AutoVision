# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference serving (roadmap F2): FastAPI + batch + metrics.

A deliberately MINIMAL, testable reference — the prerequisite for
inference monitoring, not a production gateway:

- ``POST /predict`` — JSON body ``{"images": ["<base64>", ...]}`` (or a
  single ``{"image": ...}``); results come back in input order. Pre/post
  processing is the network's own (the same source as evaluation and the
  stream runtime), so served predictions match offline readings.
- ``GET /healthz`` — liveness + the loaded backend/model identity.
- ``GET /metrics`` — Prometheus exposition text (requests, errors, images,
  latency summary) for scraping.

Explicitly OUT of scope (documented boundary, README): authentication,
multi-worker orchestration, TLS, and the monitoring layers built on top of
``/metrics`` (drift detection, A/B, canary) — those are product decisions
that start from this endpoint.
"""

import base64
import io
import time
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.serving.metrics import MetricsRegistry

__all__ = ["create_app"]

_REQUESTS_HELP = "Served requests by endpoint and status"
_ERRORS_HELP = "Failed predictions"


def _decode_image(payload: str) -> Any:
    """base64 image payload → HWC uint8 numpy array (RGB)."""
    import numpy as np
    from PIL import Image

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as error:  # noqa: BLE001 - malformed payload is a 400
        raise AutoVisionError(f"Invalid base64 image payload: {error}") from error
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as error:  # noqa: BLE001 - undecodable payload is a 400
        # PIL's own message embeds an object address — keep the body stable.
        raise AutoVisionError(
            "Unsupported image payload — expected base64-encoded PNG/JPEG bytes."
        ) from error
    return np.asarray(image)


def create_app(
    model: Any = None,
    *,
    engine: Any = None,
    backend: str = "torch",
    artifact: "str | Path | None" = None,
    device: str | None = None,
    providers: "list[str] | None" = None,
    openvino_device: str = "CPU",
    threshold: float = 0.5,
    max_batch: int = 16,
    class_names: dict[int, str] | None = None,
) -> Any:
    """Build the FastAPI app around a task model or a ready inference engine.

    ``model`` may be an exported ``.pt`` path or a ``TaskNetwork`` (it gets
    wrapped in an :class:`~autovision.inference.InferenceEngine`); pass a
    ready engine via ``engine=`` instead. The engine path
    (``backend``/``artifact``) selects the torch / ONNX / TensorRT /
    OpenVINO forward, matching the stream runtime. ``max_batch`` bounds one
    request's image count (a 413 otherwise).
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as error:  # pragma: no cover - import guard path
        raise AutoVisionError(
            "The 'fastapi' package is required for the serving reference. "
            "Install it with: pip install 'autovision[serving]'."
        ) from error

    if max_batch < 1:
        raise AutoVisionError(f"max_batch must be >= 1, got {max_batch}.")

    import torch

    from autovision.inference import InferenceEngine

    if engine is None:
        if model is None:
            raise AutoVisionError("create_app needs either engine= or model=.")
        engine = InferenceEngine(
            model, backend=backend, artifact=artifact, device=device,
            providers=providers, openvino_device=openvino_device,
        )
    names = dict(class_names or {})
    metrics = MetricsRegistry()
    metrics.set_gauge("autovision_model_loaded", 1.0, help_text="1 when a model is loaded")
    app = FastAPI(title="AutoVision reference serving", version="0.1")

    def _predict_one(images_raw: list[str]) -> list[dict[str, Any]]:
        decoded = [_decode_image(payload) for payload in images_raw]
        tensors = [torch.from_numpy(frame).permute(2, 0, 1) for frame in decoded]
        return engine.run_batch(tensors, threshold=threshold)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        network = getattr(engine, "network", None)
        return {
            "status": "ok",
            "backend": engine.backend,
            "model": type(network).__name__ if network is not None else None,
            "artifact": str(engine.artifact) if getattr(engine, "artifact", None) else None,
            "openvino_device": getattr(engine, "openvino_device", None),
            "threshold": threshold,
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Any:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict")
    async def predict(request: Request) -> Any:
        started = time.perf_counter()
        try:
            payload = await request.json()
        except Exception as error:  # noqa: BLE001
            metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "400"},
                        help_text=_REQUESTS_HELP)
            return JSONResponse({"error": f"Invalid JSON body: {error}"}, status_code=400)
        if isinstance(payload, dict) and "image" in payload and "images" not in payload:
            payload = {"images": [payload["image"]]}
        images_raw = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images_raw, list) or not images_raw:
            metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "400"},
                        help_text=_REQUESTS_HELP)
            return JSONResponse(
                {"error": "Body must be {\"images\": [<base64>, ...]} (or {\"image\": ...})."},
                status_code=400,
            )
        if len(images_raw) > max_batch:
            metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "413"},
                        help_text=_REQUESTS_HELP)
            return JSONResponse(
                {"error": f"Batch of {len(images_raw)} exceeds max_batch={max_batch}."},
                status_code=413,
            )
        try:
            records = _predict_one(images_raw)
        except AutoVisionError as error:
            metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "400"},
                        help_text=_REQUESTS_HELP)
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse({"error": str(error)}, status_code=400)
        except Exception:
            # Anything else becomes FastAPI's 500 — count it before re-raising
            # so a failing deployment stays visible on /metrics.
            metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "500"},
                        help_text=_REQUESTS_HELP)
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            metrics.observe("autovision_predict_latency_seconds",
                            time.perf_counter() - started,
                            help_text="Per-request predict latency (seconds)")
            raise
        elapsed = time.perf_counter() - started
        metrics.inc("autovision_requests_total", {"endpoint": "predict", "status": "200"},
                    help_text=_REQUESTS_HELP)
        metrics.inc("autovision_images_total", value=float(len(records)),
                    help_text="Images served")
        metrics.observe("autovision_predict_latency_seconds", elapsed,
                        help_text="Per-request predict latency (seconds)")
        results = []
        for record in records:
            entry = {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in record.items()
            }
            if names and "labels" in entry:
                entry["class_names"] = [names.get(int(label), str(label)) for label in entry["labels"]]
            results.append(entry)
        return {"results": results, "count": len(results), "latency_s": elapsed}

    return app
