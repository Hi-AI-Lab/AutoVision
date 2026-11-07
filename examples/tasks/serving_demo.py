#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference serving smoke test (``serving/`` subsystem).

Wraps a trained checkpoint in the FastAPI reference app and exercises the
HTTP contract the production server serves: ``POST /predict`` with a base64
image and ``GET /healthz``. ``fastapi.testclient.TestClient`` runs the ASGI
app in-process, so the script exercises the real route handlers (decode →
engine → postprocess → JSON) and still exits cleanly — no port or process
management needed.

    python examples/tasks/serving_demo.py
    python examples/tasks/serving_demo.py \
        --artifact runs/examples/classification_train/exported_models/exported_last.pt

For a real deployment (not this smoke test) launch the app with uvicorn:

    from autovision.serving import create_app, serve
    app = create_app(model=<artifact>, device="cpu")
    serve(model=app, host="127.0.0.1", port=8000)
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from autovision.serving import create_app

try:
    from fastapi.testclient import TestClient
except ImportError as error:  # pragma: no cover - import guard path
    raise SystemExit(
        "The 'fastapi' package is required for the serving demo. "
        "Install it with: pip install 'autovision[serving]'."
    ) from error

ASSET = Path(__file__).resolve().parent / "assets" / "airport_aerial.jpg"
DEFAULT_ARTIFACT = (
    Path("runs/examples/classification_train/exported_models/exported_last.pt")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="exported .pt to serve (default: classification smoke export)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="serving decision threshold"
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        raise SystemExit(
            f"Artifact not found: {args.artifact}\n"
            "Train one first, e.g. classification_train.py --export."
        )

    app = create_app(
        model=args.artifact,
        backend="torch",
        device=args.device,
        threshold=args.threshold,
    )
    client = TestClient(app)

    health = client.get("/healthz")
    print("GET  /healthz ->", health.status_code, health.json().get("status"))

    payload = base64.b64encode(Path(ASSET).read_bytes()).decode()
    response = client.post("/predict", json={"image": payload})
    body = response.json()
    print("POST /predict ->", response.status_code, "| count =", body.get("count"))
    results = body.get("results") or []
    if results:
        print("first result keys:", sorted(results[0].keys())[:8])


if __name__ == "__main__":
    main()
