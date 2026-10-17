# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference serving (roadmap F2): FastAPI app + metrics registry.

``create_app(...)`` builds the app (``/predict`` batch endpoint,
``/healthz``, Prometheus ``/metrics``); the ``serving`` extra carries
fastapi/uvicorn. Boundaries (auth, orchestration, drift/A-B/canary) are
documented in the README and module docstring.
"""

from autovision.serving.app import create_app
from autovision.serving.metrics import MetricsRegistry

__all__ = ["MetricsRegistry", "create_app"]
