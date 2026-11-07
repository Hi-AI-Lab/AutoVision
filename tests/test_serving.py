# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference serving (roadmap F2): /predict batching, /healthz, /metrics."""

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from typing import Any

import httpx
import numpy as np
import pytest
from tests.test_tracking import tiny_model  # noqa: F401 - local fixture reuse

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import create_app  # noqa: E402
from autovision.serving import server as serving_server  # noqa: E402
from autovision.serving.app import (  # noqa: E402
    _DEFAULT_MAX_IMAGE_PIXELS,
    _sanitize_request_id,
)
from autovision.serving.server import serve  # noqa: E402
from autovision.serving.worker import SERVE_CONFIG_ENV  # noqa: E402


def _b64_image(size: int = 64, seed: int = 0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def client(tiny_model):  # noqa: F811 - pytest fixture parameter
    app = create_app(tiny_model, threshold=0.0, max_batch=4, class_names={0: "cat", 1: "dog"})
    return TestClient(app)


class TestServing:
    def test_healthz_reports_the_backend(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["backend"] == "torch"
        assert response.json()["status"] == "ok"

    def test_probes_do_not_leak_artifact_paths(self):
        class _PathEngine:
            backend = "onnx"
            artifact = "/srv/private/models/model.onnx"
            network = None

            def run_batch(self, images, *, threshold=0.5):
                return []

        client = TestClient(create_app(engine=_PathEngine(), threshold=0.0))
        payload = client.get("/healthz").json()
        assert payload["artifact"] == "model.onnx"
        assert "/srv/private" not in str(payload)

    def test_predict_single_and_batch_order(self, client):
        first = _b64_image(seed=1)
        second = _b64_image(seed=2)
        single = client.post("/predict", json={"image": first})
        assert single.status_code == 200, single.text
        payload = single.json()
        assert payload["count"] == 1
        assert set(payload["results"][0]) >= {"bboxes", "scores", "labels", "class_names"}
        assert len(payload["results"][0]["scores"]) == 10  # num_top_queries

        batch = client.post("/predict", json={"images": [first, second, first]})
        assert batch.status_code == 200, batch.text
        results = batch.json()["results"]
        assert batch.json()["count"] == 3
        # Input order is preserved: A, B, A → A, B, A (not sorted/deduped).
        # Batched forwards may differ from single ones in the last float bits,
        # hence allclose (bit-equality is the export gate's job, not a
        # different-batch-size run's).
        assert np.allclose(results[0]["scores"], payload["results"][0]["scores"], atol=1e-6)
        assert np.allclose(results[0]["scores"], results[2]["scores"], atol=1e-12)
        assert not np.allclose(results[0]["scores"], results[1]["scores"], atol=1e-3)

    def test_metrics_endpoint_renders_prometheus_text(self, client):
        client.post("/predict", json={"image": _b64_image(seed=4)})
        text = client.get("/metrics").text
        assert "# TYPE autovision_requests_total counter" in text
        assert 'autovision_requests_total{endpoint="predict",status="200"}' in text
        assert "# HELP autovision_requests_total" in text  # never order-dependent
        assert "autovision_images_total" in text
        assert "# TYPE autovision_predict_latency_seconds summary" in text
        assert "autovision_predict_latency_seconds_count" in text
        assert "autovision_predict_latency_seconds_sum" in text
        # The peak rides in its own gauge family (not inside the summary).
        assert "# TYPE autovision_predict_latency_seconds_max gauge" in text
        assert "autovision_predict_latency_seconds_max " in text
        assert "autovision_model_loaded 1" in text

    def test_bad_payloads_are_400_and_counted(self, client):
        missing = client.post("/predict", json={"nope": 1})
        assert missing.status_code == 400
        assert "images" in missing.json()["error"]
        bad_json = client.post(
            "/predict", content=b"not json", headers={"content-type": "application/json"}
        )
        assert bad_json.status_code == 400
        bad_b64 = client.post("/predict", json={"images": ["!!not-base64!!"]})
        assert bad_b64.status_code == 400
        text = client.get("/metrics").text
        assert 'status="400"' in text

    def test_batch_limit_is_413(self, client):
        response = client.post(
            "/predict", json={"images": [_b64_image(seed=5)] * 5}
        )
        assert response.status_code == 413
        assert "max_batch" in response.json()["error"]

    def test_batch_matches_single_calls(self, client):
        """Serving a batch equals serving the images one by one."""
        images = [_b64_image(seed=6), _b64_image(seed=7)]
        batch = client.post("/predict", json={"images": images}).json()["results"]
        singles = [
            client.post("/predict", json={"image": payload}).json()["results"][0]
            for payload in images
        ]
        for batched, single in zip(batch, singles):
            assert np.allclose(batched["scores"], single["scores"], atol=1e-6)
            assert np.allclose(batched["bboxes"], single["bboxes"], atol=1e-4)


class TestFailureVisibility:
    def test_engine_errors_are_500_and_counted(self):
        """A crashing forward stays visible on /metrics (no silent 5xx)."""

        class _BrokenEngine:
            backend = "torch"
            artifact = None
            network = None

            def run_batch(self, images, *, threshold=0.5):
                raise RuntimeError("engine exploded")

        app = create_app(engine=_BrokenEngine(), threshold=0.0, max_batch=4)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/predict", json={"image": _b64_image(seed=8)})
        assert response.status_code == 500
        assert response.headers["X-Request-ID"]
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        text = client.get("/metrics").text
        assert 'autovision_requests_total{endpoint="predict",status="500"}' in text
        assert "autovision_errors_total 1" in text
        # H2-5d: even 500s land in the latency histogram.
        assert 'autovision_predict_latency_seconds_count{error_class="internal"}' in text

    def test_inference_timeout_returns_504_then_poisons_the_engine(self):
        class _HungEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.release = threading.Event()

            def run_batch(self, images, *, threshold=0.5):
                self.release.wait(timeout=5.0)
                return [{} for _ in images]

        engine = _HungEngine()
        app = create_app(
            engine=engine,
            threshold=0.0,
            inference_timeout_s=0.05,
            queue_timeout_s=0.05,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            first = client.post("/predict", json={"image": _b64_image(seed=21)})
            assert first.status_code == 504
            assert "timed out" in first.json()["error"]
            assert first.headers["X-Request-ID"]
            second = client.post("/predict", json={"image": _b64_image(seed=22)})
            assert second.status_code == 503
            assert "restart" in second.json()["error"]
            text = client.get("/metrics").text
            assert 'status="504"' in text
            assert "autovision_inference_timeouts_total 1" in text
            # H2-5d: timeout-bound failures carry their latency class.
            assert 'autovision_predict_latency_seconds_count{error_class="timeout"}' in text
            assert 'autovision_predict_latency_seconds_count{error_class="poisoned"}' in text
        finally:
            engine.release.set()

    def test_poisoned_engine_reports_unready(self):
        """After a timeout, /readyz must 503 so orchestrators restart."""

        class _HungEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.release = threading.Event()

            def run_batch(self, images, *, threshold=0.5):
                self.release.wait(timeout=5.0)
                return [{} for _ in images]

        engine = _HungEngine()
        app = create_app(
            engine=engine,
            threshold=0.0,
            inference_timeout_s=0.05,
            queue_timeout_s=0.05,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            first = client.post("/predict", json={"image": _b64_image(seed=31)})
            assert first.status_code == 504
            ready = client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["poisoned"] is True
            assert "restart" in ready.json()["error"]
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["poisoned"] is True
            text = client.get("/metrics").text
            assert "autovision_model_ready 0" in text
            assert 'autovision_variant_ready{variant="default"} 0' in text
        finally:
            engine.release.set()


class TestVariantIsolation:
    """H2-3/H2-5a: per-variant poison isolation and A/B routing control."""

    class _OkEngine:
        backend = "torch"
        artifact = None
        network = None

        def run_batch(self, images, *, threshold=0.5):
            return [{"depth": np.zeros((2, 2), dtype=np.float32)} for _ in images]

    class _HungEngine:
        backend = "torch"
        artifact = None
        network = None

        def __init__(self):
            self.release = threading.Event()

        def run_batch(self, images, *, threshold=0.5):
            self.release.wait(timeout=5.0)
            return [{} for _ in images]

    def test_poisoned_variant_does_not_take_down_the_other(self):
        """H2-3: one wedged variant is 503/readyz-flagged; the other serves."""
        hung = self._HungEngine()
        app = create_app(
            variants={"a": hung, "b": self._OkEngine()},
            threshold=0.0,
            allow_variant_override=True,
            inference_timeout_s=0.05,
            queue_timeout_s=0.5,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            first = client.post(
                "/predict",
                json={"image": _b64_image(seed=41)},
                headers={"X-AB-Variant": "a"},
            )
            assert first.status_code == 504
            # The healthy variant still answers (single-variant poison used
            # to 503 the whole process).
            second = client.post(
                "/predict",
                json={"image": _b64_image(seed=42)},
                headers={"X-AB-Variant": "b"},
            )
            assert second.status_code == 200, second.text
            assert second.json()["variant"] == "b"
            # Partial poison keeps the process ready, with per-variant detail.
            ready = client.get("/readyz")
            assert ready.status_code == 200
            assert ready.json()["poisoned_variants"] == ["a"]
            assert ready.json()["variants"]["b"] is True
            health = client.get("/healthz")
            assert health.json()["poisoned"] is True
            assert health.json()["poisoned_variants"] == ["a"]
            text = client.get("/metrics").text
            assert 'autovision_variant_ready{variant="a"} 0' in text
            assert 'autovision_variant_ready{variant="b"} 1' in text
            assert "autovision_model_ready 1" in text
        finally:
            hung.release.set()

    def test_readyz_503_only_when_every_variant_is_poisoned(self):
        """H2-3: /readyz stays 200 until all variants are wedged."""
        hung_a = self._HungEngine()
        hung_b = self._HungEngine()
        app = create_app(
            variants={"a": hung_a, "b": hung_b},
            threshold=0.0,
            allow_variant_override=True,
            inference_timeout_s=0.05,
            queue_timeout_s=0.5,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            client.post(
                "/predict",
                json={"image": _b64_image(seed=43)},
                headers={"X-AB-Variant": "a"},
            )
            ready = client.get("/readyz")
            assert ready.status_code == 200
            client.post(
                "/predict",
                json={"image": _b64_image(seed=44)},
                headers={"X-AB-Variant": "b"},
            )
            ready = client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["poisoned"] is True
            assert ready.json()["poisoned_variants"] == ["a", "b"]
        finally:
            hung_a.release.set()
            hung_b.release.set()

    def test_ab_variant_header_requires_override(self):
        """H2-5a: X-AB-Variant steering is opt-in; the bucket key is not
        the client-controlled X-Request-ID."""
        engines = {"a": self._OkEngine(), "b": self._OkEngine()}
        default_app = create_app(variants=dict(engines), threshold=0.0)
        default_client = TestClient(default_app)
        # Without the opt-in the header is ignored (unknown names included):
        # the request routes normally instead of failing.
        response = default_client.post(
            "/predict",
            json={"image": _b64_image(seed=45)},
            headers={"X-AB-Variant": "nonexistent"},
        )
        assert response.status_code == 200
        assert response.json()["variant"] in {"a", "b"}

        override_app = create_app(
            variants=dict(engines), threshold=0.0, allow_variant_override=True
        )
        override_client = TestClient(override_app)
        forced = override_client.post(
            "/predict",
            json={"image": _b64_image(seed=46)},
            headers={"X-AB-Variant": "b"},
        )
        assert forced.status_code == 200
        assert forced.json()["variant"] == "b"
        unknown = override_client.post(
            "/predict",
            json={"image": _b64_image(seed=47)},
            headers={"X-AB-Variant": "nonexistent"},
        )
        assert unknown.status_code == 400
        assert "X-AB-Variant" in unknown.json()["error"]

    def test_summarized_depth_is_flagged_as_truncated(self):
        """H2-5f: a depth map summarized by max_elements must be flagged —
        a shape digest is not pixels, and silent truncation is unusable."""

        class _DepthEngine:
            backend = "torch"
            artifact = None
            network = None

            def run_batch(self, images, *, threshold=0.5):
                return [{"depth": np.full((8, 8), 0.5, dtype=np.float32)}]

        truncated = TestClient(
            create_app(engine=_DepthEngine(), threshold=0.0, max_json_elements=16)
        )
        response = truncated.post("/predict", json={"image": _b64_image(seed=48)})
        assert response.status_code == 200
        entry = response.json()["results"][0]
        assert entry["depth"]["summarized"] is True
        assert entry["depth_truncated"] is True

        intact = TestClient(
            create_app(engine=_DepthEngine(), threshold=0.0, max_json_elements=100)
        )
        response = intact.post("/predict", json={"image": _b64_image(seed=49)})
        assert response.status_code == 200
        assert "depth_truncated" not in response.json()["results"][0]


class TestAppValidation:
    def test_missing_model_fails_loud(self):
        with pytest.raises(AutoVisionError, match="engine= or model="):
            create_app()

    def test_bad_max_batch_fails_loud(self, tiny_model):  # noqa: F811 - fixture param
        with pytest.raises(AutoVisionError, match="max_batch"):
            create_app(tiny_model, max_batch=0)

    def test_bad_guards_fail_loud(self, tiny_model):  # noqa: F811 - fixture param
        with pytest.raises(AutoVisionError, match="api_key"):
            create_app(tiny_model, api_key="")
        with pytest.raises(AutoVisionError, match="api_key"):
            create_app(tiny_model, api_key=123)  # type: ignore[arg-type]
        with pytest.raises(AutoVisionError, match="max_body_bytes"):
            create_app(tiny_model, max_body_bytes=0)
        with pytest.raises(AutoVisionError, match="max_image_pixels"):
            create_app(tiny_model, max_image_pixels=0)
        with pytest.raises(AutoVisionError, match="max_concurrency"):
            create_app(tiny_model, max_concurrency=0)


class TestProductionGuards:
    def test_readyz_reports_ready(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert "autovision_model_ready 1" in client.get("/metrics").text

    def test_readyz_is_503_without_a_callable_engine(self):
        class _NotReady:
            backend = "torch"
            artifact = None
            network = None

        client = TestClient(create_app(engine=_NotReady()))
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_api_key_is_opt_in_and_constant_time_checked(self, tiny_model):  # noqa: F811
        app = create_app(tiny_model, threshold=0.0, api_key="s3cret")
        client = TestClient(app)
        payload = {"image": _b64_image(seed=11)}

        assert client.post("/predict", json=payload).status_code == 401
        assert (
            client.post(
                "/predict", json=payload, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/predict", json=payload, headers={"Authorization": "Bearer s3cret"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/predict", json=payload, headers={"X-API-Key": "s3cret"}
            ).status_code
            == 200
        )
        assert 'status="401"' in client.get(
            "/metrics", headers={"X-API-Key": "s3cret"}
        ).text

    def test_api_key_can_come_from_env(self, tiny_model, monkeypatch):  # noqa: F811
        monkeypatch.setenv("AUTOVISION_SERVING_API_KEY", "env-secret")
        client = TestClient(create_app(tiny_model, threshold=0.0))
        payload = {"image": _b64_image(seed=17)}
        assert client.post("/predict", json=payload).status_code == 401
        assert (
            client.post(
                "/predict", json=payload, headers={"X-API-Key": "env-secret"}
            ).status_code
            == 200
        )

    def test_body_limit_is_413(self, tiny_model):  # noqa: F811
        client = TestClient(
            create_app(tiny_model, threshold=0.0, max_body_bytes=64)
        )
        response = client.post("/predict", json={"image": _b64_image(seed=12)})
        assert response.status_code == 413
        assert "max_body_bytes" in response.json()["error"]
        assert 'status="413"' in client.get("/metrics").text

    def test_image_pixel_limit_is_413(self, tiny_model):  # noqa: F811
        client = TestClient(
            create_app(tiny_model, threshold=0.0, max_image_pixels=100)
        )
        response = client.post("/predict", json={"image": _b64_image(seed=13)})
        assert response.status_code == 413
        assert "max_image_pixels" in response.json()["error"]

    def test_request_id_is_echoed(self, client):
        response = client.post(
            "/predict",
            json={"image": _b64_image(seed=14)},
            headers={"X-Request-ID": "req-abc"},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "req-abc"
        assert response.json()["request_id"] == "req-abc"

    def test_inference_runs_off_the_event_loop_and_is_serialized(self):
        class _SlowEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.active = 0
                self.max_active = 0

            def run_batch(self, images, *, threshold=0.5):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                time.sleep(0.5)
                self.active -= 1
                return [{"score": 1.0} for _ in images]

        engine = _SlowEngine()
        app = create_app(engine=engine, threshold=0.0)

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as async_client:
                first = asyncio.create_task(
                    async_client.post("/predict", json={"image": _b64_image(seed=15)})
                )
                second = asyncio.create_task(
                    async_client.post("/predict", json={"image": _b64_image(seed=16)})
                )
                await asyncio.sleep(0.05)
                started = time.perf_counter()
                health = await async_client.get("/healthz")
                health_elapsed = time.perf_counter() - started
                responses = await asyncio.gather(first, second)
                return health, health_elapsed, responses

        health, health_elapsed, responses = asyncio.run(scenario())
        # The blocking forward must not stall the event loop: /healthz answers
        # while both predictions are in flight.
        assert health.status_code == 200
        assert health_elapsed < 0.35, (
            f"/healthz took {health_elapsed:.3f}s while /predict was blocked"
        )
        assert [response.status_code for response in responses] == [200, 200]
        # Default max_concurrency=1 serializes engine calls (TRT/OVN runners
        # are not thread-safe).
        assert engine.max_active == 1


class TestServingHardening:
    def test_docs_endpoints_are_disabled_by_default(self, client):
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_docs_can_be_enabled_explicitly(self, tiny_model):  # noqa: F811
        client = TestClient(create_app(tiny_model, threshold=0.0, docs_enabled=True))
        assert client.get("/openapi.json").status_code == 200

    def test_non_ascii_credentials_are_rejected_not_raised(self):
        # httpx refuses non-ASCII header values, but a raw HTTP client can
        # send them (latin-1 bytes). compare_digest would raise TypeError on
        # str inputs; the app must treat them as a normal 401 credential.
        from autovision.serving.app import _authorized

        class _Request:
            headers = {"authorization": "Bearer caf\u00e9"}

        assert _authorized(_Request(), "s3cret") is False
        assert _authorized(_Request(), "other-caf\u00e9") is False


class TestQueueBackpressure:
    def test_saturated_queue_returns_503(self, tiny_model, monkeypatch):  # noqa: F811
        import autovision.serving.app as app_module

        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            async def slow_threadpool(*args, **kwargs):
                started.set()
                await release.wait()
                return [{}]

            monkeypatch.setattr(app_module, "run_in_threadpool", slow_threadpool)
            app = create_app(
                tiny_model,
                threshold=0.0,
                max_concurrency=1,
                queue_timeout_s=0.05,
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/predict", json={"image": _b64_image(seed=31)})
                )
                await asyncio.wait_for(started.wait(), timeout=2.0)
                second = await client.post(
                    "/predict", json={"image": _b64_image(seed=32)}
                )
                release.set()
                first_response = await first
                # H2-5d: queue timeouts are observable in the latency histogram.
                metrics_text = (await client.get("/metrics")).text
            return first_response, second, metrics_text

        first_response, second, metrics_text = asyncio.run(scenario())
        assert first_response.status_code == 200
        assert second.status_code == 503
        assert second.headers["Retry-After"] == "1"
        assert "saturated" in second.json()["error"]
        assert 'autovision_predict_latency_seconds_count{error_class="queue_timeout"}' in metrics_text


class TestObservabilityAuth:
    def test_metrics_and_drift_follow_the_api_key(self, tiny_model):  # noqa: F811
        app = create_app(tiny_model, threshold=0.0, api_key="s3cret")
        client = TestClient(app)
        assert client.get("/metrics").status_code == 401
        assert client.get("/drift").status_code == 401
        headers = {"X-API-Key": "s3cret"}
        assert client.get("/metrics", headers=headers).status_code == 200
        assert client.get("/drift", headers=headers).status_code == 200
        # Kubernetes probes remain unauthenticated.
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200

    def test_observability_auth_can_be_disabled_explicitly(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model,
            threshold=0.0,
            api_key="s3cret",
            protect_observability_endpoints=False,
        )
        assert TestClient(app).get("/metrics").status_code == 200

    def test_observability_failures_are_rate_limited(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model,
            threshold=0.0,
            api_key="s3cret",
            rate_limit_per_minute=1,
            rate_limit_burst=1,
        )
        client = TestClient(app)
        assert client.get("/metrics").status_code == 401
        throttled = client.get("/metrics")
        assert throttled.status_code == 429
        assert throttled.headers["Retry-After"]


class _StubEngine:
    """Minimal callable engine: plumbing tests only need decode + dispatch."""

    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        return [{"score": 1.0} for _ in images]


class _UvicornCapture:
    """Record a single uvicorn.run invocation without binding a socket."""

    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture()
def uvicorn_run(monkeypatch):
    capture = _UvicornCapture()
    monkeypatch.setattr("uvicorn.run", capture)
    return capture


@pytest.fixture()
def stub_app(monkeypatch):
    """Replace the real app build inside serve(): launcher tests must not
    load a model onto a device just to assert transport options."""
    monkeypatch.setattr(
        serving_server,
        "create_app",
        lambda model, **kwargs: ("APP", model, kwargs),
    )


class TestServeLauncherHardening:
    """H2-1/H2-4/H2-2a/H2-5e: launch-time guards of autovision.serving.server."""

    def test_non_loopback_without_a_key_is_fail_closed(self, uvicorn_run):
        """H2-1: 0.0.0.0 + no key + no env key refuses to start (P0 fix)."""
        with pytest.raises(AutoVisionError, match="AUTOVISION_SERVE_ALLOW_INSECURE"):
            serve("model.pt", host="0.0.0.0")
        assert uvicorn_run.args == ()  # uvicorn.run was never reached

    def test_non_loopback_with_env_key_starts(self, monkeypatch, uvicorn_run, stub_app):
        """H2-1: AUTOVISION_SERVING_API_KEY satisfies the non-loopback guard."""
        monkeypatch.setenv("AUTOVISION_SERVING_API_KEY", "env-secret")
        serve("model.pt", host="0.0.0.0", threshold=0.2)
        assert uvicorn_run.kwargs["host"] == "0.0.0.0"
        assert uvicorn_run.args[0] == ("APP", "model.pt", {"threshold": 0.2})

    def test_non_loopback_with_explicit_key_starts(self, uvicorn_run, stub_app):
        """H2-1: api_key= keeps working for non-loopback single-process serving."""
        serve("model.pt", host="192.168.1.10", api_key="k")
        assert uvicorn_run.kwargs["host"] == "192.168.1.10"

    def test_allow_insecure_env_opt_in_starts(self, monkeypatch, uvicorn_run, stub_app):
        """H2-1: the explicit open-endpoint escape hatch is honored."""
        monkeypatch.setenv("AUTOVISION_SERVE_ALLOW_INSECURE", "1")
        serve("model.pt", host="0.0.0.0")
        assert uvicorn_run.kwargs["host"] == "0.0.0.0"

    def test_loopback_without_a_key_keeps_the_legacy_behavior(
        self, uvicorn_run, stub_app
    ):
        """H2-1: loopback binding stays unauthenticated by default."""
        serve("model.pt")
        assert uvicorn_run.kwargs["host"] == "127.0.0.1"

    def test_non_loopback_is_guarded_for_multi_worker_too(self, uvicorn_run):
        """H2-1: the import-string factory path is checked before uvicorn.run."""
        with pytest.raises(AutoVisionError, match="AUTOVISION_SERVING_API_KEY"):
            serve("model.pt", host="0.0.0.0", workers=2)
        assert uvicorn_run.args == ()

    def test_api_key_is_rejected_for_multi_worker(self, uvicorn_run):
        """H2-4: the key would be serialized into the env and leak via /proc."""
        with pytest.raises(AutoVisionError, match="AUTOVISION_SERVING_API_KEY"):
            serve("model.pt", workers=2, api_key="secret")
        assert uvicorn_run.args == ()

    def test_multi_worker_env_key_stays_out_of_the_serialized_config(
        self, monkeypatch, uvicorn_run
    ):
        """H2-4: with the env channel the config payload carries no credential."""
        monkeypatch.setenv("AUTOVISION_SERVING_API_KEY", "env-secret")
        monkeypatch.setenv(SERVE_CONFIG_ENV, "")  # tracked for teardown restore
        observed: dict[str, str] = {}
        real_run = uvicorn_run

        def recording_run(*args, **kwargs):
            # SERVE-09: serve() cleans the config env up on exit, so the
            # payload is observed at uvicorn.run call time.
            observed["payload"] = os.environ[SERVE_CONFIG_ENV]
            real_run(*args, **kwargs)

        import uvicorn as _uvicorn

        monkeypatch.setattr(_uvicorn, "run", recording_run)
        serve("model.pt", workers=2)
        payload = json.loads(observed["payload"])
        assert payload == {"model": "model.pt"}
        assert uvicorn_run.kwargs["workers"] == 2
        assert uvicorn_run.kwargs["factory"] is True
        assert SERVE_CONFIG_ENV not in os.environ

    def test_graceful_shutdown_defaults_to_thirty_seconds(
        self, uvicorn_run, stub_app
    ):
        """H2-5e: the SIGTERM drain is bounded by default, not unbounded."""
        serve("model.pt")
        assert uvicorn_run.kwargs["timeout_graceful_shutdown"] == 30

    def test_graceful_shutdown_none_restores_unbounded_wait(
        self, uvicorn_run, stub_app
    ):
        """H2-5e: explicit None still restores uvicorn's unbounded drain."""
        serve("model.pt", timeout_graceful_shutdown=None)
        assert "timeout_graceful_shutdown" not in uvicorn_run.kwargs

    def test_connection_limits_are_forwarded_to_uvicorn(self, uvicorn_run, stub_app):
        """H2-2a: concurrency/keep-alive bounds reach uvicorn (single process)."""
        serve("model.pt", limit_concurrency=8, timeout_keep_alive=5)
        assert uvicorn_run.kwargs["limit_concurrency"] == 8
        assert uvicorn_run.kwargs["timeout_keep_alive"] == 5

    def test_connection_limits_have_safe_defaults(self, uvicorn_run, stub_app):
        """H2-2a: unbounded queues are not the default launch posture."""
        serve("model.pt")
        assert uvicorn_run.kwargs["limit_concurrency"] == 64
        assert uvicorn_run.kwargs["timeout_keep_alive"] == 30

    def test_connection_limits_forwarded_for_multi_worker(self, monkeypatch, uvicorn_run):
        """H2-2a: the import-string factory path forwards the bounds too."""
        monkeypatch.setenv(SERVE_CONFIG_ENV, "")  # tracked for teardown restore
        serve("model.pt", workers=2, limit_concurrency=8)
        assert uvicorn_run.kwargs["limit_concurrency"] == 8

    def test_bad_connection_limits_fail_loud(self):
        with pytest.raises(AutoVisionError, match="limit_concurrency"):
            serve("model.pt", limit_concurrency=0)
        with pytest.raises(AutoVisionError, match="timeout_keep_alive"):
            serve("model.pt", timeout_keep_alive=0)


class TestPixelBudget:
    """H2-2b/c: request-wide pixel budget and the 16M single-image default."""

    @staticmethod
    def _flat_image(width: int, height: int) -> str:
        """Solid-color PNG: tiny payload, exact decoded pixel count."""
        from PIL import Image

        image = Image.new("RGB", (width, height), (17, 34, 51))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_default_cap_is_16_megapixels(self):
        """H2-2c: the default now covers 4000x4000 frames, not 6000x6000+."""
        assert _DEFAULT_MAX_IMAGE_PIXELS == 16_000_000

    def test_batch_total_over_budget_is_413(self):
        """H2-2b: four 15M-pixel images exceed a (60M - 1) request budget."""
        client = TestClient(
            create_app(
                engine=_StubEngine(),
                threshold=0.0,
                max_batch=4,
                max_image_pixels=16_000_000,
                max_request_pixels=59_999_999,
            )
        )
        images = [self._flat_image(3750, 4000) for _ in range(4)]  # 4 x 15M
        response = client.post("/predict", json={"images": images})
        assert response.status_code == 413
        error = response.json()["error"]
        # The aggregate message must be distinguishable from the per-image one.
        assert "max_request_pixels" in error
        assert "in total" in error
        assert 'status="413"' in client.get("/metrics").text

    def test_batch_total_at_the_boundary_passes(self):
        """H2-2b: exactly max_request_pixels decodes and serves."""
        client = TestClient(
            create_app(
                engine=_StubEngine(),
                threshold=0.0,
                max_batch=4,
                max_image_pixels=16_000_000,
                max_request_pixels=60_000_000,
            )
        )
        images = [self._flat_image(3750, 4000) for _ in range(4)]  # 4 x 15M
        response = client.post("/predict", json={"images": images})
        assert response.status_code == 200, response.text

    def test_default_budget_is_batch_times_image_pixels(self):
        """H2-2b: with no explicit budget, max_batch x max_image_pixels fits."""
        client = TestClient(
            create_app(
                engine=_StubEngine(), threshold=0.0, max_batch=3, max_image_pixels=100
            )
        )
        images = [self._flat_image(10, 10) for _ in range(3)]  # 300 == 3 x 100
        assert client.post("/predict", json={"images": images}).status_code == 200

    def test_single_image_at_the_per_image_limit_passes(self):
        """H2-2b: the per-image check stays inclusive at the exact boundary."""
        client = TestClient(
            create_app(engine=_StubEngine(), threshold=0.0, max_image_pixels=100)
        )
        response = client.post("/predict", json={"image": self._flat_image(10, 10)})
        assert response.status_code == 200

    def test_request_budget_below_the_image_cap_is_rejected(self, tiny_model):  # noqa: F811
        """H2-2b: a total budget under the per-image cap is incoherent."""
        with pytest.raises(AutoVisionError, match="max_request_pixels"):
            create_app(tiny_model, max_request_pixels=5, max_image_pixels=100)


class TestRequestIdHardening:
    """H2-5b: client-supplied request ids cannot forge log lines."""

    def test_malformed_ids_are_truncated_and_neutralized(self):
        dirty = "evil\nFAKE status=200 request_id=forged"
        clean = _sanitize_request_id(dirty)
        assert clean == "evil?FAKE?status?200?request?id?forged"
        assert _sanitize_request_id("req-abc") == "req-abc"  # conforming ids pass
        assert len(_sanitize_request_id("x" * 500)) == 64
        assert _sanitize_request_id("a b/c!d") == "a?b?c?d"

    def test_injected_request_id_never_reaches_the_log(self, client, caplog):
        dirty = "evil\nFAKE status=200 request_id=forged"
        with caplog.at_level(logging.INFO, logger="autovision.serving"):
            response = client.post(
                "/predict",
                json={"image": _b64_image(seed=18)},
                headers={"X-Request-ID": dirty},
            )
        assert response.status_code == 200
        echoed = response.headers["X-Request-ID"]
        assert echoed == "evil?FAKE?status?200?request?id?forged"
        assert response.json()["request_id"] == echoed
        for record in caplog.records:
            assert "\n" not in record.getMessage()
        assert any("evil?FAKE" in record.getMessage() for record in caplog.records)


class TestErrorDisclosure:
    """H2-5c: framework-internal 400s cannot echo server internals."""

    def test_internal_engine_error_is_masked_with_a_stable_phrase(self, caplog):
        class _LeakyEngine:
            backend = "torch"
            artifact = None
            network = None

            def run_batch(self, images, *, threshold=0.5):
                raise AutoVisionError(
                    "cannot load engine artifact /srv/private/secret.engine"
                )

        app = create_app(engine=_LeakyEngine(), threshold=0.0)
        client = TestClient(app)
        with caplog.at_level(logging.ERROR, logger="autovision.serving"):
            response = client.post("/predict", json={"image": _b64_image(seed=19)})
        # SERVE-02: an engine-boundary failure is a SERVER error now —
        # the stable-phrase masking is unchanged, the status moved 400→500.
        assert response.status_code == 500
        body = response.json()
        assert body["request_id"] == response.headers["X-Request-ID"]
        assert "/srv/private" not in body["error"]
        assert "Prediction failed" in body["error"]
        # The original text stays server-side, keyed by request id.
        assert any("/srv/private" in record.getMessage() for record in caplog.records)

    def test_client_input_errors_keep_the_original_text(self, client):
        """Validation messages are built from client input and stay verbatim."""
        response = client.post("/predict", json={"images": ["!!not-base64!!"]})
        assert response.status_code == 400
        assert "Invalid base64 image payload" in response.json()["error"]


class TestErrorPathMetrics:
    """H2-5d: failed requests land in the latency histogram with error_class."""

    def test_rate_limited_predict_latency_is_observed(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model,
            threshold=0.0,
            rate_limit_per_minute=1,
            rate_limit_burst=1,
            trust_proxy_headers=True,
        )
        client = TestClient(app)
        first = client.post("/predict", json={"image": _b64_image(seed=23)})
        assert first.status_code == 200
        throttled = client.post("/predict", json={"image": _b64_image(seed=24)})
        assert throttled.status_code == 429
        # The limiter is per client key: scrape from a different key so the
        # shared bucket does not throttle the /metrics read itself.
        text = client.get(
            "/metrics", headers={"X-Forwarded-For": "10.0.0.9"}
        ).text
        assert 'autovision_predict_latency_seconds_count{error_class="rate_limited"}' in text

    def test_validation_error_latency_is_observed(self, client):
        response = client.post("/predict", json={"nope": 1})
        assert response.status_code == 400
        assert 'error_class="validation"' in client.get("/metrics").text
