# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference serving (roadmap F2): /predict batching, /healthz, /metrics."""

import base64
import io

import numpy as np
import pytest
from tests.test_tracking import tiny_model  # noqa: F401 - local fixture reuse

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import create_app  # noqa: E402


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
        text = client.get("/metrics").text
        assert 'autovision_requests_total{endpoint="predict",status="500"}' in text
        assert "autovision_errors_total 1" in text


class TestAppValidation:
    def test_missing_model_fails_loud(self):
        with pytest.raises(AutoVisionError, match="engine= or model="):
            create_app()

    def test_bad_max_batch_fails_loud(self, tiny_model):  # noqa: F811 - fixture param
        with pytest.raises(AutoVisionError, match="max_batch"):
            create_app(tiny_model, max_batch=0)
