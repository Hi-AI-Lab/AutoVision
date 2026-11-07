# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Second-round serving-surface hardening: state consistency, error
semantics, decode/warmup budgets, rate-limit topology, parent-side config
validation.

Covers the SERVE-01..09 / 11 / 12 / 13 hardening designs. Every
behavioral test here fails on
the pre-fix code by construction (typo applied-then-rejected, engine 500s
reported as 400s, decode timeouts poisoning healthy engines, ...).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from autovision.cli._app import (  # noqa: E402
    _parse_bool_kwarg,
    _parse_int_kwarg,
    app,
)
from autovision.errors import AutoVisionError, ConfigValidationError  # noqa: E402
from autovision.serving import create_app  # noqa: E402
from autovision.serving.app import validate_serve_config  # noqa: E402
from autovision.serving.server import serve  # noqa: E402


def _b64_image(size: int = 64, seed: int = 0) -> str:
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(seed)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class _OkEngine:
    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        return [
            {
                "bboxes": __import__("torch").tensor([[1.0, 2.0, 8.0, 8.0]]),
                "scores": __import__("torch").tensor([0.9]),
                "labels": __import__("torch").tensor([0]),
            }
            for _ in images
        ]


class _EngineAutoVisionError:
    """A healthy-looking engine whose forward violates its own contract."""

    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        raise AutoVisionError("engine returned an inconsistent record layout")


class _SlowDecodeEngine:
    """A healthy engine whose IMAGE DECODE phase is slow (SERVE-04)."""

    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        return []


class _SlowForwardEngine:
    """A healthy engine whose FORWARD is slow (the poison contract)."""

    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        time.sleep(0.4)
        return []


class _SlowWarmupEngine:
    """A healthy engine whose FIRST call (the warmup) never returns."""

    backend = "torch"
    artifact = None
    network = None

    def run_batch(self, images, *, threshold=0.5):
        if not getattr(self, "_warmed", False):
            time.sleep(1.0)
        return []


# ── SERVE-01: registry CLI validates before it mutates ────────────────────


class TestRegistryCliValidationOrder:
    def test_typo_key_does_not_apply_the_change(self, tmp_path, monkeypatch):
        """Pre-fix: `registry action=register ... stae=dev` registered the
        model FIRST and failed on the unknown key second — the change was
        already applied when the usage error printed."""
        import autovision.registry as registry_module
        from autovision.registry.model_registry import RegistryEntry

        calls: list[tuple] = []

        class _SpyRegistry:
            def __init__(self, root):
                pass

            def register(self, *args, **kwargs):
                calls.append(("register", kwargs))
                return RegistryEntry(
                    name=kwargs["name"], version=kwargs["version"],
                    stage=kwargs.get("stage", "dev"),
                    path=Path("x"), sha256="0" * 64, registered_at="now",
                )

        monkeypatch.setattr(registry_module, "ModelRegistry", _SpyRegistry)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["registry", "action=register", "file=best.pt", "name=det",
             "v=1.0.0", f"root={tmp_path}", "stae=dev"],  # typo
        )
        assert result.exit_code == 2
        assert "stae" in result.output
        assert calls == [], "register must NOT run when a key is unknown"

    def test_valid_keys_still_register(self, tmp_path, monkeypatch):
        import autovision.registry as registry_module
        from autovision.registry.model_registry import RegistryEntry

        calls: list[tuple] = []

        class _SpyRegistry:
            def __init__(self, root):
                pass

            def register(self, *args, **kwargs):
                calls.append(("register", kwargs))
                return RegistryEntry(
                    name=kwargs["name"], version=kwargs["version"],
                    stage=kwargs.get("stage", "dev"),
                    path=Path("x"), sha256="0" * 64, registered_at="now",
                )

        monkeypatch.setattr(registry_module, "ModelRegistry", _SpyRegistry)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["registry", "action=register", "file=best.pt", "name=det",
             "v=1.0.0", f"root={tmp_path}"],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1


# ── SERVE-02: engine failures are 5xx, client failures stay 4xx ───────────


class TestErrorSemantics:
    def test_engine_contract_violation_returns_500(self):
        """Pre-fix: an AutoVisionError crossing the engine boundary was
        mapped to 400 — a SERVER failure reported as the client's fault,
        poisoning retry policies and SLO splits."""
        with TestClient(
            create_app(engine=_EngineAutoVisionError(), threshold=0.0)
        ) as client:
            response = client.post(
                "/predict", json={"images": [_b64_image()]}
            )
        assert response.status_code == 500
        body = response.json()
        assert body["error"].startswith("Prediction failed")
        assert body["request_id"]

    def test_client_input_error_stays_400(self):
        with TestClient(create_app(engine=_OkEngine(), threshold=0.0)) as client:
            response = client.post(
                "/predict", json={"images": ["!!not-base64!!"]}
            )
        assert response.status_code == 400

    def test_error_class_split_visible_on_metrics(self):
        with TestClient(
            create_app(engine=_EngineAutoVisionError(), threshold=0.0)
        ) as client:
            client.post("/predict", json={"images": [_b64_image()]})
            client.post("/predict", json={"images": ["!!not-base64!!"]})
            metrics = client.get("/metrics").text
        assert 'error_class="internal"' in metrics
        assert 'error_class="validation"' in metrics


# ── SERVE-03: serve CLI numeric guards ────────────────────────────────────


class TestServeCliNumericGuards:
    def test_port_abc_is_a_usage_error_not_a_traceback(self, tmp_path):
        """Pre-fix: the real terminal printed a raw ValueError traceback
        for port=abc (CliRunner-based tests masked it — this test runs the
        actual entrypoint in a subprocess)."""
        result = subprocess.run(
            [sys.executable, "-m", "autovision.cli._app",
             "serve", "model=x.pt", "port=abc"],
            capture_output=True, text=True, timeout=120,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 2, result.stderr
        assert "port= must be an integer" in result.stderr
        assert "Traceback" not in result.stderr

    def test_port_out_of_range_is_a_usage_error(self):
        import typer

        kwargs = {"port": "99999"}
        with pytest.raises(typer.BadParameter, match="port"):
            _parse_int_kwarg(kwargs, "port", default=8000, low=1, high=65535)

    def test_parse_int_kwarg_unit(self):
        import typer

        assert _parse_int_kwarg({"port": "8123"}, "port", default=8000) == 8123
        with pytest.raises(typer.BadParameter, match="must be an integer"):
            _parse_int_kwarg({"workers": "abc"}, "workers", default=1, low=1)


# ── SERVE-04: decode budget is separate; poison is forward-only ───────────


class TestDecodeBudget:
    def test_decode_timeout_does_not_poison_the_engine(self, monkeypatch):
        """Pre-fix: decode ran inside the inference timeout, so a slow
        decode permanently poisoned a HEALTHY engine (503 until restart)."""
        import autovision.serving.app as app_module

        real_decode = app_module._decode_image

        def slow_decode(payload, *, max_pixels):
            time.sleep(0.3)
            return real_decode(payload, max_pixels=max_pixels)

        monkeypatch.setattr(app_module, "_decode_image", slow_decode)
        with TestClient(
            create_app(
                engine=_SlowDecodeEngine(), threshold=0.0,
                decode_timeout_s=0.05,
            )
        ) as client:
            timed_out = client.post("/predict", json={"images": [_b64_image()]})
            assert timed_out.status_code == 504
            assert "Decoding" in timed_out.json()["error"]
            # The engine stays healthy: readyz keeps 200 and a fast decode
            # (patch removed via new client) would serve — the poisoned
            # contract did NOT trip.
            assert client.get("/readyz").status_code == 200
            assert client.get("/healthz").json()["status"] == "ok"

    def test_decode_timeouts_counted_on_metrics(self, monkeypatch):
        import autovision.serving.app as app_module

        real_decode = app_module._decode_image
        monkeypatch.setattr(
            app_module, "_decode_image",
            lambda payload, *, max_pixels: (
                time.sleep(0.3) or real_decode(payload, max_pixels=max_pixels)
            ),
        )
        with TestClient(
            create_app(
                engine=_SlowDecodeEngine(), threshold=0.0,
                decode_timeout_s=0.05,
            )
        ) as client:
            client.post("/predict", json={"images": [_b64_image()]})
            metrics = client.get("/metrics").text
        assert "autovision_decode_timeouts_total" in metrics
        assert 'error_class="decode_timeout"' in metrics

    def test_forward_timeout_still_poisons(self):
        """The poison contract stays reserved for stuck forwards (H2-3)."""
        with TestClient(
            create_app(
                engine=_SlowForwardEngine(), threshold=0.0,
                inference_timeout_s=0.05,
            )
        ) as client:
            timed_out = client.post("/predict", json={"images": [_b64_image()]})
            assert timed_out.status_code == 504
            assert "unresponsive" in timed_out.json()["error"]
            # The variant is poisoned now.
            ready = client.get("/readyz")
            assert ready.status_code == 503


# ── SERVE-05: warmup timeout ──────────────────────────────────────────────


class TestWarmupTimeout:
    def test_wedged_warmup_fails_boot_with_a_named_error(self):
        with pytest.raises(AutoVisionError, match="warmup_timeout_s"):
            with TestClient(
                create_app(
                    engine=_SlowWarmupEngine(), threshold=0.0,
                    warmup_timeout_s=0.05,
                )
            ):
                pass

    def test_zero_disables_warmup(self):
        """warmup_timeout_s=0 skips warmup entirely (documented opt-out)."""
        with TestClient(
            create_app(engine=_OkEngine(), threshold=0.0, warmup_timeout_s=0)
        ) as client:
            assert client.get("/healthz").status_code == 200

    def test_negative_warmup_timeout_rejected(self):
        with pytest.raises(AutoVisionError, match="warmup_timeout_s"):
            create_app(engine=_OkEngine(), threshold=0.0, warmup_timeout_s=-1)


# ── SERVE-06: rate-limit topology ─────────────────────────────────────────


class TestRateLimitTopology:
    def test_proxy_hops_split_clients_into_distinct_buckets(self):
        """Pre-fix: behind a proxy every client shared the socket-peer
        bucket — one noisy client 429'd everyone. With hops=2 the client
        is the SECOND entry from the right (each trusted proxy appends the
        peer it saw), so the two distinct client entries select distinct
        buckets."""
        app = create_app(
            engine=_OkEngine(), threshold=0.0,
            rate_limit_per_minute=2, rate_limit_burst=2,
            trusted_proxy_hops=2,
        )
        with TestClient(app) as client:
            # Chain shape: [client-injected?, client_ip, proxy2_ip] — the
            # client is parts[-2] once two trusted proxies have appended.
            headers_a = {"X-Forwarded-For": "10.0.0.1, 172.17.0.9"}
            headers_b = {"X-Forwarded-For": "10.0.0.2, 172.17.0.9"}
            codes_a = [
                client.post("/predict", json={"images": [_b64_image()]},
                            headers=headers_a).status_code
                for _ in range(3)
            ]
            codes_b = [
                client.post("/predict", json={"images": [_b64_image()]},
                            headers=headers_b).status_code
                for _ in range(3)
            ]
        assert codes_a == [200, 200, 429]
        assert codes_b == [200, 200, 429]  # independent buckets

    def test_endpoints_have_distinct_buckets(self):
        """/metrics scraping must not consume /predict quota (pre-fix they
        shared one per-client bucket)."""
        app = create_app(
            engine=_OkEngine(), threshold=0.0,
            rate_limit_per_minute=1, rate_limit_burst=1,
        )
        with TestClient(app) as client:
            assert client.post(
                "/predict", json={"images": [_b64_image()]}
            ).status_code == 200
            # The predict bucket is empty now, but metrics has its own
            # bucket — the first scrape would have been 429 under the old
            # shared per-client key.
            assert client.get("/metrics").status_code == 200

    def test_deprecated_bool_maps_to_one_hop_with_a_warning(self, caplog):
        with caplog.at_level("WARNING", logger="autovision"):
            create_app(
                engine=_OkEngine(), threshold=0.0, trust_proxy_headers=True,
            )
        assert any("deprecated in 0.6.1" in r.getMessage() for r in caplog.records)

    def test_negative_hops_rejected(self):
        with pytest.raises(AutoVisionError, match="trusted_proxy_hops"):
            create_app(engine=_OkEngine(), threshold=0.0, trusted_proxy_hops=-1)


# ── SERVE-07: parent-side config validation ───────────────────────────────


class TestServeConfigValidation:
    def test_pure_function_rejects_bad_knobs(self):
        with pytest.raises(AutoVisionError, match="max_batch"):
            validate_serve_config(max_batch=0)
        with pytest.raises(AutoVisionError, match="mask_format"):
            validate_serve_config(mask_format="bogus")
        validate_serve_config()  # defaults pass

    def test_workers_gt_one_rejects_bad_config_in_the_parent(self, monkeypatch):
        """Pre-fix: the bad config exploded inside every worker (uvicorn
        restart loop); now the parent fails with one clean error before
        uvicorn.run is reached."""
        import sys

        ran_uvicorn: list[int] = []
        monkeypatch.setitem(
            sys.modules, "uvicorn",
            SimpleNamespace(run=lambda *a, **k: ran_uvicorn.append(1)),
        )
        with pytest.raises(AutoVisionError, match="mask_format"):
            serve(
                str(Path("unused.pt")),
                host="127.0.0.1", port=8901, workers=2,
                mask_format="bogus",
            )
        assert ran_uvicorn == []

    def test_unknown_serving_option_named_in_the_parent(self, monkeypatch):
        import sys

        monkeypatch.setitem(
            sys.modules, "uvicorn",
            SimpleNamespace(run=lambda *a, **k: None),
        )
        with pytest.raises(AutoVisionError, match="Unknown or misspelled"):
            serve(
                str(Path("unused.pt")),
                host="127.0.0.1", port=8902, workers=2,
                not_a_real_option=1,
            )


# ── SERVE-08: 429 joins the latency histogram ─────────────────────────────


class TestThrottleLatency:
    def test_rate_limited_error_class_on_metrics(self):
        app = create_app(
            engine=_OkEngine(), threshold=0.0,
            rate_limit_per_minute=1, rate_limit_burst=1,
        )
        with TestClient(app) as client:
            client.post("/predict", json={"images": [_b64_image()]})
            client.post("/predict", json={"images": [_b64_image()]})  # 429
            metrics = client.get("/metrics").text
        assert 'error_class="rate_limited"' in metrics


# ── SERVE-11: strict bool parsing ─────────────────────────────────────────


class TestBoolParsing:
    def test_shell_style_false_is_false(self):
        kwargs = {"strict": "False"}
        assert _parse_bool_kwarg(kwargs, "strict", default=True) is False

    def test_garbage_is_a_usage_error(self):
        import typer

        with pytest.raises(typer.BadParameter, match="boolean"):
            _parse_bool_kwarg({"force": "banana"}, "force", default=False)


# ── SERVE-12/13: CLI helper exception family + epochs guard ───────────────


class TestCliHelpers:
    def test_parallel_rejects_with_config_error(self):
        from autovision.cli.common_helpers import resolve_parallel

        with pytest.raises(ConfigValidationError):
            resolve_parallel({"parallel": "deepspeed"})

    def test_epochs_reject_floats_and_garbage(self):
        from autovision.cli.train_helpers import get_epochs
        from autovision.pretext.recipe import PretextRecipe

        class _Recipe(PretextRecipe):
            pass

        with pytest.raises(ConfigValidationError, match="whole number"):
            get_epochs(_Recipe, 2.5, dataset_size=100, batch_size=8)
        with pytest.raises(ConfigValidationError, match="integer or 'auto'"):
            get_epochs(_Recipe, "abc", dataset_size=100, batch_size=8)
        assert get_epochs(_Recipe, 5, dataset_size=100, batch_size=8) == 5




# ── review follow-ups: P1-1 label pollution, P2-1 validate coverage ───────


class TestReviewFollowUps:
    def test_admitted_request_never_produces_rate_limited_series(self):
        """Review P1-1: the error-latency observation used to sit OUTSIDE
        the 429 branch — every ALLOWED request fired the rate_limited label
        and double-counted into the histogram, poisoning the very SLO
        split SERVE-08 exists to protect."""
        app = create_app(
            engine=_OkEngine(), threshold=0.0,
            rate_limit_per_minute=10, rate_limit_burst=10,
        )
        with TestClient(app) as client:
            assert client.post(
                "/predict", json={"images": [_b64_image()]}
            ).status_code == 200
            metrics = client.get("/metrics").text
        assert 'error_class="rate_limited"' not in metrics

    def test_rate_limited_series_only_from_actual_429s(self):
        app = create_app(
            engine=_OkEngine(), threshold=0.0,
            rate_limit_per_minute=1, rate_limit_burst=1,
        )
        with TestClient(app) as client:
            assert client.post(
                "/predict", json={"images": [_b64_image()]}
            ).status_code == 200
            assert client.post(
                "/predict", json={"images": [_b64_image()]}
            ).status_code == 429
            metrics = client.get("/metrics").text
        assert 'error_class="rate_limited"' in metrics

    def test_deprecated_bool_param_passes_workers_gt_one_validation(
        self, monkeypatch,
    ):
        """Review P2-1: trust_proxy_headers is a SUPPORTED (deprecated)
        parameter — the workers>1 parent validation must not reject it as
        a typo while workers=1 accepts it (deprecation protocol)."""
        import sys

        ran: list[int] = []
        monkeypatch.setitem(
            sys.modules, "uvicorn",
            SimpleNamespace(run=lambda *a, **k: ran.append(1)),
        )
        serve(
            str(Path("unused.pt")),
            host="127.0.0.1", port=8903, workers=2,
            trust_proxy_headers=True,
        )
        assert ran == [1]

    def test_string_threshold_rejected_not_silently_passed(self):
        """Review P2-1: threshold="0.9" passed the old validation while the
        request path operated on the raw string (every request 500'd)."""
        with pytest.raises(AutoVisionError, match="threshold"):
            validate_serve_config(threshold="0.9")

    def test_bad_backend_rejected_in_the_parent(self):
        """Review P2-1: a workers>1 launch with a bogus backend used to
        pass parent validation and restart-loop inside every worker."""
        with pytest.raises(AutoVisionError, match="backend"):
            validate_serve_config(backend="bogus")
