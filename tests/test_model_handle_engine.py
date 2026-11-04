# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for the handle-level compiled-backend engine cache (D15).

These are mock-only tests: the recording engine replaces
``InferenceEngine`` so construction counts and lifecycle semantics are
verified without GPU, artifacts, or training runs.
"""

from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from autovision.cli.model_handle import DetectionModel, detection
from autovision.errors import AutoVisionError


class _RecordingEngine:
    """Stand-in for ``InferenceEngine`` that counts constructions."""

    builds = 0

    def __init__(
        self,
        model: Any,
        *,
        backend: str,
        artifact: Any,
        device: Any,
        providers: Any,
        openvino_device: str,
    ) -> None:
        type(self).builds += 1
        self.model = model
        self.init_args: dict[str, Any] = {
            "backend": backend,
            "artifact": artifact,
            "device": device,
            "providers": providers,
            "openvino_device": openvino_device,
        }

    def run(self, image: Any, *, threshold: float = 0.5) -> dict[str, Any]:
        """Echo the setup backend and the per-call threshold."""
        return {"backend": self.init_args["backend"], "threshold": threshold}


class _FakeNetwork:
    """Minimal task-network stand-in exposing only ``predict``."""

    def predict(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        """Echo kwargs so per-call runtime parameters are observable."""
        return {"network": True, **kwargs}


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real engine class with the recording counter."""
    _RecordingEngine.builds = 0
    monkeypatch.setattr(
        "autovision.inference.engine.InferenceEngine", _RecordingEngine
    )


def _make_handle(network: Any) -> DetectionModel:
    """Build a handle whose network is already resolved."""
    handle = detection("rtdetr/resnet18")
    handle._network = network
    return handle


class TestEngineResolution:
    """Setup-key signature semantics of ``_resolve_engine``."""

    def test_compiled_backend_builds_once_then_reuses(self) -> None:
        """Two predictions with identical setup keys hit one engine build."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        first = handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        second = handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        assert first == second == {"backend": "onnx", "threshold": 0.5}
        assert _RecordingEngine.builds == 1
        assert len(handle._engine_cache) == 1
        assert next(iter(handle._engine_cache.values())).model is handle._network

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(
                {"backend": "onnx", "artifact": "m.onnx"},
                {"backend": "onnx", "artifact": "n.onnx"},
                id="artifact-change",
            ),
            pytest.param(
                {"backend": "onnx", "artifact": "m.onnx"},
                {
                    "backend": "onnx",
                    "artifact": "m.onnx",
                    "device": "cuda:0",
                },
                id="device-change",
            ),
            pytest.param(
                {"backend": "onnx", "artifact": "m.onnx"},
                {
                    "backend": "onnx",
                    "artifact": "m.onnx",
                    "providers": ("CUDAExecutionProvider",),
                },
                id="providers-change",
            ),
            pytest.param(
                {"backend": "openvino", "artifact": "m.xml"},
                {
                    "backend": "openvino",
                    "artifact": "m.xml",
                    "openvino_device": "GPU:0",
                },
                id="openvino-device-change",
            ),
        ],
    )
    def test_setup_key_change_rebuilds(
        self, first: dict[str, Any], second: dict[str, Any]
    ) -> None:
        """Each distinct setup-key signature produces a fresh engine."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", **first)
        handle.predict("a.jpg", **second)
        assert _RecordingEngine.builds == 2

    def test_runtime_kwargs_do_not_rebuild(self) -> None:
        """Threshold-style per-call parameters never invalidate the cache."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx", threshold=0.2)
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx", threshold=0.8)
        assert _RecordingEngine.builds == 1

    def test_engine_wraps_current_network(self) -> None:
        """The cached engine is bound to the handle's network object."""
        network = torch.nn.Linear(2, 2)
        handle = _make_handle(network)
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        assert next(iter(handle._engine_cache.values())).model is network

    def test_alternating_two_signatures_hit_the_lru(self) -> None:
        """Alternating two setups stays at two builds (stress-probe regress)."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        for index in range(20):
            backend = "onnx" if index % 2 == 0 else "openvino"
            artifact = "m.onnx" if backend == "onnx" else "m.xml"
            handle.predict("a.jpg", backend=backend, artifact=artifact)
        assert _RecordingEngine.builds == 2
        assert len(handle._engine_cache) == 2

    def test_third_signature_evicts_the_least_recent(self) -> None:
        """True LRU (not FIFO): re-hitting a signature refreshes its slot.

        a, b, hit-a, c — the eviction victim must be b (least RECENT), not
        a (least recently INSERTED). A dict-assignment "LRU" is FIFO and
        evicts a here, which this test catches.
        """
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", backend="onnx", artifact="a.onnx")
        handle.predict("a.jpg", backend="onnx", artifact="b.onnx")
        handle.predict("a.jpg", backend="onnx", artifact="a.onnx")  # refresh a
        assert _RecordingEngine.builds == 2
        handle.predict("a.jpg", backend="onnx", artifact="c.onnx")  # evicts b
        assert _RecordingEngine.builds == 3
        assert len(handle._engine_cache) == 2
        # a was refreshed, so it must still be cached: no rebuild.
        handle.predict("a.jpg", backend="onnx", artifact="a.onnx")
        assert _RecordingEngine.builds == 3
        # b was the evicted one: hitting it rebuilds.
        handle.predict("a.jpg", backend="onnx", artifact="b.onnx")
        assert _RecordingEngine.builds == 4

    def test_pre_d15_pickle_roundtrip(self) -> None:
        """A handle pickled before the engine cache existed still works."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        # Simulate a pre-D15 instance: neither cache nor lock key present.
        del handle.__dict__["_engine_cache"]
        del handle.__dict__["_engine_lock"]
        restored = pickle.loads(pickle.dumps(handle))
        assert restored._engine_cache == {}
        assert isinstance(restored._engine_lock, type(threading.Lock()))
        # And it predicts without AttributeError.
        assert (
            restored.predict("a.jpg", backend="onnx", artifact="m.onnx")["backend"]
            == "onnx"
        )

    def test_concurrent_predictions_build_once(self) -> None:
        """Racing callers with the same signature produce a single build."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        barrier = threading.Barrier(4)
        results: list[dict[str, Any]] = []

        def worker() -> None:
            barrier.wait()
            results.append(handle.predict("a.jpg", backend="onnx", artifact="m.onnx"))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 4
        assert _RecordingEngine.builds == 1


class TestTorchBackendContract:
    """Default torch path stays on the network and rejects artifacts."""

    def test_default_backend_uses_network(self) -> None:
        """Default ``predict`` keeps the historical network-forward path."""
        handle = _make_handle(_FakeNetwork())
        result = handle.predict("a.jpg", threshold=0.3)
        assert result == {"network": True, "threshold": 0.3}
        assert _RecordingEngine.builds == 0

    def test_torch_backend_rejects_artifact(self) -> None:
        """``backend='torch'`` with an artifact is a configuration error."""
        handle = _make_handle(_FakeNetwork())
        with pytest.raises(AutoVisionError, match="takes no artifact"):
            handle.predict("a.jpg", artifact="m.onnx")


class TestCacheLifecycle:
    """Invalidation and serialization boundaries of the engine cache."""

    def test_train_invalidates_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-training replaces the network and must drop the stale engine."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        assert handle._engine_cache

        fake_result = SimpleNamespace(
            best_exported_model_path=None,
            exported_model_path="exported/m.pt",
            classes={0: "cat"},
        )
        monkeypatch.setattr(
            "autovision.cli.train_task.object_detection.train_object_detection",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "autovision.cli.model_handle.load_task_model",
            lambda *args, **kwargs: torch.nn.Linear(2, 2),
        )
        handle.train(out="runs/x", data="coco8")
        assert handle._engine_cache == {}

    def test_pickle_drops_engine_cache_and_rebuilds(self) -> None:
        """Unpickling yields a clean cache and a fresh lock; predict works."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        assert handle._engine_cache

        restored = pickle.loads(pickle.dumps(handle))
        assert restored._engine_cache == {}
        assert isinstance(restored._engine_lock, type(threading.Lock()))
        assert (
            restored.predict("a.jpg", backend="onnx", artifact="m.onnx")["backend"]
            == "onnx"
        )
        assert _RecordingEngine.builds == 2

    def test_invalidate_engine_clears_all_slots(self) -> None:
        """Explicit invalidation empties every LRU slot."""
        handle = _make_handle(torch.nn.Linear(2, 2))
        handle.predict("a.jpg", backend="onnx", artifact="m.onnx")
        handle.predict("a.jpg", backend="onnx", artifact="n.onnx")
        handle.invalidate_engine()
        assert handle._engine_cache == {}
