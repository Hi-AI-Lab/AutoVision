# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the TensorRT export path.

Full engine builds need the ``tensorrt`` package plus a GPU; the tests below
cover everything reachable without it: argument validation (which happens
before the TensorRT import by design) and the graceful degradation message.
"""

from __future__ import annotations

import pytest

from autovision.export.tensorrt_helpers import export_onnx_to_tensorrt


def _noop_export_fn(**kwargs):  # pragma: no cover - only reached with trt installed
    raise AssertionError("export_onnx_fn must not run when validation fails")


class TestValidation:
    def test_batch_size_ordering_enforced(self, tmp_path):
        with pytest.raises(ValueError, match="min_batch_size <= opt_batch_size"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
                min_batch_size=4,
                opt_batch_size=2,
                max_batch_size=8,
            )

    def test_strongly_typed_excludes_fp32_attention(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
                strongly_typed=True,
                fp32_attention_scores=True,
            )


class TestGracefulDegradation:
    def test_missing_tensorrt_gives_install_guidance(self, tmp_path):
        try:
            import tensorrt  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            pytest.skip("tensorrt is installed; degradation path not applicable")
        with pytest.raises(ModuleNotFoundError, match="pip install tensorrt"):
            export_onnx_to_tensorrt(
                export_onnx_fn=_noop_export_fn,
                out=tmp_path / "model.engine",
            )

    def test_task_model_method_exposed(self):
        from autovision.tasks import RTDETRObjectDetection

        assert hasattr(RTDETRObjectDetection, "export_tensorrt")
