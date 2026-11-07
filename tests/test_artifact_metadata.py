# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Artifact identity metadata (design D1): build/read/verify round-trips.

The identity gate closes at three places, each covered here: the schema
builder (duck-typed over task networks), the per-backend readers (ONNX
in-graph ``metadata_props`` vs JSON sidecars), and the
:class:`InferenceEngine` construction gate that fails a network pointed
at a foreign task's artifact.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import onnx
import onnx.helper
import pytest

from autovision.errors import AutoVisionError
from autovision.export.artifact_metadata import (
    _NAMES_LIMIT,
    META_SCHEMA_VERSION,
    build_metadata,
    infer_task_name,
    read_metadata,
    record_onnx_opset,
    toolchain_versions,
    verify_engine_artifact,
    verify_metadata,
    write_artifact_metadata,
)
from autovision.inference.engine import InferenceEngine
from autovision.tasks.task_model import TaskNetwork

# The stub's module path decides the inferred task name; any real task
# surface module works without importing the (heavy) real network.
_TASK_MODULE = "autovision.tasks.pose_estimation.gtr.task_model"


class _StubNetwork(TaskNetwork):
    """Minimal TaskNetwork standing in for a trained task model."""

    def __init__(
        self, init_args: dict[str, object] | None = None, **attrs: object
    ) -> None:
        super().__init__(init_args=init_args or {"model_name": "stub/backbone"})
        for key, value in attrs.items():
            setattr(self, key, value)


def _pose_network(**attrs: object) -> _StubNetwork:
    _StubNetwork.__module__ = _TASK_MODULE
    attrs.setdefault("image_size", (640, 640))
    attrs.setdefault("num_classes", 17)
    return _StubNetwork(**attrs)


def _write_identity_onnx(path: Path, metadata: dict[str, str]) -> None:
    """Smallest valid ONNX graph that onnxruntime can actually load."""
    value_info = onnx.helper.make_tensor_value_info
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["images"], ["outputs"])],
        "identity",
        [value_info("images", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [value_info("outputs", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
        # The installed onnxruntime may lag the onnx package's IR version;
        # a test graph only needs an old, universally supported IR.
        ir_version=9,
    )
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


@contextmanager
def _artifact_with_metadata(
    tmp_path: Path, network: _StubNetwork
) -> Iterator[Path]:
    path = tmp_path / "model.onnx"
    _write_identity_onnx(path, build_metadata(network))
    yield path


# ── schema builder ──────────────────────────────────────────────────────


def test_build_metadata_covers_the_full_schema(tmp_path: Path) -> None:
    """Legacy key names stay untouched; D1 keys use the autovision_ prefix."""
    network = _pose_network(class_names=["cat", "dog"])

    metadata = build_metadata(network)

    assert metadata["autovision_meta_version"] == str(META_SCHEMA_VERSION)
    assert metadata["autovision_task"] == "pose_estimation"
    assert metadata["autovision_class_path"].endswith("_StubNetwork")
    assert metadata["autovision_input_layout"] == "NCHW"
    assert metadata["model_name"] == "stub/backbone"
    assert metadata["num_classes"] == "17"
    assert json.loads(metadata["autovision_image_size"]) == [640, 640]
    assert json.loads(metadata["autovision_names"]) == {
        "names": ["cat", "dog"],
        "truncated": False,
    }


def test_build_metadata_normalizes_scalar_image_size() -> None:
    """A classifier-style int image_size becomes an [H, W] pair."""
    network = _pose_network(image_size=224)

    assert json.loads(build_metadata(network)["autovision_image_size"]) == [
        224, 224,
    ]


def test_build_metadata_truncates_large_class_lists() -> None:
    """Open-vocabulary heads keep metadata small, with the flag set."""
    network = _pose_network(class_names=[f"class_{i}" for i in range(600)])

    payload = json.loads(build_metadata(network)["autovision_names"])

    assert len(payload["names"]) == _NAMES_LIMIT
    assert payload["truncated"] is True


def test_infer_task_name_falls_back_to_unknown_outside_the_package() -> None:
    """A network defined outside autovision.tasks is 'unknown', not a crash."""

    class _Foreign(TaskNetwork):
        pass

    assert infer_task_name(_Foreign(init_args={})) == "unknown"
    assert infer_task_name(_pose_network()) == "pose_estimation"


def test_infer_task_name_prefers_the_explicit_attribute() -> None:
    """An artifact_task_name attribute wins over the module-path segment."""
    network = _pose_network(artifact_task_name="oriented_detection")

    assert infer_task_name(network) == "oriented_detection"


# ── toolchain record (H3-4d) ────────────────────────────────────────────


def test_build_metadata_records_toolchain_payload() -> None:
    """H3-4d: the schema carries the producing toolchain; optional packages
    that are absent stay None instead of lying about a version."""
    metadata = build_metadata(_pose_network())

    payload = json.loads(metadata["autovision_toolchain"])
    assert set(payload) == {
        "torch", "onnx", "onnx_opset", "onnxruntime", "tensorrt", "openvino",
    }
    assert payload["torch"] == toolchain_versions()["torch"]


def test_tensorrt_sidecar_carries_toolchain(tmp_path: Path) -> None:
    """H3-4d: the engine's .meta.json write point records the toolchain."""
    engine = tmp_path / "model.engine"
    write_artifact_metadata(
        _pose_network(), engine, backend="tensorrt", extra={"precision": "fp16"}
    )

    metadata = read_metadata(engine, backend="tensorrt")
    payload = json.loads(metadata["autovision_toolchain"])

    assert "torch" in payload and payload["torch"]


def test_openvino_sidecar_write_point_carries_toolchain(tmp_path: Path) -> None:
    """H3-4d: the OV sidecar spreads ``onnx_export_metadata()``, so the
    toolchain rides along without openvino_helpers changing."""
    ir = tmp_path / "model.xml"
    ir.with_suffix(".json").write_text(
        json.dumps({**build_metadata(_pose_network()), "backend": "openvino"}),
        encoding="utf-8",
    )

    metadata = read_metadata(ir, backend="openvino")

    assert json.loads(metadata["autovision_toolchain"])["torch"]


def test_record_onnx_opset_fills_the_payload() -> None:
    """H3-4d: the exporter fills the REAL graph opset; a legacy payload
    without the key is left alone."""
    metadata = build_metadata(_pose_network())
    record_onnx_opset(metadata, 18)
    assert json.loads(metadata["autovision_toolchain"])["onnx_opset"] == 18

    legacy: dict[str, str] = {}
    record_onnx_opset(legacy, 18)
    assert legacy == {}


def test_verify_metadata_warns_on_toolchain_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """H3-4d: load-time version skew warns (via the existing verify path —
    InferenceEngine constructs through it) but never fails."""
    network = _pose_network()
    metadata = build_metadata(network)
    payload = json.loads(metadata["autovision_toolchain"])
    payload["torch"] = "0.0.0-fake-build"
    metadata["autovision_toolchain"] = json.dumps(payload, sort_keys=True)

    with caplog.at_level(logging.WARNING):
        verify_metadata(metadata, network, artifact=tmp_path / "model.onnx")

    assert "different toolchain" in caplog.text
    assert "0.0.0-fake-build" in caplog.text


def test_verify_metadata_stays_silent_when_toolchain_matches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Matching toolchains (and legacy artifacts without the key) warn not."""
    network = _pose_network()

    with caplog.at_level(logging.WARNING):
        verify_metadata(
            build_metadata(network), network, artifact=tmp_path / "model.onnx"
        )
        verify_metadata({}, network, artifact=tmp_path / "model.onnx")

    assert "different toolchain" not in caplog.text


# ── read / verify round-trips ───────────────────────────────────────────


def test_onnx_metadata_roundtrip_and_verify(tmp_path: Path) -> None:
    """In-graph metadata survives a save/load and passes verification."""
    network = _pose_network()
    with _artifact_with_metadata(tmp_path, network) as path:
        metadata = read_metadata(path, backend="onnx")

        assert metadata["autovision_task"] == "pose_estimation"
        verify_metadata(metadata, network, artifact=path)


def test_tensorrt_sidecar_roundtrip_and_verify(tmp_path: Path) -> None:
    """Engines carry identity in <engine>.meta.json; verify accepts it."""
    network = _pose_network()
    engine = tmp_path / "model.engine"
    write_artifact_metadata(
        network, engine, backend="tensorrt", extra={"precision": "fp16"}
    )

    metadata = read_metadata(engine, backend="tensorrt")

    assert metadata["autovision_task"] == "pose_estimation"
    assert metadata["precision"] == "fp16"
    verify_metadata(metadata, network, artifact=engine)
    # No temp file survives the atomic publish.
    assert not list(tmp_path.glob(".*.tmp"))


def test_openvino_sidecar_is_read(tmp_path: Path) -> None:
    """The <xml>.json sidecar openvino_helpers already writes is honored."""
    network = _pose_network()
    ir = tmp_path / "model.xml"
    sidecar = ir.with_suffix(".json")
    sidecar.write_text(
        json.dumps({**build_metadata(network), "backend": "openvino"}),
        encoding="utf-8",
    )

    metadata = read_metadata(ir, backend="openvino")

    assert metadata["backend"] == "openvino"
    verify_metadata(metadata, network, artifact=ir)


def test_missing_sidecar_reads_as_empty(tmp_path: Path) -> None:
    """A TensorRT engine from before D1 simply has no identity to check."""
    assert read_metadata(tmp_path / "model.engine", backend="tensorrt") == {}


# ── verification failures ───────────────────────────────────────────────


def test_task_mismatch_fails_with_both_values(tmp_path: Path) -> None:
    """A detection artifact loaded under a pose network names both sides."""
    network = _pose_network()
    metadata = build_metadata(network)
    metadata["autovision_task"] = "detection"

    with pytest.raises(AutoVisionError, match="detection.*pose_estimation"):
        verify_metadata(metadata, network, artifact=tmp_path / "model.onnx")


def test_class_count_mismatch_fails(tmp_path: Path) -> None:
    """Different num_classes on the two sides is a hard error."""
    network = _pose_network()
    metadata = build_metadata(network)
    metadata["num_classes"] = "80"

    with pytest.raises(AutoVisionError, match="80 classes.*17"):
        verify_metadata(metadata, network, artifact=tmp_path / "model.onnx")


def test_image_size_mismatch_fails(tmp_path: Path) -> None:
    """A 640x640 artifact under a 512x512 network is a hard error."""
    network = _pose_network()
    metadata = build_metadata(network)
    metadata["autovision_image_size"] = json.dumps([512, 512])

    with pytest.raises(AutoVisionError, match="512x512.*640x640"):
        verify_metadata(metadata, network, artifact=tmp_path / "model.onnx")


def test_absent_metadata_and_foreign_networks_pass(tmp_path: Path) -> None:
    """Legacy artifacts and out-of-package stubs never fail the gate."""
    network = _pose_network()

    verify_metadata({}, network, artifact=tmp_path / "model.onnx")

    # The task comparison only fires when the network's own task is known;
    # a class defined in this test module lives outside autovision.tasks.
    class _ForeignNetwork(TaskNetwork):
        pass

    verify_metadata(
        {"autovision_task": "detection"},
        _ForeignNetwork(init_args={}),
        artifact=tmp_path / "x.onnx",
    )


def test_newer_schema_version_degrades_to_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A future writer never breaks today's loader; it warns and continues."""
    network = _pose_network()
    metadata = build_metadata(network)
    metadata["autovision_meta_version"] = str(META_SCHEMA_VERSION + 1)

    with caplog.at_level(logging.WARNING):
        verify_metadata(metadata, network, artifact=tmp_path / "model.onnx")

    assert "v" + str(META_SCHEMA_VERSION + 1) in caplog.text


def test_write_artifact_metadata_rejects_unknown_backend(tmp_path: Path) -> None:
    """Sidecar writers exist only for carriers without in-graph metadata."""
    with pytest.raises(AutoVisionError, match="no sidecar carrier"):
        write_artifact_metadata(
            _pose_network(), tmp_path / "model.onnx", backend="onnx"
        )


# ── InferenceEngine construction gate ───────────────────────────────────


def test_engine_accepts_an_artifact_from_the_same_network(
    tmp_path: Path,
) -> None:
    """Matching identity constructs the ONNX backend normally."""
    network = _pose_network()
    with _artifact_with_metadata(tmp_path, network) as path:
        engine = InferenceEngine(network, backend="onnx", artifact=path)

    assert engine.backend == "onnx"


def test_engine_rejects_a_foreign_task_artifact(tmp_path: Path) -> None:
    """A detection-labeled artifact fails construction under a pose network."""
    network = _pose_network()
    metadata = build_metadata(network)
    metadata["autovision_task"] = "detection"
    path = tmp_path / "model.onnx"
    _write_identity_onnx(path, metadata)

    with pytest.raises(AutoVisionError, match="'detection' model"):
        InferenceEngine(network, backend="onnx", artifact=path)


def test_engine_skips_the_gate_for_metadata_less_artifacts(
    tmp_path: Path,
) -> None:
    """Pre-D1 artifacts (no metadata_props) construct exactly as before."""
    network = _pose_network()
    path = tmp_path / "model.onnx"
    _write_identity_onnx(path, {})

    engine = InferenceEngine(network, backend="onnx", artifact=path)

    assert engine.network is network


def test_verify_engine_artifact_is_the_engine_entry_point(tmp_path: Path) -> None:
    """The combined read+verify helper mirrors the engine's behavior."""
    network = _pose_network()
    with _artifact_with_metadata(tmp_path, network) as path:
        verify_engine_artifact(network, path, backend="onnx")
