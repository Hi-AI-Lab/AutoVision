# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ONNX export must not destroy a previous good artifact on failure."""

import errno

import pytest

from autovision.export.export_onnx import ONNXExportMixin


class _DummyExport(ONNXExportMixin):
    pass


def test_failed_export_preserves_the_previous_artifact(tmp_path, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-good-graph")
    dummy = _DummyExport()

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(dummy, "_export_onnx_to_path", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        dummy.export_onnx(target, write_spec=False)
    assert target.read_bytes() == b"previous-good-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())


def test_enospc_export_preserves_the_previous_artifact(tmp_path, monkeypatch):
    """Disk full during export keeps the old graph and sweeps the temp."""
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-good-graph")
    dummy = _DummyExport()

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(dummy, "_export_onnx_to_path", fail)
    with pytest.raises(OSError, match="No space left"):
        dummy.export_onnx(target, write_spec=False)
    assert target.read_bytes() == b"previous-good-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())


def test_successful_export_replaces_the_target_atomically(tmp_path, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    dummy = _DummyExport()

    def succeed(out, **kwargs):
        with open(out, "wb") as handle:
            handle.write(b"new-graph")
        return out

    monkeypatch.setattr(dummy, "_export_onnx_to_path", succeed)
    result = dummy.export_onnx(target, write_spec=False)
    assert result == target
    assert target.read_bytes() == b"new-graph"
    assert not any(p.name.endswith(".tmp.onnx") for p in tmp_path.iterdir())


def _inlined_identity_model():
    """A one-node ONNX graph with a fully inlined initializer."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.ones((2, 2), np.float32), "w")
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["w"], ["y"])],
            "inlined",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))],
            initializer=[weight],
        )
    )


def test_sweep_removes_orphan_sidecar_for_inlined_graph(tmp_path):
    """After onnxslim re-inlines the dynamo external data, the ``.data``
    sidecar the success path moves alongside the artifact is dead weight:
    the sweep must delete it, and the graph must stay loadable."""
    import onnx

    from autovision.export.export_onnx import _sweep_orphan_external_data

    path = tmp_path / "model.onnx"
    onnx.save(_inlined_identity_model(), str(path))
    path.with_suffix(path.suffix + ".data").write_bytes(b"stale dynamo blob")

    _sweep_orphan_external_data(path)

    assert not path.with_suffix(path.suffix + ".data").exists()
    assert onnx.load(str(path)).graph.initializer[0].name == "w"


def test_sweep_keeps_sidecar_for_genuine_external_data(tmp_path):
    """A graph that still references external initializers (e.g. beyond the
    ~2GB protobuf limit) keeps its sidecar — the sweep only drops orphans."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from autovision.export.export_onnx import _sweep_orphan_external_data

    raw = np.ones((2, 2), np.float32)
    (tmp_path / "weights.bin").write_bytes(raw.tobytes())
    tensor = numpy_helper.from_array(raw, "w")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.ClearField("raw_data")
    for key, value in (
        ("location", "weights.bin"), ("offset", "0"), ("length", str(raw.nbytes)),
    ):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["w"], ["y"])],
            "external",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))],
            initializer=[tensor],
        )
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    path.with_suffix(path.suffix + ".data").write_bytes(b"really referenced")

    _sweep_orphan_external_data(path)

    assert path.with_suffix(path.suffix + ".data").read_bytes() == b"really referenced"


# ── H1-6: the TensorRT engine blob publish ──────────────────────────────────


def test_engine_blob_publish_replaces_target_atomically(tmp_path):
    """H1-6: the TensorRT helper stages the engine blob under ``.tmp-<pid>``
    and publishes it with ``os.replace`` — the target never holds a partial
    engine and no temp file survives the publish."""
    from autovision.export.tensorrt_helpers import _publish_engine

    target = tmp_path / "model.engine"
    target.write_bytes(b"old-engine")

    _publish_engine(target, b"new-engine-blob")

    assert target.read_bytes() == b"new-engine-blob"
    assert not any(".tmp-" in p.name for p in tmp_path.iterdir())


def test_engine_blob_publish_failure_preserves_previous_engine(
    tmp_path, monkeypatch
):
    """H1-6: a publish failure (e.g. disk full at the rename) keeps the
    previous engine bytes and leaves no ``.tmp-`` residue behind."""
    import autovision.export.tensorrt_helpers as helpers

    target = tmp_path / "model.engine"
    target.write_bytes(b"previous-good-engine")

    def enospc_replace(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(helpers.os, "replace", enospc_replace)
    with pytest.raises(OSError, match="No space left"):
        helpers._publish_engine(target, b"new-engine-blob")
    assert target.read_bytes() == b"previous-good-engine"
    assert not any(".tmp-" in p.name for p in tmp_path.iterdir())
