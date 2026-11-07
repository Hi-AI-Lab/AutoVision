# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Every task example must stay importable and print its data contract."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "tasks"

#: GTR's eager forward needs flash-linear-attention + CUDA; construction
#: and ONNX execution do not.
_GTR_READY = (
    importlib.util.find_spec("fla") is not None and torch.cuda.is_available()
)

TASKS = [
    pytest.param("detection_train", id="detection-train"),
    pytest.param("oriented_detection_train", id="oriented-detection-train"),
    pytest.param("instance_segmentation_train", id="instance-segmentation-train"),
    pytest.param("panoptic_segmentation_train", id="panoptic-segmentation-train"),
    pytest.param("semantic_segmentation_train", id="semantic-segmentation-train"),
    pytest.param("pose_estimation_train", id="pose-estimation-train"),
    pytest.param("depth_estimation_train", id="depth-estimation-train"),
    pytest.param("classification_train", id="classification-train"),
]

INFERENCE = [
    pytest.param("detection", id="detection"),
    pytest.param("oriented_detection", id="oriented-detection"),
    pytest.param("instance_segmentation", id="instance-segmentation"),
    pytest.param("panoptic_segmentation", id="panoptic-segmentation"),
    pytest.param("semantic_segmentation", id="semantic-segmentation"),
    pytest.param("pose_estimation", id="pose-estimation"),
    pytest.param("depth_estimation", id="depth-estimation"),
    pytest.param("classification", id="classification"),
    pytest.param("inference_demo", id="inference-demo-alias"),
]

YOLO_EXAMPLES = [
    pytest.param("detection", id="yolo-detection"),
    pytest.param("segmentation", id="yolo-segmentation"),
    pytest.param("pose", id="yolo-pose"),
    pytest.param("obb", id="yolo-obb"),
    pytest.param("classification", id="yolo-classification"),
]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / f"{script}.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _run_yolo(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / "yolo" / f"{script}.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("task", INFERENCE)
def test_inference_help(task: str) -> None:
    """Every inference script exposes image/weights/threshold flags."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert "--weights" in result.stdout
    assert "--threshold" in result.stdout


@pytest.mark.parametrize("script", YOLO_EXAMPLES)
def test_yolo_example_help(script: str) -> None:
    """YOLO examples parse --help without importing the AGPL package."""
    result = _run_yolo(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--image" in result.stdout
    assert "--threshold" in result.stdout


def test_detection_inference_is_not_training() -> None:
    """detection.py is inference-only; training lives in detection_train.py."""
    result = _run("detection", "--help")
    assert result.returncode == 0, result.stderr
    assert "--steps" not in result.stdout


def test_detection_offers_supervision_effects() -> None:
    """detection.py exposes the rendering effects migrated from supervision."""
    result = _run("detection", "--help")
    assert result.returncode == 0, result.stderr
    assert "--effect" in result.stdout
    for effect in ("corner", "round", "blur", "pixelate"):
        assert effect in result.stdout


def test_detection_exposes_class_names_flag() -> None:
    """detection.py accepts --class-names like the sibling inference examples."""
    result = _run("detection", "--help")
    assert result.returncode == 0, result.stderr
    assert "--class-names" in result.stdout


def test_detection_rejects_cpu_for_gtr_eager() -> None:
    """--device cpu on the GTR default exits with guidance, not a Triton crash."""
    result = _run("detection", "--device", "cpu")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    if importlib.util.find_spec("fla") is None:
        # The gtr-extra check fires first on machines without fla; the
        # CUDA/onnx guidance only appears once the dependency is present.
        assert "gtr` extra" in result.stderr
    else:
        assert "device='cuda'" in result.stderr
        assert "--onnx" in result.stderr


def test_detection_train_out_dir_collision_is_clean(tmp_path: Path) -> None:
    """A second run without --overwrite exits with one actionable line."""
    out = tmp_path / "run"
    out.mkdir()
    (out / ".keep").write_text("")
    result = _run("detection_train", "--out", str(out))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "--overwrite" in result.stderr


def test_num_classes_on_unsupported_task_is_clean(tmp_path: Path) -> None:
    """--num-classes on a data-driven task exits instead of silently no-op'ing."""
    result = _run("detection_train", "--num-classes", "5", "--dry-run")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "does not apply" in result.stderr


@pytest.mark.parametrize("task", TASKS)
def test_train_help_offers_onnx_export(task: str) -> None:
    """Every training example exposes the parity-gated --export-onnx flag."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--export-onnx" in result.stdout


@pytest.mark.parametrize("task", INFERENCE)
def test_inference_help_offers_onnx_backend(task: str) -> None:
    """Every inference example exposes the --onnx forward-swap flag."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--onnx" in result.stdout


def test_classification_onnx_round_trip(tmp_path: Path) -> None:
    """Train once, export ONNX, serve it: the CPU-friendly full walk."""
    out = tmp_path / "run"
    artifact = tmp_path / "classification.onnx"
    trained = _run(
        "classification_train", "--steps", "1", "--out", str(out),
        "--export-onnx", str(artifact),
    )
    assert trained.returncode == 0, trained.stderr
    assert artifact.is_file()
    served = _run(
        "classification",
        "--weights", str(out / "exported_models" / "exported_last.pt"),
        "--onnx", str(artifact),
        "--out", str(tmp_path / "infer"),
    )
    assert served.returncode == 0, served.stderr
    assert "top1" in served.stdout


@pytest.mark.skipif(
    not _GTR_READY, reason="GTR export needs flash-linear-attention + CUDA"
)
def test_detection_onnx_round_trip(tmp_path: Path) -> None:
    """The GTR flow: train, parity-gated ONNX export, CPU ONNX inference."""
    out = tmp_path / "run"
    artifact = tmp_path / "detection.onnx"
    trained = _run(
        "detection_train", "--steps", "1", "--out", str(out),
        "--export-onnx", str(artifact),
    )
    assert trained.returncode == 0, trained.stderr
    assert artifact.is_file()
    served = _run(
        "detection",
        "--weights", str(out / "exported_models" / "exported_last.pt"),
        "--onnx", str(artifact),
        "--threshold", "0.0",
        "--out", str(tmp_path / "infer"),
    )
    assert served.returncode == 0, served.stderr
    assert "onnx (" in served.stdout


class TestSummarizePrediction:
    """_common.summarize_prediction picks the wording from the record keys."""

    @staticmethod
    def _shaped(height: int, width: int) -> Any:
        return types.SimpleNamespace(shape=(height, width))

    def _common(self) -> Any:
        return _load_examples_common()

    def test_detection_record_counts_detections(self) -> None:
        lines = self._common().summarize_prediction(
            {"bboxes": [[0, 0, 1, 1], [2, 2, 3, 3]],
             "scores": [0.9, 0.8], "labels": [0, 1]},
            {0: "cat", 1: "dog"},
        )
        assert lines[0] == "detections  = 2"
        assert "cat" in lines[1] and "dog" in lines[1]

    def test_none_labels_degrade_to_the_fallback_summary(self) -> None:
        """A record carrying instance keys with None labels/scores (a task
        surface omitting the field) must not TypeError in ``list(None)`` —
        it degrades to the outputs summary."""
        lines = self._common().summarize_prediction(
            {"bboxes": [[0, 0, 1, 1]], "labels": None, "scores": None},
            None,
        )
        assert lines[0].startswith("outputs")
        assert "bboxes" in lines[0]

    def test_obb_polygons_count_as_detections(self) -> None:
        lines = self._common().summarize_prediction(
            {"polygons": [[0, 0, 1, 0, 1, 1, 0, 1]],
             "scores": [0.9], "labels": [3]},
            None,
        )
        assert lines[0] == "detections  = 1"
        assert "'3': 1" in lines[1]

    def test_pose_keypoints_count_as_detections(self) -> None:
        lines = self._common().summarize_prediction(
            {"keypoints": [[1, 2, 3]], "scores": [0.9], "labels": [0]},
            None,
        )
        assert lines[0] == "detections  = 1"

    def test_classification_names_the_top1(self) -> None:
        lines = self._common().summarize_prediction(
            {"labels": [1], "scores": [0.87]}, {0: "cat", 1: "dog"}
        )
        assert lines == ["top1        = dog (0.87)"]

    def test_panoptic_counts_segments(self) -> None:
        common = self._common()
        lines = common.summarize_prediction(
            {"masks": [], "scores": [0.9, 0.8, 0.7], "segment_ids": [1, 2, 3]},
            None,
        )
        assert lines[0] == "segments    = 3"
        empty = common.summarize_prediction(
            {"masks": [], "scores": [], "segment_ids": []}, None
        )
        assert any("no confident segments" in line for line in empty)

    def test_semantic_mask_reports_the_map_shape(self) -> None:
        lines = self._common().summarize_prediction(
            {"mask": self._shaped(64, 64), "input_h": 64, "input_w": 64}, None
        )
        assert lines == ["mask        = 64x64 class map"]

    def test_depth_reports_the_map_shape(self) -> None:
        lines = self._common().summarize_prediction(
            {"depth": self._shaped(128, 128)}, None
        )
        assert lines == ["depth       = 128x128 metres"]

    def test_empty_detection_uses_the_caller_note(self) -> None:
        lines = self._common().summarize_prediction(
            {"bboxes": [], "scores": [], "labels": []},
            None,
            zero_note="custom hint",
        )
        assert any("custom hint" in line for line in lines)

    def test_finish_width_alignment(self) -> None:
        lines = self._common().summarize_prediction(
            {"bboxes": [], "scores": [], "labels": []},
            None,
            label_width=16,
        )
        assert lines[0].startswith("detections      = ")


class TestRejectNumClasses:
    """_common.reject_num_classes guards the data-driven train examples."""

    def test_none_passes(self) -> None:
        common = _load_examples_common()
        args = argparse.Namespace(num_classes=None)
        common.reject_num_classes(args, "object detection")

    def test_value_exits_with_guidance(self) -> None:
        common = _load_examples_common()
        args = argparse.Namespace(num_classes=5)
        with pytest.raises(SystemExit, match="does not apply"):
            common.reject_num_classes(args, "object detection")


def _load_examples_common() -> Any:
    """Import ``examples/tasks/_common.py`` without a package install."""
    spec = importlib.util.spec_from_file_location(
        "_examples_common", EXAMPLES / "_common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGtrEagerDeviceGuard:
    """_common.gtr_eager_device_guard sniffs the family from the module tree."""

    @staticmethod
    def _network_with_module(module_name: str) -> nn.Module:
        class _Net(nn.Module):
            pass

        _Net.__module__ = module_name
        return _Net()

    def test_gtr_task_model_on_cpu_is_rejected(self) -> None:
        """A GTR task-model module path on --device cpu exits with guidance."""
        common = _load_examples_common()
        network = self._network_with_module(
            "autovision.tasks.pose_estimation.gtr.task_model"
        )
        with pytest.raises(SystemExit, match="GTR-family"):
            common.gtr_eager_device_guard(network, device="cpu")

    def test_detection_route_hides_gtr_in_a_submodule(self) -> None:
        """The generic DETR class wraps the GTR backbone as a submodule."""
        common = _load_examples_common()

        class _Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = TestGtrEagerDeviceGuard._network_with_module(
                    "autovision.tasks.detection.shared.gtr.encoder"
                )

        with pytest.raises(SystemExit):
            common.gtr_eager_device_guard(_Holder(), device="cpu")

    def test_non_gtr_network_on_cpu_passes(self) -> None:
        """A non-GTR export on CPU is not blocked."""
        common = _load_examples_common()
        common.gtr_eager_device_guard(nn.Linear(2, 2), device="cpu")

    def test_cuda_device_skips_inspection(self) -> None:
        """device='cuda' never inspects the module tree nor exits."""
        common = _load_examples_common()
        network = self._network_with_module(
            "autovision.tasks.depth_estimation.gtr.task_model"
        )
        common.gtr_eager_device_guard(network, device="cuda")


@pytest.mark.skipif(
    not _GTR_READY, reason="GTR export needs flash-linear-attention + CUDA"
)
def test_detection_rejects_cpu_for_gtr_weights(tmp_path: Path) -> None:
    """A real GTR export on --device cpu exits cleanly, not in Triton."""
    from autovision import ensure_coco8, train_object_detection

    result = train_object_detection(
        out=tmp_path / "run",
        data=ensure_coco8(tmp_path / "coco8"),
        model="detr/gtrvit-s",
        steps=1,
        batch_size=2,
        num_workers=0,
        model_args={
            "image_size": (64, 64),
            "num_queries": 10,
            "num_top_queries": 10,
            "num_denoising": 0,
            "lr_warmup_steps": 2,
        },
        transform_args={
            "scale_jitter": None,
            "photometric_distort_prob": 0.5,
            "random_zoom_out_prob": None,
            "random_iou_crop_prob": None,
        },
        trainer_args={"val_every_num_steps": 1},
    )
    assert result.exported_model_path is not None
    proc = _run(
        "detection",
        "--weights",
        str(result.exported_model_path),
        "--device",
        "cpu",
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "GTR-family" in proc.stderr
    assert "--onnx" in proc.stderr


def test_bundled_demo_asset_is_readable() -> None:
    """The bundled CC0 demo image exists with the expected dimensions."""
    from PIL import Image

    expected = {
        "airport_aerial.jpg": (1280, 606),
    }
    for filename, size in expected.items():
        path = EXAMPLES / "assets" / filename
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == size


@pytest.mark.parametrize("task", TASKS)
def test_example_help(task: str) -> None:
    """--help parses without importing a task-specific optional stack."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("task", TASKS)
def test_example_dry_run_writes_data_contract(task: str, tmp_path: Path) -> None:
    """--dry-run generates demo data and prints the config without training."""
    result = _run(task, "--dry-run", "--out", str(tmp_path / "run"))
    assert result.returncode == 0, result.stderr
    assert "demo data config" in result.stdout
    if task != "detection_train":  # coco8 comes from the shared cache, not --out
        demo_root = tmp_path / "_demo_data"
        assert demo_root.is_dir() and any(demo_root.iterdir())


def _load_example_module(name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import an examples/tasks script in-process for stubbed unit tests."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        f"_example_{name}", EXAMPLES / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDryRunNotes:
    """dry_run names ignored flags with matching singular/plural wording."""

    def test_single_flag_uses_singular(self, capsys) -> None:
        common = _load_examples_common()
        args = argparse.Namespace(
            predict_image=None, export=None, export_onnx=Path("x.onnx")
        )
        common.dry_run("demo", {}, args)
        out = capsys.readouterr().out
        assert "--export-onnx ignored" in out
        assert "it needs a trained model" in out
        assert "they need" not in out

    def test_multiple_flags_use_plural(self, capsys) -> None:
        common = _load_examples_common()
        args = argparse.Namespace(
            predict_image=Path("x.png"),
            export=Path("x.pt"),
            export_onnx=Path("x.onnx"),
        )
        common.dry_run("demo", {}, args)
        out = capsys.readouterr().out
        assert "--predict-image, --export and --export-onnx" in out
        assert "they need a trained model" in out


def _stub_export_demo(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> None:
    """Replace the exporters with file-writing stubs (no model, no CUDA)."""
    def fake_onnx(pt: Path, out: Path) -> Path:
        out = Path(out)
        if "onnx" in fail:
            raise SystemExit("parity refused")
        if "onnx" not in missing:
            out.write_bytes(b"onnx")
        return out

    def fake_export_model(pt: Path, out: Path, *, device: str = "cpu", **kwargs: Any) -> Path:
        out = Path(out)
        label = "tensorrt" if out.suffix == ".engine" else "openvino"
        if label in fail:
            raise RuntimeError(f"{label} parity refused")
        if label not in missing:
            out.write_bytes(b"model")
        return out

    monkeypatch.setattr(module, "export_task_to_onnx", fake_onnx)
    monkeypatch.setattr(module, "export_model", fake_export_model)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)


class TestExportDemo:
    """export_demo must fail closed on skipped or missing artifacts."""

    @staticmethod
    def _args(tmp_path: Path, *extra: str) -> list[str]:
        weights = tmp_path / "weights.pt"
        weights.write_bytes(b"weights")
        return ["--weights", str(weights), "--out", str(tmp_path / "out"), *extra]

    def test_all_targets_succeed(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Three healthy targets report 3 exported and exit 0."""
        module = _load_example_module("export_demo", monkeypatch)
        _stub_export_demo(module, monkeypatch)
        assert module.main(self._args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "3 exported, 0 skipped" in out
        assert (tmp_path / "out" / "model.onnx").is_file()

    @pytest.mark.parametrize(
        "fail",
        [pytest.param(("openvino",), id="openvino-gate"),
         pytest.param(("tensorrt",), id="tensorrt-gate")],
    )
    def test_refused_target_exits_nonzero(
        self, tmp_path: Path, monkeypatch, capsys, fail: tuple[str, ...]
    ) -> None:
        """A parity-gate refusal is a failed run, not a silent SKIPPED line."""
        module = _load_example_module("export_demo", monkeypatch)
        _stub_export_demo(module, monkeypatch, fail=fail)
        assert module.main(self._args(tmp_path)) == 1
        err = capsys.readouterr().err
        assert fail[0] in err
        assert "--allow-skips" in err

    def test_allow_skips_accepts_known_unhealthy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """--allow-skips is an explicit, visible opt-in."""
        module = _load_example_module("export_demo", monkeypatch)
        _stub_export_demo(module, monkeypatch, fail=("openvino",))
        assert module.main(self._args(tmp_path, "--allow-skips")) == 0

    def test_reported_success_without_artifact_is_a_skip(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """An exporter claiming success without writing a file is detected."""
        module = _load_example_module("export_demo", monkeypatch)
        _stub_export_demo(module, monkeypatch, missing=("onnx",))
        assert module.main(self._args(tmp_path)) == 1
        assert "artifact missing" in capsys.readouterr().out


def test_export_demo_help_offers_allow_skips() -> None:
    """--allow-skips is the documented escape hatch for known-bad backends."""
    result = _run("export_demo", "--help")
    assert result.returncode == 0, result.stderr
    assert "--allow-skips" in result.stdout


class _OOMOnceModel:
    """Fake handle: the first CUDA predict OOMs; the CPU retry succeeds."""

    def __init__(self) -> None:
        self.devices: list[str] = []
        self.calls = 0

    def to(self, device: str):
        self.devices.append(device)
        return self

    def predict(self, image, threshold):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return {"labels": []}


class _BrokenModel(_OOMOnceModel):
    """Fails with an unrelated kernel error on every device."""

    def predict(self, image, threshold):
        raise RuntimeError("bad kernel")


class _ToOOMModel:
    """Fails while moving to CUDA, but predicts fine once on CPU."""

    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str):
        self.devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return self

    def predict(self, image, threshold):
        return {"labels": []}


class TestDetectionBackendsFallback:
    """The CUDA OOM retry never hides an ordinary inference failure."""

    def test_cuda_oom_retries_on_cpu(self, monkeypatch, capsys) -> None:
        """A CUDA OOM moves the model to CPU and retries once."""
        module = _load_example_module("detection_backends", monkeypatch)
        model = _OOMOnceModel()
        result, device = module._predict_with_fallback(
            model, Path("x.jpg"), 0.3, "cuda"
        )
        assert result == {"labels": []}
        assert device == "cpu"
        assert model.devices == ["cuda", "cpu"]
        assert "retried on CPU" in capsys.readouterr().out

    def test_cuda_to_oom_retries_on_cpu(self, monkeypatch) -> None:
        """An OOM while moving the model to CUDA also triggers the retry."""
        module = _load_example_module("detection_backends", monkeypatch)
        model = _ToOOMModel()
        result, device = module._predict_with_fallback(
            model, Path("x.jpg"), 0.3, "cuda"
        )
        assert result == {"labels": []}
        assert device == "cpu"
        assert model.devices == ["cuda", "cpu"]

    def test_non_oom_cuda_error_propagates(self, monkeypatch) -> None:
        """Only out-of-memory errors trigger the CPU retry."""
        module = _load_example_module("detection_backends", monkeypatch)
        with pytest.raises(RuntimeError, match="bad kernel"):
            module._predict_with_fallback(
                _BrokenModel(), Path("x.jpg"), 0.3, "cuda"
            )

    def test_cpu_error_propagates(self, monkeypatch) -> None:
        """A CPU failure has no fallback device and must surface."""
        module = _load_example_module("detection_backends", monkeypatch)
        with pytest.raises(RuntimeError, match="out of memory"):
            module._predict_with_fallback(
                _OOMOnceModel(), Path("x.jpg"), 0.3, "cpu"
            )
