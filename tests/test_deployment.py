# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the deployment layer: ONNX export, SAHI, benchmark, load_model."""

import json
from pathlib import Path

import pytest
import torch

from autovision.export import benchmark_inference
from autovision.tasks import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.shared.sahi import (
    combine_object_detection_tiles,
    tile_image,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_helpers import load_task_model

onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed (pip install -e '.[export]')")
pytest.importorskip("onnxslim", reason="onnxslim not installed (pip install -e '.[export]')")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )


class TestOnnxExport:
    def test_fp32_export_with_parity_gate(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model.onnx",
            precision="fp32",
            dynamic_batch_size=True,
            format_args={"opset_version": 18},
        )
        assert out.exists()
        assert out.stat().st_size > 100_000

    def test_fp16_export(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model_fp16.onnx",
            precision="fp16",
            format_args={"opset_version": 18},
        )
        assert out.exists()
        # FP16 graph actually contains half tensors.
        import onnx

        model = onnx.load(str(out))
        dtypes = {t.data_type for i in model.graph.initializer for t in [i]}
        assert onnx.TensorProto.FLOAT16 in dtypes

    def test_dynamic_batch(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model_dyn.onnx",
            format_args={"opset_version": 18},
        )
        import onnxruntime as ort

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        for batch in (1, 3):
            outputs = session.run(
                None, {"images": torch.randn(batch, 3, 64, 64).numpy()}
            )
            assert outputs[0].shape[0] == batch
            assert outputs[1].shape[0] == batch

    def test_bad_precision_raises(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="precision"):
            tiny_model.export_onnx(tmp_path / "x.onnx", precision="int8")

    def test_fp16_requires_simplify(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="simplify"):
            tiny_model.export_onnx(
                tmp_path / "x.onnx", precision="fp16", simplify=False
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_resident_model_exports(self, tiny_model, tmp_path):
        """The example input follows the model's device (trained-on-GPU flow)."""
        import copy

        model = copy.deepcopy(tiny_model).cuda()
        out = model.export_onnx(
            tmp_path / "cuda_model.onnx",
            precision="fp32",
            format_args={"opset_version": 18},
        )
        assert out.exists() and out.stat().st_size > 100_000


class TestSAHI:
    def test_tile_image_overlap(self):
        image = torch.randn(3, 200, 300)
        tiles, coordinates = tile_image(image, overlap=0.2, tile_size=128)
        assert tiles.shape == (len(coordinates), 3, 128, 128)
        assert coordinates.min() >= 0
        # Overlap guarantees more than the no-overlap count.
        tiles_no_overlap, coords_no_overlap = tile_image(image, 0.0, 128)
        assert tiles.shape[0] >= tiles_no_overlap.shape[0]

    def test_tile_image_small_image(self):
        # Small images are upscaled (default) or padded to the tile size.
        tiles, coordinates = tile_image(torch.randn(3, 32, 32), 0.2, 64)
        assert tiles.shape == (1, 3, 64, 64)
        assert coordinates.tolist() == [[0, 0]]
        tiles_pad, _ = tile_image(torch.randn(3, 32, 32), 0.2, 64, padding_mode="pad")
        assert tiles_pad.shape == (1, 3, 64, 64)

    def test_tile_image_padding_mode(self):
        tiles, _ = tile_image(torch.randn(3, 100, 100), 0.0, 128, padding_mode="pad")
        assert tiles.shape[-2:] == (128, 128)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            tile_image(torch.randn(3, 100, 100), 1.0, 64)

    def test_combine_suppresses_duplicates(self):
        global_output = {
            "bboxes": torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        tile_outputs = [
            {  # same object seen inside a tile (near-duplicate)
                "bboxes": torch.tensor([[20.0, 20.0, 70.0, 70.0],
                                        [200.0, 200.0, 240.0, 240.0]]),
                "scores": torch.tensor([0.8, 0.7]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        coordinates = torch.tensor([[80, 80]])
        merged = combine_object_detection_tiles(
            global_output, tile_outputs, coordinates, original_size=(300, 300)
        )
        # The duplicate tile box (IoU > 0.1 with the global box) is dropped;
        # the second (different object) survives.
        assert merged["bboxes"].shape[0] == 2

    def test_combine_no_tiles(self):
        global_output = {
            "bboxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.5]),
            "labels": torch.tensor([0]),
        }
        merged = combine_object_detection_tiles(
            global_output, [], torch.zeros(0, 2), (100, 100)
        )
        assert merged["bboxes"].shape[0] == 1


class _SahiStubModel:
    """Stub wiring a TaskNetwork's three-stage contract to canned outputs,
    so the SAHI coordinate pipeline is verified without a real detector."""

    image_size = (64, 64)
    training = False

    def __init__(self, outputs):
        self.outputs = outputs
        self.seen_metadata = None
        self.seen_threshold = None

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def preprocess_image(self, image):
        from torchvision.transforms.v2 import functional as TF

        tensor = image.float() if isinstance(image, torch.Tensor) else image
        _, h, w = tensor.shape
        # Real models resize to the fixed input size here.
        resized = TF.resize(tensor, list(self.image_size))
        return resized, {"orig_w": w, "orig_h": h}

    def preprocess_batch(self, batch):
        return batch

    def forward_backend(self, batch):
        return None

    def postprocess(self, raw, metadata, threshold=0.0):
        self.seen_metadata = metadata
        self.seen_threshold = threshold
        return self.outputs

    def _module_device(self):
        return torch.device("cpu")

    # TaskNetwork unbound methods bound to the stub instance.
    predict_sahi = TaskNetwork.predict_sahi
    _load_image_unchanged = TaskNetwork._load_image_unchanged


def _det_output(boxes, scores, labels):
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32),
        "scores": torch.as_tensor(scores, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


class TestPredictSahi:
    def test_end_to_end_contract_and_clipping(self, tiny_model):
        torch.manual_seed(0)
        tiny_model.eval()
        out = tiny_model.predict_sahi(
            torch.rand(3, 220, 300), tile_size=128, overlap=0.2, threshold=0.0
        )
        assert set(out) == {"bboxes", "scores", "labels"}
        assert out["bboxes"].shape[1] == 4
        assert out["bboxes"].shape[0] == out["scores"].shape[0]
        # Boxes are clipped to the original canvas.
        assert (out["bboxes"][:, 0::2] >= 0).all()
        assert (out["bboxes"][:, 0::2] <= 300 + 1e-4).all()
        assert (out["bboxes"][:, 1::2] <= 220 + 1e-4).all()

    def test_default_tile_size_is_model_input_size(self, tiny_model, monkeypatch):
        seen = {}
        from autovision.tasks.shared import tiling_utils

        real_tile = tiling_utils.tile_image

        def spy_tile(image, overlap, tile_size, padding_mode="resize"):
            seen["tile_size"] = tile_size
            return real_tile(image, overlap, tile_size, padding_mode)

        monkeypatch.setattr(tiling_utils, "tile_image", spy_tile)
        tiny_model.predict_sahi(torch.rand(3, 70, 70), threshold=0.0)
        assert seen["tile_size"] == min(tiny_model.image_size)

    def test_tile_metas_effective_canvas(self):
        # resize mode: (3, 32, 90) with tile 64 slices into two tiles
        # (x=0 and x=26); edge tiles are interpolated, so the effective
        # canvas is the *pre-interpolation* edge, not the full tile.
        outputs = [_det_output([[1.0, 1.0, 5.0, 5.0]], [0.9], [0])] * 3
        stub = _SahiStubModel(outputs)
        stub.predict_sahi(torch.rand(3, 32, 90), tile_size=64, threshold=0.0)
        assert stub.seen_metadata[0] == {"orig_w": 90, "orig_h": 32}  # global
        assert stub.seen_metadata[1] == {"orig_w": 64, "orig_h": 32}
        assert stub.seen_metadata[2] == {"orig_w": 64, "orig_h": 32}
        # pad mode: the tile frame is the full padded tile.
        stub_pad = _SahiStubModel(outputs)
        stub_pad.predict_sahi(
            torch.rand(3, 32, 90), tile_size=64, threshold=0.0, padding_mode="pad"
        )
        assert stub_pad.seen_metadata[1] == {"orig_w": 64, "orig_h": 64}

    def test_tile_offset_and_global_passthrough(self, monkeypatch):
        from autovision.tasks.shared import tiling_utils

        seen = {}
        real_combine = tiling_utils.combine_object_detection_tiles

        def spy_combine(global_output, tile_outputs, coordinates, size, **kwargs):
            seen["global"] = global_output
            seen["tiles"] = tile_outputs
            seen["coordinates"] = coordinates
            seen["size"] = size
            seen["nms_iou_threshold"] = kwargs["nms_iou_threshold"]
            return real_combine(
                global_output, tile_outputs, coordinates, size, **kwargs
            )

        monkeypatch.setattr(
            tiling_utils, "combine_object_detection_tiles", spy_combine
        )
        global_out = _det_output([[10.0, 10.0, 50.0, 50.0]], [0.9], [0])
        tile_out = _det_output(
            [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 30.0, 30.0]], [0.8, 0.7], [0, 1]
        )
        # (3, 128, 128) with tile 64 and no overlap -> 4 tiles; tile outputs
        # must line up one-to-one with the tile coordinates.
        stub = _SahiStubModel([global_out, tile_out, tile_out, tile_out, tile_out])
        merged = stub.predict_sahi(
            torch.rand(3, 128, 128), tile_size=64, overlap=0.0, threshold=0.0
        )
        assert seen["size"] == (128, 128)
        assert seen["global"] is global_out
        assert seen["nms_iou_threshold"] == 0.3
        assert seen["coordinates"].tolist() == [[0, 0], [64, 0], [0, 64], [64, 64]]
        assert len(seen["tiles"]) == 4
        # Tile-0 output (no offset needed at (0,0)): the different-label box
        # survives merging alongside the global box.
        assert merged["bboxes"].shape[0] >= 2
        assert merged["scores"][0] == 0.9  # global first (descending)

    def test_threshold_forwarded_to_postprocess(self):
        stub = _SahiStubModel(
            [_det_output([[1.0, 1.0, 5.0, 5.0]], [0.9], [0])]
        )
        stub.predict_sahi(torch.rand(3, 32, 32), tile_size=64, threshold=0.55)
        assert stub.seen_threshold == 0.55

    def test_non_detection_model_raises(self):
        stub = _SahiStubModel([{"logits": torch.zeros(2, 3)}])
        with pytest.raises(TypeError, match="bboxes"):
            stub.predict_sahi(torch.rand(3, 32, 32), tile_size=64)


class TestBenchmark:
    def test_benchmark_latency(self, tiny_model):
        inputs = {"images": torch.randn(1, 3, 64, 64)}
        result = benchmark_inference(
            tiny_model, inputs, model_name="tiny", warmup_steps=2, steps=5
        )
        assert result.num_steps == 5
        assert result.latency_ms_p50 <= result.latency_ms_p90
        assert result.latency_ms_mean > 0
        assert result.images_per_sec > 0
        assert "tiny" in result.to_markdown()


class TestLoadTaskModel:
    def _export(self, tmp_path):
        torch.manual_seed(0)
        model = RTDETRObjectDetection(
            num_classes=2, backbone="torchvision/resnet18",
            image_size=(64, 64), hidden_dim=32, num_queries=10,
            dec_layers=1, num_heads=4, num_top_queries=10,
        )
        train_model = RTDETRObjectDetectionTrain(
            RTDETRObjectDetectionTrainArgs(
                num_classes=2, image_size=(64, 64), hidden_dim=32,
                num_queries=10, dec_layers=1, num_heads=4,
                backbone="torchvision/resnet18", num_top_queries=10,
            )
        )
        train_model.model = model
        train_model.on_train_batch_end()
        out = tmp_path / "exported_last.pt"
        torch.save(
            {
                "schema_version": 1,
                "kind": "task_model",
                "model_class_path": model.class_path,
                "model_init_args": {**model.init_args, "model_name": "rtdetr/test"},
                "train_model": train_model.get_export_state_dict(),
            },
            out,
        )
        return out

    def test_load_and_predict(self, tmp_path):
        out = self._export(tmp_path)
        model = load_task_model(out, device="cpu")  # dec_layers matches export
        assert not model.training
        prediction = model.predict(
            torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8), threshold=0.0
        )
        assert set(prediction) == {"labels", "bboxes", "scores"}

    def test_missing_file_raises(self, tmp_path):
        from autovision.errors import CheckpointError

        with pytest.raises(CheckpointError, match="does not exist"):
            load_task_model(tmp_path / "nope.pt")

    def test_foreign_file_raises(self, tmp_path):
        p = tmp_path / "foreign.pt"
        torch.save({"something": 1}, p)
        from autovision.errors import CheckpointError

        with pytest.raises(CheckpointError, match="not an exported"):
            load_task_model(p)


# ── H1-6: TensorRT engine atomic publish (mocked build + parity gate) ───────


class TestTensorRTAtomicPublish:
    """H1-6: the TRT engine is built and parity-gated under a staged
    ``.tmp-<pid>`` name; ``os.replace`` publication and the sidecars happen
    only after the gate. Build and verify are mocked — no TensorRT runtime
    or GPU is involved."""

    def test_verify_failure_preserves_previous_engine(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Injected gate failure: the previous engine bytes survive and only
        the staged engine is dropped."""
        import autovision.export.tensorrt_helpers as trt_helpers

        target = tmp_path / "model.engine"
        target.write_bytes(b"previous-good-engine")
        seen = {}

        def fake_build(**kwargs):
            seen["build_out"] = Path(kwargs["out"])
            seen["build_out"].write_bytes(b"new-bad-engine")
            return seen["build_out"]

        monkeypatch.setattr(trt_helpers, "export_onnx_to_tensorrt", fake_build)

        class _FailingRunner:
            def __init__(self, engine_path, *, device="cuda"):
                seen["verify_path"] = Path(engine_path)
                raise RuntimeError("engine parity gate failed (injected)")

        monkeypatch.setattr(trt_helpers, "TensorRTEngineRunner", _FailingRunner)

        with pytest.raises(RuntimeError, match="injected"):
            tiny_model.export_tensorrt(target, verify=True)

        # The gate ran against the STAGED engine, not the published target.
        assert seen["verify_path"] != target
        assert ".tmp-" in seen["verify_path"].name
        # A gate failure must not overwrite-then-delete the old engine.
        assert target.read_bytes() == b"previous-good-engine"
        assert not any(".tmp-" in p.name for p in tmp_path.iterdir())

    def test_success_publishes_engine_and_sidecars_together(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Success path: the staged engine is published with the identity and
        decode-contract sidecars, with no staged residue left behind."""
        import autovision.export.tensorrt_helpers as trt_helpers

        target = tmp_path / "model.engine"
        target.write_bytes(b"previous-good-engine")
        seen = {}

        def fake_build(**kwargs):
            seen["build_out"] = Path(kwargs["out"])
            seen["build_out"].write_bytes(b"new-good-engine")
            return seen["build_out"]

        monkeypatch.setattr(trt_helpers, "export_onnx_to_tensorrt", fake_build)

        class _PassingRunner:
            def __init__(self, engine_path, *, device="cuda"):
                seen["verify_path"] = Path(engine_path)

            def run(self, images):
                return [images]

        monkeypatch.setattr(trt_helpers, "TensorRTEngineRunner", _PassingRunner)
        # The publish routing is what this test pins down; the real
        # decode-contract comparison is covered by the model's own tests.
        monkeypatch.setattr(tiny_model, "postprocess", None)
        monkeypatch.setattr(
            tiny_model, "verify_onnx_export_outputs", lambda *args, **kwargs: None
        )

        result = tiny_model.export_tensorrt(target, verify=True)

        assert result == target
        assert target.read_bytes() == b"new-good-engine"
        assert seen["verify_path"] == seen["build_out"] != target
        assert ".tmp-" in seen["build_out"].name
        assert (tmp_path / "model.meta.json").exists()
        assert (tmp_path / "model.engine.postprocess.json").exists()
        assert not any(".tmp-" in p.name for p in tmp_path.iterdir())


# ── H1-7: OpenVINO IR atomic publish (mocked conversion + parity gate) ──────


class TestOpenVINOAtomicPublish:
    """H1-7: the IR pair (.xml/.bin) is staged under temp names and
    published (bin first, then xml) only after the parity gate; a gate
    failure drops the staged pair and leaves the previous IR untouched.
    Conversion/save/verify are mocked around the real orchestration."""

    def test_verify_failure_preserves_previous_ir(
        self, tiny_model, tmp_path, monkeypatch
    ):
        openvino = pytest.importorskip("openvino")
        import autovision.export.openvino_helpers as ov_helpers

        target_xml = tmp_path / "model.xml"
        target_xml.write_bytes(b"previous-good-ir")
        (tmp_path / "model.bin").write_bytes(b"previous-good-weights")
        seen = {}

        def fake_export_onnx(path, **kwargs):
            Path(path).write_bytes(b"staged-onnx")
            return Path(path)

        def fake_save(ov_model, xml_path, compress_to_fp16=True):
            staged = Path(xml_path)
            staged.write_bytes(b"new-bad-ir")
            staged.with_suffix(".bin").write_bytes(b"new-bad-weights")
            seen["save_xml"] = staged

        def failing_verify(openvino_mod, model, xml_path, device, precision="fp32"):
            seen["verify_xml"] = Path(xml_path)
            raise RuntimeError("OpenVINO parity gate failed (injected)")

        monkeypatch.setattr(tiny_model, "export_onnx", fake_export_onnx)
        monkeypatch.setattr(openvino, "convert_model", lambda path: object())
        monkeypatch.setattr(openvino, "save_model", fake_save)
        monkeypatch.setattr(ov_helpers, "_verify_openvino_parity", failing_verify)

        with pytest.raises(RuntimeError, match="injected"):
            tiny_model.export_openvino(target_xml, verify=True)

        # The gate ran against the staged IR pair.
        assert seen["save_xml"] != target_xml
        assert ".tmp-" in seen["save_xml"].name
        assert seen["verify_xml"] == seen["save_xml"]
        # A gate failure must leave the previous IR pair byte-identical.
        assert target_xml.read_bytes() == b"previous-good-ir"
        assert (tmp_path / "model.bin").read_bytes() == b"previous-good-weights"
        assert not any(".tmp-" in p.name for p in tmp_path.iterdir())

    def test_success_publishes_ir_pair_and_sidecar(
        self, tiny_model, tmp_path, monkeypatch
    ):
        openvino = pytest.importorskip("openvino")
        import autovision.export.openvino_helpers as ov_helpers

        target_xml = tmp_path / "model.xml"
        target_xml.write_bytes(b"previous-good-ir")
        (tmp_path / "model.bin").write_bytes(b"previous-good-weights")
        seen = {}

        def fake_export_onnx(path, **kwargs):
            Path(path).write_bytes(b"staged-onnx")
            return Path(path)

        def fake_save(ov_model, xml_path, compress_to_fp16=True):
            staged = Path(xml_path)
            staged.write_bytes(b"new-good-ir")
            staged.with_suffix(".bin").write_bytes(b"new-good-weights")
            seen["save_xml"] = staged

        def passing_verify(openvino_mod, model, xml_path, device, precision="fp32"):
            seen["verify_xml"] = Path(xml_path)

        monkeypatch.setattr(tiny_model, "export_onnx", fake_export_onnx)
        monkeypatch.setattr(openvino, "convert_model", lambda path: object())
        monkeypatch.setattr(openvino, "save_model", fake_save)
        monkeypatch.setattr(ov_helpers, "_verify_openvino_parity", passing_verify)

        result = tiny_model.export_openvino(target_xml, verify=True)

        assert result == target_xml
        assert seen["verify_xml"] == seen["save_xml"] != target_xml
        assert ".tmp-" in seen["save_xml"].name
        assert target_xml.read_bytes() == b"new-good-ir"
        assert (tmp_path / "model.bin").read_bytes() == b"new-good-weights"
        metadata = json.loads((tmp_path / "model.json").read_text())
        assert metadata["backend"] == "openvino"
        assert not any(".tmp-" in p.name for p in tmp_path.iterdir())


# ── H1-8: parity-gate execution-provider routing ────────────────────────────


class TestParityProviderSelection:
    """H1-8: the fp16 gate's CUDA-EP promise is wired up — the parity
    session routes through CUDAExecutionProvider when the installed
    onnxruntime offers one, and the CPU-only path keeps its warning plus
    smoke gate. Providers are simulated via monkeypatch (no real sessions)."""

    def test_provider_selection_helper(self):
        from autovision.export.export_onnx import _parity_session_providers

        assert _parity_session_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert _parity_session_providers(
            ["AzureExecutionProvider", "CPUExecutionProvider"]
        ) == ["CPUExecutionProvider"]

    def _export_gate_target(self, tiny_model, tmp_path):
        """A real (fp32) graph for the gate: provider routing is driven by
        the precision ARGUMENT and the patched providers, not the graph's
        own dtype."""
        out = tiny_model.export_onnx(
            tmp_path / "gate_probe.onnx",
            verify=False,
            format_args={"opset_version": 18},
        )
        spec = tiny_model.model_input_spec
        example_name, example = next(iter(spec.example_inputs(batch_size=1).items()))
        return out, example_name, example

    def test_fp16_gate_uses_cuda_provider_when_available(
        self, tiny_model, tmp_path, monkeypatch, caplog
    ):
        import logging

        import onnxruntime as ort

        out, example_name, example = self._export_gate_target(tiny_model, tmp_path)
        captured = {}

        class _Probe(Exception):
            pass

        def probe_session(path, *, providers=None):
            captured["providers"] = providers
            raise _Probe("stop before the parity compare")

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(ort, "InferenceSession", probe_session)
        with caplog.at_level(logging.INFO, logger="autovision"):
            with pytest.raises(_Probe):
                tiny_model._verify_onnx(out, example_name, example, "fp16")

        assert captured["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert "CUDA execution provider" in caplog.text

    def test_fp16_gate_without_cuda_keeps_warning_and_smoke(
        self, tiny_model, tmp_path, monkeypatch, caplog
    ):
        import logging

        out, example_name, example = self._export_gate_target(tiny_model, tmp_path)
        smoke_calls = []
        monkeypatch.setattr(
            tiny_model,
            "_smoke_verify_fp16",
            lambda *args, **kwargs: smoke_calls.append(args),
        )
        with caplog.at_level(logging.WARNING, logger="autovision"):
            tiny_model._verify_onnx(out, example_name, example, "fp16")

        # The smoke gate ran instead of the parity session, and the warning
        # naming the missing CUDA EP is preserved.
        assert smoke_calls
        assert "no CUDA execution provider" in caplog.text


# ── H3-4e: dynamic-axis marker Dim identity ─────────────────────────────────


class TestDimMarkerIdentity:
    """H3-4e: same-NAME axis markers must resolve to the same ``Dim`` object
    within one export — a fresh ``Dim`` with a reused name reads as a second
    unrelated axis to torch.export."""

    def test_same_marker_name_shares_one_dim(self):
        from autovision.export.export_onnx import _dim_from_marker

        cache: dict = {}
        height = _dim_from_marker("same_size", 2, cache=cache)
        width = _dim_from_marker("same_size", 3, cache=cache)

        assert height is width

    def test_different_marker_names_get_distinct_dims(self):
        from autovision.export.export_onnx import _dim_from_marker

        cache: dict = {}
        first = _dim_from_marker("height", 2, cache=cache)
        second = _dim_from_marker("width", 3, cache=cache)

        assert first is not second

    def test_cache_scopes_to_one_export(self):
        """The cache is a per-export dict: two independent exports must not
        share Dim objects (global caching would leak bounds across exports)."""
        from torch.export import Dim

        from autovision.export.export_onnx import _dim_from_marker

        first = _dim_from_marker("axis", 2, cache={})
        second = _dim_from_marker("axis", 2, cache={})

        assert isinstance(first, Dim) and isinstance(second, Dim)
        assert first is not second


# ── H3-4f: multi-input parity refused loudly ────────────────────────────────


class _FakeOrtInput:
    """Stand-in for ``ort.InferenceSession.get_inputs()`` entries."""

    def __init__(self, name: str, type_: str = "tensor(float)") -> None:
        self.name = name
        self.type = type_


class _FakeOrtSession:
    def __init__(self, inputs: list[_FakeOrtInput]) -> None:
        self._inputs = inputs

    def get_inputs(self) -> list[_FakeOrtInput]:
        return self._inputs

    def run(self, outputs: list[str], feeds: dict) -> list:
        import numpy as np

        return [np.zeros(1, dtype=np.float32)]


def _write_tiny_onnx(path: Path) -> None:
    """Smallest valid ONNX graph so ``onnx.checker`` passes before the gate."""
    import onnx

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
        ir_version=9,
    )
    onnx.save(model, str(path))


class TestParityMultiInput:
    """H3-4f: feeding one tensor to every graph input verifies nothing for
    the second input; the gate now refuses instead of pretending."""

    def _verify_with_session(self, tiny_model, tmp_path, monkeypatch, inputs):
        import onnxruntime as ort

        path = tmp_path / "graph.onnx"
        _write_tiny_onnx(path)
        spec = tiny_model.model_input_spec
        example_name, example = next(
            iter(spec.example_inputs(batch_size=1).items())
        )
        # The fake session's canned outputs are not real detection tensors;
        # the hook contract itself is covered by the task tests.
        monkeypatch.setattr(
            tiny_model,
            "verify_onnx_export_outputs",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(
            ort, "InferenceSession", lambda p, providers=None: _FakeOrtSession(inputs)
        )
        tiny_model._verify_onnx(path, example_name, example, "fp32")

    def test_two_graph_inputs_raise(self, tiny_model, tmp_path, monkeypatch):
        from autovision.errors import AutoVisionError

        with pytest.raises(AutoVisionError, match="多输入对拍"):
            self._verify_with_session(
                tiny_model, tmp_path, monkeypatch,
                [_FakeOrtInput("images"), _FakeOrtInput("intrinsics")],
            )

    def test_single_graph_input_still_passes(self, tiny_model, tmp_path, monkeypatch):
        """The supported contract (one image input) keeps working."""

        self._verify_with_session(
            tiny_model, tmp_path, monkeypatch, [_FakeOrtInput("images")]
        )


# ── H3-4c: verify hooks without a precision parameter ───────────────────────


class TestVerifyHookPrecisionFallback:
    """H3-4c: hooks that predate the ``precision`` keyword are called without
    it (legacy plugins keep working) and warned about ONCE — their gate then
    runs at fp32 tolerances even for an fp16 export."""

    @pytest.fixture(autouse=True)
    def _reset_warned_hooks(self, monkeypatch):
        """Isolate the module-level warn-once registry per test."""
        from autovision.export import export_onnx

        monkeypatch.setattr(export_onnx, "_FP32_TOLERANCE_WARNED", set())

    def _run_fp16_gate(self, tiny_model, tmp_path, monkeypatch, hook):
        import onnxruntime as ort

        path = tmp_path / "graph.onnx"
        _write_tiny_onnx(path)
        spec = tiny_model.model_input_spec
        example_name, example = next(
            iter(spec.example_inputs(batch_size=1).items())
        )
        monkeypatch.setattr(tiny_model, "verify_onnx_export_outputs", hook)
        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(
            ort,
            "InferenceSession",
            lambda p, providers=None: _FakeOrtSession([_FakeOrtInput("images")]),
        )
        tiny_model._verify_onnx(path, example_name, example, "fp16")

    def test_hook_without_precision_warns_once(
        self, tiny_model, tmp_path, monkeypatch, caplog
    ):
        import logging

        calls = []

        def hook_no_precision(torch_outputs, onnx_outputs):
            """Legacy verify hook: no precision keyword, fp32 tolerances."""
            calls.append((torch_outputs, onnx_outputs))

        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run_fp16_gate(tiny_model, tmp_path, monkeypatch, hook_no_precision)
            self._run_fp16_gate(tiny_model, tmp_path, monkeypatch, hook_no_precision)

        assert len(calls) == 2
        assert "fp32 tolerance" in caplog.text
        assert caplog.text.count("does not accept") == 1

    def test_hook_with_precision_gets_it_and_stays_silent(
        self, tiny_model, tmp_path, monkeypatch, caplog
    ):
        import logging

        seen = []

        def hook_with_precision(torch_outputs, onnx_outputs, *, precision="fp32"):
            """Modern verify hook: consumes the precision keyword."""
            seen.append(precision)

        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run_fp16_gate(
                tiny_model, tmp_path, monkeypatch, hook_with_precision
            )

        assert seen == ["fp16"]
        assert "does not accept" not in caplog.text


# ── H3-4a/H3-4b: TensorRT resolved precision + staged ONNX lifecycle ───────


def _install_fake_trt(monkeypatch, *, fp16_flag: bool = True, fast_fp16=None):
    """Install a fake ``tensorrt`` module driving the REAL build pipeline:
    Builder/Config/Profile/Parser no-ops, ``build_serialized_network``
    returns a fixed blob. TensorRT and a GPU are not needed."""
    import sys
    import types

    fake = types.ModuleType("tensorrt")
    fake.__version__ = "10.14.2.2"

    class Logger:
        INFO = 3
        VERBOSE = 4

        def __init__(self, severity=None):
            self.severity = severity

    fake.Logger = Logger

    class BuilderFlag:
        TF32 = 1 << 11
        DEBUG = 1 << 9

    if fp16_flag:
        BuilderFlag.FP16 = 1 << 5
        BuilderFlag.OBEY_PRECISION_CONSTRAINTS = 1 << 7
    fake.BuilderFlag = BuilderFlag

    class MemoryPoolType:
        WORKSPACE = 0

    fake.MemoryPoolType = MemoryPoolType

    class ProfilingVerbosity:
        DETAILED = 1

    fake.ProfilingVerbosity = ProfilingVerbosity

    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 1
        STRONGLY_TYPED = 2

    fake.NetworkDefinitionCreationFlag = NetworkDefinitionCreationFlag

    class ITensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class INetworkDefinition:
        def __init__(self):
            self._inputs = [ITensor("images", (-1, 3, 64, 64))]
            self.num_inputs = 1
            self.num_outputs = 0
            self.num_layers = 0

        def get_input(self, index):
            return self._inputs[index]

        def get_output(self, index):
            raise IndexError(index)

        def get_layer(self, index):
            raise IndexError(index)

    class OnnxParser:
        def __init__(self, network, logger):
            self.network = network

        def parse(self, blob):
            return True

    class OptimizationProfile:
        def set_shape(self, name, **bounds):
            pass

    class BuilderConfig:
        def clear_flag(self, flag):
            pass

        def set_flag(self, flag):
            pass

        def set_memory_pool_limit(self, kind, value):
            pass

        def add_optimization_profile(self, profile):
            pass

    class Builder:
        platform_has_fast_fp16 = fast_fp16

        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return INetworkDefinition()

        def create_builder_config(self):
            return BuilderConfig()

        def create_optimization_profile(self):
            return OptimizationProfile()

        def build_serialized_network(self, network, config):
            return Builder.serialized_blob

    fake.Builder = Builder
    fake.Builder.serialized_blob = b"engine-blob"
    fake.OnnxParser = OnnxParser
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    return fake


class TestTensorRTResolvedPrecision:
    """H3-4a: ``precision="auto"`` used to be gated with the fp16 tolerance
    band unconditionally — an fp32 engine got verified 10x looser than it
    needs. The build decision point now reports the resolved precision and
    the gate picks tolerances from it."""

    @staticmethod
    def _stub_export_onnx(payload: bytes):
        def stub(out, **kwargs):
            """Write the intermediate graph without running a real export."""
            from pathlib import Path

            Path(out).write_bytes(payload)
            return Path(out)

        return stub

    @staticmethod
    def _stub_postprocess():
        """Well-formed detection records for the e2e gate. The engine
        runner's canned output cannot go through the real decode, so the
        postprocess is stubbed — the tolerance VALUES are what this pins."""
        import torch

        def stub_postprocess(raw, metadata, threshold=0.0):
            return [
                {
                    "bboxes": torch.zeros(1, 4),
                    "scores": torch.ones(1),
                    "labels": torch.zeros(1, dtype=torch.long),
                }
            ]

        return stub_postprocess

    @staticmethod
    def _patch_gate(monkeypatch, captured):
        """Passing engine runner + captured e2e tolerance comparison."""
        import types

        import autovision.export.tensorrt_helpers as trt_helpers

        class _PassingRunner:
            def __init__(self, engine_path, *, device="cuda"):
                pass

            def run(self, images):
                return [images]

        monkeypatch.setattr(trt_helpers, "TensorRTEngineRunner", _PassingRunner)

        import autovision.export.e2e_parity as e2e_parity

        def capture_compare(reference_records, engine_records, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(passed=True)

        monkeypatch.setattr(
            e2e_parity, "compare_detection_records", capture_compare
        )

    def _export_with_gate_capture(
        self, tiny_model, tmp_path, monkeypatch, *, precision
    ) -> dict:
        _install_fake_trt(monkeypatch)
        captured: dict = {}
        self._patch_gate(monkeypatch, captured)
        monkeypatch.setattr(
            tiny_model,
            "export_onnx",
            self._stub_export_onnx(b"staged-graph"),
        )
        monkeypatch.setattr(tiny_model, "postprocess", self._stub_postprocess())
        monkeypatch.setattr(
            tiny_model,
            "verify_onnx_export_outputs",
            lambda *args, **kwargs: None,
        )
        tiny_model.export_tensorrt(
            tmp_path / "model.engine", precision=precision, verify=True
        )
        return captured

    def test_auto_with_fp32_weights_gates_at_fp32_tolerance(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """auto + fp32 model: the engine really is fp32, so the score
        tolerance must be the fp32 band (1e-2), not the fp16 band (5e-2)."""
        captured = self._export_with_gate_capture(
            tiny_model, tmp_path, monkeypatch, precision="auto"
        )

        assert captured["score_tolerance"] == 1e-2
        assert captured["box_tolerance_px"] == 1.0

    def test_explicit_fp16_gates_at_fp16_tolerance(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Explicit fp16: the wide band is kept."""
        captured = self._export_with_gate_capture(
            tiny_model, tmp_path, monkeypatch, precision="fp16"
        )

        assert captured["score_tolerance"] == 5e-2

    def test_pipeline_reports_resolved_precision_per_scenario(
        self, tmp_path, monkeypatch
    ):
        """Unit sweep of the resolution contract over the flag/precision
        matrix (no model, no export): auto+fp32->fp32, auto+fp16->fp16,
        explicit fp16->fp16, platform downgrade->fp32, TRT11 graph
        route->fp16, int8->int8."""
        import torch

        from autovision.export import tensorrt_helpers as th

        def stub_export(out, **kwargs):
            from pathlib import Path

            Path(out).write_bytes(b"staged")
            return Path(out)

        for fake_kwargs, call_kwargs, expected in (
            (
                dict(fp16_flag=True, fast_fp16=None),
                dict(precision="auto", model_dtype=torch.float32),
                "fp32",
            ),
            (
                dict(fp16_flag=True, fast_fp16=None),
                dict(precision="auto", model_dtype=torch.float16),
                "fp16",
            ),
            (
                dict(fp16_flag=True, fast_fp16=None),
                dict(precision="fp16", model_dtype=torch.float32),
                "fp16",
            ),
            (
                dict(fp16_flag=True, fast_fp16=False),
                dict(precision="fp16", model_dtype=torch.float32),
                "fp32",
            ),
            (
                dict(fp16_flag=False, fast_fp16=None),
                dict(precision="auto", model_dtype=torch.float16),
                "fp16",
            ),
            (
                dict(fp16_flag=True, fast_fp16=None),
                dict(precision="int8", model_dtype=torch.float32),
                "int8",
            ),
        ):
            fake = _install_fake_trt(monkeypatch, **fake_kwargs)
            fake.Builder.serialized_blob = b"engine-blob"
            resolved: dict = {}
            extra: dict = {}
            if call_kwargs["precision"] == "int8":
                # int8 consumes a caller-staged QDQ graph (lifecycle owned
                # by the caller, so the helper must not delete it). The QDQ
                # sanitizer loads the graph, so it must be a valid ONNX.
                staged = tmp_path / "model.int8.onnx"
                _write_tiny_onnx(staged)
                extra["pre_staged_onnx"] = staged
            th.export_onnx_to_tensorrt(
                out=tmp_path / "model.engine",
                export_onnx_fn=stub_export,
                resolved_precision_out=resolved,
                **extra,
                **call_kwargs,
            )
            assert resolved["precision"] == expected, call_kwargs


class TestTensorRTStagedOnnx:
    """H3-4b: a self-staged intermediate ONNX (exported with verify=False)
    used to live on next to the parity-verified engine under the same stem.
    After a successful build it is removed; explicit keeps and failed
    builds keep it."""

    @staticmethod
    def _stub_export(payload: bytes = b"staged-onnx-bytes"):
        def stub(out, **kwargs):
            from pathlib import Path

            path = Path(out)
            path.write_bytes(payload)
            path.with_suffix(path.suffix + ".data").write_bytes(b"sidecar")
            return path

        return stub

    def test_success_removes_staged_graph_and_sidecar(self, tmp_path, monkeypatch):
        from autovision.export import tensorrt_helpers as th

        _install_fake_trt(monkeypatch)

        th.export_onnx_to_tensorrt(
            out=tmp_path / "model.engine",
            export_onnx_fn=self._stub_export(),
            precision="fp32",
        )

        assert (tmp_path / "model.engine").read_bytes() == b"engine-blob"
        assert not (tmp_path / "model.onnx").exists()
        assert not (tmp_path / "model.onnx.data").exists()

    def test_keep_staged_onnx_preserves_the_graph(self, tmp_path, monkeypatch):
        from autovision.export import tensorrt_helpers as th

        _install_fake_trt(monkeypatch)

        th.export_onnx_to_tensorrt(
            out=tmp_path / "model.engine",
            export_onnx_fn=self._stub_export(),
            precision="fp32",
            keep_staged_onnx=True,
        )

        assert (tmp_path / "model.onnx").read_bytes() == b"staged-onnx-bytes"

    def test_failed_build_keeps_the_staged_graph(self, tmp_path, monkeypatch):
        """A failed build is debugged against its staged graph, so only the
        SUCCESS path cleans it."""
        from autovision.export import tensorrt_helpers as th

        fake = _install_fake_trt(monkeypatch)
        fake.Builder.serialized_blob = None

        with pytest.raises(RuntimeError, match="engine building failed"):
            th.export_onnx_to_tensorrt(
                out=tmp_path / "model.engine",
                export_onnx_fn=self._stub_export(),
                precision="fp32",
            )

        assert (tmp_path / "model.onnx").exists()

    def test_pre_staged_int8_graph_is_not_touched(self, tmp_path, monkeypatch):
        """The int8 route stages its QDQ graph explicitly (caller-owned
        lifecycle with keep_int8_onnx); the helper must never delete it."""
        from autovision.export import tensorrt_helpers as th

        fake = _install_fake_trt(monkeypatch)
        fake.Builder.serialized_blob = b"int8-engine"
        staged = tmp_path / "model.int8.onnx"
        _write_tiny_onnx(staged)

        th.export_onnx_to_tensorrt(
            out=tmp_path / "model.engine",
            export_onnx_fn=self._stub_export(),
            precision="int8",
            pre_staged_onnx=staged,
        )

        assert staged.exists()

    def test_public_export_tensorrt_cleans_the_staged_graph(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Wiring through the public path: after a successful build the
        intermediate graph next to the target is gone."""

        import autovision.export.tensorrt_helpers as trt_helpers

        _install_fake_trt(monkeypatch)
        monkeypatch.setattr(
            tiny_model,
            "export_onnx",
            TestTensorRTResolvedPrecision._stub_export_onnx(b"staged-graph"),
        )

        class _PassingRunner:
            def __init__(self, engine_path, *, device="cuda"):
                pass

            def run(self, images):
                return [images]

        monkeypatch.setattr(trt_helpers, "TensorRTEngineRunner", _PassingRunner)
        monkeypatch.setattr(tiny_model, "postprocess", None)
        monkeypatch.setattr(
            tiny_model,
            "verify_onnx_export_outputs",
            lambda *args, **kwargs: None,
        )

        tiny_model.export_tensorrt(tmp_path / "model.engine", verify=True)

        assert (tmp_path / "model.engine").exists()
        # Staged name derived from the tmp engine ("model.engine.tmp-<pid>"
        # -> ".onnx"): it must not survive the success.
        assert not list(tmp_path.glob("*.onnx"))
        assert not list(tmp_path.glob("*.onnx.data"))


# ── H3-4d: exported ONNX carries the toolchain record ───────────────────────


class TestOnnxToolchainMetadata:
    """H3-4d: the ONNX carrier records the producing toolchain in-graph,
    with the REAL graph opset (not the argument default guess)."""

    def test_export_records_toolchain_with_graph_opset(
        self, tiny_model, tmp_path
    ):
        import json

        import onnx

        out = tiny_model.export_onnx(
            tmp_path / "model.onnx",
            precision="fp32",
            verify=False,
            format_args={"opset_version": 18},
        )
        model = onnx.load(str(out), load_external_data=False)
        metadata = {prop.key: prop.value for prop in model.metadata_props}

        payload = json.loads(metadata["autovision_toolchain"])
        assert payload["onnx_opset"] == 18
        assert payload["torch"]
        assert set(payload) == {
            "torch", "onnx", "onnx_opset", "onnxruntime", "tensorrt", "openvino",
        }
