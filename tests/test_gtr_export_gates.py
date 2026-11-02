# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR family export gates: verify hooks + end-to-end ONNX export per route.

The eager GLA runs fla's Triton / CUDA kernels, which have no ONNX
representation; the export pipeline swaps them for the pure-ONNX lowering
in :mod:`autovision.tasks.detection.shared.gtr.gla_onnx` via the
``prepare_onnx_export`` module-tree hook. These tests pin that contract:

- every ``gtr`` route (det / obb / pose / depth / semseg) must pass
  ``export_onnx(..., verify=True)`` on CPU — the gate runs the real
  torch-vs-onnxruntime parity check, and the swap makes the torch
  reference device-independent (no CUDA needed);
- the lowered GLA kernel must match a naive per-token reference (fp64)
  and fla's ``chunk_gla`` on CUDA, including with strongly negative gates
  (the naive factorized decay overflows there; the explicit per-channel
  form does not);
- plain models keep a clean ``_export_blocker()``.
"""

from __future__ import annotations

import pytest
import torch

from autovision.errors import AutoVisionError

try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_gtr_extra = pytest.mark.skipif(
    not _fla, reason="GTR construction needs flash-linear-attention (pip install -e '.[gtr]')"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fla's Triton GLA kernels need CUDA"
)


def _build_pose():
    from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

    return GTRPoseEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_depth():
    from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

    return GTRDepthEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_semseg():
    from autovision.tasks.semantic_segmentation.gtr import GTRSemanticSegmentation

    return GTRSemanticSegmentation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_obb():
    from autovision.tasks.oriented_detection.gtr import GTRROBBOrientedDetection

    model = GTRROBBOrientedDetection(
        variant="s", image_size=(64, 64), num_queries=10, load_checkpoint=False
    )
    # The D-FINE-style zero-init reg heads make every decoded box exactly
    # square at initialization, and the long-edge normalization's ``w < h``
    # comparison then flips on ulp-level torch-vs-ORT differences (the angle
    # jumps by exactly pi/2). Real trained weights never sit on that
    # boundary; break the degeneracy so the random-weight parity gate is
    # meaningful (same spirit as the exporter's seeded compare input).
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for module in model.modules():
            for name, child in module.named_children():
                if "bbox_head" in name or "pre_bbox_head" in name or "enc_bbox_head" in name:
                    for parameter in child.parameters():
                        parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _build_detection():
    from autovision.tasks.detection.detr.task_model import DETRObjectDetection

    return DETRObjectDetection(
        num_classes=2,
        model_name="gtrvit-s",
        image_size=(64, 64),
        decoder="dfine_official",
        num_queries=10,
        load_weights=False,
        backbone_pretrained=False,
    )


_TASK_MODELS = [
    pytest.param(_build_pose, ((2, 60, 2), (2, 60, 34)), id="pose"),
    pytest.param(_build_depth, ((2, 64, 64),), id="depth"),
    pytest.param(_build_semseg, ((2, 19, 64, 64),), id="semseg"),
    pytest.param(_build_obb, ((2, 10, 15), (2, 10, 5)), id="obb"),
]

#: One builder per deployable GTR route, for the end-to-end export gate.
_EXPORT_BUILDERS = [
    pytest.param(_build_detection, id="det"),
    pytest.param(_build_pose, id="pose"),
    pytest.param(_build_depth, id="depth"),
    pytest.param(_build_semseg, id="semseg"),
    pytest.param(_build_obb, id="obb"),
]


def _as_forward_output(tensors: list[torch.Tensor]):
    """Single-output models return a bare tensor; multi-output ones a tuple."""
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


@requires_gtr_extra
@pytest.mark.parametrize(("build", "shapes"), _TASK_MODELS)
def test_verify_hook_contract(build, shapes):
    torch.manual_seed(0)
    model = build()
    torch_outputs = [torch.randn(shape) for shape in shapes]
    onnx_outputs = [tensor.numpy() for tensor in torch_outputs]

    model.verify_onnx_export_outputs(_as_forward_output(torch_outputs), onnx_outputs)

    perturbed = [array.copy() for array in onnx_outputs]
    perturbed[-1] += 1.0
    with pytest.raises(AssertionError):
        model.verify_onnx_export_outputs(
            _as_forward_output(torch_outputs), perturbed
        )


@requires_gtr_extra
def test_pose_fp32_gate_is_strict_but_permutation_tolerant():
    """The pose hook (no decode in-graph) compares element-wise at fp32:
    a drift that cancels inside the row-sum tolerance — the historical gate
    — must now fail, while a reordered-but-equal row set still passes."""
    torch.manual_seed(0)
    model = _build_pose()
    reference = (torch.randn(2, 60, 2), torch.randn(2, 60, 17, 2))
    model.verify_onnx_export_outputs(reference, [t.numpy() for t in reference])

    # legitimate permutation: swap three queries identically in both outputs
    order = torch.arange(60)
    order[[3, 4, 5]] = torch.tensor([5, 3, 4])
    permuted = (
        reference[0][:, order].clone(),
        reference[1][:, order].clone(),
    )
    model.verify_onnx_export_outputs(reference, [t.numpy() for t in permuted])

    # ±0.5 across two rows, same class: row-sums unchanged (old gate passes)
    drifted = (reference[0].clone(), reference[1].clone())
    drifted[0][:, 0, 0] += 0.5
    drifted[0][:, 1, 0] -= 0.5
    from autovision.export.parity_checks import assert_outputs_close_sum

    assert_outputs_close_sum(reference, drifted)
    with pytest.raises(AssertionError):
        model.verify_onnx_export_outputs(reference, [t.numpy() for t in drifted])

    # the fp16 path keeps the drift-tolerant row-sum invariant
    model.verify_onnx_export_outputs(
        reference, [t.numpy() for t in drifted], precision="fp16"
    )


@requires_gtr_extra
def test_depth_fp32_gate_rejects_cancelling_drift():
    """The depth hook compares the dense map element-wise at fp32: a ±0.5
    pair that cancels in the spatial sum must fail (the historical gate
    passed it); fp16 keeps the sum form."""
    torch.manual_seed(0)
    model = _build_depth()
    reference = torch.rand(2, 64, 64)
    model.verify_onnx_export_outputs(reference, [reference.numpy()])

    drifted = reference.clone()
    drifted[:, 0, 0] += 0.5
    drifted[:, 1, 0] -= 0.5
    from autovision.export.parity_checks import assert_outputs_close_sum

    assert_outputs_close_sum((reference,), (drifted,))
    with pytest.raises(AssertionError):
        model.verify_onnx_export_outputs(reference, [drifted.numpy()])
    model.verify_onnx_export_outputs(
        reference, [drifted.numpy()], precision="fp16"
    )


@requires_gtr_extra
@pytest.mark.parametrize("build", _EXPORT_BUILDERS)
def test_export_onnx_end_to_end(build, tmp_path):
    """Every GTR route exports through the real gated pipeline: dynamo trace,
    onnxslim, onnx.checker and the torch-vs-ORT parity gate — on CPU."""
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = build()
    out = model.export_onnx(tmp_path / "model.onnx", verify=True)
    assert out.exists()
    # onnxslim re-inlined the dynamo external data: no orphan .data sidecar
    # may survive next to the artifact (regression: ~45MB litter per model).
    assert not (tmp_path / "model.onnx.data").exists()

    # The whole point of the lowering: no custom-domain nodes, so the graph
    # runs on any ONNX runtime without plugins.
    import onnx

    graph = onnx.load(str(out)).graph
    custom = [node.op_type for node in graph.node if node.domain not in ("", "ai.onnx")]
    assert custom == []


def _naive_gla_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         gk: torch.Tensor, scale: float) -> torch.Tensor:
    """Per-token GLA recurrence in fp64 — the oracle for :func:`chunk_gla_pure`."""
    q, k, v, gk = (t.double() for t in (q, k, v, gk))
    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    out = torch.zeros(batch, seq, heads, v_dim, dtype=torch.float64)
    state = torch.zeros(batch, heads, k_dim, v_dim, dtype=torch.float64)
    for t in range(seq):
        decay = gk[:, t].unsqueeze(-1).exp()          # (B, H, K, 1)
        state = state * decay + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(2)
        out[:, t] = scale * (state * q[:, t].unsqueeze(-1)).sum(dim=2)
    return out.float()


@requires_gtr_extra
@pytest.mark.parametrize("seq_len", [16, 200], ids=["single-chunk", "pad-path"])
def test_chunk_gla_pure_matches_naive_reference(seq_len):
    """The lowered kernel equals the per-token recurrence (fp64 oracle),
    including the T % chunk padding path."""
    from autovision.tasks.detection.shared.gtr import chunk_gla_pure

    torch.manual_seed(0)
    q = torch.randn(2, seq_len, 3, 32)
    k = torch.randn(2, seq_len, 3, 32)
    v = torch.randn(2, seq_len, 3, 64)
    gk = torch.nn.functional.logsigmoid(torch.randn(2, seq_len, 3, 32)) / 16.0
    out = chunk_gla_pure(q, k, v, gk, 32**-0.5)
    ref = _naive_gla_reference(q, k, v, gk, 32**-0.5)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@requires_gtr_extra
@requires_cuda
def test_chunk_gla_pure_matches_fla_with_strong_gates():
    """fla parity, including strongly negative gates where a factorized decay
    overflows to inf*0=NaN — the explicit per-channel decay must not."""
    from fla.ops.gla import chunk_gla

    from autovision.tasks.detection.shared.gtr import chunk_gla_pure

    torch.manual_seed(0)
    q = torch.randn(1, 256, 2, 32, device="cuda")
    k = torch.randn(1, 256, 2, 32, device="cuda")
    v = torch.randn(1, 256, 2, 64, device="cuda")
    scale = 32**-0.5
    # fla's Triton kernel accumulates in a different order (BT=64 blocks, BC=16
    # sub-blocks), so fp32 outputs differ from ours by accumulation noise that
    # grows with the state magnitude along the sequence (observed max_abs
    # ~0.01-0.03 at cos > 0.9999). Exact math is pinned by the fp64-oracle
    # test above; this test pins cross-implementation agreement and — the
    # regression it exists for — no NaN where a factorized decay overflows.
    for name, gk in (
        # the distribution the trained models produce (normalizer 16)
        ("model-like",
         torch.nn.functional.logsigmoid(torch.randn(1, 256, 2, 32, device="cuda")) / 16.0),
        # strong gates: logsigmoid(x ~ -100) drives per-chunk decays past -6000,
        # where fp32 exp() underflows to exact zero and the two orders diverge
        # further in the last ulp.
        ("strong",
         torch.nn.functional.logsigmoid(torch.randn(1, 256, 2, 32, device="cuda") - 100.0)),
    ):
        ref = chunk_gla(q=q, k=k, v=v, g=gk)[0]
        out = chunk_gla_pure(q, k, v, gk, scale)
        assert not torch.isnan(out).any(), f"{name}: lowered GLA produced NaN"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=5e-2)


@requires_gtr_extra
def test_pose_postprocess_handles_deploy_keypoint_layout():
    """forward_backend emits (B, N, K, 2) keypoints; postprocess must gather per
    point (upstream deploy_mode branch) — the flat-layout gather regressed to a
    RuntimeError on every real pose inference."""
    torch.manual_seed(0)
    model = _build_pose()
    logits = torch.randn(2, 60, 2)
    keypoints = torch.randn(2, 60, 17, 2)
    records = model.postprocess(
        (logits, keypoints),
        [{"orig_w": 64.0, "orig_h": 64.0}] * 2,
        threshold=0.0,
    )
    assert len(records) == 2
    for record in records:
        # per-record contract: flattened COCO triplets (n, K*3) with v=1
        assert record["keypoints"].shape[1] == 17 * 3
        assert record["keypoints"].shape[0] == record["scores"].shape[0]


@requires_gtr_extra
def test_gtr_rejects_dynamic_spatial_export(tmp_path):
    """GTR graphs are fixed-shape (the GLA recurrence unrolls for the variant's
    token count): dynamic H/W export is refused fast with a readable error
    instead of grinding through symbolic-shape tracing."""
    model = _build_depth()
    with pytest.raises(AutoVisionError, match="dynamic_spatial"):
        model.export_onnx(tmp_path / "model.onnx", dynamic_spatial=True, verify=False)


@requires_gtr_extra
def test_torch_backend_cpu_engine_rejected_readably():
    """Eager GLA runs fla's Triton kernels (CUDA-only): a CPU torch-backend
    engine is refused at construction with install/ONNX guidance instead of an
    opaque Triton pointer error at the first forward."""
    from autovision.inference.engine import InferenceEngine

    model = _build_depth()
    with pytest.raises(AutoVisionError, match="backend='onnx'"):
        InferenceEngine(model, backend="torch")


@requires_gtr_extra
def test_fp16_export_with_gla_fp32_pins(tmp_path):
    """The GLA's ``onnx_fp32_op_types`` pins keep the fp16 conversion
    consistent (regression: the unpinned converter produced mixed-type
    MatMul/Resize nodes that failed onnx.checker) and the artifact runs in
    onnxruntime on CPU."""
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = _build_depth()
    out = model.export_onnx(tmp_path / "model_fp16.onnx", precision="fp16", verify=True)
    assert out.exists()

    import numpy as np
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(out)).graph
    fp16_init = [t for t in graph.initializer if t.data_type == onnx.TensorProto.FLOAT16]
    assert fp16_init, "expected real fp16 initializers in the graph"

    # The pins are graph-global: EVERY MatMul must stay fp32 (GTR is
    # GEMM-dominated, so a single mixed-dtype MatMul means the dtype
    # propagation across the GLA fp32-state boundary broke again).
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(str(out)), strict_mode=False, data_prop=True
    )
    dtypes = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input)
    }
    dtypes.update({t.name: t.data_type for t in inferred.graph.initializer})
    matmul_input_dtypes = {
        dtypes.get(name)
        for node in inferred.graph.node
        if node.op_type == "MatMul"
        for name in node.input
    }
    assert matmul_input_dtypes == {onnx.TensorProto.FLOAT}, (
        f"fp16 graph carries non-fp32 MatMul inputs: {matmul_input_dtypes}"
    )

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.rand(1, 3, 64, 64)
    feed = x.numpy().astype(np.float16)
    pred = session.run(None, {"images": feed})[0]
    assert pred.shape[0] == 1 and np.isfinite(pred).all()


@requires_gtr_extra
def test_fp16_smoke_gate_rejects_nonfinite_graph(tmp_path):
    """The no-CUDA-EP fp16 fallback runs the execute-and-finiteness smoke
    gate (regression: verify=True skipped silently on CPU-only onnxruntime
    builds). A clean fp16 graph passes; NaN-poisoning one initializer must
    fail the gate instead of shipping an unverified artifact."""
    pytest.importorskip("onnxruntime")
    import numpy as np
    import onnx
    from onnx import numpy_helper

    torch.manual_seed(0)
    model = _build_depth()
    out = tmp_path / "model_fp16.onnx"
    model.export_onnx(out, precision="fp16", verify=False)
    example = torch.rand(1, 3, 64, 64)
    model._smoke_verify_fp16(out, example)

    # Poison the largest fp16 initializer (a main weight, so the NaN
    # provably reaches the outputs through the random compare input).
    graph = onnx.load(str(out))
    target = max(
        (t for t in graph.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16),
        key=lambda t: int(np.prod(t.dims)) if t.dims else 1,
    )
    poisoned = numpy_helper.to_array(target).copy()
    poisoned.reshape(-1)[0] = np.float16(np.nan)
    target.CopyFrom(numpy_helper.from_array(poisoned, target.name))
    onnx.save(graph, str(out))
    with pytest.raises(AssertionError, match="non-finite"):
        model._smoke_verify_fp16(out, example)


def test_plain_models_have_no_blocker():
    from autovision.tasks import RTDETRObjectDetection

    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    assert model._export_blocker() is None


@requires_gtr_extra
def test_openvino_export_end_to_end(tmp_path):
    """OpenVINO export's parity gate replays the export-safe reference path
    (regression: the gate deep-copied the model to CPU and ran the eager
    fla kernels there, killing every GTR route with an opaque Triton
    pointer error after the IR had already been produced)."""
    pytest.importorskip("openvino")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = _build_depth()
    out = model.export_openvino(
        tmp_path / "model.xml", precision="fp32", device="CPU", verify=True
    )
    assert out.exists()
    assert out.with_suffix(".bin").exists()


@requires_gtr_extra
def test_onnx_gate_reference_runs_after_hooks_restored(tmp_path):
    """The ONNX gate's reference hooks itself explicitly instead of relying
    on deepcopy-inheritance from the trace window: ``_verify_onnx`` must
    also work invoked OUTSIDE a trace (hooks restored), where an inherited
    reference would run the eager fla kernels and die on CPU."""
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = _build_depth()
    out = tmp_path / "model.onnx"
    model.export_onnx(out, verify=False)
    spec = model.model_input_spec
    example_name, example_tensor = next(
        iter(spec.example_inputs(batch_size=1).items())
    )
    # Post-export re-verify: must pass, not raise the Triton pointer error.
    model._verify_onnx(out, example_name, example_tensor, "fp32")


@requires_gtr_extra
def test_backend_parity_reference_runs_on_cpu_model():
    """The backend-parity reference helpers (TensorRT gate path) detect a
    CPU-resident device-bound model and replay the export-safe path instead
    of the eager fla kernels — outputs must be finite tensors."""
    from autovision.export.export_onnx import (
        _eager_reference_device_bound,
        _forward_with_export_hooks,
    )

    torch.manual_seed(0)
    model = _build_depth()
    model.eval()
    # The real gate runs post-ONNX-export, where deploy() has already fused
    # the GLA projections; mirror that state here.
    model.deploy()
    x = torch.rand(2, 3, 64, 64)
    assert _eager_reference_device_bound(model), (
        "the deployed GTR model must report its eager forward as device-bound"
    )
    with torch.no_grad():
        outputs = _forward_with_export_hooks(model, x)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    for tensor in tensors:
        assert torch.isfinite(tensor).all()


@requires_gtr_extra
def test_spatial_swiglu_deploy_static_perm_matches_flip_path():
    """The CLS-free deploy path honors ``_deploy_static_perm``: the static
    index_select reproduces the flip/transpose chain bit-for-bit per layer,
    and convert_to_deploy builds a CLS-free (length-L) perm buffer
    (regression: the override ignored the flag, so the static permutation —
    and with it the whole eval_spatial_size deploy stamping — was dead code
    for every GTR route's backbone)."""
    torch.manual_seed(0)
    model = _build_depth()
    model.eval()
    adapter = model.backbone._adapter
    vit = adapter.backbone
    assert not hasattr(vit._model, "cls_token")
    assert getattr(vit, "_deploy_static_perm", False) is False

    model.deploy()
    assert vit._deploy_static_perm is True
    L = (adapter.eval_spatial_size[0] // vit.patch_size) ** 2
    assert tuple(vit._bid_step_perm_full.shape) == (vit.depth, L)

    tokens = torch.randn(2, L, vit.embed_dim)
    for i in range(vit.depth):
        static = vit._apply_bid_scan(tokens, i)
        flipped = tokens.flip(1)
        if i % 2 == 1 and vit.if_quad_dir:
            gs = int(flipped.shape[1] ** 0.5)
            flipped = flipped.reshape(flipped.shape[0], gs, gs, -1)
            flipped = flipped.transpose(1, 2).flatten(1, 2)
        # Both are pure permutations: equality, not allclose.
        assert torch.equal(static, flipped), f"bid_scan layer {i} mismatch"
