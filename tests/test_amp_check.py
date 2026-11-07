# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for the pre-flight AMP numeric probe (D16).

All tests run on CPU hardware: the probe paths that need a "CUDA-like"
device use a duck-typed device stand-in (``SimpleNamespace(type="cuda")``)
so the fp32-vs-autocast comparison logic is exercised without a GPU.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch.nn import Linear, Module

from autovision.engine.amp_check import (
    _flatten_outputs,
    _probe_batch,
    check_amp_support,
    resolve_amp_precision,
)
from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec
from autovision.tasks.train_model import TrainModel

CUDA_LIKE = SimpleNamespace(type="cuda")
CPU = torch.device("cpu")


class _AmpBroken(Module):
    """Module whose autocast outputs diverge from fp32 (broken stack)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(4, 2)
        self.model_input_spec = ModelInputSpec(
            input_specs={
                "x": TensorSpec(shape=(4,), dtype=torch.float32, is_batched=True)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            return self.linear(x) + 1e3  # simulate autocast corruption
        return self.linear(x)


class _SpecModel(Module):
    """Module exposing a ``model_input_spec`` for probe-input derivation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(4, 2)
        self.model_input_spec = ModelInputSpec(
            input_specs={
                "x": TensorSpec(shape=(4,), dtype=torch.float32, is_batched=True)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestCheckAmpSupport:
    """Device gates, probe comparison, and failure degradation."""

    def test_cpu_device_returns_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CPU devices have no autocast training path; probe reports False."""
        with caplog.at_level(logging.INFO):
            assert check_amp_support(Linear(2, 2), device=CPU) is False
        assert "no autocast support" in caplog.text

    def test_divergent_autocast_outputs_fail(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output divergence beyond atol degrades the decision to False."""
        # torch disables autocast("cuda") on CUDA-less hosts, which would
        # void the probe into a vacuous fp32-vs-fp32 tie; claim CUDA
        # availability so the comparison logic itself is exercised here.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        batch = torch.randn(2, 4)
        assert (
            check_amp_support(_AmpBroken(), device=CUDA_LIKE, batch=batch) is False
        )
        assert "divergence" in caplog.text

    def test_agreeing_outputs_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A well-behaved module passes the fp32-vs-autocast comparison."""
        # Keep the probe honest on CUDA-less hosts (the fail-closed guard
        # would otherwise short-circuit it before the comparison runs).
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        batch = torch.randn(2, 4)
        assert check_amp_support(Linear(4, 2), device=CUDA_LIKE, batch=batch) is True

    def test_missing_probe_inputs_are_inconclusive(self) -> None:
        """No batch and no model_input_spec: probe is None, not unsafe."""
        assert (
            check_amp_support(Linear(4, 2), device=CUDA_LIKE, batch=None) is None
        )

    def test_model_input_spec_supplies_the_probe_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The declared input spec is used when no batch is passed."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert check_amp_support(_SpecModel(), device=CUDA_LIKE) is True

    def test_forward_exception_fails_closed(self) -> None:
        """A probe-time exception degrades to False, never raises."""

        class _Boom(Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = Linear(4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("boom")

        assert (
            check_amp_support(
                _Boom(), device=CUDA_LIKE, batch=torch.randn(2, 4)
            )
            is False
        )

    def test_training_mode_restored_after_probe(self) -> None:
        """The probe runs in eval mode but restores the previous mode."""
        model = Linear(4, 2)
        model.train()
        check_amp_support(model, device=CUDA_LIKE, batch=torch.randn(2, 4))
        assert model.training is True


class TestResolveAmpPrecision:
    """Gate ordering and the DDP broadcast of the probe decision."""

    def test_fp32_request_passes_through(self) -> None:
        """Non-mixed precisions never trigger the probe."""
        assert (
            resolve_amp_precision(Linear(2, 2), device=CPU, precision="32-true")
            == "32-true"
        )

    def test_unsupported_device_degrades_to_fp32(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Mixed precision on a CPU device falls back with the old warning."""
        result = resolve_amp_precision(
            Linear(2, 2), device=CPU, precision="16-mixed"
        )
        assert result == "32-true"
        assert "without CUDA/NPU" in caplog.text

    def test_amp_check_false_skips_probe(self) -> None:
        """``amp_check=False`` keeps mixed precision without probing."""
        result = resolve_amp_precision(
            Linear(2, 2), device=CUDA_LIKE, precision="16-mixed", amp_check=False
        )
        assert result == "16-mixed"

    def test_failed_probe_degrades_to_fp32(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing numeric probe forces fp32."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        result = resolve_amp_precision(
            _AmpBroken(), device=CUDA_LIKE, precision="16-mixed"
        )
        assert result == "32-true"

    def test_cuda_device_without_cuda_fails_closed(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cuda-like device on a CUDA-less host has no autocast path.

        torch disables autocast("cuda") there, which would otherwise turn
        the probe into a vacuous fp32-vs-fp32 tie reporting "supported"."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with caplog.at_level(logging.WARNING):
            assert (
                check_amp_support(
                    Linear(4, 2), device=CUDA_LIKE, batch=torch.randn(2, 4)
                )
                is False
            )
        assert "CUDA is unavailable" in caplog.text

    def test_passing_probe_keeps_mixed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A passing probe keeps the requested mixed precision."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        result = resolve_amp_precision(
            _SpecModel(), device=CUDA_LIKE, precision="16-mixed"
        )
        assert result == "16-mixed"

    def test_ddp_broadcasts_rank0_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-zero ranks follow the rank-0 decision via int8 broadcast."""
        broadcast_calls: list[torch.Tensor] = []

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

        def fake_broadcast(tensor: torch.Tensor, src: int = 0) -> None:
            broadcast_calls.append(tensor.clone())
            tensor.fill_(0)  # rank 0 said "unsafe"; every rank must adopt it

        monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
        result = resolve_amp_precision(
            _SpecModel(), device=CUDA_LIKE, precision="16-mixed"
        )
        assert result == "32-true"
        assert len(broadcast_calls) == 1
        assert broadcast_calls[0].dtype == torch.int8

    def test_ddp_inconclusive_probe_keeps_precision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An inconclusive probe broadcasts -1 and never downgrades."""
        broadcast_calls: list[torch.Tensor] = []

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

        def fake_broadcast(tensor: torch.Tensor, src: int = 0) -> None:
            broadcast_calls.append(tensor.clone())
            tensor.fill_(int(tensor.item()))  # adopt rank 0's value as-is

        monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
        result = resolve_amp_precision(
            Linear(4, 2), device=CUDA_LIKE, precision="16-mixed"
        )
        assert result == "16-mixed"
        assert len(broadcast_calls) == 1
        assert int(broadcast_calls[0].item()) == -1  # three-state encoding


class TestFlattenOutputs:
    """Float-tensor extraction from nested output containers."""

    def test_nested_containers_and_dtype_filter(self) -> None:
        """Dicts, tuples, and object state are walked; non-float is skipped."""

        class _Out:
            def __init__(self) -> None:
                self.score = torch.randn(3)
                self.label = torch.randint(0, 2, (3,))

        out: list[torch.Tensor] = []
        _flatten_outputs(
            {"a": (torch.randn(2), torch.arange(4)), "b": _Out(), "c": None}, out
        )
        assert len(out) == 2


class TestTrainModelSpecForwarding:
    """The TrainModel wrapper must forward ``model_input_spec`` upstream.

    The engine hands the probe the wrapper, not the inner TaskNetwork;
    without forwarding, every family that declares a spec degrades to an
    inconclusive probe (observed on detection runs, 2026-11-04).
    """

    def test_wrapper_forwards_spec_to_the_probe(self) -> None:
        """A spec'd task model surfaces its spec through the wrapper."""
        wrapper = TrainModel(task_model=_SpecModel())
        assert wrapper.model_input_spec is not None
        assert _probe_batch(wrapper, None) is not None

    def test_wrapper_without_inner_spec_stays_none(self) -> None:
        """A bare task model keeps the probe-input path unavailable."""
        wrapper = TrainModel(task_model=Linear(4, 2))
        assert wrapper.model_input_spec is None
        assert _probe_batch(wrapper, None) is None


class TestProbeUnwrapsTaskModel:
    """Engine call sites pass the TrainModel wrapper; the probe must
    forward the inner TaskNetwork, whose ``forward(x: Tensor)`` is the
    contract the probe actually exercises."""

    def test_wrapper_probe_never_hits_the_training_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tensor input routed to the wrapper forward would crash and
        fail the probe closed to False; unwrapping keeps the verdict True."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        wrapper = TrainModel(task_model=_SpecModel())

        def _must_not_be_called(batch, step):
            raise AssertionError("probe must not call the wrapper forward")

        wrapper.forward = _must_not_be_called  # type: ignore[assignment]
        assert check_amp_support(wrapper, device=CUDA_LIKE, batch=None) is True

    def test_bare_network_behavior_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Modules without get_task_model keep the previous probe path."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert (
            check_amp_support(_SpecModel(), device=CUDA_LIKE, batch=None) is True
        )


class TestAmpProbeAbstentionAndDtype:
    """Declared probe abstention (amp_probe='skip') and dtype threading."""

    def test_skip_declaration_is_inconclusive_not_unsafe(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """amp_probe='skip' abstains (None), never downgrades to False."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        model = Linear(4, 2)
        model.amp_probe = "skip"
        with caplog.at_level(logging.INFO):
            result = check_amp_support(model, device=CUDA_LIKE, batch=torch.randn(2, 4))
        assert result is None
        assert "declined by model declaration" in caplog.text

    def test_skip_declaration_reaches_resolution_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An abstaining probe keeps the requested precision end to end."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        model = Linear(4, 2)
        model.amp_probe = "skip"
        assert (
            resolve_amp_precision(model, device=CUDA_LIKE, precision="bf16-mixed")
            == "bf16-mixed"
        )

    def test_resolve_threads_the_requested_autocast_dtype(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bf16-mixed runs probe the bf16 path, not the fp16 default."""
        captured: list[torch.dtype | None] = []
        real = check_amp_support

        def _spy(model, **kwargs):
            captured.append(kwargs.get("dtype"))
            return real(model, **kwargs)

        monkeypatch.setattr(
            "autovision.engine.amp_check.check_amp_support", _spy
        )
        resolve_amp_precision(
            Linear(4, 2), device=CUDA_LIKE, precision="bf16-mixed"
        )
        assert captured == [torch.bfloat16]
