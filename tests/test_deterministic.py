# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Determinism is an opt-in environment switch, not a default side effect."""

import os

import torch

from autovision.engine.engine import enable_cuda_math_modes, seed_everything


def _restore(old_deterministic: bool, old_cudnn: bool, old_benchmark: bool) -> None:
    torch.use_deterministic_algorithms(old_deterministic)
    torch.backends.cudnn.deterministic = old_cudnn
    torch.backends.cudnn.benchmark = old_benchmark


def _restore_cublas(old: str | None) -> None:
    if old is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = old


def test_deterministic_env_enables_torch_flags(monkeypatch):
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setenv("AUTOVISION_DETERMINISTIC", "1")
    try:
        seed_everything(123)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        _restore(*old)
        _restore_cublas(old_cublas)


def test_default_does_not_turn_determinism_on(monkeypatch):
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.delenv("AUTOVISION_DETERMINISTIC", raising=False)
    try:
        seed_everything(7)
        assert (
            torch.are_deterministic_algorithms_enabled()
            == old[0]
        )
        assert torch.backends.cudnn.deterministic == old[1]
        assert torch.backends.cudnn.benchmark == old[2]
    finally:
        _restore(*old)


def _restore_math_modes(old_tf32_matmul, old_tf32_cudnn, old_benchmark) -> None:
    torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
    torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
    torch.backends.cudnn.benchmark = old_benchmark


def test_cuda_math_modes_enabled_without_deterministic(monkeypatch):
    """Non-deterministic CUDA runs opt into TF32 + cudnn autotuning."""
    old = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.delenv("AUTOVISION_DETERMINISTIC", raising=False)
    try:
        enable_cuda_math_modes(torch.device("cuda"))
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
    finally:
        _restore_math_modes(*old)


def test_cuda_math_modes_kept_off_under_deterministic(monkeypatch):
    """AUTOVISION_DETERMINISTIC=1 leaves every math-mode flag untouched."""
    old = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv("AUTOVISION_DETERMINISTIC", "1")
    try:
        enable_cuda_math_modes(torch.device("cuda"))
        assert torch.backends.cuda.matmul.allow_tf32 == old[0]
        assert torch.backends.cudnn.allow_tf32 == old[1]
        assert torch.backends.cudnn.benchmark == old[2]
    finally:
        _restore_math_modes(*old)


def test_cuda_math_modes_ignore_non_cuda_devices():
    """CPU/NPU devices never touch the cudnn/cublas switches."""

    class _CpuLike:
        type = "cpu"

    old = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        enable_cuda_math_modes(_CpuLike())
        assert torch.backends.cuda.matmul.allow_tf32 == old[0]
        assert torch.backends.cudnn.allow_tf32 == old[1]
        assert torch.backends.cudnn.benchmark == old[2]
    finally:
        _restore_math_modes(*old)
