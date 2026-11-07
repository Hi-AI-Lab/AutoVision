# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Registry hardening regressions (H3-4g): version-ordering total order.

A stage whose versions mix numeric strings (``1.0.0``) with non-numeric
markers (``rc1``) used to crash ``get(version=None)`` with a TypeError
because the sort key mixed tuple shapes. The fix gives every version a
comparable key and warns once when the ordering becomes mixed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from autovision.registry.model_registry import ModelRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    """Empty filesystem registry under the test tmp dir."""
    return ModelRegistry(tmp_path / "registry")


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    """A minimal (content-arbitrary) model artifact to register."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    return path


def _register(registry: ModelRegistry, model_file: Path, version: str) -> None:
    registry.register(model_file, name="det", version=version, stage="dev")


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(["1.0.0", "rc1"], id="numeric-first"),
        pytest.param(["rc1", "1.0.0"], id="marker-first"),
    ],
)
def test_get_latest_prefers_numeric_version_over_nonnumeric(
    registry: ModelRegistry, model_file: Path, order: list[str]
) -> None:
    """H3-4g: mixed numeric/non-numeric stages resolve latest without TypeError."""
    for version in order:
        _register(registry, model_file, version)
    entry = registry.get(name="det", stage="dev", version=None)
    assert entry.version == "1.0.0"


def test_get_latest_keeps_numeric_semantic_ordering(
    registry: ModelRegistry, model_file: Path
) -> None:
    """Pure numeric stages keep natural ordering (10.0.0 > 9.0.0 > 1.2.0)."""
    for version in ("1.2.0", "10.0.0", "9.0.0"):
        _register(registry, model_file, version)
    entry = registry.get(name="det", stage="dev", version=None)
    assert entry.version == "10.0.0"


def test_get_latest_mixed_stage_warns_once(
    registry: ModelRegistry,
    model_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """H3-4g: the mixed numeric/non-numeric ordering is announced once."""
    for version in ("1.0.0", "rc1", "rc2"):
        _register(registry, model_file, version)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        entry = registry.get(name="det", stage="dev", version=None)
    # The EXPORT-audit P3-20 stability warning fires on fresh fixtures —
    # it is not part of this test's contract.
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "modified less than 2s ago" not in record.getMessage()
    ]
    assert entry.version == "1.0.0"
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "rc1" in message and "rc2" in message


def test_get_latest_nonnumeric_only_stage_keeps_string_order(
    registry: ModelRegistry,
    model_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All-marker stages keep string ordering and stay silent."""
    for version in ("rc1", "rc2"):
        _register(registry, model_file, version)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        entry = registry.get(name="det", stage="dev", version=None)
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "modified less than 2s ago" not in record.getMessage()
    ]
    assert entry.version == "rc2"
    assert warnings == []
