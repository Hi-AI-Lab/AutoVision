# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Private registry: promotion gates, current pointer, rollback, audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.registry import (
    BenchmarkGate,
    LicenseGate,
    ModelCardGate,
    ModelRegistry,
    ParityGate,
    RegistryError,
    Sha256Gate,
)
from autovision.registry.model_card import LatencyRecord, ModelCard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(tmp_path: Path, name: str = "model.pt", license_id: str = "Apache-2.0") -> Path:
    path = tmp_path / name
    path.write_bytes((name + "-bytes").encode() * 8)
    card = ModelCard(
        model="detr/s",
        name="Registry demo",
        train_dataset="COCO 2017 train2017",
        schedule="1 step",
        seed=0,
        metrics={"coco_map": 50.0},
        latency=[
            LatencyRecord(
                backend="onnxruntime", precision="fp32", hardware="CPU",
                batch_size=1, p50_ms=10.0, script="scripts/measure_spectrum_benchmarks.py",
            )
        ],
        license=license_id,
        weights_sha256={path.name: _sha(path)},
    )
    card.save(path)
    Path(str(path) + ".parity.json").write_text(json.dumps({"passed": True}))
    return path


def _gates():
    return [Sha256Gate(), ModelCardGate(), LicenseGate(), BenchmarkGate(), ParityGate()]


def test_parity_gate_requires_the_report_file(tmp_path):
    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"plain-model")
    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(plain, name="plain", version="1.0.0")
    problems = ParityGate().check(entry)
    assert problems and "no parity_report" in problems[0]


def test_parity_gate_accepts_legacy_flag_only_when_explicitly_allowed(tmp_path):
    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"plain-model")
    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(
        plain, name="plain", version="1.0.0", metadata={"parity_passed": True}
    )
    assert ParityGate(require_report=False).check(entry) == []
    assert ParityGate().check(entry)


def test_promotion_current_and_audit(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model = _model(tmp_path)
    registry.register(
        model, name="det", version="1.0.0", metadata={"parity_passed": True}
    )
    assert registry.current(name="det", stage="dev").version == "1.0.0"
    staging = registry.promote(
        name="det", version="1.0.0", to_stage="staging", gates=_gates()
    )
    assert staging.stage == "staging"
    assert registry.current(name="det", stage="staging").version == "1.0.0"
    registry.promote(name="det", version="1.0.0", to_stage="prod", gates=_gates())
    assert registry.current(name="det", stage="prod").version == "1.0.0"
    actions = [record["action"] for record in registry.audit_log()]
    assert actions == ["register", "promote", "promote"]


def test_rollback_to_previous_version(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    first = _model(tmp_path, "first.pt")
    second = _model(tmp_path, "second.pt")
    registry.register(first, name="det", version="1.0.0", metadata={"parity_passed": True})
    registry.promote(name="det", version="1.0.0", to_stage="staging", gates=_gates())
    registry.promote(name="det", version="1.0.0", to_stage="prod", gates=_gates())
    registry.register(second, name="det", version="2.0.0", metadata={"parity_passed": True})
    registry.promote(name="det", version="2.0.0", to_stage="staging", gates=_gates())
    registry.promote(name="det", version="2.0.0", to_stage="prod", gates=_gates())
    assert registry.current(name="det", stage="prod").version == "2.0.0"
    rolled = registry.rollback(
        name="det", stage="prod", to_version="1.0.0", reason="regression"
    )
    assert rolled.version == "1.0.0"
    assert registry.current(name="det", stage="prod").version == "1.0.0"
    assert registry.audit_log()[-1]["action"] == "rollback"


def test_gate_failure_removes_target(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model = _model(tmp_path)
    registry.register(model, name="det", version="1.0.0")

    class FailingGate:
        name = "always-fail"

        def check(self, entry):
            return ["intentional gate failure"]

    with pytest.raises(RegistryError, match="Deployment gates failed"):
        registry.promote(
            name="det", version="1.0.0", to_stage="staging",
            gates=[FailingGate()],
        )
    assert not registry.exists(name="det", version="1.0.0", stage="staging")


def test_missing_card_and_license_gate(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    bare = tmp_path / "bare.pt"
    bare.write_bytes(b"bare")
    registry.register(bare, name="bare", version="1.0.0")
    with pytest.raises(RegistryError, match="model card"):
        registry.promote(
            name="bare", version="1.0.0", to_stage="staging",
            gates=[ModelCardGate()],
        )

    gpl = _model(tmp_path, "gpl.pt", license_id="GPL-3.0")
    registry.register(gpl, name="gpl", version="1.0.0")
    with pytest.raises(RegistryError, match="license"):
        registry.promote(
            name="gpl", version="1.0.0", to_stage="staging",
            gates=[LicenseGate()],
        )


def test_cli_registry_register_promote_current(tmp_path):
    registry_root = tmp_path / "registry"
    model = _model(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "registry", "action=register", f"file={model}", "name=det",
            "v=1.0.0", f"root={registry_root}",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "registry", "action=promote", "name=det", "v=1.0.0",
            "to=staging", "gates=none", f"root={registry_root}",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["registry", "action=current", "name=det", "stage=staging", f"root={registry_root}"],
    )
    assert result.exit_code == 0, result.output
    assert "1.0.0" in result.output


# ── MetricFloorGate (design D5, opt-in) ─────────────────────────────────


def test_metric_floor_gate_passes_when_metrics_clear_the_floor(tmp_path):
    """A card recording 50 mAP passes a 40 mAP floor."""
    from autovision.registry import MetricFloorGate

    model = _model(tmp_path)
    entry = ModelRegistry(tmp_path / "registry").register(
        model, name="det", version="1.0.0"
    )

    assert MetricFloorGate({"coco_map": 40.0}).check(entry) == []


def test_metric_floor_gate_fails_below_the_floor(tmp_path):
    """A regression below the floor fails naming both numbers."""
    from autovision.registry import MetricFloorGate

    model = _model(tmp_path)
    entry = ModelRegistry(tmp_path / "registry").register(
        model, name="det", version="1.0.0"
    )

    problems = MetricFloorGate({"coco_map": 60.0}).check(entry)

    assert problems and "50.0000" in problems[0] and "60.0000" in problems[0]


def test_metric_floor_gate_requires_recorded_metrics(tmp_path):
    """A metric the card never recorded is itself a gate failure."""
    from autovision.registry import MetricFloorGate

    model = _model(tmp_path)
    entry = ModelRegistry(tmp_path / "registry").register(
        model, name="det", version="1.0.0"
    )

    problems = MetricFloorGate({"aabb_map": 10.0}).check(entry)

    assert problems == ["model card does not record 'aabb_map'"]


# ── H3-4h: single-source allowlist + normalized model-card licenses ──────


def test_license_allowlist_is_the_compliance_single_source():
    """H3-4h: the registry gate reads compliance/licenses' table — the two
    hand-maintained copies had drifted (Python-2.0 / PSF-2.0)."""
    from autovision.compliance.licenses import DEFAULT_ALLOWLIST
    from autovision.registry.deployment_gates import _DEFAULT_ALLOWLIST

    assert _DEFAULT_ALLOWLIST is DEFAULT_ALLOWLIST


@pytest.mark.parametrize(
    "license_id",
    [
        pytest.param("MPL-2.0", id="mpl-2.0"),
        pytest.param("Python-2.0", id="python-2.0"),
        pytest.param("PSF-2.0", id="psf-2.0"),
        pytest.param("Apache Software License", id="vendor-alias"),
    ],
)
def test_license_gate_judges_edge_licenses_like_the_compliance_report(
    tmp_path, license_id
):
    """H3-4h: gate and report agree on edge licenses — previously MPL-2.0
    passed one allowlist but drifted out of the other, and vendor aliases
    were judged on the raw string."""
    model = _model(tmp_path, "edge.pt", license_id=license_id)
    entry = ModelRegistry(tmp_path / "registry").register(
        model, name="edge", version="1.0.0"
    )
    gate_problems = LicenseGate().check(entry)

    from autovision.compliance import build_compliance_report

    report = build_compliance_report(
        model=model, include_dependencies=False, timestamp=False
    )
    review_issues = [
        issue for issue in report.issues if issue.kind == "model_license_review"
    ]

    assert gate_problems == []
    assert review_issues == []


def test_license_gate_still_rejects_unknown_licenses(tmp_path):
    """H3-4h: normalization must not widen the gate — UNKNOWN and copyleft
    stay outside the allowlist."""
    gpl = _model(tmp_path, "gpl.pt", license_id="GPL-3.0")
    entry = ModelRegistry(tmp_path / "registry").register(
        gpl, name="gpl", version="1.0.0"
    )
    problems = LicenseGate().check(entry)

    assert problems and "not in the deployment allowlist" in problems[0]
