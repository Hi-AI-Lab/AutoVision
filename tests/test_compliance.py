# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Compliance pack: license normalization, model card/sha, snapshot, CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.compliance import build_compliance_report, normalize_license
from autovision.registry.model_card import LatencyRecord, ModelCard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_with_card(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not-a-real-checkpoint" * 16)
    card = ModelCard(
        model="detr/s",
        name="Demo detector",
        train_dataset="COCO 2017 train2017",
        schedule="1 step smoke",
        seed=0,
        metrics={"coco_map": 50.0},
        latency=[
            LatencyRecord(
                backend="onnxruntime",
                precision="fp32",
                hardware="CPU",
                batch_size=1,
                p50_ms=10.0,
                script="scripts/measure_spectrum_benchmarks.py",
            )
        ],
        license="Apache-2.0",
        weights_sha256={path.name: _sha(path)},
        version="demo-v1",
    )
    card.save(path)
    return path


def test_license_normalization():
    assert normalize_license("Apache Software License") == "Apache-2.0"
    assert normalize_license("MIT License") == "MIT"
    assert normalize_license("BSD 3-Clause") == "BSD-3-Clause"
    assert normalize_license("some custom terms") == "UNKNOWN"
    assert normalize_license(None) == "UNKNOWN"


def test_report_with_model_card_and_snapshot(tmp_path):
    model = _model_with_card(tmp_path)
    snapshot = tmp_path / "data.snap.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "splits": []}))
    report = build_compliance_report(
        model=model,
        snapshot=snapshot,
        include_dependencies=False,
        timestamp=False,
    )
    assert report.passed
    assert report.generated_at is None
    assert report.models[0]["license"] == "Apache-2.0"
    assert report.models[0]["sha256"] == _sha(model)
    assert report.datasets[0]["snapshot"]["schema_version"] == 1
    report.save(tmp_path / "compliance")
    payload = json.loads((tmp_path / "compliance" / "compliance.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert (tmp_path / "compliance" / "compliance.md").is_file()


def test_missing_model_card_is_an_error(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_bytes(b"bare")
    report = build_compliance_report(
        model=model, include_dependencies=False, timestamp=False
    )
    assert not report.passed
    assert any(issue.kind == "missing_model_card" for issue in report.issues)
    assert any(issue.kind == "unknown_model_license" for issue in report.issues)


def test_cli_compliance_report_and_strict(tmp_path):
    good = _model_with_card(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compliance-report", f"model={good}", f"out={tmp_path / 'ok'}",
            "include_dependencies=false", "no_timestamp=true",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ok" / "compliance.json").is_file()

    bad = tmp_path / "bare.pt"
    bad.write_bytes(b"bare")
    result = runner.invoke(
        app,
        [
            "compliance-report", f"model={bad}", f"out={tmp_path / 'bad'}",
            "include_dependencies=false", "strict=true",
        ],
    )
    assert result.exit_code == 1
