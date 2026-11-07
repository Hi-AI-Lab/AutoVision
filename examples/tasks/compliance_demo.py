#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Compliance subsystem smoke test (``compliance/``).

Exercises the license-normalization + compliance-report API WITHOUT any
trained weights. It shows ``normalize_license`` (SPDX expression / alias
resolution), the ``DEFAULT_ALLOWLIST`` gate, the installed-dependency
inventory, and ``build_compliance_report`` across three states (pass /
model-card sha256 mismatch / missing model file) plus the audit pack written
by ``ComplianceReport.save``.

    python examples/tasks/compliance_demo.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autovision.compliance import (
    build_compliance_report,
)
from autovision.compliance.licenses import (
    DEFAULT_ALLOWLIST,
    dependency_inventory,
    normalize_license,
)
from autovision.encoders.weight_verification import compute_file_sha256
from autovision.registry.model_card import LatencyRecord, ModelCard


def _norm_cases() -> None:
    assert normalize_license("MIT") == "MIT"
    assert normalize_license("Apache 2.0") == "Apache-2.0"
    assert normalize_license("the MIT License") == "MIT"
    assert normalize_license("GPL-3.0-or-later") == "GPL-3.0-or-later"
    assert normalize_license("MIT OR Apache-2.0") == "MIT OR Apache-2.0"
    # Unresolvable atoms collapse the whole expression to UNKNOWN (fail-closed).
    assert normalize_license("some made-up license") == "UNKNOWN"
    assert normalize_license("") == "UNKNOWN"
    print("normalize_license: aliases + SPDX operators + UNKNOWN fail-closed ok")

    assert "MIT" in DEFAULT_ALLOWLIST
    assert "Apache-2.0" in DEFAULT_ALLOWLIST
    assert "GPL-3.0-or-later" not in DEFAULT_ALLOWLIST
    print(f"allowlist: {sorted(DEFAULT_ALLOWLIST)}")


def _deps() -> None:
    entries = dependency_inventory(skip=("autovision-framework",))
    assert entries, "expected an installed-dependency inventory"
    by_name = {entry.name: entry for entry in entries}
    assert "torch" in by_name, "torch should be in the dependency inventory"
    # Every inventory entry carries a normalized license id (never a raw string).
    assert all(isinstance(entry.license, str) for entry in entries)
    print(f"dependency_inventory: {len(entries)} packages "
          f"(torch -> {by_name['torch'].license})")


def _valid_card(artifact: Path, *, sha: str) -> None:
    card = ModelCard(
        model="detr/s",
        name="DETR-S (compliance demo)",
        train_dataset="COCO 2017 train2017",
        schedule="6ep finetune (demo)",
        license="MIT",
        metrics={"coco_map": 0.55},
        latency=[
            LatencyRecord(
                backend="onnx", precision="fp32",
                hardware="NVIDIA T4", batch_size=1, p50_ms=3.0, script="bench.py",
            )
        ],
        weights_sha256={artifact.name: sha},
    )
    card.save(artifact)


def _report_pass(out: Path) -> None:
    artifact = out / "weights_pass.pt"
    artifact.write_bytes(b"compliance demo weights")
    _valid_card(artifact, sha=compute_file_sha256(artifact))

    report = build_compliance_report(
        model=artifact, include_dependencies=False, timestamp=True
    )
    assert report.passed, f"expected PASS, got errors: {report.errors}"
    assert report.framework.get("version"), "framework identity missing"
    assert report.models and report.models[0]["sha256"] == compute_file_sha256(artifact)

    pack_dir = out / "report_pass"
    report.save(pack_dir)
    for name in ("compliance.json", "compliance.md", "compliance.sbom.json"):
        path = pack_dir / name
        assert path.is_file(), f"missing artifact {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        declared = json.loads((pack_dir / "compliance.manifest.json").read_text())[
            "sha256"
        ][name]
        assert actual == declared, f"manifest sha256 mismatch for {name}"
    print(f"report(pass): passed; pack written to {pack_dir} "
          f"(errors={len(report.errors)}, warnings={len(report.warnings)})")


def _report_mismatch(out: Path) -> None:
    artifact = out / "weights_mismatch.pt"
    artifact.write_bytes(b"compliance demo weights (other)")
    # Declare a WRONG sha256 -> model_inventory records a hard error.
    _valid_card(artifact, sha="0" * 64)

    report = build_compliance_report(model=artifact, include_dependencies=False)
    assert not report.passed, "mismatched card sha256 must fail the report"
    assert report.errors, "expected at least one error issue"
    assert report.passes("error") is False
    print(f"report(mismatch): {len(report.errors)} error(s) — "
          f"{report.errors[0].kind}")


def _report_missing() -> None:
    report = build_compliance_report(
        model="does_not_exist.pt", include_dependencies=False
    )
    assert not report.passed
    assert any(issue.kind == "missing_model_file" for issue in report.errors)
    print("report(missing model): missing_model_file error recorded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/examples/compliance_demo")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _norm_cases()
    _deps()
    _report_pass(out)
    _report_mismatch(out)
    _report_missing()

    print("\nOK compliance subsystem (license normalization + report + gate)")


if __name__ == "__main__":
    main()
