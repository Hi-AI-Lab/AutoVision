# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Deployment gates for the private model registry (roadmap 90d, phase 4).

A gate returns a list of human-readable failure reasons; an empty list is a
pass. Gates read the immutable registry entry (file + sidecars + metadata),
so a promotion cannot pass on a stale source path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol

from autovision.compliance.licenses import DEFAULT_ALLOWLIST, normalize_license
from autovision.registry.model_card import load_model_card
from autovision.registry.model_registry import RegistryEntry, compute_file_sha256

__all__ = [
    "BenchmarkGate",
    "DataSnapshotGate",
    "DeploymentGate",
    "LicenseGate",
    "MetricFloorGate",
    "ModelCardGate",
    "ParityGate",
    "Sha256Gate",
    "default_gates",
    "VerifyStrengthGate",]

# H3-4h: one allowlist. The registry gate and the compliance report used to
# carry two hand-maintained copies that had already drifted (Python-2.0 /
# PSF-2.0 present in one, missing in the other); both now read the
# authoritative table from compliance/licenses. (registry -> compliance is a
# same-rank import: check_layering puts both at rank 5.)
_DEFAULT_ALLOWLIST = DEFAULT_ALLOWLIST


class DeploymentGate(Protocol):
    name: str

    def check(self, entry: RegistryEntry) -> list[str]: ...


class Sha256Gate:
    name = "sha256"

    def check(self, entry: RegistryEntry) -> list[str]:
        if not entry.path.is_file():
            return [f"artifact missing: {entry.path}"]
        actual = compute_file_sha256(entry.path)
        return [] if actual == entry.sha256 else [f"sha256 mismatch ({actual})"]


class ModelCardGate:
    name = "model_card"

    def check(self, entry: RegistryEntry) -> list[str]:
        card_path = Path(str(entry.path) + ".modelcard.json")
        if not card_path.is_file():
            return [f"missing model card: {card_path.name}"]
        try:
            card = load_model_card(entry.path)
            card.validate()
        except Exception as error:  # noqa: BLE001 - card content is the finding
            return [f"invalid model card: {error}"]
        declared = card.weights_sha256.get(entry.path.name)
        if declared and declared != entry.sha256:
            return ["card sha256 does not match registered artifact"]
        if not declared:
            return [f"card does not list '{entry.path.name}' in weights_sha256"]
        return []


class LicenseGate:
    name = "license"

    def __init__(self, allowlist: set[str] | frozenset[str] | None = None) -> None:
        self.allowlist = frozenset(allowlist or _DEFAULT_ALLOWLIST)

    def check(self, entry: RegistryEntry) -> list[str]:
        try:
            card = load_model_card(entry.path)
        except Exception as error:  # noqa: BLE001
            return [f"cannot read model card: {error}"]
        if not card.license:
            return ["model card has no license"]
        # H3-4h: cards may declare a vendor alias ("Apache Software License");
        # judging the raw string would fail-closed on a license the aliases
        # table already maps into the allowlist. UNKNOWN never matches.
        if normalize_license(card.license) not in self.allowlist:
            return [
                f"license '{card.license}' is not in the deployment allowlist"
            ]
        return []


class BenchmarkGate:
    name = "benchmark"

    def check(self, entry: RegistryEntry) -> list[str]:
        try:
            card = load_model_card(entry.path)
        except Exception as error:  # noqa: BLE001
            return [f"cannot read model card: {error}"]
        problems: list[str] = []
        if float(card.metrics.get("coco_map", 0.0)) <= 0.0:
            problems.append("model card has no positive coco_map")
        if not card.latency:
            problems.append("model card has no latency record")
        return problems


def _metadata_path(entry: RegistryEntry, key: str) -> Path | None:
    value = entry.metadata.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else entry.path.parent / path


class ParityGate:
    name = "parity"

    def __init__(self, *, require_report: bool = True) -> None:
        # A self-asserted metadata flag is not evidence: require the report
        # file (and, when recorded, its sha256) by default.
        self.require_report = require_report

    def check(self, entry: RegistryEntry) -> list[str]:
        if not self.require_report and entry.metadata.get("parity_passed") is True:
            return []
        path = _metadata_path(entry, "parity_report")
        if path is None:
            candidate = Path(str(entry.path) + ".parity.json")
            path = candidate if candidate.is_file() else None
        if path is None:
            return ["no parity_report file (or artifact.parity.json sidecar)"]
        if not path.is_file():
            return [f"parity report missing: {path}"]
        expected_sha = entry.metadata.get("parity_report_sha256")
        if expected_sha and compute_file_sha256(path) != expected_sha:
            return ["parity report sha256 does not match registry metadata"]
        try:
            payload = json.loads(path.read_text())
        except Exception as error:  # noqa: BLE001
            return [f"invalid parity report: {error}"]
        return [] if payload.get("passed") is True else ["parity report did not pass"]


class VerifyStrengthGate:
    """EXPORT-02: require artifacts whose numbers were actually verified.

    Opt-in (design 5.2): ``check`` reads the artifact's identity metadata
    (in-graph for ONNX, sidecar for TRT/OV via the same
    :func:`autovision.export.artifact_metadata.read_metadata` the engine
    loader uses) and rejects anything exported with
    ``verify=smoke-only`` — an fp16 export made on a machine without a
    CUDA execution provider, where the parity gate only executed the
    graph. Artifacts without any recorded strength (legacy exports) are
    reported too: the gate is about KNOWING, not about trusting defaults.
    """

    name = "verify_strength"

    def __init__(self, *, require_full: bool = True) -> None:
        self.require_full = require_full

    def check(self, entry: RegistryEntry) -> list[str]:
        from autovision.export.artifact_metadata import read_metadata

        path = entry.path
        if Path(path).suffix.lower() in (".engine", ".trt"):
            backend = "tensorrt"
        elif Path(path).suffix.lower() == ".xml":
            backend = "openvino"
        elif Path(path).suffix.lower() == ".onnx":
            backend = "onnx"
        else:
            return [f"unknown artifact carrier '{Path(path).suffix}'"]
        metadata = read_metadata(path, backend=backend)
        strength = metadata.get("autovision_verify_strength")
        if strength is None:
            return ["artifact records no verify strength (pre-0.6.1 export)"]
        if self.require_full and strength != "full":
            # Review P2-1: "unverified" (verify=False debug exports) joins
            # "smoke-only" on the rejection list.
            return [
                f"verify strength is {strength!r}; full numerical parity "
                "required (re-export with verify=True on a CUDA-enabled "
                "onnxruntime build)"
            ]
        return []


class DataSnapshotGate:
    name = "data_snapshot"

    def check(self, entry: RegistryEntry) -> list[str]:
        path = _metadata_path(entry, "data_snapshot")
        if path is None:
            return ["no data_snapshot in registry metadata"]
        return [] if path.is_file() else [f"data snapshot missing: {path}"]


class MetricFloorGate:
    """Metric floors over the model card (design D5; opt-in).

    The CI form of "this model must not regress": each declared metric
    (``{"coco_map": 0.3}``) becomes a hard floor over the card's recorded
    metrics — a promotion whose card reports a value below its floor fails
    with both numbers in the message. Not part of :func:`default_gates`
    because floors are per-model policy; compose it explicitly for the
    models you gate.
    """

    name = "metric_floor"

    def __init__(self, floors: Mapping[str, float]) -> None:
        self.floors = dict(floors)

    def check(self, entry: RegistryEntry) -> list[str]:
        try:
            card = load_model_card(entry.path)
        except Exception as error:  # noqa: BLE001
            return [f"cannot read model card: {error}"]
        problems: list[str] = []
        for metric, floor in sorted(self.floors.items()):
            recorded = card.metrics.get(metric)
            if recorded is None:
                problems.append(f"model card does not record '{metric}'")
            elif float(recorded) < float(floor):
                problems.append(
                    f"{metric}={float(recorded):.4f} is below the "
                    f"{float(floor):.4f} floor"
                )
        return problems


def default_gates() -> list[DeploymentGate]:
    """The standard production promotion set (no dataset-specific gate)."""
    return [Sha256Gate(), ModelCardGate(), LicenseGate(), BenchmarkGate(), ParityGate()]
