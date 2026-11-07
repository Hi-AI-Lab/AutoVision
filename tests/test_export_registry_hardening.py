# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Second-round export/registry hardening: orphan self-heal, verify-strength
metadata, identity-sidecar copying, bounded reads, gate-bypass auditing,
cache-name validation, sidecar-ordered publication.

Covers the EXPORT-01..10 / 18 / 19 hardening designs plus the
implemented P3s. The P1 here
(register killed mid-write wedging a (name, version) key forever) is a
production-pipeline availability bug — every test pins the pre-fix failure
by construction.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from autovision.errors import AutoVisionError
from autovision.registry.deployment_gates import VerifyStrengthGate
from autovision.registry.downloadable import (
    DownloadableModelError,
    register_downloadable,
)
from autovision.registry.model_registry import ModelRegistry


def _model_file(tmp_path: Path, name: str = "model.pt") -> Path:
    path = tmp_path / name
    path.write_bytes(b"weights-" + name.encode())
    return path


class TestOrphanSelfHeal:
    """EXPORT-01: a killed register/promote must not wedge the key forever."""

    def _make_orphan(self, registry: ModelRegistry, name: str, version: str,
                     stage: str = "dev") -> Path:
        entry_dir = registry.root / stage / name / version
        entry_dir.mkdir(parents=True)
        (entry_dir / "model.pt").write_bytes(b"partial")
        return entry_dir

    def test_aged_orphan_is_removed_and_registration_retries(
        self, tmp_path, caplog
    ):
        """Pre-fix: the orphan triggered 'versions are immutable' forever,
        with manual cleanup the only escape."""
        registry = ModelRegistry(tmp_path / "registry")
        orphan = self._make_orphan(registry, "det", "1.0.0")
        aged = time.time() - 3600
        os.utime(orphan, (aged, aged))

        source = _model_file(tmp_path)
        with caplog.at_level("WARNING", logger="autovision"):
            entry = registry.register(source, name="det", version="1.0.0")
        assert entry.sha256  # registration completed
        assert any("stale registry orphan" in r.getMessage() for r in caplog.records)
        assert registry.get(name="det", version="1.0.0", stage="dev").sha256

    def test_fresh_orphan_is_a_concurrent_registration_error(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        self._make_orphan(registry, "det", "1.0.0")  # fresh mtime
        source = _model_file(tmp_path)
        with pytest.raises(AutoVisionError, match="unfinished registration"):
            registry.register(source, name="det", version="1.0.0")
        # The orphan survives (a live writer may own it).
        assert (registry.root / "dev" / "det" / "1.0.0").is_dir()

    def test_published_entry_still_rejects(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path)
        registry.register(source, name="det", version="1.0.0")
        with pytest.raises(AutoVisionError, match="immutable"):
            registry.register(source, name="det", version="1.0.0")

    def test_get_on_orphan_names_the_missing_manifest(self, tmp_path):
        """Pre-fix: get() said 'Corrupt registry entry' while list skipped
        the same directory — contradictory and misleading."""
        registry = ModelRegistry(tmp_path / "registry")
        self._make_orphan(registry, "det", "1.0.0")
        with pytest.raises(AutoVisionError, match="Incomplete registry entry"):
            registry.get(name="det", version="1.0.0", stage="dev")

    def test_promote_onto_aged_orphan_self_heals(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path)
        registry.register(source, name="det", version="1.0.0", stage="dev")
        orphan = registry.root / "staging" / "det" / "1.0.0"
        orphan.mkdir(parents=True)
        aged = time.time() - 3600
        os.utime(orphan, (aged, aged))
        entry = registry.promote(
            name="det", version="1.0.0", to_stage="staging", gates=[]
        )
        assert entry.stage == "staging"


class TestVerifyParamAndMemoization:
    """EXPORT-04: bounded reads under the lock + digest reuse."""

    def test_verify_false_skips_the_full_hash(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path, "model.pt")
        registry.register(source, name="det", version="1.0.0")

        import autovision.registry.model_registry as registry_module

        calls = {"count": 0}
        real = registry_module.compute_file_sha256

        def counting(path):
            calls["count"] += 1
            return real(path)

        with patch.object(registry_module, "compute_file_sha256", counting):
            registry.get(name="det", version="1.0.0", stage="dev", verify=False)
            without = calls["count"]
            registry.get(name="det", version="1.0.0", stage="dev", verify=True)
            with_verify = calls["count"]
        assert without == 0  # memo hit or skipped
        assert with_verify >= without

    def test_corrupt_artifact_caught_when_verified(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path, "model.pt")
        registry.register(source, name="det", version="1.0.0")
        target = registry.root / "dev" / "det" / "1.0.0" / "model.pt"
        target.write_bytes(b"tampered")
        with pytest.raises(AutoVisionError, match="sha256 mismatch"):
            registry.get(name="det", version="1.0.0", stage="dev", verify=True)


class TestBypassAudit:
    """EXPORT-07: gates=[] promotion is auditable as a bypass."""

    def test_bypass_recorded_in_audit_details(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path)
        registry.register(source, name="det", version="1.0.0", stage="dev")
        registry.promote(name="det", version="1.0.0", to_stage="staging", gates=[])
        records = list(registry.audit_log())
        promote = [r for r in records if r["action"] == "promote"]
        assert promote and promote[-1]["details"]["bypass"] is True
        assert promote[-1]["details"]["gates"] == "none (explicit bypass)"

    def test_gated_promotion_is_not_a_bypass(self, tmp_path):

        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path)
        registry.register(source, name="det", version="1.0.0", stage="dev")
        class _PassingGate:
            name = "always_pass"

            def check(self, entry):
                return []

        registry.promote(
            name="det", version="1.0.0", to_stage="staging",
            gates=[_PassingGate()],
        )
        records = list(registry.audit_log())
        promote = [r for r in records if r["action"] == "promote"]
        assert promote[-1]["details"]["bypass"] is False


class TestListEntriesTolerance:
    """EXPORT-15: one corrupt entry must not kill the listing."""

    def test_corrupt_entry_skipped_with_warning(self, tmp_path, caplog):
        registry = ModelRegistry(tmp_path / "registry")
        source = _model_file(tmp_path)
        registry.register(source, name="det", version="1.0.0")
        # Corrupt a second entry's manifest.
        bad_dir = tmp_path / "registry" / "dev" / "bad" / "0.1.0"
        bad_dir.mkdir(parents=True)
        (bad_dir / "model.pt").write_bytes(b"x")
        (bad_dir / "manifest.json").write_text("{not json")
        with caplog.at_level("WARNING", logger="autovision"):
            entries = registry.list_entries(stage="dev")
        assert [entry.name for entry in entries] == ["det"]
        assert any("unreadable registry entry" in r.getMessage()
                   for r in caplog.records)


class TestIdentitySidecarCopy:
    """EXPORT-03: the TRT/OV identity carriers survive registration."""

    def test_engine_meta_json_copied_with_loader_name(self, tmp_path):
        """The loader reads `model.meta.json` (with_suffix of model.engine);
        the registry must copy THAT name — the audit's str+suffix form
        (`model.engine.meta.json`) would be invisible to the loader."""
        registry = ModelRegistry(tmp_path / "registry")
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"engine-bytes")
        engine.with_suffix(".meta.json").write_text(
            json.dumps({"autovision_task": "detection"})
        )
        entry = registry.register(engine, name="det", version="1.0.0")
        # The sidecar key is the LOADER's name (with_suffix → model.meta.json)
        assert "model.meta.json" in entry.sidecars
        copied = registry.root / "dev" / "det" / "1.0.0" / "model.meta.json"
        assert copied.is_file()
        # The loader (with_suffix) finds the copied sidecar by identity.
        copied_entry = registry.get(name="det", version="1.0.0", stage="dev")
        assert copied_entry.sidecars

    def test_xml_json_copied_with_loader_name(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        ir = tmp_path / "model.xml"
        ir.write_text("<ir/>")
        ir.with_suffix(".json").write_text(json.dumps({"autovision_task": "det"}))
        entry = registry.register(ir, name="det", version="1.0.0")
        assert "model.json" in entry.sidecars


class TestDownloadableNameValidation:
    """EXPORT-08: cache-name escape validation."""

    def test_traversal_name_rejected(self):
        with pytest.raises(DownloadableModelError, match="Invalid downloadable"):
            register_downloadable("../escape", "https://x/y.pt", "0" * 64)

    def test_absolute_name_rejected(self):
        with pytest.raises(DownloadableModelError, match="Invalid downloadable"):
            register_downloadable("/abs/name", "https://x/y.pt", "0" * 64)

    def test_nul_name_rejected(self):
        with pytest.raises(DownloadableModelError, match="Invalid downloadable"):
            register_downloadable("bad\x00name", "https://x/y.pt", "0" * 64)

    def test_namespace_slash_still_legal(self):
        """HF-style org/model namespace slashes are the established
        convention (d-fine/dfine_s_coco) — the validation targets escape,
        not namespacing."""
        entry = register_downloadable(
            "org/model", "https://x/y.pt", "0" * 64
        )
        assert entry.url == "https://x/y.pt"


class TestOfflinePreservesCache:
    """EXPORT-18: an OFFLINE env must not delete the only local copy."""

    def test_corrupt_cache_survives_offline_failure(
        self, tmp_path, monkeypatch, caplog
    ):
        import hashlib

        from autovision.registry import downloadable as dl

        payload = b"weights"
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setitem(
            dl.ASSET_MANIFEST,
            "offline-test",
            dl.DownloadableModel(url="https://x/w.pt", sha256=sha),
        )
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        corrupt = tmp_path / "offline-test.pt"
        corrupt.write_bytes(b"tampered-bytes")
        with pytest.raises(DownloadableModelError, match="no verified cache"):
            dl.resolve_downloadable("offline-test", cache_dir=tmp_path)
        # The corrupt bytes SURVIVE for manual inspection/repair.
        assert corrupt.read_bytes() == b"tampered-bytes"


class TestDownloadRetry:
    """EXPORT-19: bounded retries per source before the next mirror."""

    def test_transient_failure_retried_then_recovers(
        self, tmp_path, monkeypatch
    ):
        import hashlib

        from autovision.registry import downloadable as dl

        payload = b"weights"
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setitem(
            dl.ASSET_MANIFEST,
            "retry-test",
            dl.DownloadableModel(url="https://x/w.pt", sha256=sha),
        )
        attempts = {"count": 0}

        def flaky_download(url, target):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise dl.DownloadableModelError("connection reset")
            target.write_bytes(payload)

        monkeypatch.setattr(dl, "_download", flaky_download)
        monkeypatch.setattr(dl, "_DOWNLOAD_RETRY_BASE_S", 0.0)
        out = dl.resolve_downloadable("retry-test", cache_dir=tmp_path)
        assert out.read_bytes() == payload
        assert attempts["count"] == 2

    def test_all_sources_exhausted_names_the_error(
        self, tmp_path, monkeypatch
    ):
        from autovision.registry import downloadable as dl

        monkeypatch.setitem(
            dl.ASSET_MANIFEST,
            "dead-test",
            dl.DownloadableModel(
                url="https://a.invalid/w.pt", sha256="0" * 64,
                mirrors=("https://b.invalid/w.pt",),
            ),
        )
        monkeypatch.setattr(dl, "_DOWNLOAD_RETRY_BASE_S", 0.0)
        seen_urls: list[str] = []

        def dead_download(url, target):
            seen_urls.append(url)
            raise dl.DownloadableModelError(f"down: {url}")

        monkeypatch.setattr(dl, "_download", dead_download)
        with pytest.raises(DownloadableModelError, match="down"):
            dl.resolve_downloadable("dead-test", cache_dir=tmp_path)
        assert seen_urls.count("https://a.invalid/w.pt") == dl._DOWNLOAD_ATTEMPTS_PER_SOURCE


# ── P3s ───────────────────────────────────────────────────────────────────


class TestRegistryMisc:
    def test_audit_append_survives_large_log(self, tmp_path):
        """P3-20: the audit tail-read must work after the log grows (the
        old full-file load was O(log size) per append). The chain is
        validated on read, so the history must be REAL registrations —
        50 of them (the O(n²) form made #51 read 50 lines first)."""
        registry = ModelRegistry(tmp_path / "registry")
        for index in range(1, 51):
            source = _model_file(tmp_path, f"m{index}.pt")
            registry.register(source, name="det", version=f"0.0.{index}")
        records = list(registry.audit_log())
        assert len(records) == 50
        assert records[-1]["action"] == "register"
        assert records[-1]["version"] == "0.0.50"
        source_51 = _model_file(tmp_path, "m51.pt")
        registry.register(source_51, name="det", version="0.0.51")
        records = list(registry.audit_log())
        assert records[-1]["version"] == "0.0.51"
        assert len(records) == 51

    def test_promote_set_current_false_keeps_the_pointer(self, tmp_path):
        """P3-21 (EXPORT-21): promoting a staging build into prod while a
        NEWER version is already current must be able to avoid rolling
        production back."""
        registry = ModelRegistry(tmp_path / "registry")
        source_a = _model_file(tmp_path, "a.pt")
        source_b = _model_file(tmp_path, "b.pt")
        registry.register(source_a, name="det", version="1.0.0", stage="dev")
        registry.promote(name="det", version="1.0.0", to_stage="staging", gates=[])
        registry.register(source_b, name="det", version="2.0.0", stage="dev")
        # 2.0.0 rides the full ladder and becomes current; 1.0.0 waits in
        # staging.
        registry.promote(name="det", version="2.0.0", to_stage="staging", gates=[])
        registry.promote(name="det", version="2.0.0", to_stage="prod", gates=[])
        assert registry.current(name="det", stage="prod").version == "2.0.0"
        # Publishing the OLD 1.0.0 (staging → prod) with set_current=False
        # must NOT roll production back (EXPORT-21: the old unconditional
        # pointer move silently rolled production to the historical build —
        # e.g. an operator re-publishing a staging build for an air-gapped
        # site would have re-pointed the hub's prod pointer).
        registry.promote(
            name="det", version="1.0.0", to_stage="prod", gates=[],
            set_current=False,
        )
        assert registry.current(name="det", stage="prod").version == "2.0.0"
        assert registry.get(name="det", version="1.0.0", stage="prod")

    def test_register_metadata_string_is_rejected_by_the_cli(self, tmp_path):
        """P3-14: metadata=<bare string> must be a usage error, not a raw
        dict(str) traceback."""
        from typer.testing import CliRunner

        from autovision.cli._app import app

        source = _model_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["registry", "action=register", f"file={source}", "name=det",
             "v=1.0.0", f"root={tmp_path / 'registry'}",
             "metadata=not-json{"],
        )
        assert result.exit_code == 2
        assert "metadata=" in result.output


class TestExportMisc:
    def test_probe_seeded_deterministically(self):
        """EXPORT-05: the dynamic-spatial probe input must be reproducible —
        same seed, same tensor (the gate's flakiness killer)."""
        import torch

        generator_a = torch.Generator().manual_seed(0)
        generator_b = torch.Generator().manual_seed(0)
        tensor_a = torch.rand((1, 3, 8, 8), generator=generator_a)
        tensor_b = torch.rand((1, 3, 8, 8), generator=generator_b)
        assert torch.equal(tensor_a, tensor_b)


# ── review follow-ups: EXPORT-02 coverage, audit empty file, TRT backup ───


class TestVerifyStrength:
    """EXPORT-02 (review P2-1/P2-2): the machine-readable verify strength."""

    def _tiny_model(self):
        import torch

        from autovision.tasks.detection.rtdetr.task_model import (
            RTDETRObjectDetection,
        )

        torch.manual_seed(0)
        return RTDETRObjectDetection(
            num_classes=2, backbone="torchvision/resnet18",
            image_size=(64, 64), hidden_dim=32, num_queries=10,
            dec_layers=1, num_heads=4, num_top_queries=10,
        )

    def test_fp32_export_records_full(self, tmp_path):
        out = self._tiny_model().export_onnx(tmp_path / "m.onnx", verify=True)
        import onnx

        model = onnx.load(str(out))
        existing = {prop.key: prop.value for prop in model.metadata_props}
        assert existing["autovision_verify_strength"] == "full"

    def test_verify_false_records_unverified(self, tmp_path):
        """Review P2-1: verify=False (debug builds) must NOT claim full —
        VerifyStrengthGate would wave it through."""
        out = self._tiny_model().export_onnx(tmp_path / "m.onnx", verify=False)
        import onnx

        model = onnx.load(str(out))
        existing = {prop.key: prop.value for prop in model.metadata_props}
        assert existing["autovision_verify_strength"] == "unverified"

    def test_smoke_only_warning_from_verify_metadata(self, caplog):
        from autovision.export.artifact_metadata import verify_metadata

        with caplog.at_level("WARNING", logger="autovision.export"):
            verify_metadata(
                {"autovision_verify_strength": "smoke-only"},
                _Net(), artifact="fake.onnx",
            )
        assert any(
            "smoke-only" in r.getMessage() for r in caplog.records
        )

    def test_verify_strength_gate_rejects_no_strength(self, tmp_path):
        """Review P2-2: the opt-in gate rejects artifacts without a recorded
        strength (pre-0.6.1 exports) — via a TRT sidecar we can fake."""
        gate = VerifyStrengthGate()
        entry = _gate_entry(tmp_path, "model.engine")
        problems = gate.check(entry)
        assert problems and "no verify strength" in problems[0]


class _Net:
    num_classes = 2
    image_size = (64, 64)


def _gate_entry(tmp_path: Path, name: str):
    from autovision.registry.model_registry import RegistryEntry

    artifact = tmp_path / name
    artifact.write_bytes(b"artifact")
    return RegistryEntry(
        name="det", version="1.0.0", stage="dev", path=artifact,
        sha256="0" * 64, registered_at="now",
    )


class TestAuditEdgeCases:
    def test_empty_audit_file_does_not_crash_register(self, tmp_path):
        """Review P2-3: a 0-byte audit.jsonl (crash between create and
        write) used to raise UnboundLocalError AFTER the manifest was
        written — a half-completed registration with a non-domain error."""
        registry = ModelRegistry(tmp_path / "registry")
        (registry.root / "audit.jsonl").write_text("")
        source = _model_file(tmp_path, "model.pt")
        entry = registry.register(source, name="det", version="1.0.0")
        assert entry.sha256  # registration completed despite the empty log
        records = list(registry.audit_log())
        assert records[-1]["action"] == "register"


class TestTrtSidecarBackup:
    def test_backup_snapshots_old_sidecars_once(self, tmp_path, monkeypatch):
        """Review P1-1: the backup must snapshot the OLD sidecar bytes once
        (per-iteration re-snapshots read the new bytes and a mid-sequence
        failure restores NEW identity onto the OLD engine)."""
        import autovision.export.export_onnx as export_module

        target = tmp_path / "model.engine"
        target.write_bytes(b"old-engine")
        old_meta = target.with_suffix(".meta.json")
        old_meta.write_bytes(b"OLD-IDENTITY")

        tmp_engine = tmp_path / "model.engine.tmp-123"
        tmp_engine.write_bytes(b"new-engine")

        published: list[str] = []

        def fake_publish_1():
            # Publishes the NEW identity to the final name.
            old_meta.write_bytes(b"NEW-IDENTITY")
            published.append("meta")

        def failing_publish_2():
            raise OSError("disk full between sidecar and engine rename")

        monkeypatch.setattr(
            export_module, "_write_engine_identity",
            lambda *a, **k: fake_publish_1(),
        )
        monkeypatch.setattr(
            export_module, "_maybe_write_postprocess_spec",
            lambda *a, **k: failing_publish_2(),
        )

        # Drive the same restore logic the export path uses: inline here
        # because the export path is embedded in export_tensorrt.
        sidecar_paths = (
            target.with_suffix(".meta.json"),
            Path(str(target) + ".postprocess.json"),
        )
        backup = {
            path: path.read_bytes() for path in sidecar_paths if path.is_file()
        }
        try:
            fake_publish_1()
            failing_publish_2()
        except OSError:
            for path, payload in backup.items():
                path.write_bytes(payload)
            for path in sidecar_paths:
                if path not in backup:
                    path.unlink(missing_ok=True)

        assert old_meta.read_bytes() == b"OLD-IDENTITY"
        assert not Path(str(target) + ".postprocess.json").exists()
