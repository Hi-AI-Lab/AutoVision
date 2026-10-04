# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for verified pretrained-weight loading (supply-chain integrity)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from autovision._models.weight_hashes import WEIGHT_MANIFEST
from autovision._models.weight_verification import (
    WeightArtifact,
    WeightVerificationError,
    _render_manifest,
    compute_file_sha256,
    get_weight_artifact,
    load_pretrained_verified,
)

pytest.importorskip("huggingface_hub")

GOOD_CONTENT = b"weights-bytes"
BAD_CONTENT = b"corrupted-bytes"
GOOD_HASH = hashlib.sha256(GOOD_CONTENT).hexdigest()
BAD_HASH = hashlib.sha256(BAD_CONTENT).hexdigest()


class FakeModelCls:
    """Stands in for a transformers model class."""

    calls: list[Path] = []

    @classmethod
    def from_pretrained(cls, path: Any, **kwargs: Any) -> "FakeModelCls":
        cls.calls.append(Path(path))
        return cls()


def make_artifact(**overrides: Any) -> WeightArtifact:
    defaults = dict(
        repo_id="org/model",
        license="apache-2.0",
        source="huggingface",
        revision=None,
        sha256={"model.safetensors": GOOD_HASH},
    )
    defaults.update(overrides)
    return WeightArtifact(**defaults)


@pytest.fixture()
def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Queue-driven fake snapshot_download: each call writes one content set.

    ``contents`` items are dicts of filename -> bytes, or an Exception to
    raise instead.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[list] = []

    def download(repo_id: str, revision: str | None) -> Path:
        snapshot = tmp_path / "cache" / f"models--{repo_id.replace('/', '--')}" / "snap"
        payload = calls.pop(0)
        if isinstance(payload, Exception):
            raise payload
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename, content in payload.items():
            (snapshot / filename).write_bytes(content)
        return snapshot

    monkeypatch.setattr(
        "autovision._models.weight_verification._download_snapshot",
        lambda repo_id, revision, *a, **k: download(repo_id, revision),
    )
    monkeypatch.setattr(
        "autovision._models.weight_verification._lfs_sha256",
        lambda repo_id, revision: {"model.safetensors": GOOD_HASH},
    )
    monkeypatch.setattr(
        "autovision._models.weight_verification.get_weight_artifact",
        lambda repo_id: make_artifact(),
    )
    calls.append({"config.json": b"{}", "model.safetensors": GOOD_CONTENT})
    return calls


class TestManifest:
    def test_covers_dinov2_registry(self):
        from autovision._models.dinov2 import dinov2_package

        repos = {
            info["hf"] if use_reg else info["hf_noreg"]
            for info in dinov2_package._MODEL_INFO.values()
            for use_reg in (True, False)
        }
        for repo in repos:
            artifact = get_weight_artifact(repo)
            assert artifact is not None, f"missing manifest entry: {repo}"
            assert artifact.license
            assert artifact.source

    def test_covers_dinov3_registry(self):
        from autovision._models.dinov3 import dinov3_package

        for info in dinov3_package._MODEL_INFO.values():
            artifact = get_weight_artifact(info["hf"])
            assert artifact is not None, f"missing manifest entry: {info['hf']}"
            assert artifact.license
            assert artifact.source

    def test_renderer_round_trip(self):
        rendered = _render_manifest(dict(WEIGHT_MANIFEST))
        assert f"'{next(iter(WEIGHT_MANIFEST))}'" in rendered
        # Re-rendering the rendered file's data must be byte-identical.
        namespace: dict[str, Any] = {}
        exec(compile(rendered, "<manifest>", "exec"), namespace)  # noqa: S102
        assert _render_manifest(namespace["WEIGHT_MANIFEST"]) == rendered


class TestSha256:
    def test_known_vector(self, tmp_path: Path):
        path = tmp_path / "blob"
        path.write_bytes(b"abc")
        assert compute_file_sha256(path) == hashlib.sha256(b"abc").hexdigest()

    def test_large_file_chunked(self, tmp_path: Path):
        path = tmp_path / "blob"
        payload = GOOD_CONTENT * 500_000  # ~6 MB: multiple chunks
        path.write_bytes(payload)
        assert compute_file_sha256(path) == hashlib.sha256(payload).hexdigest()


class TestLoadPretrainedVerified:
    def test_happy_path_pinned(self, fake_download):
        calls = fake_download
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert FakeModelCls.calls[-1].name == "snap"
        assert (FakeModelCls.calls[-1] / "model.safetensors").read_bytes() == GOOD_CONTENT
        assert len(calls) == 0  # exactly the one queued download consumed

    def test_mismatch_retries_once_then_succeeds(self, fake_download):
        calls = fake_download
        calls.insert(0, {"config.json": b"{}", "model.safetensors": BAD_CONTENT})
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert len(calls) == 0  # both queued downloads consumed
        final_snapshot = FakeModelCls.calls[-1]
        assert (final_snapshot / "model.safetensors").read_bytes() == GOOD_CONTENT

    def test_double_mismatch_raises_with_guidance(
        self, monkeypatch, fake_download, tmp_path: Path
    ):
        monkeypatch.setattr(
            "autovision._models.weight_verification._download_snapshot",
            lambda *a, **k: _write_snapshot(tmp_path, BAD_CONTENT),
        )
        with pytest.raises(WeightVerificationError, match="sha256 mismatch"):
            load_pretrained_verified(FakeModelCls, "org/model")

    def test_unpinned_uses_lfs_metadata_tier(self, monkeypatch, fake_download):
        calls = fake_download
        monkeypatch.setattr(
            "autovision._models.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(sha256={}),
        )
        lfs_calls: list[str] = []
        monkeypatch.setattr(
            "autovision._models.weight_verification._lfs_sha256",
            lambda repo_id, revision: lfs_calls.append(repo_id)
            or {"model.safetensors": GOOD_HASH},
        )
        load_pretrained_verified(FakeModelCls, "org/model")
        assert lfs_calls == ["org/model"]
        assert len(calls) == 0  # single queued download consumed

    def test_pinned_skips_lfs_metadata(self, monkeypatch, fake_download):
        monkeypatch.setattr(
            "autovision._models.weight_verification._lfs_sha256",
            lambda repo_id, revision: (_ for _ in ()).throw(
                AssertionError("network metadata must not be fetched when pinned")
            ),
        )
        load_pretrained_verified(FakeModelCls, "org/model")

    def test_unknown_repo_loads_unverified_with_warning(
        self, monkeypatch, caplog: pytest.LogCaptureFixture
    ):
        seen: list[Any] = []

        class PlainCls:
            @classmethod
            def from_pretrained(cls, repo: str, **kwargs: Any) -> "PlainCls":
                seen.append(repo)
                return cls()

        with caplog.at_level("WARNING", logger="autovision"):
            load_pretrained_verified(PlainCls, "org/unlisted")
        assert seen == ["org/unlisted"]
        assert "WITHOUT" in caplog.text


class TestOfflineAndMirror:
    def test_offline_uses_cache_without_network(self, monkeypatch, tmp_path: Path):
        snapshot = _write_snapshot(tmp_path, GOOD_CONTENT, write=True)
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        network_attempts: list[tuple] = []

        def fake_download(
            repo_id: str, revision: str | None, local_files_only: bool = False
        ) -> Path:
            # Offline lookups arrive with local_files_only=True; anything
            # else would be a network attempt.
            assert local_files_only, "offline mode must not hit the network"
            network_attempts.append((repo_id, revision, local_files_only))
            return snapshot

        monkeypatch.setattr(
            "autovision._models.weight_verification._download_snapshot",
            fake_download,
        )
        monkeypatch.setattr(
            "autovision._models.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(
                sha256={"model.safetensors": hashlib.sha256(GOOD_CONTENT).hexdigest()}
            ),
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert network_attempts == [("org/model", None, True)]

    def test_offline_missing_cache_actionable_error(
        self, monkeypatch, fake_download, tmp_path: Path
    ):
        from huggingface_hub.errors import LocalEntryNotFoundError

        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        monkeypatch.setattr(
            "autovision._models.weight_verification._download_snapshot",
            lambda *a, **k: (_ for _ in ()).throw(
                LocalEntryNotFoundError("not cached")
            ),
        )
        with pytest.raises(RuntimeError, match="AUTOVISION_OFFLINE"):
            load_pretrained_verified(FakeModelCls, "org/model")

    def test_offline_unpinned_skips_verification(self, monkeypatch, tmp_path: Path):
        snapshot = _write_snapshot(tmp_path, BAD_CONTENT, write=True)
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        monkeypatch.setattr(
            "autovision._models.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(sha256={}),
        )
        monkeypatch.setattr(
            "autovision._models.weight_verification._download_snapshot",
            lambda *a, **k: snapshot,
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)  # no hash check, loads anyway

    def test_mirror_url_bridges_to_hf_endpoint(
        self, monkeypatch, fake_download
    ):
        monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://mirror.example/hf")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        load_pretrained_verified(FakeModelCls, "org/model")
        import os

        assert os.environ["HF_ENDPOINT"] == "https://mirror.example/hf"

    def test_existing_hf_endpoint_wins(self, monkeypatch, fake_download):
        import os

        monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://mirror.example/hf")
        monkeypatch.setenv("HF_ENDPOINT", "https://explicit.example")
        load_pretrained_verified(FakeModelCls, "org/model")
        assert os.environ["HF_ENDPOINT"] == "https://explicit.example"


def _write_snapshot(root: Path, content: bytes, write: bool = True) -> Path:
    snapshot = root / "cache" / "models--org--model" / "snap"
    if write:
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b"{}")
        (snapshot / "model.safetensors").write_bytes(content)
    return snapshot
