# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the model asset registry and downloadable model cache."""

import json

import pytest

from autovision._models.weight_verification import compute_file_sha256
from autovision._registry import (
    DOWNLOADABLE_MODEL_URL_AND_HASH,
    DownloadableModelError,
    ModelRegistry,
    RegistryError,
    register_downloadable,
    resolve_downloadable,
)


@pytest.fixture()
def model_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"model-bytes-" + b"x" * 32)
    return path


# ── registration ────────────────────────────────────────────────────────


class TestRegister:
    def test_creates_manifest_and_layout(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(
            model_file, name="detector", version="1.0.0", metadata={"map": 0.42}
        )
        assert entry.stage == "dev"
        assert entry.sha256 == compute_file_sha256(model_file)
        assert entry.path.is_file()
        manifest = json.loads(
            (entry.path.parent / "manifest.json").read_text()
        )
        assert manifest["name"] == "detector"
        assert manifest["version"] == "1.0.0"
        assert manifest["metadata"] == {"map": 0.42}

    def test_versions_are_immutable(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="immutable"):
            registry.register(model_file, name="detector", version="1.0.0")

    def test_missing_file_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="does not exist"):
            registry.register(tmp_path / "nope.pt", name="m", version="1.0.0")

    def test_invalid_key_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.register(model_file, name="a/b", version="1.0.0")

    def test_unknown_stage_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Unknown stage"):
            registry.register(
                model_file, name="m", version="1.0.0", stage="qa"
            )


# ── promotion gate ──────────────────────────────────────────────────────


class TestPromote:
    def test_single_step_promotion(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        entry = registry.promote(name="detector", version="1.0.0", to_stage="staging")
        assert entry.stage == "staging"
        assert entry.sha256 == compute_file_sha256(model_file)
        entry = registry.promote(name="detector", version="1.0.0", to_stage="prod")
        assert entry.stage == "prod"

    def test_skip_stage_rejected(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="not found in 'staging'"):
            registry.promote(name="detector", version="1.0.0", to_stage="prod")

    def test_promotion_requires_existing_source(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="not found in 'dev'"):
            registry.promote(name="ghost", version="1.0.0", to_stage="staging")

    def test_cannot_promote_to_dev(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="Cannot promote to 'dev'"):
            registry.promote(name="detector", version="1.0.0", to_stage="dev")

    def test_promotion_is_idempotent_guarded(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        registry.promote(name="detector", version="1.0.0", to_stage="staging")
        with pytest.raises(RegistryError, match="immutable"):
            registry.promote(name="detector", version="1.0.0", to_stage="staging")


# ── lookup ──────────────────────────────────────────────────────────────


class TestLookup:
    def test_get_latest_version(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        for version in ("1.0.0", "1.2.0", "1.10.0"):
            registry.register(model_file, name="detector", version=version)
        entry = registry.get(name="detector", stage="dev")
        assert entry.version == "1.10.0"

    def test_get_explicit_version(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        registry.register(model_file, name="detector", version="2.0.0")
        entry = registry.get(name="detector", stage="dev", version="1.0.0")
        assert entry.version == "1.0.0"

    def test_get_missing_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="No versions"):
            registry.get(name="ghost", stage="prod")

    def test_list_entries_filters_by_stage(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="a", version="1.0.0")
        registry.register(model_file, name="b", version="1.0.0")
        registry.promote(name="a", version="1.0.0", to_stage="staging")
        assert {e.name for e in registry.list_entries(stage="dev")} == {"a", "b"}
        assert {e.name for e in registry.list_entries(stage="staging")} == {"a"}
        assert len(registry.list_entries()) == 3

    def test_corrupt_entry_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(model_file, name="detector", version="1.0.0")
        entry.path.unlink()
        with pytest.raises(RegistryError, match="missing"):
            registry.get(name="detector", stage="dev", version="1.0.0")


# ── downloadable models ─────────────────────────────────────────────────


class TestDownloadable:
    def test_unknown_name_raises(self, tmp_path):
        with pytest.raises(DownloadableModelError, match="Unknown downloadable"):
            resolve_downloadable("nope", cache_dir=tmp_path)

    def test_register_and_resolve_local_url(self, tmp_path):
        import hashlib
        import urllib.request

        source = tmp_path / "hosted.pt"
        payload = b"hosted-checkpoint"
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # A file:// URL keeps the download fully offline for the test.
        register_downloadable(
            "test-model", source.as_uri(), digest
        )
        try:
            path = resolve_downloadable("test-model", cache_dir=tmp_path / "cache")
            assert path.read_bytes() == payload
            # Second resolve hits the verified cache.
            again = resolve_downloadable("test-model", cache_dir=tmp_path / "cache")
            assert again == path
        finally:
            DOWNLOADABLE_MODEL_URL_AND_HASH.pop("test-model", None)

    def test_corrupt_cache_is_redownloaded(self, tmp_path):
        import hashlib

        source = tmp_path / "hosted.pt"
        payload = b"hosted-checkpoint"
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "test-corrupt.pt").write_bytes(b"corrupted")
        register_downloadable(
            "test-corrupt", source.as_uri(), digest
        )
        try:
            path = resolve_downloadable("test-corrupt", cache_dir=cache)
            assert path.read_bytes() == payload
        finally:
            DOWNLOADABLE_MODEL_URL_AND_HASH.pop("test-corrupt", None)
