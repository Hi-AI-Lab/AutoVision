# Copyright (c) 2026 Autovision contributors. Apache-2.0.
import pytest

from autovision.runtime.env import Env


class TestDefaults:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AUTOVISION_LOG_LEVEL", raising=False)
        assert Env.LOG_LEVEL.value == "INFO"
        assert Env.CACHE_DIR.value.name == "autovision-cache"  # from conftest
        assert Env.MODEL_CACHE_DIR.value == Env.CACHE_DIR.value / "models"
        assert Env.DATA_CACHE_DIR.value == Env.TMP_DIR.value
        assert Env.DATALOADER_TIMEOUT_SEC.value == 180.0
        assert Env.MAX_NUM_WORKERS_AUTO.value == 8
        assert Env.MMAP_TIMEOUT_SEC.value == 300.0
        assert Env.MMAP_REUSE_FILE.value is False
        assert Env.OFFLINE.value is False
        assert Env.TELEMETRY.value is False  # compliance: opt-in only
        assert Env.MIRROR_URL.value is None

    def test_values_snapshot_covers_all_vars(self):
        values = Env.values()
        assert values["AUTOVISION_LOG_LEVEL"] == "INFO"
        assert "AUTOVISION_TELEMETRY" in values


class TestParsing:
    def test_bool_lenient(self, monkeypatch):
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("", False),  # empty string falls back to default -> False
        ]:
            monkeypatch.setenv("AUTOVISION_MMAP_REUSE_FILE", raw)
            assert Env.MMAP_REUSE_FILE.value is expected, raw

    def test_empty_string_converts_to_default(self, monkeypatch):
        monkeypatch.setenv("AUTOVISION_LOG_LEVEL", "")
        assert Env.LOG_LEVEL.value == "INFO"

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv("AUTOVISION_MAX_NUM_WORKERS_AUTO", "eight")
        with pytest.raises(ValueError, match="AUTOVISION_MAX_NUM_WORKERS_AUTO"):
            Env.MAX_NUM_WORKERS_AUTO.value

    def test_set_and_unset(self, monkeypatch):
        Env.OFFLINE.set(True)
        assert Env.OFFLINE.value is True
        Env.OFFLINE.set(None)
        assert Env.OFFLINE.value is False

    def test_mirror_url(self, monkeypatch):
        monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://mirror.internal")
        assert Env.MIRROR_URL.value == "https://mirror.internal"
