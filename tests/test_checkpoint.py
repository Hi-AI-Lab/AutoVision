# Copyright (c) 2026 Autovision contributors. Apache-2.0.
import pytest
import torch

from autovision._checkpoint import (
    CHECKPOINT_AUTOVISION_KEY,
    Checkpoint,
    CheckpointMetadata,
)
from autovision.errors import CheckpointError


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "checkpoints" / "last.ckpt"
        metadata = CheckpointMetadata(epoch=3, global_step=42, normalize_args={"mean": [0.5]})
        checkpoint = Checkpoint(state_dict={"weight": torch.randn(4, 4)}, autovision=metadata)
        checkpoint.save(path)
        restored = Checkpoint.from_path(path)
        assert torch.allclose(restored.state_dict["weight"], checkpoint.state_dict["weight"])
        assert restored.autovision.epoch == 3
        assert restored.autovision.global_step == 42
        assert restored.autovision.license_info
        assert restored.autovision.version

    def test_atomic_save_leaves_no_tmp(self, tmp_path):
        path = tmp_path / "last.ckpt"
        Checkpoint(state_dict={}, autovision=CheckpointMetadata()).save(path)
        assert path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="does not exist"):
            Checkpoint.from_path(tmp_path / "nope.ckpt")

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "bad.ckpt"
        path.write_bytes(b"not-a-checkpoint")
        with pytest.raises(CheckpointError, match="Failed to load"):
            Checkpoint.from_path(path)

    def test_non_autovision_file_raises(self, tmp_path):
        path = tmp_path / "foreign.ckpt"
        torch.save({"some_key": 1}, path)
        with pytest.raises(CheckpointError, match=CHECKPOINT_AUTOVISION_KEY):
            Checkpoint.from_path(path)

    def test_metadata_ignores_unknown_fields(self):
        metadata = CheckpointMetadata.from_dict(
            {"epoch": 2, "future_field": "whatever"}
        )
        assert metadata.epoch == 2

    def test_default_metadata(self):
        metadata = CheckpointMetadata()
        assert metadata.epoch == -1
        assert metadata.date  # ISO timestamp set
