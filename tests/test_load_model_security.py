# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Security contract of ``load_task_model`` on untrusted exported files.

Exported ``*.pt`` files are treated as untrusted input: they must be
read with the pickle-free safe loader and may only ever rebuild concrete
``TaskModel`` classes inside ``autovision._task_models``. These tests pin
that contract against regressions (a naive ``weights_only=False`` here
is arbitrary code execution via crafted pickle).
"""

import torch
import pytest

from autovision import load_task_model
from autovision.errors import CheckpointError


def _write_export(path, model_class_path):
    torch.save(
        {
            "schema_version": 1,
            "kind": "task_model",
            "model_class_path": model_class_path,
            "model_init_args": {},
            "train_model": {},
            "normalize_args": {},
            "version": "test",
        },
        path,
    )


def test_rejects_class_path_outside_task_models(tmp_path):
    path = tmp_path / "exported.pt"
    _write_export(path, "subprocess.Popen")
    with pytest.raises(CheckpointError, match="only autovision task-model"):
        load_task_model(path)


def test_rejects_non_concrete_task_model_class(tmp_path):
    path = tmp_path / "exported.pt"
    # Inside the package, but the abstract base itself is not a
    # constructible task model.
    _write_export(path, "autovision._task_models.task_model.TaskModel")
    with pytest.raises(CheckpointError, match="not a concrete"):
        load_task_model(path)


def test_rejects_function_target(tmp_path):
    path = tmp_path / "exported.pt"
    _write_export(
        path, "autovision._task_models.task_model_helpers.load_task_model"
    )
    with pytest.raises(CheckpointError, match="not a concrete"):
        load_task_model(path)


# Module-level so pickle can resolve it by reference at load time.
_CANARY_HITS: list[str] = []


def _record_canary(hits):
    hits.append("executed")


class _CanaryPickle:
    """Unpickling this object would call ``_record_canary``."""

    def __reduce__(self):
        return _record_canary, (_CANARY_HITS,)


def test_safe_loader_blocks_pickle_execution(tmp_path):
    path = tmp_path / "malicious.pt"
    torch.save({"payload": _CanaryPickle()}, path)
    with pytest.raises(CheckpointError):
        load_task_model(path)
    assert _CANARY_HITS == [], "pickle payload must never execute"


def test_not_an_exported_model(tmp_path):
    path = tmp_path / "random.pt"
    torch.save({"unrelated": torch.zeros(1)}, path)
    with pytest.raises(CheckpointError, match="not an exported"):
        load_task_model(path)
