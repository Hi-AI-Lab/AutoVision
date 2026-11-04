# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Artifact identity metadata (design D1; lineage: docs/provenance/LEDGER.md).

An exported artifact is self-describing: the task surface, class count,
input geometry and class names ride inside the artifact (ONNX
``metadata_props``) or beside it (a JSON sidecar — ``<xml>.json`` for
OpenVINO, ``<engine>.meta.json`` for TensorRT engines, which cannot carry
in-graph metadata). The inference runtime reads it back and verifies
network-vs-artifact identity at construction time, so a pose network
pointed at a detection engine fails with both conflicting values in the
message instead of an opaque shape error at the first forward.

The schema is flat string key/value with an ``autovision_`` prefix so
every carrier holds it unchanged. ``META_SCHEMA_VERSION`` gates evolution:
a newer writer schema degrades the check to a warning; absent metadata
(legacy exports, hand-built graphs) skips verification entirely — the
pre-D1 behavior. Identity checking is hardening, not the corruption gate:
numerical parity stays owned by the export parity tests.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError

__all__ = [
    "META_SCHEMA_VERSION",
    "build_metadata",
    "infer_task_name",
    "read_metadata",
    "verify_engine_artifact",
    "verify_metadata",
    "write_artifact_metadata",
]

logger = logging.getLogger(__name__)

#: Bump when a key changes meaning (adding keys is compatible, per the
#: same rule as the postprocess spec's frozen schema).
META_SCHEMA_VERSION = 1

#: Carrier per backend: ONNX carries metadata in-graph; OpenVINO rides the
#: existing ``<xml>.json`` sidecar that ``openvino_helpers`` already writes;
#: TensorRT engines get their own ``<engine>.meta.json``.
_SIDECAR_SUFFIX = {"openvino": ".json", "tensorrt": ".meta.json"}

#: Class names above this count are truncated (open-vocabulary heads can
#: carry thousands; metadata is for humans and sanity checks, not lookups).
_NAMES_LIMIT = 512


def infer_task_name(network: Any) -> str:
    """Task-surface name for a network: explicit attribute, else module path.

    A task model may pin the name with an ``artifact_task_name`` class
    attribute; otherwise the name is the ``autovision.tasks.<task>``
    module segment (``detection``, ``pose_estimation``, ...). Anything
    outside the package maps to ``"unknown"`` — never a crash, the value
    is informational for humans and compared only when both sides know it.
    """
    explicit = getattr(network, "artifact_task_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    module = type(network).__module__
    parts = module.split(".")
    if len(parts) > 3 and parts[0] == "autovision" and parts[1] == "tasks":
        return parts[2]
    return "unknown"


def _class_names_list(network: Any) -> list[str] | None:
    """Duck-typed class names (mapping id->name, or a sequence)."""
    names = getattr(network, "class_names", None)
    if isinstance(names, Mapping):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(name) for name in names]
    return None


def _image_size_pair(network: Any) -> tuple[int, int] | None:
    """``image_size`` normalized to a positive ``(H, W)`` pair."""
    size = getattr(network, "image_size", None)
    if size is None:
        return None
    if isinstance(size, (int, float)):
        size = (size, size)
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def build_metadata(network: Any) -> dict[str, str]:
    """Full identity schema for a network, as writable string key/values.

    Pre-D1 keys keep their exact names (``model_name``, ``image_normalize``,
    ``num_classes`` are unprefixed); readers of existing sidecars keep
    working. Everything added by D1 uses the ``autovision_`` prefix.
    """
    from autovision._license import LICENSE_INFO
    from autovision._version import __version__

    metadata = {
        "autovision_meta_version": str(META_SCHEMA_VERSION),
        "autovision_task": infer_task_name(network),
        "autovision_class_path": (
            f"{type(network).__module__}.{type(network).__name__}"
        ),
        "autovision_input_layout": "NCHW",
        "autovision_version": __version__,
        "license_info": LICENSE_INFO,
    }
    init_args = getattr(network, "init_args", None) or {}
    if isinstance(init_args, Mapping):
        model_name = init_args.get("model_name") or init_args.get("backbone")
        if model_name:
            metadata["model_name"] = str(model_name)
    normalize = getattr(network, "normalize", None)
    if normalize is not None:
        metadata["image_normalize"] = json.dumps(
            normalize.to_dict(), sort_keys=True
        )
    num_classes = getattr(network, "num_classes", None)
    if num_classes is not None:
        metadata["num_classes"] = str(int(num_classes))
    image_size = _image_size_pair(network)
    if image_size is not None:
        metadata["autovision_image_size"] = json.dumps(list(image_size))
    names = _class_names_list(network)
    if names:
        metadata["autovision_names"] = json.dumps(
            {
                "names": names[:_NAMES_LIMIT],
                "truncated": len(names) > _NAMES_LIMIT,
            }
        )
    return metadata


def read_metadata(artifact: Path | str, *, backend: str) -> dict[str, str]:
    """Read artifact identity; ``{}`` when absent or unreadable.

    Unreadable metadata is a debug-level skip, not an error: identity
    checking is hardening, and the artifact itself fails loudly at session
    creation if it is actually corrupt.
    """
    artifact = Path(artifact)
    try:
        if backend == "onnx":
            return _read_onnx(artifact)
        suffix = _SIDECAR_SUFFIX.get(backend)
        if suffix is not None:
            return _read_sidecar(artifact.with_suffix(suffix))
    except Exception as error:
        logger.debug(
            "Artifact metadata unreadable for '%s' (%s: %s); skipping the "
            "identity check.", artifact, type(error).__name__, error,
        )
    return {}


def _read_onnx(path: Path) -> dict[str, str]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def _read_sidecar(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def write_artifact_metadata(
    network: Any,
    artifact: Path | str,
    *,
    backend: str,
    extra: Mapping[str, str] | None = None,
) -> Path:
    """Write the identity sidecar for carriers without in-graph metadata.

    Only TensorRT engines need this today (OpenVINO's sidecar is written by
    ``openvino_helpers``, ONNX carries ``metadata_props`` in-graph). The
    write is atomic: a killed exporter never leaves a partial sidecar next
    to a good engine.
    """
    if backend not in _SIDECAR_SUFFIX:
        raise AutoVisionError(
            f"write_artifact_metadata: no sidecar carrier for backend "
            f"'{backend}' (expected one of {sorted(_SIDECAR_SUFFIX)}); ONNX "
            "metadata is written in-graph by the ONNX exporter."
        )
    payload: dict[str, str] = {**build_metadata(network), **(extra or {})}
    target = Path(artifact).with_suffix(_SIDECAR_SUFFIX[backend])
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def verify_metadata(
    metadata: dict[str, str], network: Any, *, artifact: Path | str
) -> None:
    """Raise on network-vs-artifact identity mismatch; skip when unknown.

    Only fields present on BOTH sides are compared, so legacy artifacts and
    networks without the attribute (e.g. a classifier without
    ``num_classes``) never fail on missing information.
    """
    if not metadata:
        logger.debug("No artifact metadata; skipping the identity check.")
        return
    version = metadata.get("autovision_meta_version", "")
    if version.isdigit() and int(version) > META_SCHEMA_VERSION:
        logger.warning(
            "Artifact '%s' uses metadata schema v%s (this runtime knows "
            "v%s); the identity check may be incomplete.",
            Path(artifact).name, version, META_SCHEMA_VERSION,
        )

    actual_task = metadata.get("autovision_task")
    expected_task = infer_task_name(network)
    if actual_task and expected_task != "unknown" and actual_task != expected_task:
        raise AutoVisionError(
            f"Artifact '{Path(artifact).name}' was exported from a "
            f"'{actual_task}' model, but the loaded network is "
            f"'{expected_task}' ({_class_path(network)}). Re-export the "
            "artifact from this network, or load the network the artifact "
            "was exported from."
        )

    actual_classes = metadata.get("num_classes")
    expected_classes = getattr(network, "num_classes", None)
    if (
        actual_classes is not None
        and expected_classes is not None
        and int(actual_classes) != int(expected_classes)
    ):
        raise AutoVisionError(
            f"Artifact '{Path(artifact).name}' has {actual_classes} classes "
            f"but the loaded network has {expected_classes}. Re-export the "
            "artifact or load a network with the matching class count."
        )

    actual_size = metadata.get("autovision_image_size")
    expected_size = _image_size_pair(network)
    if actual_size is not None and expected_size is not None:
        try:
            exported = [int(value) for value in json.loads(actual_size)]
        except (ValueError, TypeError):
            exported = []
        if len(exported) == 2 and list(expected_size) != exported:
            raise AutoVisionError(
                f"Artifact '{Path(artifact).name}' expects input "
                f"{exported[0]}x{exported[1]} (HxW) but the loaded network "
                f"uses {expected_size[0]}x{expected_size[1]}. Re-export the "
                "artifact or pass a network with the matching image_size."
            )


def _class_path(network: Any) -> str:
    return f"{type(network).__module__}.{type(network).__name__}"


def verify_engine_artifact(
    network: Any, artifact: Path | str, *, backend: str
) -> None:
    """Read + verify in one call (the :class:`InferenceEngine` entry point)."""
    verify_metadata(read_metadata(artifact, backend=backend), network, artifact=artifact)
