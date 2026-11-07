#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model registry + catalog smoke test (``registry/``).

Exercises the asset registry (filesystem backend) end to end WITHOUT any
trained weights: a synthetic artifact file is registered in ``dev``, promoted
through ``staging`` -> ``prod`` (passing the default deployment gates), then
rolled back; a second artifact with no model card demonstrates a
gate-rejected promotion that lands in the append-only audit log. Also shows
the curated catalog (``MODEL_CATALOG`` / ``get_model_entry`` / table) and a
``ModelCard`` round-trip.

    python examples/tasks/registry_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.registry import (
    ModelRegistry,
    RegistryError,
    describe_model,
    format_model_table,
    get_model_entry,
    list_task_models,
)
from autovision.registry.model_card import (
    LatencyRecord,
    ModelCard,
    ModelCardError,
    load_model_card,
)


def _make_artifact(path: Path, size: int) -> None:
    """Write random bytes and age the mtime so register() doesn't warn."""
    path.write_bytes(bytes((i * 37 + 11) % 256 for i in range(size)))
    old = time.time() - 10.0
    os.utime(path, (old, old))


def _write_valid_card(artifact: Path) -> None:
    """Attach a publishable model card sidecar (reviewed by the gates)."""
    card = ModelCard(
        model="detr/s",
        name="DETR-S (registry demo)",
        train_dataset="COCO 2017 train2017",
        schedule="6ep finetune (demo)",
        license="MIT",
        metrics={"coco_map": 0.55},
        latency=[
            LatencyRecord(
                backend="tensorrt", precision="fp16",
                hardware="NVIDIA T4", batch_size=1, p50_ms=2.1,
                script="bench.py",
            )
        ],
        weights_sha256={artifact.name: compute_file_sha256(artifact)},
    )
    card.save(artifact)


def _write_parity(artifact: Path) -> None:
    """Attach a passing parity sidecar (read by ParityGate)."""
    (Path(str(artifact) + ".parity.json")).write_text(
        json.dumps({"passed": True, "backend": "onnx"}, indent=2)
    )


def _run_registry(out: Path) -> None:
    root = out / "registry"
    if root.exists():  # versions are immutable; clear before a re-run
        shutil.rmtree(root)
    reg = ModelRegistry(root)

    # ── passing artifact: dev -> staging -> prod (default gates) ────────────
    a = out / "artifact_a.bin"
    _make_artifact(a, 4096)
    _write_valid_card(a)
    _write_parity(a)

    reg.register(a, name="detector", version="1.0.0", stage="dev", actor="demo")
    reg.promote(name="detector", version="1.0.0", to_stage="staging", actor="demo")
    reg.promote(name="detector", version="1.0.0", to_stage="prod", actor="demo")
    assert reg.current(name="detector", stage="prod").version == "1.0.0"

    # A second version promoted to prod becomes the current pointer.
    a2 = out / "artifact_a2.bin"
    _make_artifact(a2, 4096)
    _write_valid_card(a2)
    _write_parity(a2)
    reg.register(a2, name="detector", version="2.0.0", stage="dev", actor="demo")
    reg.promote(name="detector", version="2.0.0", to_stage="staging", actor="demo")
    reg.promote(name="detector", version="2.0.0", to_stage="prod", actor="demo")
    assert reg.current(name="detector", stage="prod").version == "2.0.0"

    # Rollback points prod's current pointer at the older, still-present 1.0.0.
    reg.rollback(name="detector", stage="prod", to_version="1.0.0", reason="demo")
    assert reg.current(name="detector", stage="prod").version == "1.0.0"

    entries = reg.list_entries(stage="prod")
    assert len(entries) == 2, f"expected 2 prod entries, got {len(entries)}"
    print(f"registry: promoted detector 1.0.0 -> 2.0.0 -> rolled back to 1.0.0; "
          f"{len(entries)} prod entries")

    # ── failing artifact: a promotion rejected by the deployment gates ──────
    b = out / "artifact_b.bin"
    _make_artifact(b, 2048)  # no model card / no parity -> gates must fail
    reg.register(b, name="detector_b", version="1.0.0", stage="dev", actor="demo")
    try:
        reg.promote(name="detector_b", version="1.0.0", to_stage="staging", actor="demo")
        raise AssertionError("promotion should have been gate-rejected")
    except RegistryError as exc:
        assert "model card" in str(exc).lower() or "gate" in str(exc).lower()
    log = reg.audit_log()
    assert any(entry["action"] == "promote_failed" for entry in log), \
        "gate-rejected promotion must be recorded in the audit log"
    print(f"registry: gate-rejected promotion recorded in audit log "
          f"({len(log)} audit events)")


def _run_catalog() -> None:
    released = list_task_models(task="detection", pretrained_only=True)
    assert released, "expected released detection models in the catalog"
    table = format_model_table(released)
    print(f"catalog: {len(released)} released detection models")
    print(table)
    entry = get_model_entry("detr/gtrvit-s")
    assert entry.license == "MIT"
    print("catalog describe detr/gtrvit-s:\n" + describe_model("detr/gtrvit-s"))


def _run_model_card() -> None:
    # Valid card passes validation.
    card = ModelCard(
        model="detr/s", name="DETR-S", train_dataset="COCO", schedule="12ep",
        license="Apache-2.0", metrics={"coco_map": 0.58},
        latency=[LatencyRecord("onnx", "fp32", "T4", 1, 3.0, "bench.py")],
        weights_sha256={"detr_s.pt": "0" * 64},
    )
    card.validate()
    # Missing license / bad sha256 must fail validation.
    bad = ModelCard(
        model="x", name="", train_dataset="", schedule="",
        license="", weights_sha256={"f": "zzz"},
    )
    try:
        bad.validate()
        raise AssertionError("invalid card should not validate")
    except ModelCardError:
        pass
    # Round-trip save -> load.
    weights = Path(__file__).parent / "runs" / "examples" / "registry_modelcard_demo.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"demo")
    card.save(weights)
    loaded = load_model_card(weights)
    assert loaded.license == "Apache-2.0" and loaded.metrics["coco_map"] == 0.58
    print("modelcard: validate + save/load round-trip ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/examples/registry_demo")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _run_registry(out)
    _run_catalog()
    _run_model_card()

    print("\nOK registry subsystem (filesystem registry + gates + audit, "
          "catalog, model card)")


if __name__ == "__main__":
    main()
