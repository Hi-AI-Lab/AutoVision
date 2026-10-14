#!/usr/bin/env python3
"""Layering-contract checker (design doc 28, section 4.2 / rule R4).

Enforces the one-way layering of the ``autovision`` package with a
dependency-free AST scan (no import-linter dependency — this environment
is offline):

    cli -> engine -> {pretext, tasks} -> {augment, features} -> encoders
        -> {data, optim, metrics, config, registry, scaling, observability,
            export, debug, visualize}

A module may import its own layer or any LOWER layer. Upward imports must
appear in the ``ALLOWED_UPWARD`` table below with a reason (they mark
shared-utility placements to be tightened later). Two hard bans are
enforced regardless of ranks: ``tasks -> pretext`` (teacher construction
goes through protocols) and ``encoders`` importing any framework layer
above it.

Exit code 1 on any violation not covered by ``ALLOWED_UPWARD``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "autovision"

#: Layer ranks — higher number = lower layer. Same-layer imports are free.
LAYERS: dict[str, int] = {
    "cli": 0,
    "engine": 1,
    "pretext": 2,
    "tasks": 2,
    "augment": 3,
    "features": 3,
    "encoders": 4,
    "data": 5,
    "optim": 5,
    "metrics": 5,
    "config": 5,
    "registry": 5,
    "scaling": 5,
    "observability": 5,
    "export": 5,  # export mixin sits below tasks (task models inherit it)
    "debug": 5,
    "visualize": 5,
}
_TOP_LEVEL_RANK = 6  # errors, types, scaling-style single modules

#: Hard bans regardless of ranks: (source package, target package).
HARD_BANS: set[tuple[str, str]] = {
    ("tasks", "pretext"),
    ("encoders", "tasks"),
    ("encoders", "pretext"),
    ("encoders", "engine"),
    ("encoders", "cli"),
}

#: Sanctioned upward imports: (source module prefix, target module prefix,
#: reason). Keep this table SHORT — entries are debt markers, not policy.
ALLOWED_UPWARD: dict[tuple[str, str], str] = {
    ("augment.detection", "tasks.detection.components"): (
        "shared detection box math (box_ops); move to a common layer later"
    ),
    ("features.embed", "cli.common_helpers"): (
        "embed is a command implementation reusing CLI helpers"
    ),
    ("features.embed", "engine.engine"): (
        "embed reuses device/seed helpers"
    ),
    ("data.image_dataset", "augment.method_transform"): (
        "dataset consumes the multi-view transform protocol"
    ),
    ("metrics.oriented_detection", "tasks.detection.components"): (
        "shared OBB math (obb.py); move to a common layer later"
    ),
    ("registry", "encoders.weight_verification"): (
        "sha256 utility currently lives under encoders; relocate later"
    ),
    ("_torch_compile", "config"): (
        "runtime torch.compile helpers read the shared config base"
    ),
    ("visualize", "augment"): (
        "detection visualization consumes transform args for display"
    ),
    ("export.benchmark", "tasks"): (
        "benchmark is a runtime entrypoint that loads task models"
    ),
    ("export.benchmark", "engine"): (
        "benchmark drives the training engine for speed measurement"
    ),
}

def layer_of(top_segment: str) -> int | None:
    return LAYERS.get(top_segment, _TOP_LEVEL_RANK)


def imported_autovision_targets(tree: ast.AST) -> list[str]:
    """Dotted ``autovision.*`` targets referenced by import statements."""
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "autovision" or alias.name.startswith("autovision."):
                    targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (
                module == "autovision" or module.startswith("autovision.")
            ):
                targets.append(module)
            elif node.level > 0:
                # Relative import: resolve against the file's package later.
                targets.append(f".{module}")
    return targets


def check() -> list[str]:
    problems: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT)
        parts = rel.with_suffix("").parts
        if len(parts) == 1 and parts[0] == "__init__":
            continue  # package composition root: imports every layer
        if len(parts) == 1:
            # Top-level module; the legacy re-export shim is exempt.
            if parts[0] == "_model_wrappers":
                continue
        source_pkg = parts[0]
        source_rank = layer_of(source_pkg)
        source_mod = ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            problems.append(f"{rel}: syntax error: {error}")
            continue
        for target in imported_autovision_targets(tree):
            if target.startswith("."):
                continue  # relative intra-package import: same layer by construction
            segments = target.split(".")
            if len(segments) < 2:
                continue  # bare `import autovision`
            target_pkg = segments[1]
            target_rank = layer_of(target_pkg)
            if target_rank is None:
                continue
            if source_rank is not None and target_rank < source_rank:
                if (source_pkg, target_pkg) in HARD_BANS:
                    problems.append(
                        f"{rel}: HARD BAN {source_pkg} -> {target_pkg} "
                        f"(design doc 28 §4.2)"
                    )
                    continue
                reason = _allowed_upward_reason(source_mod, target)
                if reason is None:
                    problems.append(
                        f"{rel}: upward import {source_pkg}({source_rank}) -> "
                        f"{target_pkg}({target_rank}): from {target}"
                    )
    return problems


def _allowed_upward_reason(source_mod: str, target: str) -> str | None:
    for (src_prefix, dst_prefix), reason in ALLOWED_UPWARD.items():
        if source_mod.startswith(src_prefix) and target.startswith("autovision." + dst_prefix):
            return reason
    return None


def main() -> int:
    problems = check()
    if problems:
        print(f"{len(problems)} layering violation(s):")
        for problem in problems:
            print(f"  {problem}")
        print(
            "Fix the import direction, or add a reasoned entry to "
            "ALLOWED_UPWARD in this script if the coupling is sanctioned debt."
        )
        return 1
    print("layering contract satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
