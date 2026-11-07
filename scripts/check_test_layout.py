#!/usr/bin/env python3
"""Test-layout guard: ``test_*`` definitions must be discoverable (H4-4).

Historical accident (H0-1): an indentation slip nested 8 pipeline tests
inside a module-level function's body, so pytest silently collected 8/16
cases and the affected runtime guards lost their regression coverage.
This guard turns that class of mistake into a CI lint failure instead of
a silent suite shrinkage.

Rule: every ``test_*`` function (sync or async) must be defined either at
module top level or directly in the body of a module-level class. A
``test_*`` definition nested in a function body, or inside a class that is
itself nested (pytest collects neither), is reported as a violation with
``file:line:name``.

Pure standard library (``ast``); cost is O(number of test files).

CI hook: the ``lint`` job in ``.github/workflows/ci.yml``.

Usage:
    python scripts/check_test_layout.py             # scan <repo>/tests
    python scripts/check_test_layout.py --root DIR  # scan an explicit tree
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "tests"

_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "node_modules"})

#: Scopes in which pytest actually discovers ``test_*`` definitions:
#: module top level and the body of a module-level class.
_DISCOVERED_SCOPES = frozenset({"module", "class"})

#: Human-readable names for the scopes that hide tests from collection.
_HIDDEN_SCOPE_DESCRIPTIONS = {
    "function": "a function body",
    "nested_class": "a nested class body (not collected by pytest)",
}


def _iter_python_files(root: Path) -> list[Path]:
    """All ``*.py`` files under ``root`` in stable order, caches skipped."""
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*.py")
        if not _SKIP_DIRS.intersection(path.parts)
    )


def _rel_path(path: Path) -> str:
    """Repo-relative POSIX path (falls back to absolute outside the repo)."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _walk_body(
    body: list[ast.stmt],
    scope: str,
    rel_path: str,
    violations: list[str],
) -> None:
    """Collect ``test_*`` definitions that pytest would not discover.

    ``scope`` is the enclosing container: ``module`` (top level), ``class``
    (module-level class body), ``function`` or ``nested_class``.
    """
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name.startswith("test_") and scope not in _DISCOVERED_SCOPES:
                violations.append(
                    f"{rel_path}:{stmt.lineno}:{stmt.name} (defined in "
                    f"{_HIDDEN_SCOPE_DESCRIPTIONS[scope]}; pytest will not "
                    "collect it)"
                )
            _walk_body(stmt.body, "function", rel_path, violations)
        elif isinstance(stmt, ast.ClassDef):
            child_scope = "class" if scope == "module" else "nested_class"
            _walk_body(stmt.body, child_scope, rel_path, violations)


def scan_file(path: Path) -> list[str]:
    """Violations for one file, as ``file:line:name`` strings."""
    rel_path = _rel_path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [f"{rel_path}:0:<module> (SyntaxError: {error})"]
    violations: list[str] = []
    _walk_body(tree.body, "module", rel_path, violations)
    return violations


def main(argv: list[str] | None = None) -> int:
    """Scan the test tree and exit 1 when any ``test_*`` definition nests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="file or directory tree to scan (default: <repo>/tests)",
    )
    options = parser.parse_args(argv)
    root: Path = options.root
    if not root.is_file() and not root.is_dir():
        parser.error(f"--root is not a file or directory: {root}")

    files = _iter_python_files(root)
    violations: list[str] = []
    for path in files:
        violations.extend(scan_file(path))
    if violations:
        print(f"{len(violations)} test-layout violation(s):")
        for violation in violations:
            print(f"  {violation}")
        print(
            "Move each test to module top level or into a module-level class "
            "body (H4-4: nested test definitions are never collected)."
        )
        return 1
    print(
        f"OK: {len(files)} file(s) scanned; every test_* definition is "
        "discoverable."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
