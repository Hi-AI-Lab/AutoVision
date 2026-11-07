#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hyperparameter search smoke test (``hpo/``).

Exercises the search API without training a model: a synthetic trial
function returns a cheap objective, so all three strategies (random / grid /
asha) and the artifact writers (``trials.jsonl`` / ``report.md`` /
``best.json``) run offline. It also shows ``load_space`` (yaml spec),
``split_overrides`` (the CLI override surface), the provenance-based resume
(re-run reuses finished trials), and the ``task_trial`` helper that drives a
real training entry.

    python examples/tasks/hpo_demo.py
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from autovision.hpo import (
    HpoResult,
    SearchSpace,
    load_space,
    run_search,
    split_overrides,
    task_trial,
)
from autovision.hpo.search import TrialFn

OBJECTIVE = "val_metric/accuracy_top1"


def _synthetic_trial(params: Mapping[str, object], steps: int, run_dir: Path) -> Mapping[str, float]:
    """Cheap deterministic objective; longer budgets score marginally higher.

    Peaks near ``model_args.lr=1e-2``, ``model_args.dropout=0.1``,
    ``batch_size=8`` (and ``x`` high / ``y`` low for the grid space), so every
    strategy has a discoverable optimum. The ``steps`` term lets ASHA
    promotions (more budget) improve a configuration's read.
    """
    lr = float(params.get("model_args.lr", 1e-3))
    bs = float(params.get("batch_size", 4))
    do = float(params.get("model_args.dropout", 0.1))
    x = float(params.get("x", 1))
    y = float({"a": 0.0, "b": 1.0, "c": 2.0}.get(str(params.get("y", "b")), 1.0))
    score = (
        -((lr - 1e-2) ** 2) * 500.0
        - ((do - 0.1) ** 2) * 10.0
        + bs * 0.005
        + x * 0.01
        - y * 0.005
        - steps * 1e-7
    )
    return {OBJECTIVE: float(score)}


def _continuous_space() -> SearchSpace:
    """A space mixing all four distribution kinds."""
    return SearchSpace.from_mapping(
        {
            "model_args.lr": {"loguniform": [1e-4, 1e-2]},
            "model_args.dropout": {"uniform": [0.0, 0.5]},
            "batch_size": [2, 4, 8],
            "x": {"randint": [0, 3]},
        }
    )


def _grid_space() -> SearchSpace:
    """All-choice axes so ``strategy='grid'`` enumerates the product."""
    return SearchSpace.from_mapping(
        {
            "x": [0, 1, 2],
            "y": ["a", "b", "c"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/examples/hpo_demo")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── load_space from a yaml spec ────────────────────────────────────────
    space_yaml = out / "space.yaml"
    space_yaml.write_text(
        "model_args.lr: {loguniform: [1.0e-4, 1.0e-2]}\n"
        "batch_size: [2, 4, 8]\n"
        "model_args.dropout: {uniform: [0.0, 0.5]}\n",
        encoding="utf-8",
    )
    space = load_space(space_yaml)
    print(f"loaded space from yaml: {space!r}")

    # ── split_overrides rides the CLI override surface ─────────────────────
    assert split_overrides({"model_args.lr": 1e-3, "batch_size": 4}) == {
        "model_args": {"lr": 1e-3},
        "batch_size": 4,
    }, "split_overrides must nest dotted keys"
    print("split_overrides ok: dotted keys -> nested kwargs")

    # ── strategy: random ───────────────────────────────────────────────────
    random_dir = out / "random"
    random_res: HpoResult = run_search(
        _synthetic_trial,
        _continuous_space(),
        random_dir,
        strategy="random",
        trials=8,
        budget=50,
        objective=OBJECTIVE,
        maximize=True,
        seed=0,
    )
    _check_artifacts(random_dir, random_res, "random")
    print(f"random best: #{random_res.best.index} -> {random_res.best.params}")

    # ── resume: re-running reuses finished trials (provenance fingerprint) ──
    repeated = run_search(
        _synthetic_trial,
        _continuous_space(),
        random_dir,
        strategy="random",
        trials=8,
        budget=50,
        objective=OBJECTIVE,
        maximize=True,
        seed=0,
    )
    reused = sum(1 for t in repeated.trials if t.status == "reused")
    assert reused == len(repeated.trials), f"expected all trials reused, got {reused}"
    print(f"resume: {reused}/{len(repeated.trials)} trials reused (provenance match)")

    # ── strategy: grid (all-choice axes) ───────────────────────────────────
    grid_dir = out / "grid"
    grid_res = run_search(
        _synthetic_trial,
        _grid_space(),
        grid_dir,
        strategy="grid",
        trials=9,
        budget=30,
        objective=OBJECTIVE,
        maximize=True,
        seed=0,
    )
    assert len(grid_res.trials) == 9, f"grid should enumerate 9 points, got {len(grid_res.trials)}"
    _check_artifacts(grid_dir, grid_res, "grid")
    print(f"grid enumerated {len(grid_res.trials)} points (3x3)")

    # ── strategy: asha (successive halving) ────────────────────────────────
    asha_dir = out / "asha"
    asha_res = run_search(
        _synthetic_trial,
        _continuous_space(),
        asha_dir,
        strategy="asha",
        trials=6,
        budget=10,
        asha_min_budget=10,
        asha_max_budget=40,
        asha_eta=2,
        objective=OBJECTIVE,
        maximize=True,
        seed=0,
    )
    _check_artifacts(asha_dir, asha_res, "asha")
    print(f"asha best: #{asha_res.best.index} -> {asha_res.best.params}")

    # ── task_trial: the real training integration helper ───────────────────
    fn: TrialFn = task_trial(
        "object_detection",
        base={"data": "coco8", "model": "gtr_tiny", "devices": [0]},
    )
    assert callable(fn), "task_trial must return a trial function"
    print("task_trial('object_detection', ...) built (not executed: would train)")

    print("\nOK hpo subsystem (random / grid / asha + artifacts + resume + task_trial)")


def _check_artifacts(out_dir: Path, result: HpoResult, strategy: str) -> None:
    assert result.best is not None, f"{strategy}: no best trial selected"
    assert (out_dir / "trials.jsonl").is_file(), f"{strategy}: trials.jsonl missing"
    assert (out_dir / "report.md").is_file(), f"{strategy}: report.md missing"
    assert (out_dir / "best.json").is_file(), f"{strategy}: best.json missing"


if __name__ == "__main__":
    main()
