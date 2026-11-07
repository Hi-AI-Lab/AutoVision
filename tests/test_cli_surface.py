# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CLI error-surface regressions (H3-6a / H3-6b) and CLI-side hardening.

All invocations go through typer's CliRunner — the same path as the
console entry point after ``get_command(app)`` — and pin exit code plus
stderr text instead of the raw-exception channel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import create_image
from typer.testing import CliRunner

import autovision.cli.model_handle as model_handle
import autovision.data.downloads as downloads
import autovision.hpo as hpo
from autovision.cli._app import app


def test_val_missing_model_reports_error_line_and_exit_1(tmp_path: Path) -> None:
    """H3-6a: a bad model path is one stderr line + exit 1, not a traceback."""
    result = CliRunner().invoke(
        app, ["val", f"model={tmp_path / 'missing.pt'}", "data=coco8"]
    )
    assert result.exit_code == 1
    assert "error: " in result.stderr
    assert str(tmp_path / "missing.pt") in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout
    # A clean exit (SystemExit) proves the error was converted, not left to
    # escape as a raw framework exception.
    assert isinstance(result.exception, SystemExit)


def test_plot_curves_without_run_is_usage_error() -> None:
    """H3-6a: a missing run= is a usage error (exit 2), not a KeyError."""
    result = CliRunner().invoke(app, ["plot-curves"])
    assert result.exit_code == 2
    assert "run=" in result.output
    assert not isinstance(result.exception, KeyError)


def test_download_rejects_unknown_key() -> None:
    """H3-6b: unknown download keys are usage errors, not silently dropped."""
    result = CliRunner().invoke(
        app, ["download", "dataset=coco2017-val", "sha256=deadbeef"]
    )
    assert result.exit_code == 2
    assert "sha256" in result.output
    assert "download" in result.output


def test_download_valid_keys_reach_download_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-6b: accepted download keys keep reaching download_dataset unchanged."""
    calls: list[tuple[str, Path]] = []

    def fake_download_dataset(name: str, out_dir: Path) -> Path:
        calls.append((name, out_dir))
        return out_dir / name

    monkeypatch.setattr(downloads, "download_dataset", fake_download_dataset)
    result = CliRunner().invoke(
        app, ["download", "dataset=coco2017-val", f"out={tmp_path / 'ds'}"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("coco2017-val", tmp_path / "ds")]


def test_predict_rejects_unknown_key(tmp_path: Path) -> None:
    """H3-6b: predict rejects typos as usage errors instead of TypeError."""
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "model=m.pt",
            f"images={tmp_path}",
            "out=preds/",
            "threshhold=0.5",
        ],
    )
    assert result.exit_code == 2
    assert "threshhold" in result.output
    assert "predict" in result.output


def test_predict_valid_keys_reach_predict_to_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-6b: accepted predict keys keep the out= -> out_dir= mapping."""
    calls: list[dict[str, Any]] = []

    def fake_predict_to_files(**kwargs: Any) -> Path:
        calls.append(dict(kwargs))
        return tmp_path / "preds"

    monkeypatch.setattr(model_handle, "predict_to_files", fake_predict_to_files)
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "model=m.pt",
            f"images={tmp_path}",
            f"out={tmp_path / 'preds'}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "model": "m.pt",
            "images": str(tmp_path),
            "out_dir": str(tmp_path / "preds"),
        }
    ]


def test_val_output_declares_headline_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-6c + METRIC-04: the val output carries the headline protocol
    note (single operating point since the 0.6.1 rewrite), and the out=
    json keeps float-only metrics (the protocol travels beside the
    numbers, not inside the metric dict — the engine rounds every entry)."""
    monkeypatch.setattr(
        model_handle,
        "evaluate_object_detection",
        lambda **kwargs: {"map": 0.5, "f1": 0.4},
    )
    out = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app, ["val", "model=m.pt", "data=coco8", f"out={out}"]
    )
    assert result.exit_code == 0, result.output
    assert "ONE operating point" in result.output
    assert "not an operating point" in result.output
    payload = json.loads(out.read_text())
    assert payload == {"map": 0.5, "f1": 0.4}  # no str-valued protocol key


class TestAnalyzeErrorsOfflineCategories:
    """H3-6e: offline mode label-id mapping between predictions and COCO."""

    @staticmethod
    def _annotations(tmp_path: Path, images: list[Path]) -> Path:
        """COCO annotations with ids 1=cat, 2=dog (sorted-id rank: cat=0)."""
        annotations = {
            "images": [
                {"id": index + 1, "file_name": Path(path).name, "width": 64, "height": 64}
                for index, path in enumerate(images)
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(annotations))
        return path

    def test_categories_envelope_maps_labels_by_name(
        self, tmp_path: Path
    ) -> None:
        """H3-6e: an envelope's ids map through class names, so a training
        order unlike the COCO id order cannot mislabel every class.

        The envelope declares dog=0/cat=1 while the annotations rank
        cat=0/dog=1; both detections must land as TPs of their own class.
        """
        images = [
            create_image(tmp_path / "im" / f"img_{i}.jpg", size=(64, 64), seed=i)
            for i in range(2)
        ]
        annotations_path = self._annotations(tmp_path, images)
        envelope = {
            "categories": {"0": "dog", "1": "cat"},
            "records": [
                {"image": str(images[0]), "bboxes": [[10, 10, 50, 50]],
                 "scores": [0.9], "labels": [1]},  # 1 = cat -> cat GT: TP
                {"image": str(images[1]), "bboxes": [[0, 0, 10, 10]],
                 "scores": [0.8], "labels": [0]},  # 0 = dog -> dog GT: TP
            ],
        }
        predictions_path = tmp_path / "preds.json"
        predictions_path.write_text(json.dumps(envelope))
        result = CliRunner().invoke(
            app,
            [
                "analyze-errors",
                f"predictions={predictions_path}",
                f"annotations={annotations_path}",
                f"out={tmp_path / 'report'}",
            ],
        )
        assert result.exit_code == 0, result.output
        # A mangled mapping would swap the classes into classification FPs.
        assert '"tp": 2' in result.output
        assert (tmp_path / "report" / "report.md").exists()

    def test_bare_record_list_warns_about_sorted_position_assumption(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """H3-6e: without a categories envelope the sorted-position
        assumption is stated in a warning, and the run still completes."""
        images = [
            create_image(tmp_path / "im" / f"img_{i}.jpg", size=(64, 64), seed=i)
            for i in range(2)
        ]
        annotations_path = self._annotations(tmp_path, images)
        records = [
            {"image": str(images[0]), "bboxes": [[10, 10, 50, 50]],
             "scores": [0.9], "labels": [0]},  # rank 0 = cat
            {"image": str(images[1]), "bboxes": [], "scores": [], "labels": []},
        ]
        predictions_path = tmp_path / "preds.json"
        predictions_path.write_text(json.dumps(records))
        with caplog.at_level(logging.WARNING):
            result = CliRunner().invoke(
                app,
                [
                    "analyze-errors",
                    f"predictions={predictions_path}",
                    f"annotations={annotations_path}",
                    f"out={tmp_path / 'report'}",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "no 'categories'" in caplog.text
        assert "sorted by id" in caplog.text


class TestHpoFailureAggregation:
    """H3-6f: swallowed per-trial failures surface in the final error."""

    @staticmethod
    def _space(tmp_path: Path) -> Path:
        path = tmp_path / "space.yaml"
        path.write_text("batch_size: [2, 4]\n")
        return path

    @staticmethod
    def _result(tmp_path: Path, trials: list[hpo.TrialRecord]) -> hpo.HpoResult:
        return hpo.HpoResult(
            out_dir=tmp_path / "hpo",
            objective="val_metric/map",
            maximize=True,
            strategy="random",
            trials=trials,
            report_md=tmp_path / "hpo" / "report.md",
            trials_jsonl=tmp_path / "hpo" / "trials.jsonl",
            best_json=tmp_path / "hpo" / "best.json",
            space="batch_size: [2, 4]",
        )

    def test_mistyped_kwarg_surfaces_the_original_typeerror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H3-6f: stepps=100 fails every trial with a TypeError — the final
        error must carry that message, not just "no successful trial"."""
        monkeypatch.setattr(hpo, "resolve_task_entry", lambda task: object())
        trials = [
            hpo.TrialRecord(
                index=0, params={"batch_size": 2}, steps=5, status="failed",
                error="AutoVisionError: data config not found.",  # intentional
            ),
            hpo.TrialRecord(
                index=1, params={"batch_size": 4, "stepps": 100}, steps=5,
                status="failed",
                error=(
                    "TypeError: train_object_detection() got an unexpected "
                    "keyword argument 'stepps'"
                ),
            ),
        ]
        monkeypatch.setattr(
            hpo, "run_search",
            lambda *args, **kwargs: self._result(tmp_path, trials),
        )
        result = CliRunner().invoke(
            app,
            [
                "hpo", f"space={self._space(tmp_path)}", f"out={tmp_path / 'hpo'}",
                "task=object_detection", "stepps=100",
            ],
        )
        assert result.exit_code == 1
        # The accidental TypeError beats the intentional AutoVisionError.
        assert "2/2 trials failed" in result.stderr
        assert "first failure: TypeError" in result.stderr
        assert "stepps" in result.stderr
        assert "no successful trial" not in result.stderr

    def test_successful_trial_output_is_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H3-6f: a search with a best trial keeps the exit-0 surface."""
        monkeypatch.setattr(hpo, "resolve_task_entry", lambda task: object())
        best = hpo.TrialRecord(
            index=0, params={"batch_size": 2}, steps=5, status="ok",
            metrics={"val_metric/map": 0.42},
        )
        monkeypatch.setattr(
            hpo, "run_search",
            lambda *args, **kwargs: self._result(tmp_path, [best]),
        )
        result = CliRunner().invoke(
            app,
            [
                "hpo", f"space={self._space(tmp_path)}", f"out={tmp_path / 'hpo'}",
                "task=object_detection",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "best trial #0" in result.output
