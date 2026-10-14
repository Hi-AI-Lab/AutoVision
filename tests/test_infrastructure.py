# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Infrastructure checks: CI configs, Docker, docs, packaging metadata."""

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: stdlib tomllib landed in 3.11
    import tomli as tomllib

ROOT = Path(__file__).parent.parent


class TestCI:
    def test_gitlab_ci_parses(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
        assert set(config["stages"]) == {"lint", "test"}
        assert "pytest" in config["test"]["script"][-1]

    def test_github_workflow_parses(self):
        pytest.importorskip("yaml")
        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert "test" in workflow["jobs"] and "lint" in workflow["jobs"]

    def test_telemetry_disabled_in_ci(self):
        import yaml

        config = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
        variables = config["test"].get("variables", {})
        # Compliance: CI must not enable telemetry.
        assert variables.get("AUTOVISION_TELEMETRY", "0") != "1"


class TestDocker:
    def test_dockerfile_exists_and_smoke_wired(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        assert "smoke_test.sh" in dockerfile
        assert (ROOT / "docker" / "smoke_test.sh").exists()

    def test_smoke_script_exercises_pretrain(self):
        script = (ROOT / "docker" / "smoke_test.sh").read_text()
        assert "autovision.pretrain" in script
        assert "checkpoint" in script


class TestDocs:
    def test_mkdocs_config_parses_and_links_design(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        assert config["site_name"] == "Autovision"
        nav_str = str(config["nav"])
        assert "design/" in nav_str
        for part in range(1, 12):
            assert any(
                f"{part:02d}" in str(entry) for entry in config["nav"]
            ), f"design part {part:02d} missing from nav"


class TestPackaging:
    def test_cli_entry_point_registered(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        scripts = config["project"].get("scripts", {})
        assert scripts.get("autovision-cli") == "autovision.cli._app:cli_entrypoint"

    def test_optional_extras_declared(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extras = config["project"]["optional-dependencies"]
        for name in ("tensorboard", "mlflow", "wandb", "bitsandbytes",
                     "ultralytics", "transformers", "rfdetr", "timm"):
            assert name in extras, f"extra '{name}' missing"

    def test_public_api_surface(self):
        import autovision

        for name in (
            "pretrain", "train_object_detection", "embed",
            "load_task_model", "load_embedding_model",
            "list_models", "list_methods", "extract_video_frames",
        ):
            assert callable(getattr(autovision, name, None)), f"{name} missing"
