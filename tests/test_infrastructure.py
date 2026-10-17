# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Infrastructure checks: CI configs, Docker, docs, release package, packaging."""

import fnmatch
import re
import subprocess
import sys
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
        assert "mkdocs" in str(config["docs"]["script"])

    def test_github_workflow_parses(self):
        pytest.importorskip("yaml")
        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert {"test", "lint", "docs"} <= set(workflow["jobs"])
        # PyYAML 1.1 parses the `on:` key as boolean True.
        trigger = workflow.get("on", workflow.get(True))
        # Release tags trigger the suite, which carries the tag ↔ version check.
        assert "tags" in trigger["push"]

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


def _nav_paths(nav, prefix: str = "") -> list[str]:
    """Flatten a mkdocs nav tree into the referenced page paths."""
    paths: list[str] = []
    if isinstance(nav, list):
        for entry in nav:
            paths.extend(_nav_paths(entry, prefix))
    elif isinstance(nav, dict):
        for entry in nav.values():
            paths.extend(_nav_paths(entry, prefix))
    elif isinstance(nav, str):
        paths.append(nav)
    return paths


class TestDocs:
    """Site contract (E1 release package): nav ⊆ docs/, docs/ ⊆ nav ∪ excluded.

    `mkdocs build --strict` is the renderer-level gate; these checks are the
    fast structural contract and run without mkdocs installed.
    """

    def test_mkdocs_config_nav_targets_exist(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        assert config["site_name"] == "AutoVision"
        nav_files = _nav_paths(config["nav"])
        assert nav_files, "nav must reference at least one page"
        for target in nav_files:
            assert (ROOT / "docs" / target).is_file(), (
                f"nav references missing page: {target}"
            )

    def test_every_docs_page_in_nav_or_excluded(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        in_nav = set(_nav_paths(config["nav"]))
        patterns = [
            line.strip()
            for line in str(config.get("exclude_docs", "")).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        pages = {
            path.relative_to(ROOT / "docs").as_posix()
            for path in (ROOT / "docs").rglob("*.md")
        }
        # mkdocs exclude_docs takes gitignore-style patterns (docs: "a
        # pattern may match a directory"). `dir/` excludes the whole dir;
        # anything else is fnmatch-style.
        excluded: set[str] = set()
        matched = {pattern: False for pattern in patterns}
        for page in pages:
            for pattern in patterns:
                hit = (
                    page.startswith(pattern.rstrip("/") + "/")
                    if pattern.endswith("/")
                    else fnmatch.fnmatch(page, pattern)
                )
                if hit:
                    excluded.add(page)
                    matched[pattern] = True
        orphans = pages - in_nav - excluded
        assert not orphans, f"pages neither in nav nor excluded: {sorted(orphans)}"
        # Exclusions must not be stale (they double as the "internal, not
        # published" register).
        stale = sorted(pattern for pattern, hit in matched.items() if not hit)
        assert not stale, f"stale exclude_docs entries: {stale}"

    def test_strategy_docs_stay_out_of_the_site(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        # The internal analysis draft is decision material, not a published
        # page — excluded from the site, still present in the repository.
        assert "strategy/gap-analysis.md" in str(config["exclude_docs"])
        assert not any(
            "strategy/" in target for target in _nav_paths(config["nav"])
        )

    def test_site_builds_strictly(self, tmp_path):
        pytest.importorskip("mkdocs")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-d",
                str(tmp_path / "site"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_no_broken_references_in_public_tree(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_docs_links.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_link_checker_detects_breakage(self, tmp_path, monkeypatch):
        """The guard itself: fences skipped for links, inline code skipped,
        repo paths checked everywhere (root + ``src/autovision/``
        convention), `<!-- docpath-ok -->` exempts one line."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "check_docs_links", ROOT / "scripts" / "check_docs_links.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ok.md").write_text(
            "see [x](index.md) and `scripts/real.py`\n"
            "inline code is fine: `[y](nowhere.md)`\n"
            "angle target: [z](<index.md>)\n"
            "line suffix: `scripts/real.py:12-34`\n"
            "module-map path: `cli/real.py`\n"
            "illustrative output path: `report/report.md`\n"
            "comment is fine: <!-- [w](nowhere.md) -->\n"
        )
        (tmp_path / "docs" / "index.md").write_text("# home\n")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "real.py").write_text("")
        (tmp_path / "src" / "autovision" / "cli").mkdir(parents=True)
        (tmp_path / "src" / "autovision" / "cli" / "real.py").write_text("")
        (tmp_path / "docs" / "bad.md").write_text(
            "fenced is fine:\n```\n[x](nowhere.md)\n```\n"
            "but this is not: [y](missing.md)\n"
            "and `scripts/gone.py` too\n"
            "and `cli/gone.py` (package-map miss)\n"
            "history: `scripts/gone.py` <!-- docpath-ok -->\n"
        )
        problems = module.check()
        assert any("missing.md" in p for p in problems), problems
        assert any("scripts/gone.py" in p for p in problems), problems
        assert any("cli/gone.py" in p for p in problems), problems
        assert not any("nowhere.md" in p for p in problems), problems
        assert sum("scripts/gone.py" in p for p in problems) == 1, problems
        assert not any(p.startswith("docs/ok.md") for p in problems), problems


class TestRelease:
    """Release-package contract (E1): version source of truth, changelog,
    contributor entry points and issue/PR templates."""

    def test_version_has_a_single_source(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert "version" not in config["project"]
        assert config["project"]["dynamic"] == ["version"]
        assert (
            config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
            == "autovision._version.__version__"
        )
        import autovision

        assert re.fullmatch(r"\d+\.\d+\.\d+", autovision.__version__), (
            "release tags are vX.Y.Z; the version string must be plain semver"
        )

    def test_changelog_follows_keep_a_changelog(self):
        text = (ROOT / "CHANGELOG.md").read_text()
        assert "Keep a Changelog" in text
        assert "## [Unreleased]" in text
        import autovision

        assert f"## [{autovision.__version__}]" in text, (
            "the current version needs a dated CHANGELOG section"
        )

    def test_release_tag_matches_version(self):
        """A HEAD tagged with a release tag must carry the same version.

        Development commits ship untagged and are unaffected; the release
        checklist runs this against the tagged commit (docs/release_process.md).
        """
        import autovision

        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("git unavailable")
        release_tags = [
            tag
            for tag in result.stdout.split()
            if re.fullmatch(r"v\d+\.\d+\.\d+", tag)
        ]
        if not release_tags:
            pytest.skip("HEAD is not a release-tagged commit")
        assert f"v{autovision.__version__}" in release_tags, (
            f"release tag/version mismatch: HEAD has {release_tags}, "
            f"package version is {autovision.__version__}"
        )

    def test_contributor_entry_points_exist(self):
        contributing = (ROOT / "CONTRIBUTING.md").read_text()
        for gate in (
            "ruff check src tests scripts",
            "pytest tests -q",
            "scripts/check_provenance.py",
            "scripts/check_layering.py",
            "scripts/check_docs_links.py",
            "mkdocs build --strict",
        ):
            assert gate in contributing, f"CONTRIBUTING.md must document: {gate}"

    def test_issue_and_pr_templates_parse(self):
        pytest.importorskip("yaml")
        import yaml

        template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
        for name in ("bug_report.yml", "feature_request.yml"):
            data = yaml.safe_load((template_dir / name).read_text())
            assert data["name"] and data["description"] and data["body"]
            for field in data["body"]:
                if field.get("type") in ("input", "textarea", "checkboxes"):
                    assert field.get("id"), f"{name}: a field misses its id"
        template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
        assert "ruff check src tests scripts" in template
        assert "CHANGELOG.md" in template


class TestPackaging:
    def test_cli_entry_point_registered(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        scripts = config["project"].get("scripts", {})
        assert scripts.get("autovision-cli") == "autovision.cli._app:cli_entrypoint"

    def test_optional_extras_declared(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extras = config["project"]["optional-dependencies"]
        for name in ("tensorboard", "mlflow", "wandb", "bitsandbytes",
                     "ultralytics", "transformers", "rfdetr", "timm",
                     "video", "docs", "openvino"):
            assert name in extras, f"extra '{name}' missing"

    def test_docs_extra_pins_the_site_toolchain(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        docs_extra = config["project"]["optional-dependencies"]["docs"]
        assert any(dep.startswith("mkdocs-material") for dep in docs_extra)

    def test_public_api_surface(self):
        import autovision

        for name in (
            "pretrain", "train_object_detection", "embed",
            "load_task_model", "load_embedding_model",
            "list_models", "list_methods", "extract_video_frames",
        ):
            assert callable(getattr(autovision, name, None)), f"{name} missing"
