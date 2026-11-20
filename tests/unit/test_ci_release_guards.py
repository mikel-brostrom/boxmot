"""Exercise required database coverage and the public CI/developer checks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.unit import conftest

ROOT = Path(__file__).resolve().parents[2]


def test_required_postgres_mode_rejects_missing_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOXMOT_REQUIRE_POSTGRES_TESTS", "1")
    monkeypatch.delenv("BOXMOT_TEST_POSTGRES_URL", raising=False)
    with pytest.raises(pytest.UsageError, match="need BOXMOT_TEST_POSTGRES_URL"):
        conftest.pytest_configure(None)


def test_required_postgres_mode_rejects_missing_optional_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOXMOT_REQUIRE_POSTGRES_TESTS", "1")
    monkeypatch.setenv("BOXMOT_TEST_POSTGRES_URL", "postgresql://unused")

    def missing_driver(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(conftest.importlib, "import_module", missing_driver)
    with pytest.raises(pytest.UsageError, match="need psycopg"):
        conftest.pytest_configure(None)


def test_local_tests_keep_database_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOXMOT_REQUIRE_POSTGRES_TESTS", raising=False)
    monkeypatch.delenv("BOXMOT_TEST_POSTGRES_URL", raising=False)
    conftest.pytest_configure(None)


@pytest.mark.parametrize(("workflow_name", "job_name"), (("wheels", "source_gates"), ("ci", "tests")))
def test_ci_and_release_require_real_database_coverage(workflow_name: str, job_name: str) -> None:
    job = yaml.safe_load((ROOT / f".github/workflows/{workflow_name}.yml").read_text())["jobs"][job_name]
    assert job["services"]["postgres"]["env"]["POSTGRES_DB"] == "boxmot_test"
    assert job["env"]["BOXMOT_REQUIRE_POSTGRES_TESTS"] == "1"
    assert job["env"]["BOXMOT_TEST_POSTGRES_URL"].endswith(":5432/boxmot_test")
    assert "--extra attributes" in "\n".join(step.get("run", "") for step in job["steps"])


def test_boundary_python_contracts_are_required_for_prs() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert jobs["python_contracts"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.13"]
    assert "python_contracts" in jobs["check-failures"]["needs"]
    assert "needs.python_contracts.result" in jobs["check-failures"]["steps"][0]["run"]


def test_precommit_uses_locked_ruff_and_renders_helm_before_yaml_validation() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    repositories = config["repos"]
    ruff = next(repo for repo in repositories if repo["repo"].endswith("/ruff-pre-commit"))
    lock_version = re.search(r'name = "ruff"\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text()).group(1)
    assert ruff["rev"] == f"v{lock_version}"
    yaml_hook = next(hook for repo in repositories for hook in repo["hooks"] if hook["id"] == "check-yaml")
    assert "--allow-multiple-documents" in yaml_hook["args"]
    assert re.search(yaml_hook["exclude"], "deploy/helm/boxmot/templates/cameras.yaml")
    assert not re.search(yaml_hook["exclude"], "deploy/helm/boxmot/values.yaml")
    helm_hook = next(hook for repo in repositories for hook in repo["hooks"] if hook["id"] == "helm-contract")
    assert helm_hook["pass_filenames"] is False
    assert "python -m tests.ci.helm_contract" in helm_hook["entry"]


@pytest.mark.parametrize("name", ["helm", "helm-release"])
def test_chart_validation_installs_real_config_loader_dependencies(name: str) -> None:
    workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
    jobs = workflow["jobs"]
    job = jobs["configuration"] if name == "helm" else jobs["publish"]
    installation = next(
        step["run"] for step in job["steps"] if step.get("name") == "Install chart validation dependencies"
    )
    assert "PyYAML==6.0.2" in installation
    assert "platformdirs==4.9.4" in installation
