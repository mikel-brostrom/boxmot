"""Test namespace ownership and recovery-harness cleanup without cluster access."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.helm_streaming import HelmTest
from tests.e2e.support import OWNER_LABEL


@pytest.fixture
def harness(tmp_path: Path) -> HelmTest:
    """Prepare filesystem state only; no commands or image pulls are performed."""
    test = HelmTest(argparse.Namespace(
        output=str(tmp_path / "evidence"),
        namespace="boxmot-helm-e2e-unit",
        context="isolated-test-context",
        image="boxmot/boxmot@sha256:" + "a" * 64,
        vllm_image=None,
        external_vllm_url=None,
        prepare_only=False,
    ))
    test.values = {"vllm": {"image": test.defaults["vllm"]["image"]}}
    test.runtime_version = test.chart["appVersion"]
    return test


def test_diagnostic_timeouts_do_not_prevent_owned_namespace_cleanup(
    harness: HelmTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.owned = True
    calls = []

    def kubectl(*arguments: str, **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(arguments)
        if arguments[:2] == ("get", "namespace"):
            output = json.dumps({"metadata": {"labels": {OWNER_LABEL: harness.owner}}})
        elif arguments[:2] == ("delete", "namespace"):
            output = "deleted"
        else:
            raise subprocess.TimeoutExpired(["kubectl", *arguments], 120)
        return subprocess.CompletedProcess(arguments, 0, output)

    monkeypatch.setattr(harness, "kubectl", kubectl)
    monkeypatch.setattr("tests.e2e.support.subprocess.check_output", lambda *_args, **_kwargs: "d" * 40)
    harness.finish(None)

    result = json.loads((harness.directory / "result.json").read_text())
    assert len(result["diagnostic_errors"]) == 2
    assert result["cleanup_error"] is None
    assert any(call[:3] == ("delete", "namespace", harness.namespace) for call in calls)


def test_changed_namespace_owner_prevents_deletion_and_promotion_evidence(
    harness: HelmTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.owned = True
    calls = []

    def kubectl(*arguments: str, **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(arguments)
        output = json.dumps({"metadata": {"labels": {OWNER_LABEL: "another-run"}}})
        return subprocess.CompletedProcess(arguments, 0, output)

    monkeypatch.setattr(harness, "kubectl", kubectl)
    with pytest.raises(RuntimeError, match="Namespace owner changed"):
        harness.finish(None)

    assert not any(call[0] == "delete" for call in calls)
    assert not (harness.directory / "deployment-images.json").exists()
    assert "Namespace owner changed" in json.loads((harness.directory / "result.json").read_text())["cleanup_error"]


def test_failed_run_does_not_emit_promotable_image_evidence(
    harness: HelmTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_cluster_access(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A run that never owned a namespace must not mutate the cluster")

    monkeypatch.setattr(harness, "kubectl", no_cluster_access)
    harness.finish("Fixture validation failed")

    assert not (harness.directory / "deployment-images.json").exists()
    assert json.loads((harness.directory / "result.json").read_text())["error"] == "Fixture validation failed"


def test_unexecuted_run_does_not_emit_promotable_image_evidence(harness: HelmTest) -> None:
    harness.finish(None)

    assert not (harness.directory / "deployment-images.json").exists()
    assert json.loads((harness.directory / "result.json").read_text())["completed"] is False


def test_capacity_success_never_emits_promotable_image_evidence(harness: HelmTest, monkeypatch) -> None:
    from tests.e2e.helm_capacity import CapacityTest

    test = CapacityTest(argparse.Namespace(**{**vars(harness.args), "output": str(harness.directory / "capacity")}))
    test.owned = True
    test.scenario_completed = True
    monkeypatch.setattr(test, "kubectl", lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 0, json.dumps({"metadata": {"labels": {OWNER_LABEL: test.owner}}}),
    ))

    test.finish(None)

    assert not (test.directory / "deployment-images.json").exists()
    result = json.loads((test.directory / "result.json").read_text())
    assert result["scenario"] == "capacity" and result["completed"] is True


def test_owned_manifests_use_server_side_apply_without_recording_secret_input(harness: HelmTest, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(harness, "kubectl", lambda *args, **kwargs: calls.append((args, kwargs)))
    fixture = {"kind": "ConfigMap", "binaryData": {"frame.jpg": "a" * 300_000}}

    harness.apply(fixture)

    arguments, keywords = calls[0]
    assert "--server-side" in arguments
    assert "a" * 100 not in " ".join(arguments)
    assert json.loads(keywords["input_text"])["items"] == [fixture]


def test_existing_namespace_is_never_adopted_or_modified(
    harness: HelmTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def kubectl(*arguments: str, **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, f"namespace/{harness.namespace}\n")

    monkeypatch.setattr(harness, "kubectl", kubectl)
    with pytest.raises(RuntimeError, match="Refusing to adopt existing namespace"):
        harness.run()

    assert len(calls) == 1 and calls[0][:2] == ("get", "namespace")
    assert harness.owned is False


@pytest.mark.parametrize("external_endpoint", [None, "http://existing-vllm:8000"])
def test_initial_install_waits_only_for_running_dependencies(
    harness: HelmTest, monkeypatch: pytest.MonkeyPatch, external_endpoint: str | None
) -> None:
    """Suspended camera PVCs cannot bind on WaitForFirstConsumer storage."""
    harness.args.external_vllm_url = external_endpoint
    calls = []

    def command(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "ready")

    monkeypatch.setattr(harness, "command", command)
    harness.install(first=True)

    assert calls[0][0] == "helm" and "--wait" not in calls[0]
    dependencies = [call[call.index("status") + 1] for call in calls[1:]]
    assert dependencies == ["deployment/smoke-postgres", "deployment/fixture"] + (
        [] if external_endpoint else ["deployment/smoke-vllm"]
    )
    assert all("isolated-test-context" in call for call in calls)
    calls.clear()
    harness.install()
    assert len(calls) == 1 and "--wait" in calls[0]
