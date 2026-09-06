from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import boxmot.engine.materialization.workflow as materialize_workflow
import boxmot.engine.ui.core.ui as ui
import boxmot.native._common as native_common
from boxmot.engine.materialization.ids import fingerprint
from boxmot.engine.materialization.plan import BuildPlan, PublishOptions, StagePlan
from boxmot.engine.materialization.stages.base import StageOutcome
from boxmot.engine.ui.reporters.materialize import MATERIALIZE, MaterializeWorkflowReporter
from boxmot.engine.ui.workflow.steps import MATERIALIZE_STEPS
from boxmot.resources import download


class FixedClock:
    def __init__(self, *values: float) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _plan(tmp_path: Path) -> tuple[BuildPlan, StagePlan, StagePlan]:
    detect = StagePlan.create("detect", batch_size=2, workers=1)
    finalize = StagePlan.create(
        "finalize",
        upstream_fingerprints=(detect.fingerprint,),
        depends_on=("detect",),
    )
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="mot17",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(embeddings=False),
        stages=(detect, finalize),
        metadata={
            "source_count": 4,
            "split": "ablation",
            "components": {
                "detector": {"spec": {"backend": "fixture", "device": "mps"}},
                "segmentor": None,
                "reid": None,
            },
        },
    )
    return plan, detect, finalize


def _reporter(monkeypatch, tmp_path: Path, *, clock: FixedClock) -> MaterializeWorkflowReporter:
    workflow = ui.create_workflow_progress(
        "Dataset Materialization",
        (),
        steps=MATERIALIZE_STEPS,
        stderr=True,
    )
    monkeypatch.setattr(workflow, "start", lambda: workflow)
    monkeypatch.setattr(workflow, "stop", lambda **_kwargs: None)
    monkeypatch.setattr(ui, "print_renderable", lambda *args, **kwargs: None)
    reporter = MaterializeWorkflowReporter(
        SimpleNamespace(),
        workflow=workflow,
        clock=clock,
        refresh_interval_s=None,
    )
    reporter.start()
    return reporter


def test_rich_progress_renders_resume_stage_eta_and_atomic_finalize(monkeypatch, tmp_path) -> None:
    reporter = _reporter(monkeypatch, tmp_path, clock=FixedClock(10.0, 20.0, 30.0, 40.0, 50.0))
    plan, detect, finalize = _plan(tmp_path)

    reporter.build_started(plan)
    reporter.build_lock_waiting(plan.build_root / ".locks" / f"{plan.build_id}.lock")
    assert "waiting for build lock" in str(reporter.stage_progress.tasks[0].fields["summary"])
    reporter.build_lock_acquired()
    assert "build lock acquired" in str(reporter.stage_progress.tasks[0].fields["summary"])
    reporter.stage_started(plan, detect, completed_shards=1)
    reporter.shard_completed("detect", "00001", completed_shards=2, items=2, rows=7)
    progress = reporter.stage_progress
    assert progress is not None
    shard_summary = str(progress.tasks[0].fields["summary"])
    assert "2/2 shards" in shard_summary
    assert "2 items" in shard_summary
    assert "ETA 0.0s" in shard_summary
    assert reporter.workflow is not None
    rendered = ui.capture_renderable(reporter.workflow.renderable(compact=True), width=160)
    assert "Dataset Materialization" in rendered
    assert "fixture · mps" in rendered
    assert "Detect" in rendered
    assert "2/2 shards" in rendered
    assert "Finalize" in rendered
    reporter.stage_completed(detect, StageOutcome(metrics={"samples": 4}), completed_shards=2)
    reporter.stage_started(plan, finalize, completed_shards=0)
    reporter.stage_completed(finalize, StageOutcome(), completed_shards=0)

    detect_task, finalize_task = progress.tasks
    assert detect_task.completed == detect_task.total == 2
    assert "complete in 20.0s" in str(detect_task.fields["summary"])
    assert finalize_task.completed == finalize_task.total == 1

    reporter.build_completed(plan.output_root)
    assert reporter.workflow.steps == [("Set up", "done"), (MATERIALIZE, "done")]
    assert reporter.workflow.detail_title == "Build complete"
    reporter.stop()


def test_rich_progress_preserves_resume_location_on_failure(monkeypatch, tmp_path) -> None:
    reporter = _reporter(monkeypatch, tmp_path, clock=FixedClock(1.0))
    plan, detect, _finalize = _plan(tmp_path)
    reporter.build_started(plan)
    reporter.stage_started(plan, detect, completed_shards=1)
    error = RuntimeError("detector failed")

    reporter.stage_failed(
        "detect",
        error,
        attempt=3,
        max_attempts=3,
        staging_root=plan.staging_root,
    )
    reporter.unhandled_failure(error)

    assert getattr(error, "_workflow_rendered_error") is True
    assert any(str(plan.staging_root) in note for note in error.__notes__)
    assert reporter.workflow is not None
    assert reporter.workflow.steps[-1] == (MATERIALIZE, "failed")
    assert "detector failed" in (reporter.workflow.detail_text or "")
    reporter.stop()


def test_materialize_main_selects_rich_reporter_for_terminal(monkeypatch, tmp_path) -> None:
    events: list[object] = []
    output = tmp_path / "build"

    class FakeProgress:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

        def unhandled_failure(self, error: BaseException) -> None:
            events.append(error)

    progress = FakeProgress()
    monkeypatch.setattr(materialize_workflow, "get_console", lambda **_kwargs: SimpleNamespace(is_terminal=True))
    monkeypatch.setattr(materialize_workflow, "MaterializeWorkflowReporter", lambda _args: progress)

    def materialize(_args, *, progress):
        events.append(progress)
        return output

    monkeypatch.setattr(materialize_workflow, "materialize", materialize)

    result = materialize_workflow.main(SimpleNamespace())

    assert result == output
    assert events == ["start", progress, "stop"]


def test_materialize_main_uses_plain_progress_without_terminal(monkeypatch) -> None:
    events: list[object] = []

    class FakeProgress:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

        def unhandled_failure(self, error: BaseException) -> None:
            events.append(error)

    progress = FakeProgress()
    monkeypatch.setattr(materialize_workflow, "get_console", lambda **_kwargs: SimpleNamespace(is_terminal=False))
    monkeypatch.setattr(materialize_workflow, "MaterializationProgress", lambda: progress)
    monkeypatch.setattr(
        materialize_workflow,
        "materialize",
        lambda _args, *, progress: events.append(progress),
    )

    materialize_workflow.main(SimpleNamespace())

    assert events == ["start", progress, "stop"]


def test_materialize_main_stops_rich_reporter_after_failure(monkeypatch) -> None:
    events: list[object] = []
    error = RuntimeError("broken setup")

    class FakeProgress:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

        def unhandled_failure(self, received: BaseException) -> None:
            events.append(received)

    progress = FakeProgress()
    monkeypatch.setattr(materialize_workflow, "get_console", lambda **_kwargs: SimpleNamespace(is_terminal=True))
    monkeypatch.setattr(materialize_workflow, "MaterializeWorkflowReporter", lambda _args: progress)

    def fail(_args, *, progress) -> None:
        del progress
        raise error

    monkeypatch.setattr(materialize_workflow, "materialize", fail)

    with pytest.raises(RuntimeError, match="broken setup"):
        materialize_workflow.main(SimpleNamespace())

    assert events == ["start", error, "stop"]


def test_rich_progress_routes_shared_download_and_build_status(monkeypatch, tmp_path) -> None:
    download_callbacks: list[object] = []
    build_callbacks: list[object] = []
    monkeypatch.setattr(download, "set_download_status_fn", download_callbacks.append)
    monkeypatch.setattr(native_common, "set_build_status_fn", build_callbacks.append)
    monkeypatch.setattr(ui, "print_renderable", lambda *args, **kwargs: None)
    workflow = ui.create_workflow_progress(
        "Dataset Materialization",
        (),
        steps=MATERIALIZE_STEPS,
        stderr=True,
    )
    monkeypatch.setattr(workflow, "start", lambda: workflow)
    monkeypatch.setattr(workflow, "stop", lambda **_kwargs: None)
    reporter = MaterializeWorkflowReporter(
        SimpleNamespace(),
        workflow=workflow,
        clock=FixedClock(1.0),
        refresh_interval_s=None,
    )

    reporter.start()
    assert download_callbacks[-1].step == "Set up"
    assert build_callbacks[-1].step == "Set up"

    plan, _detect, _finalize = _plan(tmp_path)
    reporter.build_started(plan)
    assert download_callbacks[-1].step == MATERIALIZE
    assert build_callbacks[-1].step == MATERIALIZE

    reporter.stop()
    assert download_callbacks[-1] is None
    assert build_callbacks[-1] is None
