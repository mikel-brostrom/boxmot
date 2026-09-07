from __future__ import annotations

from collections.abc import Iterator

from boxmot.engine.materialization.builder import DatasetMaterializer
from boxmot.engine.materialization.ids import fingerprint
from boxmot.engine.materialization.plan import BuildPlan, PublishOptions, StagePlan
from boxmot.engine.materialization.progress import MaterializationProgress
from boxmot.engine.materialization.stages.base import StageOutcome


class RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)


class FixedClock:
    def __init__(self, *values: float) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _plan(tmp_path, stage: StagePlan, *, source_count: int = 4) -> BuildPlan:
    return BuildPlan.create(
        build_root=tmp_path,
        dataset_name="progress",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(),
        stages=(stage,),
        metadata={"source_count": source_count},
    )


def test_progress_reports_build_resume_and_durable_shards(tmp_path) -> None:
    stage = StagePlan.create("detect", batch_size=2, workers=2, executor="thread")
    plan = _plan(tmp_path, stage)
    logger = RecordingLogger()
    progress = MaterializationProgress(logger=logger, clock=FixedClock(10.0, 12.0, 14.0))

    progress.build_started(plan)
    progress.stage_started(plan, stage, completed_shards=1)
    progress.shard_completed("detect", "00001", completed_shards=2, items=2, rows=7)
    progress.stage_completed(stage, StageOutcome(metrics={"samples": 4}), completed_shards=2)

    assert logger.info_messages == [
        f"Materialization build: {plan.build_id}",
        "Source samples: 4",
        f"Output directory: {plan.output_root}",
        f"Staging directory: {plan.staging_root}",
        "Stage detect starting: 1/2 shards complete, 1 pending; batch=2, workers=2, executor=thread.",
        "Stage detect shard 00001: 2/2 (100.0%), items=2, rows=7, elapsed=2.0s, "
        "rate=0.50 shards/s, ETA=0.0s.",
        "Stage detect complete: 2/2 shards; elapsed=4.0s; samples=4.",
    ]


def test_materializer_reports_retry_and_completion(tmp_path) -> None:
    stage_plan = StagePlan.create("prepare", max_attempts=2)
    plan = _plan(tmp_path, stage_plan)
    logger = RecordingLogger()
    progress = MaterializationProgress(logger=logger, clock=FixedClock(1.0, 3.0))

    class FlakyStage:
        name = "prepare"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, context) -> StageOutcome:
            del context
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary outage")
            return StageOutcome(metrics={"recovered": True})

    result = DatasetMaterializer(plan, [FlakyStage()], progress=progress).run()

    assert result == plan.staging_root
    assert logger.warning_messages == [
        "Stage prepare attempt 1/2 failed (RuntimeError: temporary outage); "
        "retrying attempt 2/2 in 0.0s."
    ]
    assert logger.error_messages == []
    assert logger.info_messages[-1] == "Stage prepare complete: 0 shards; elapsed=2.0s; recovered=True."


def test_progress_reports_terminal_failure_and_complete_build_reuse(tmp_path) -> None:
    stage = StagePlan.create("prepare")
    plan = _plan(tmp_path, stage)
    logger = RecordingLogger()
    progress = MaterializationProgress(logger=logger, clock=FixedClock(0.0))

    progress.stage_started(plan, stage, completed_shards=0)
    progress.stage_failed(
        stage.name,
        ValueError("bad input"),
        attempt=1,
        max_attempts=1,
        staging_root=plan.staging_root,
    )
    progress.build_reused(plan.output_root)

    assert logger.error_messages == [
        f"Stage prepare failed after attempt 1/1 (ValueError: bad input); "
        f"resumable state remains at {plan.staging_root}."
    ]
    assert logger.info_messages[-1] == f"Reusing complete materialization build: {plan.output_root}"
