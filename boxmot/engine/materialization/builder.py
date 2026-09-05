"""Resumable local orchestration for materialization stage DAGs."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from filelock import FileLock

from boxmot.datasets import DatasetManifest
from boxmot.datasets.manifest import PublishedContent, StageProvenance
from boxmot.datasets.schema import ARTIFACT_PATHS, MANIFEST_FILENAME, SUCCESS_FILENAME
from boxmot.datasets.validation import validate_published_build

from .executor import ExecutorSpec, StageExecutor, create_executor
from .plan import BuildPlan, StagePlan
from .progress import MaterializationProgress, MaterializationProgressReporter
from .stages.base import MaterializationContext, MaterializationStage, StageOutcome
from .state import MaterializationStateStore

ExecutorFactory = Callable[[StagePlan], StageExecutor]


class DatasetMaterializer:
    """Execute a deterministic stage plan with atomic resumability state."""

    def __init__(
        self,
        plan: BuildPlan,
        stages: Sequence[MaterializationStage],
        *,
        executor_factory: ExecutorFactory | None = None,
        progress: MaterializationProgressReporter | None = None,
    ) -> None:
        self.plan = plan
        self._stages = {stage.name: stage for stage in stages}
        if len(self._stages) != len(stages):
            raise ValueError("Materialization stage implementations must have unique names.")
        expected = set(plan.stage_by_name)
        actual = set(self._stages)
        if actual != expected:
            raise ValueError(
                f"Stage implementations do not match the plan; missing={sorted(expected - actual)!r}, "
                f"extra={sorted(actual - expected)!r}."
            )
        ordered = plan.ordered_stages()
        finalize_indices = [index for index, stage in enumerate(ordered) if stage.name == "finalize"]
        if finalize_indices and finalize_indices != [len(ordered) - 1]:
            raise ValueError("The finalize stage must be last.")
        self._executor_factory = executor_factory or self._default_executor
        self.progress = progress or MaterializationProgress()
        self.state = MaterializationStateStore(plan.state_path)
        self._lock_path = plan.build_root / ".locks" / f"{plan.build_id}.lock"

    @staticmethod
    def _default_executor(stage: StagePlan) -> StageExecutor:
        kind = stage.executor
        max_workers = 1 if kind == "inline" else stage.workers
        return create_executor(ExecutorSpec(kind=kind, max_workers=max_workers))

    def run(self) -> Path:
        """Run pending stages and return the published or staging root."""

        self.progress.build_started(self.plan)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress.build_lock_waiting(self._lock_path)
        with FileLock(self._lock_path):
            self.progress.build_lock_acquired()
            return self._run_locked()

    def _run_locked(self) -> Path:
        """Execute while holding the per-build inter-process lock."""

        if self.plan.output_root.is_dir():
            output = self._validate_published()
            self.progress.build_reused(output)
            return output

        self.state.initialize(self.plan)
        self.plan.staging_root.mkdir(parents=True, exist_ok=True)
        for stage_plan in self.plan.ordered_stages():
            revalidate_shards = stage_plan.name in {"detect", "segment", "embed"}
            recover_finalize = stage_plan.name == "finalize" and not self.plan.output_root.is_dir()
            if self.state.is_completed(stage_plan.name) and not (revalidate_shards or recover_finalize):
                self.progress.stage_skipped(stage_plan.name)
                continue
            self.progress.stage_started(
                self.plan,
                stage_plan,
                completed_shards=len(self.state.state.by_name[stage_plan.name].completed_shards),
            )
            published = self._run_stage_with_retries(stage_plan)
            if published is not None:
                return published
        result = self.plan.output_root if self.plan.output_root.is_dir() else self.plan.staging_root
        if result == self.plan.output_root:
            self.progress.build_completed(result)
        return result

    def _run_stage_with_retries(self, stage_plan: StagePlan) -> Path | None:
        """Run one stage, retaining its runtime across retries and releasing it afterward."""

        implementation = self._stages[stage_plan.name]
        try:
            for attempt in range(stage_plan.max_attempts):
                executor: StageExecutor | None = None
                try:
                    self.state.begin(stage_plan.name)
                    executor = self._executor_factory(stage_plan)
                    context = MaterializationContext(
                        build_plan=self.plan,
                        stage_plan=stage_plan,
                        staging_root=self.plan.staging_root,
                        state=self.state,
                        executor=executor,
                        progress=self.progress,
                    )
                    outcome = implementation.run(context)
                    if not isinstance(outcome, StageOutcome):
                        raise TypeError(
                            f"Stage {stage_plan.name!r} returned {type(outcome).__name__}, expected StageOutcome."
                        )
                    executor.close()
                    executor = None
                    if not self.state.is_completed(stage_plan.name):
                        self.state.complete(stage_plan.name, artifacts=outcome.artifacts)
                    if context.repaired_shards:
                        self._invalidate_descendants(stage_plan.name)
                    self.progress.stage_completed(
                        stage_plan,
                        outcome,
                        completed_shards=len(self.state.state.by_name[stage_plan.name].completed_shards),
                    )
                    break
                except BaseException as exc:
                    if executor is not None:
                        try:
                            executor.close()
                        except BaseException as close_error:
                            exc.add_note(f"Executor cleanup also failed: {close_error}")
                    if self.plan.output_root.is_dir():
                        output = self._validate_published()
                        self.progress.build_reused(output)
                        return output
                    self.state.fail(stage_plan.name, exc)
                    if not isinstance(exc, Exception):
                        self.progress.stage_failed(
                            stage_plan.name,
                            exc,
                            attempt=attempt + 1,
                            max_attempts=stage_plan.max_attempts,
                            staging_root=self.plan.staging_root,
                        )
                        raise
                    if attempt + 1 >= stage_plan.max_attempts:
                        self.progress.stage_failed(
                            stage_plan.name,
                            exc,
                            attempt=attempt + 1,
                            max_attempts=stage_plan.max_attempts,
                            staging_root=self.plan.staging_root,
                        )
                        raise
                    delay_s = stage_plan.retry_backoff_s * (2**attempt)
                    self.progress.stage_retry(
                        stage_plan.name,
                        exc,
                        attempt=attempt + 1,
                        max_attempts=stage_plan.max_attempts,
                        delay_s=delay_s,
                    )
                    if delay_s:
                        time.sleep(delay_s)
        finally:
            release = getattr(implementation, "release", None)
            if callable(release):
                release()
        return None

    def _invalidate_descendants(self, stage_name: str) -> None:
        """Discard every transitive consumer of a repaired upstream shard."""

        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            for candidate in self.plan.stages:
                if candidate.name == stage_name or candidate.name in descendants:
                    continue
                if stage_name in candidate.depends_on or descendants.intersection(candidate.depends_on):
                    descendants.add(candidate.name)
                    changed = True
        if not descendants:
            return

        current_states = self.state.state.by_name
        owned_artifacts: set[str] = set()
        default_artifacts = {
            "segment": ("masks",),
            "embed": ("embeddings",),
            "finalize": (MANIFEST_FILENAME, SUCCESS_FILENAME),
        }
        for descendant in descendants:
            owned_artifacts.update(current_states[descendant].artifacts)
            owned_artifacts.update(default_artifacts.get(descendant, ()))
        self.state.invalidate(tuple(sorted(descendants)))

        for artifact in owned_artifacts:
            if artifact in ARTIFACT_PATHS:
                relative = ARTIFACT_PATHS[artifact]
            elif artifact in {MANIFEST_FILENAME, SUCCESS_FILENAME}:
                relative = artifact
            else:
                continue
            path = self.plan.staging_root / relative
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _validate_published(self) -> Path:
        manifest = DatasetManifest.load(self.plan.output_root)
        expected_stages = tuple(
            StageProvenance(
                name=stage.name,
                fingerprint=stage.fingerprint,
                batch_size=stage.batch_size,
                inputs=stage.depends_on,
                component=stage.component,
                config=stage.config,
            )
            for stage in self.plan.ordered_stages()
        )
        expected_publish = PublishedContent(
            image_references=self.plan.publish.image_references,
            masks=self.plan.publish.masks,
            embeddings=self.plan.publish.embeddings,
        )
        provenance_matches = (
            manifest.build_id == self.plan.build_id
            and manifest.box_type == self.plan.box_type
            and manifest.publish == expected_publish
            and manifest.stages == expected_stages
            and manifest.metadata.get("dataset_name") == self.plan.dataset_name
            and manifest.metadata.get("source_fingerprint") == self.plan.source_fingerprint
            and manifest.metadata.get("experiment_id") == self.plan.metadata.get("experiment_id")
        )
        if not provenance_matches:
            raise RuntimeError("Published build provenance does not match the requested materialization plan.")
        validate_published_build(self.plan.output_root, manifest=manifest)
        return self.plan.output_root


__all__ = ("DatasetMaterializer", "ExecutorFactory")
