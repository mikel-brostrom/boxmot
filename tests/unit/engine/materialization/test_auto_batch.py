"""Automatic materialization batching without loading models or CUDA hardware."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from boxmot.detectors.protocols import DetectorCapabilities
from boxmot.detectors.specs import DetectorSpec
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    PublishOptions,
    StagePlan,
    fingerprint,
    workflow,
)
from boxmot.engine.materialization.catalog import SourceCatalog
from boxmot.engine.materialization.detection_cache import DetectionCache
from boxmot.engine.materialization.settings import load_executor_settings
from boxmot.engine.materialization.source import SourceSample
from boxmot.engine.materialization.state import MaterializationStateStore
from boxmot.engine.profiling.batching import profile_detector
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.utils.batching import BatchMeasurement, BatchProfile


@pytest.fixture
def materialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Exercise workflow, builder locks, and state with inert stage execution."""

    sample = SourceSample(
        sample_id="train:seq:0",
        split="train",
        sequence_id="seq",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "frame.png").as_uri(),
        source_sha256=fingerprint("frame"),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint("catalog"),
        source_root=tmp_path,
        metadata={"source_catalog_digest": fingerprint("source")},
    )
    detector = DetectorSpec(backend="ultralytics", artifact="model.pt", device="cuda:0")
    encoder = ReIDEncoderSpec(backend="pytorch", artifact="osnet.pt", device="cuda:0")
    context = SimpleNamespace(calls=[], instances=[], detector=detector, encoder=encoder, catalog=catalog)
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda *args, **kwargs: ("fixture", "aabb", catalog, "detector", None, "reid", {}),
    )
    monkeypatch.setattr(
        workflow, "resolve_detector_spec", lambda *args, **kwargs: (detector, {"spec": asdict(detector)})
    )
    monkeypatch.setattr(workflow, "resolve_reid_spec", lambda *args, **kwargs: (encoder, {"spec": asdict(encoder)}))
    monkeypatch.setattr(workflow, "detector_capabilities", lambda spec: DetectorCapabilities())
    monkeypatch.setattr(DetectionCache, "available", lambda self: False)

    class CapturingMaterializer(DatasetMaterializer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            context.instances.append(self)

        def _run_stage_with_retries(self, stage_plan: StagePlan) -> Path | None:
            self.state.begin(stage_plan.name)
            self.state.complete(stage_plan.name)
            return self.plan.output_root if stage_plan.name == "finalize" else None

    monkeypatch.setattr(workflow, "DatasetMaterializer", CapturingMaterializer)

    def profile(kind: str, spec: object, **options: object) -> BatchProfile:
        # A separate lock object cannot enter the same build while profiling.
        with pytest.raises(Timeout), FileLock(context.instances[-1]._lock_path, timeout=0):
            pass
        context.calls.append((kind, spec, options))
        batch_size = 32 if kind == "detect" else 128
        return BatchProfile(
            batch_size=batch_size, measurements=(BatchMeasurement(batch_size, 0.5, batch_size * 2, 10),)
        )

    monkeypatch.setattr(workflow, "profile_detector", lambda spec, **kwargs: profile("detect", spec, **kwargs))
    monkeypatch.setattr(workflow, "profile_reid", lambda spec, **kwargs: profile("embed", spec, **kwargs))
    context.args = SimpleNamespace(
        device="cuda:0",
        materialize_explicit_keys=(),
        build_root=tmp_path / "builds",
        plan_path=None,
        plan_overrides=(),
        publish_image_refs=True,
        publish_masks=False,
        publish_embeddings=True,
        resume=True,
        auto_batch=False,
        auto_batch_max_size=256,
        auto_batch_memory_fraction=0.8,
    )
    return context


def test_auto_batch_profiles_active_stages_and_persists_selected_sizes(materialization: SimpleNamespace) -> None:
    materialization.args.auto_batch = True
    materialization.args.auto_batch_max_size = 128
    materialization.args.auto_batch_memory_fraction = 0.7

    workflow.materialize(materialization.args)

    instance = materialization.instances[-1]
    assert [call[0] for call in materialization.calls] == ["detect", "embed"]
    assert all(call[2] == {"max_batch_size": 128, "memory_fraction": 0.7} for call in materialization.calls)
    assert instance.plan.stage_by_name["detect"].batch_size == 32
    assert instance.plan.stage_by_name["embed"].batch_size == 128
    persisted = MaterializationStateStore(instance.plan.state_path).initialize(instance.plan)
    assert persisted.by_name["detect"].batch_size == 32
    assert persisted.by_name["embed"].batch_size == 128


def test_numeric_defaults_do_not_profile_and_keep_content_identity(materialization: SimpleNamespace) -> None:
    workflow.materialize(materialization.args)
    numeric_plan = materialization.instances[-1].plan
    assert materialization.calls == []
    assert numeric_plan.stage_by_name["detect"].batch_size == 16
    assert numeric_plan.stage_by_name["embed"].batch_size == 64

    materialization.args.auto_batch = True
    materialization.args.resume = False
    workflow.materialize(materialization.args)
    assert materialization.instances[-1].plan.build_id == numeric_plan.build_id
    assert materialization.instances[-1].plan.stage_by_name["detect"].batch_size == 32


@pytest.mark.parametrize("stage", ["detect", "embed"])
def test_per_stage_auto_profiles_only_requested_stage(materialization: SimpleNamespace, stage: str) -> None:
    materialization.args.plan_overrides = (f"{stage}.batch_size=auto",)

    workflow.materialize(materialization.args)

    assert [call[0] for call in materialization.calls] == [stage]
    other = "embed" if stage == "detect" else "detect"
    assert materialization.instances[-1].plan.stage_by_name[other].batch_size == (64 if other == "embed" else 16)


def test_native_embeddings_do_not_profile_an_unused_reid(
    materialization: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization.args.auto_batch = True
    monkeypatch.setattr(workflow, "detector_capabilities", lambda spec: DetectorCapabilities(provides_embeddings=True))

    workflow.materialize(materialization.args)

    assert [call[0] for call in materialization.calls] == ["detect"]
    assert "embed" not in materialization.instances[-1].plan.stage_by_name


def test_published_build_reuse_does_not_profile(
    materialization: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materialization.args.auto_batch = True
    cached = tmp_path / "published" / fingerprint("build")
    monkeypatch.setattr(workflow, "find_matching_build", lambda *args, **kwargs: cached)

    assert workflow.materialize(materialization.args) == cached
    assert materialization.calls == []
    assert materialization.instances == []


def test_shared_detection_cache_skips_detector_profiling(
    materialization: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization.args.auto_batch = True
    monkeypatch.setattr(DetectionCache, "available", lambda self: True)

    workflow.materialize(materialization.args)

    assert [call[0] for call in materialization.calls] == ["embed"]
    assert materialization.instances[-1].plan.stage_by_name["detect"].batch_size == 16


def test_resumed_completed_stages_keep_sizes_without_profiling(materialization: SimpleNamespace) -> None:
    materialization.args.auto_batch = True
    workflow.materialize(materialization.args)
    materialization.calls.clear()

    workflow.materialize(materialization.args)

    assert materialization.calls == []
    assert materialization.instances[-1].plan.stage_by_name["detect"].batch_size == 32
    assert materialization.instances[-1].plan.stage_by_name["embed"].batch_size == 128


def test_checkpointed_stage_preserves_grouping_but_unstarted_stage_profiles(tmp_path: Path) -> None:
    detect = StagePlan.create("detect", batch_size=3)
    embed = StagePlan.create("embed", batch_size=8, depends_on=("detect",))
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("samples"),
        publish=PublishOptions(),
        stages=(detect, embed),
    )
    store = MaterializationStateStore(plan.state_path)
    store.initialize(plan)
    store.begin("detect")
    store.record_shard("detect", "00000")
    store.fail("detect", "interrupted")
    requested = replace(plan, stages=(replace(detect, batch_size=16), replace(embed, batch_size=64)))

    class StopBeforeStages(DatasetMaterializer):
        def _run_stage_with_retries(self, stage_plan: StagePlan) -> Path:
            return self.plan.staging_root

    materializer = StopBeforeStages(
        requested,
        [SimpleNamespace(name="detect"), SimpleNamespace(name="embed")],
        batch_size_resolvers={"detect": lambda: pytest.fail("Checkpointed stage was profiled"), "embed": lambda: 128},
    )
    materializer.run()

    assert materializer.plan.stage_by_name["detect"].batch_size == 3
    assert materializer.plan.stage_by_name["embed"].batch_size == 128
    assert materializer.state.state.by_name["detect"].completed_shards == ("00000",)


def test_profiling_failure_does_not_start_or_complete_stages(
    materialization: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization.args.auto_batch = True
    materialization.args.device = "cpu"
    materialization.args.materialize_explicit_keys = ("device",)
    monkeypatch.setattr(workflow, "profile_detector", profile_detector)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        workflow.materialize(materialization.args)

    state = materialization.instances[-1].state.state
    assert all(stage.status == "pending" and stage.attempts == 0 for stage in state.stages)


def test_auto_rejects_multiple_worker_models(materialization: SimpleNamespace) -> None:
    materialization.args.auto_batch = True
    materialization.args.plan_overrides = ("detect.executor=process", "detect.workers=2")

    with pytest.raises(ValueError, match="independent worker models share GPU memory"):
        workflow.materialize(materialization.args)

    assert materialization.calls == []


@pytest.mark.parametrize("option", ["auto_batch_max_size", "auto_batch_memory_fraction"])
def test_profiling_options_preserve_invalid_booleans_for_validation(
    materialization: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    materialization.args.auto_batch = True
    setattr(materialization.args, option, True)
    monkeypatch.setattr(workflow, "profile_detector", profile_detector)

    with pytest.raises(ValueError, match="max_batch_size|memory_fraction"):
        workflow.materialize(materialization.args)


def test_settings_accept_auto_from_yaml_and_override(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("detect:\n  batch_size: auto\n", encoding="utf-8")

    settings = load_executor_settings(plan, ("embed.batch_size=auto",))

    assert settings["detect"]["batch_size"] == settings["embed"]["batch_size"] == "auto"


@pytest.mark.parametrize(
    "overrides",
    [
        ("segment.batch_size=auto",),
        ("finalize.batch_size=auto",),
        ("detect.batch_size=automatic",),
        ("embed.batch_size=auto", "embed.executor=process", "embed.workers=2"),
    ],
)
def test_settings_reject_invalid_auto_use(overrides: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        load_executor_settings(overrides=overrides)
