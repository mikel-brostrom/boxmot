"""Python workflow options share CLI defaults without requiring a namespace."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from boxmot.engine.config.runtime import resolve_mode_args


class _StopWorkflow(Exception):
    """Stop after configuration reaches the first external-work boundary."""


def test_keyword_defaults_and_explicit_overrides() -> None:
    """Defaults fill omitted values while authored overrides remain distinguishable."""
    args = resolve_mode_args("tune", None, {"tracker": "botsort", "n_trials": 200, "project": "custom-runs"})

    assert args.project == Path("custom-runs")
    assert args.tracker == "botsort"
    assert args.n_trials == 200
    assert args.device == "cpu"
    assert args.tracker_explicit is True
    assert args.device_explicit is False


def test_prepared_arguments_keep_execution_state() -> None:
    """Internal callers retain their namespace identity and resolved execution state."""
    args = SimpleNamespace(materialize_mode="eval", materialize_split="val", _build_validated=True)

    assert resolve_mode_args("materialize", args, {}) is args
    assert args.materialize_mode == "eval"
    assert args._build_validated is True


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    (
        ("tracking.workflow", "run_track"),
        ("materialization.workflow", "materialize"),
        ("eval.evaluator", "run_eval"),
        ("tuning.tuner", "run_tune"),
        ("research.runner", "run_research"),
        ("dataset_variants.workflow", "main"),
    ),
)
def test_workflows_reject_mixed_argument_styles(module: str, entrypoint: str) -> None:
    """Options cannot silently replace fields on an already prepared namespace."""
    workflow = importlib.import_module(f"boxmot.engine.{module}")
    run = getattr(workflow, entrypoint)

    with pytest.raises(TypeError, match="either prepared workflow arguments or keyword options"):
        run(SimpleNamespace(), experiment="mot17/ablation-yolox-lmbn.yaml")


@pytest.mark.parametrize("entrypoint", ("eval", "tune"))
def test_replay_workflows_accept_minimal_keywords(monkeypatch: pytest.MonkeyPatch, entrypoint: str) -> None:
    """Evaluation and tuning dispatch with normalized public options and execution hooks."""
    module = "eval.evaluator" if entrypoint == "eval" else "tuning.tuner"
    workflow = importlib.import_module(f"boxmot.engine.{module}")
    result = object()
    dispatch = Mock(return_value=result)
    monkeypatch.setattr(workflow, f"_run_sensor_{'evaluation' if entrypoint == 'eval' else 'tuning'}", dispatch)
    extra = {"verbose": True} if entrypoint == "eval" else {"n_trials": 200, "baseline_config": {"track_buffer": 30}}

    actual = getattr(workflow, f"run_{entrypoint}")(
        experiment="mot17/ablation-yolox-lmbn.yaml", build="BUILD_ID", tracker="botsort", **extra
    )

    assert actual is result
    args = dispatch.call_args.args[0]
    assert args.experiment == "mot17/ablation-yolox-lmbn.yaml"
    assert args.build == "BUILD_ID"
    assert args.tracker == "botsort"
    assert args.tracker_explicit is True
    assert args.project == Path("runs")
    if entrypoint == "eval":
        assert args.postprocessing == ()
        assert args.verbose is True
        assert dispatch.call_args.kwargs["verbose"] is True
    else:
        assert args.n_trials == 200
        assert dispatch.call_args.kwargs["baseline_config"] == {"track_buffer": 30}


@pytest.mark.parametrize("explicit_device", (False, True))
def test_materialize_keywords_preserve_authored_device(
    monkeypatch: pytest.MonkeyPatch, explicit_device: bool
) -> None:
    """The default CPU policy must not override an experiment's authored device."""
    from boxmot.engine.materialization import workflow

    captured: dict[str, Any] = {}

    def resolve_inputs(args: Any, **_kwargs: Any) -> None:
        captured["args"] = args
        raise _StopWorkflow

    monkeypatch.setattr(workflow, "_resolved_inputs", resolve_inputs)
    options = {"device": "cpu"} if explicit_device else {}
    with pytest.raises(_StopWorkflow):
        workflow.materialize(experiment="mot17/ablation-yolox-lmbn.yaml", build_root="builds", **options)

    args = captured["args"]
    assert args.build_root == Path("builds")
    assert args.resume is True
    assert workflow._device_override(args) == ("cpu" if explicit_device else None)


def test_research_keywords_build_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Research receives its structured defaults without invoking model providers."""
    from boxmot.engine.research import runner

    result = object()
    researcher = Mock()
    researcher.return_value.run.return_value = result
    monkeypatch.setattr(runner, "TrackerResearcher", researcher)

    actual = runner.run_research(
        experiment="mot17/ablation-yolox-lmbn.yaml", build="BUILD_ID", tracker="botsort", max_metric_calls=4
    )

    assert actual is result
    config = researcher.call_args.args[0]
    assert config.tracker == "botsort"
    assert config.build == "BUILD_ID"
    assert config.project == Path("runs")
    assert config.max_metric_calls == 4


def test_time_variant_keywords_default_optional_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A variant needs only its dataset, split, sequence and parent build."""
    from boxmot.engine.dataset_variants import workflow

    dataset = {"id": "mot17", "box_type": "aabb"}
    catalog = SimpleNamespace(fingerprint="catalog", metadata={"class_taxonomy_digest": "classes"})
    resolve_build = Mock(return_value=tmp_path / "parent-build")
    create_raw = Mock(side_effect=_StopWorkflow)
    monkeypatch.setattr(workflow, "load_dataset_config", Mock(return_value=dataset))
    monkeypatch.setattr(workflow, "default_data_root", Mock(return_value=tmp_path / "data"))
    monkeypatch.setattr(workflow, "catalog_mot_dataset", Mock(return_value=catalog))
    monkeypatch.setattr(workflow, "resolve_build_path", resolve_build)
    monkeypatch.setattr(
        workflow, "DatasetManifest", SimpleNamespace(load=Mock(return_value=SimpleNamespace(build_id="ID")))
    )
    monkeypatch.setattr(workflow, "validate_build_compatibility", Mock())
    monkeypatch.setattr(workflow, "validate_published_build", Mock())
    monkeypatch.setattr(workflow, "create_raw_variant", create_raw)

    with pytest.raises(_StopWorkflow):
        workflow.main(dataset="mot17", split="train", sequence="MOT17-02-FRCNN", build="BUILD_ID")

    workflow.default_data_root.assert_called_once_with(None)
    resolve_build.assert_called_once_with("BUILD_ID", build_root=None)
    assert create_raw.call_args.kwargs["seed"] == 0
    assert create_raw.call_args.kwargs["name"] is None
    assert create_raw.call_args.kwargs["sequence"] == "MOT17-02-FRCNN"


@pytest.mark.parametrize("prepared_args", (False, True))
def test_track_preserves_injected_components(monkeypatch: pytest.MonkeyPatch, prepared_args: bool) -> None:
    """Component instances stay injectable with either prepared args or keyword options."""
    from boxmot.engine.tracking import workflow

    detector = SimpleNamespace(capabilities=SimpleNamespace(provides_masks=False, provides_embeddings=False))
    tracker = SimpleNamespace(requirements=SimpleNamespace(embeddings=False, masks=False), generates_embeddings=False)
    source = object()
    segmentor = object()
    pipeline = Mock()
    runner = Mock(return_value=SimpleNamespace(run=lambda: iter(()), summary="summary"))
    monkeypatch.setattr(workflow, "TrackingPipeline", pipeline)
    monkeypatch.setattr(workflow, "TrackingRunner", runner)
    monkeypatch.setattr(
        workflow, "profile_components", lambda _profiler, **components: (detector, segmentor, None, tracker)
    )
    arguments = {"args": SimpleNamespace()} if prepared_args else {"classes": [0]}

    actual = workflow.run_track(
        detector=detector, tracker=tracker, segmentor=segmentor, source=source, sinks=(), **arguments
    )

    assert actual.summary == "summary"
    assert pipeline.call_args.kwargs["detector"] is detector
    assert pipeline.call_args.kwargs["tracker"] is tracker
    assert pipeline.call_args.kwargs["segmentor"] is segmentor
    assert runner.call_args.args[0] is source


@pytest.mark.parametrize("source", ("video.mp4", Path("video.mp4"), 0))
def test_track_accepts_component_names_and_source_options(monkeypatch: pytest.MonkeyPatch, source: Any) -> None:
    """Authored component references are options, rather than injected component instances."""
    from boxmot.engine.tracking import workflow

    capture = Mock(side_effect=_StopWorkflow)
    monkeypatch.setattr(workflow, "_tracker_spec", capture)

    with pytest.raises(_StopWorkflow):
        workflow.run_track(
            source=source, detector=Path("detector.yaml"), segmentor="segmentor.yaml", tracker="botsort", classes="0,1"
        )

    args = capture.call_args.args[0]
    assert args.source == source
    assert args.detector == Path("detector.yaml")
    assert args.segmentor == "segmentor.yaml"
    assert args.tracker == "botsort"
    assert args.classes == [0, 1]
    assert args.detector_explicit is True
    assert args.tracker_explicit is True


@pytest.mark.parametrize("options", ({"source": "video.mp4"}, {"detector": "yolov8n"}, {"tracker": "botsort"}))
def test_track_rejects_scalar_options_with_prepared_args(options: dict[str, Any]) -> None:
    """Names cannot bypass the mixed-style check via reserved injection arguments."""
    from boxmot.engine.tracking.workflow import run_track

    with pytest.raises(TypeError, match="either prepared workflow arguments or keyword options"):
        run_track(SimpleNamespace(), **options)
