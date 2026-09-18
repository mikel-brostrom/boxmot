"""Evaluator owns optional visualization lifetime and leaves metrics replay intact."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.replay import ReplayProgressEvent, ReplayResult
from boxmot.engine.eval.results import ValidationResult
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _args(tmp_path, **overrides):
    return SimpleNamespace(
        **{
            "_build_validated": True,
            "build_path": tmp_path / "build",
            "dataset_id": "fixture",
            "experiment_id": None,
            "geometry": "aabb",
            "sequence_workers": 3,
            "seq_info": {"sequence": 2},
            "sequence_names": None,
            "split": "validation",
            "tracker": "bytetrack",
            "tracker_class_ids": (1,),
            "tracker_class_names": ((1, "person"),),
            **overrides,
        }
    )


@pytest.mark.parametrize("source", ("build", "saved_detections", "sensor"))
def test_evaluation_cli_shows_sequences_for_named_class_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    """A class-keyed summary retains each sequence in the final evaluation panel."""
    args = _args(tmp_path)
    if source == "sensor":
        args = SimpleNamespace(dataset=sensor_dataset_fixture(tmp_path / "dataset").dataset, tracker="eagermot")
    result = ValidationResult(
        benchmark="fixture",
        raw={
            "pedestrian": {
                "HOTA": 71.1,
                "per_sequence": {"sequence-a": {"HOTA": 70.0}, "sequence-b": {"HOTA": 72.2}},
            }
        },
        summary_label="pedestrian",
        summary={"HOTA": 71.1},
        exp_dir=tmp_path / "output",
        args=args,
    )

    def evaluate(*_args: Any, **_kwargs: Any) -> ValidationResult:
        """Leave rendering real while avoiding tracking and metric computation."""
        return result

    if source == "saved_detections":
        from boxmot.engine.eval import saved_detections

        monkeypatch.setattr(saved_detections, "run_saved_detections", evaluate)
        entrypoint = saved_detections.main
    else:
        if source == "sensor":
            monkeypatch.setitem(
                sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=evaluate)
            )
        else:
            monkeypatch.setattr(evaluator, "run_eval", evaluate)
        entrypoint = evaluator.main

    assert entrypoint(args) is result
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "sequence-a" in output
    assert "sequence-b" in output
    assert "COMBINED (pedestrian)" in output
    assert "71.10" in output


@pytest.mark.parametrize("show,save", [(False, False), (True, False), (False, True), (True, True)])
def test_evaluator_opens_visualization_only_when_requested(monkeypatch, tmp_path, show, save) -> None:
    events = []
    video = tmp_path / "output" / "videos" / "sequence.mp4"

    class Visualization:
        def __init__(self, output_dir, **kwargs):
            events.append(("open", output_dir, kwargs))
            self.video_paths = (video,) if kwargs["save"] else ()

        def __enter__(self):
            events.append("enter")
            return self

        def __call__(self, frame):
            events.append(("frame", frame))

        def __exit__(self, *exc):
            events.append("close")

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.eval.visualization", SimpleNamespace(ReplayVisualization=Visualization)
    )
    frame = object()

    def replay(build, spec, **kwargs):
        assert kwargs["workers"] == 3
        if show or save:
            kwargs["frame_callback"](frame)
        else:
            assert "frame_callback" not in kwargs
        return ReplayResult(build, kwargs["output_dir"], (), 2, 0)

    def metrics(*args, **kwargs):
        if show or save:
            assert events[-1] == "close"
        return {"HOTA": 60.0}

    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", metrics)
    args = _args(tmp_path, show=show, save=save)

    result = evaluator.run_eval(args, setup=False, output_dir=tmp_path / "output")

    assert result.summary == {"HOTA": 60.0}
    assert result.timings["frames"] == 2
    assert args.video_paths == ((video,) if save else ())
    if show or save:
        assert events == [
            ("open", tmp_path / "output", {"show": show, "save": save, "class_names": {1: "person"}}),
            "enter",
            ("frame", frame),
            "close",
        ]
    else:
        assert events == []


def test_evaluator_closes_visualization_when_replay_fails(monkeypatch, tmp_path) -> None:
    closed = []

    class Visualization:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed.append(exc[1])

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.eval.visualization", SimpleNamespace(ReplayVisualization=Visualization)
    )

    def replay(*args, **kwargs):
        raise ValueError("Invalid source frame")

    monkeypatch.setattr(evaluator, "replay_build", replay)
    with pytest.raises(ValueError, match="Invalid source frame"):
        evaluator.run_eval(_args(tmp_path, save=True), setup=False, output_dir=tmp_path / "output")
    assert len(closed) == 1
    assert isinstance(closed[0], ValueError)


@pytest.mark.parametrize("with_pipeline,show_progress", [(False, False), (True, False), (True, True)])
def test_evaluator_forwards_sequence_events_independently_of_display(
    monkeypatch, tmp_path, with_pipeline: bool, show_progress: bool
) -> None:
    """Tuning receives replay events while ordinary evaluation keeps its own rows."""
    received = []
    displayed = []
    presenter_lifetime = []
    stored = []
    events = [
        ReplayProgressEvent("sequence", "queued", 0, 2, 0, None, 0),
        ReplayProgressEvent("sequence", "running", 1, 2, 1, None, 0),
        ReplayProgressEvent("sequence", "completed", 2, 2, 2, None, 0),
    ]
    renderable = object()
    workflow_callback = object()

    class Presenter:
        def __init__(self, callback, sequence_totals):
            assert callback is workflow_callback
            assert sequence_totals == {"sequence": 2}
            self.renderable = renderable

        def __enter__(self):
            presenter_lifetime.append("enter")
            return self

        def __call__(self, event):
            displayed.append(event)

        def __exit__(self, *exc):
            presenter_lifetime.append("exit")

    def replay(build, spec, **kwargs):
        for event in events:
            kwargs["progress_callback"](event)
            assert received[-1] is event
            if with_pipeline and show_progress:
                assert displayed[-1] is event
        return ReplayResult(build, kwargs["output_dir"], (), 2, 2)

    pipeline = (
        SimpleNamespace(
            callback=lambda: workflow_callback,
            advance=lambda *_args: None,
            store_step_info=lambda value, **_kwargs: stored.append(value),
        )
        if with_pipeline
        else None
    )
    monkeypatch.setattr(evaluator, "EvalSequenceProgressPresenter", Presenter)
    monkeypatch.setattr(evaluator, "_refresh_eval_pipeline_intro", lambda *_args: None)
    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", lambda *_args, **_kwargs: {"HOTA": 60.0})

    result = evaluator.run_eval(
        _args(tmp_path),
        setup=False,
        output_dir=tmp_path / "output",
        show_progress=show_progress,
        pipeline=pipeline,
        progress_callback=received.append,
    )

    assert received == events
    assert result.summary == {"HOTA": 60.0}
    if with_pipeline and show_progress:
        assert displayed == events
        assert presenter_lifetime == ["enter", "exit"]
        assert stored == [renderable]
    else:
        assert displayed == presenter_lifetime == stored == []
