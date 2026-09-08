"""Evaluator owns optional visualization lifetime and leaves metrics replay intact."""

import sys
from types import SimpleNamespace

import pytest

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.replay import ReplayResult


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
