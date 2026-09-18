"""Offline processing must precede scoring and preserve immutable replay inputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.engine.eval import evaluator, postprocessing
from boxmot.engine.eval.replay import ReplayResult
from boxmot.engine.ui.core import ui
from boxmot.engine.ui.reporters import postprocessing as postprocessing_ui
from boxmot.engine.ui.reporters.eval import EvalWorkflowReporter
from tests.unit.datasets import conftest as dataset_fixtures
from tests.unit.engine.eval.test_replay_callback import timestamped_build as timestamped_build


def _replay(build: Path, output: Path, *, split_ids: bool = False, two_sequences: bool = False) -> ReplayResult:
    """Keep a detection gap, fractional boxes, and valid source detection indices."""
    output.mkdir()
    paths = (output / "a.txt",)
    rows = np.array(
        [[1, 10, 1.25, 2.5, 10.5, 9.75, 0.9, 1, 0], [3, 20 if split_ids else 10, 1.25, 2.5, 10.5, 9.75, 0.9, 1, 0]]
    )
    np.savetxt(paths[0], rows, delimiter=",")
    if two_sequences:
        paths += (output / "b.txt",)
        np.savetxt(paths[1], rows[:1], delimiter=",")
    return ReplayResult(build, output, paths, 4 if two_sequences else 3, 3 if two_sequences else 2)


@pytest.mark.parametrize("steps", [("gsi",), ("gbrc",), ("gta",), ("gta", "gsi"), ("gta", "gbrc")])
def test_postprocessing_scores_derived_tracks_and_preserves_raw_and_build(timestamped_build, tmp_path, steps) -> None:
    replay = _replay(timestamped_build, tmp_path / "results", split_ids="gta" in steps)
    raw = replay.sequence_files[0].read_bytes()
    immutable = {path: path.read_bytes() for path in timestamped_build.rglob("*") if path.is_file()}
    events = []

    elapsed = postprocessing.postprocess_replay(
        replay, steps, split="variable", workers=1, progress_callback=events.append
    )

    assert elapsed > 0
    processed = np.loadtxt(replay.sequence_files[0], delimiter=",", ndmin=2)
    assert len(processed) == (2 if steps == ("gta",) else 3)
    assert np.unique(processed[:, 1]).size == 1
    assert np.all(processed[:, 7] == 1)
    if len(processed) == 3:
        assert processed[1, 8] == -1
    assert (replay.output_dir / "raw/a.txt").read_bytes() == raw
    assert all(path.read_bytes() == payload for path, payload in immutable.items())
    record = json.loads((replay.output_dir / "postprocessing.json").read_text())
    assert record["methods"] == list(steps)
    assert set(record["parameters"]) == set(steps)
    assert record["sequences"][0]["input_rows"] == 2
    assert record["sequences"][0]["output_rows"] == len(processed)
    assert events[0].status == "queued"
    assert events[-1].status == "completed"
    assert {event.sequence_id for event in events} == {"a"}
    assert [event.phase_index for event in events] == sorted(event.phase_index for event in events)
    assert any(event.status == "running" for event in events)
    for step in steps:
        assert any(step.upper() in (event.detail or "") for event in events if event.status == "running")


def test_sequence_parallel_processing_matches_serial_results(timestamped_build, tmp_path) -> None:
    serial = _replay(timestamped_build, tmp_path / "serial", two_sequences=True)
    parallel = _replay(timestamped_build, tmp_path / "parallel", two_sequences=True)

    postprocessing.postprocess_replay(serial, ("gta", "gsi"), split="variable", workers=1)
    postprocessing.postprocess_replay(parallel, ("gta", "gsi"), split="variable", workers=2)

    assert [path.read_bytes() for path in parallel.sequence_files] == [
        path.read_bytes() for path in serial.sequence_files
    ]


def test_failed_sequence_does_not_publish_a_partially_processed_run(timestamped_build, tmp_path, monkeypatch) -> None:
    replay = _replay(timestamped_build, tmp_path / "results", two_sequences=True)
    original = {path: path.read_bytes() for path in replay.sequence_files}
    process = postprocessing._process_sequence

    def fail_second(task, progress_callback=None):
        if task.source.stem == "b":
            raise RuntimeError("processing failed")
        return process(task, progress_callback)

    monkeypatch.setattr(postprocessing, "_process_sequence", fail_second)
    with pytest.raises(RuntimeError, match="processing failed"):
        postprocessing.postprocess_replay(replay, ("gsi",), workers=1)

    assert all(path.read_bytes() == payload for path, payload in original.items())
    assert not (replay.output_dir / "raw").exists()
    assert not (replay.output_dir / "postprocessing.json").exists()
    assert not tuple(replay.output_dir.glob(".postprocess-*"))


def _synchronized_sequence_task(task):
    """Require the parent to observe both running workers before either finishes."""
    release = task.source.parent / "release-workers"
    deadline = time.monotonic() + 20
    while not release.exists():
        postprocessing._emit_worker_progress(
            postprocessing.PostprocessingProgressEvent(
                task.source.stem, "running", 0, None, "Waiting for peer", task.ordinal, 0
            )
        )
        if time.monotonic() > deadline:
            raise RuntimeError("Concurrent worker progress did not reach the parent.")
        time.sleep(0.02)
    return postprocessing._process_sequence(task)


def test_parallel_progress_arrives_before_sequences_finish(timestamped_build, tmp_path, monkeypatch) -> None:
    replay = _replay(timestamped_build, tmp_path / "results", two_sequences=True)
    events = []
    running = set()

    def receive(event):
        events.append(event)
        if event.status == "running" and event.detail == "Waiting for peer":
            running.add(event.sequence_id)
        if running == {"a", "b"}:
            staging = next(replay.output_dir.glob(".postprocess-*"))
            (staging / "raw/release-workers").touch()

    monkeypatch.setattr(postprocessing, "_process_sequence", _synchronized_sequence_task)
    postprocessing.postprocess_replay(replay, ("gbrc",), workers=2, progress_callback=receive)

    assert running == {"a", "b"}
    first_terminal = next(index for index, event in enumerate(events) if event.status == "completed")
    assert {event.sequence_id for event in events[:first_terminal] if event.status == "running"} == {"a", "b"}
    assert {event.sequence_id for event in events if event.status == "completed"} == {"a", "b"}
    assert all(event.status != "failed" for event in events)
    assert (replay.output_dir / "postprocessing.json").is_file()


def test_parallel_failure_reports_failed_sequence_and_preserves_inputs(timestamped_build, tmp_path) -> None:
    replay = _replay(timestamped_build, tmp_path / "results", two_sequences=True)
    replay.sequence_files[1].write_text("invalid tracking rows\n")
    before = {path: path.read_bytes() for path in replay.sequence_files}
    events = []

    with pytest.raises(RuntimeError, match="Postprocessing failed for sequence\\(s\\): b"):
        postprocessing.postprocess_replay(replay, ("gsi",), workers=2, progress_callback=events.append)

    assert {event.sequence_id for event in events if event.status == "failed"} == {"b"}
    assert {event.sequence_id for event in events if event.status == "completed"} == {"a"}
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not (replay.output_dir / "postprocessing.json").exists()
    assert not (replay.output_dir / "raw").exists()


def test_progress_callback_failure_does_not_invalidate_processing(timestamped_build, tmp_path) -> None:
    replay = _replay(timestamped_build, tmp_path / "results")

    def failing_reporter(_event):
        raise ValueError("Display closed")

    postprocessing.postprocess_replay(replay, ("gsi",), workers=1, progress_callback=failing_reporter)
    assert (replay.output_dir / "postprocessing.json").is_file()


@pytest.mark.parametrize("steps", [("gsi",), ("gbrc",), ("gta",)])
def test_empty_sequence_is_preserved(timestamped_build, tmp_path, steps) -> None:
    replay = _replay(timestamped_build, tmp_path / "results")
    replay.sequence_files[0].write_text("")
    postprocessing.postprocess_replay(replay, steps, workers=1)
    assert replay.sequence_files[0].read_bytes() == b""
    assert (replay.output_dir / "raw/a.txt").read_bytes() == b""


def _args(build: Path, **overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "_build_validated": True,
            "build_path": build,
            "dataset_id": "fixture",
            "experiment_id": None,
            "geometry": "aabb",
            "sequence_workers": 1,
            "sequence_names": ("a",),
            "seq_info": {"a": 3},
            "split": "variable",
            "tracker": "bytetrack",
            "tracker_class_ids": (1,),
            "tracker_class_names": ((1, "person"),),
            "postprocessing": ("gsi",),
            **overrides,
        }
    )


def test_evaluator_applies_postprocessing_before_metrics(timestamped_build, tmp_path, monkeypatch) -> None:
    replay = _replay(timestamped_build, tmp_path / "results")
    monkeypatch.setattr(evaluator, "replay_build", lambda *_args, **_kwargs: replay)

    def score(args, **_kwargs):
        rows = np.loadtxt(args.exp_dir / "a.txt", delimiter=",")
        assert rows[:, 0].tolist() == [1, 2, 3]
        assert (args.exp_dir / "raw/a.txt").is_file()
        return {"HOTA": 90.0}

    monkeypatch.setattr(evaluator, "run_motmetrics", score)
    result = evaluator.run_eval(_args(timestamped_build), setup=False, output_dir=replay.output_dir)

    assert result.summary == {"HOTA": 90.0}
    assert result.timings["totals_ms"]["postprocess"] > 0
    assert result.timings["totals_ms"]["total"] > result.timings["totals_ms"]["track"]


@pytest.mark.parametrize("method", ["gta", "gsi", "gbrc"])
def test_evaluator_embeds_live_sequence_bars_in_postprocessing_step(
    timestamped_build, tmp_path, monkeypatch, method
) -> None:
    replay = _replay(timestamped_build, tmp_path / "results", two_sequences=True)
    args = _args(timestamped_build, postprocessing=(method,))
    pipeline = EvalWorkflowReporter(args).pipeline(auto_start=False, wire_status_fns=False)
    rendered_updates = []

    class UnthrottledPresenter(postprocessing_ui.EvalPostprocessingProgressPresenter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, refresh_interval_s=None)

    def refresh(**_kwargs):
        if pipeline.current_step == "Postprocess tracks" and pipeline.workflow.detail_renderable is not None:
            rendered_updates.append(ui.capture_renderable(pipeline.workflow.detail_renderable, width=180))

    def score(*_args, **_kwargs):
        assert pipeline.current_step == "Evaluate results"
        detail = pipeline.step_records["Postprocess tracks"].detail_renderable
        rendered = ui.capture_renderable(detail, width=180)
        assert "Postprocessing: 2/2 sequences done" in rendered
        assert method.upper() in rendered
        assert "a" in rendered and "b" in rendered
        return {"HOTA": 90.0}

    monkeypatch.setattr(postprocessing_ui, "EvalPostprocessingProgressPresenter", UnthrottledPresenter)
    monkeypatch.setattr(pipeline.workflow, "_update_live", refresh)
    monkeypatch.setattr(evaluator, "replay_build", lambda *_args, **_kwargs: replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", score)

    with pipeline:
        evaluator.run_eval(args, setup=False, output_dir=replay.output_dir, pipeline=pipeline)

    assert any("pending" in rendered for rendered in rendered_updates)
    assert any("running" in rendered and method.upper() in rendered for rendered in rendered_updates)
    assert any("Postprocessing: 2/2 sequences done" in rendered for rendered in rendered_updates)
    assert not any("Tracking:" in rendered for rendered in rendered_updates)


@pytest.mark.parametrize("fail_processing", [False, True])
def test_reused_output_does_not_retain_previous_postprocessing_metadata(
    timestamped_build, tmp_path, monkeypatch, fail_processing
) -> None:
    replay = _replay(timestamped_build, tmp_path / "results")
    original = replay.sequence_files[0].read_bytes()
    postprocessing.postprocess_replay(replay, ("gsi",), workers=1)

    def fresh_replay(*_args, **_kwargs):
        replay.sequence_files[0].write_bytes(original)
        return replay

    def score(*_args, **_kwargs):
        assert not fail_processing, "Failed processing must not reach scoring."
        return {"HOTA": 90.0}

    def fail(*_args, **_kwargs):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(evaluator, "replay_build", fresh_replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", score)
    monkeypatch.setattr(postprocessing, "_process_sequence", fail)
    args = _args(timestamped_build, postprocessing=("gsi",) if fail_processing else ())
    if fail_processing:
        with pytest.raises(RuntimeError, match="processing failed"):
            evaluator.run_eval(args, setup=False, output_dir=replay.output_dir)
    else:
        evaluator.run_eval(args, setup=False, output_dir=replay.output_dir)
    assert replay.sequence_files[0].read_bytes() == original
    assert not (replay.output_dir / "postprocessing.json").exists()
    assert not (replay.output_dir / "raw").exists()


@pytest.mark.parametrize("overrides", [{"geometry": "obb"}, {"eval_masks": True}])
def test_incompatible_output_is_rejected_before_replay(timestamped_build, tmp_path, monkeypatch, overrides) -> None:
    monkeypatch.setattr(evaluator, "replay_build", lambda *a, **kw: pytest.fail("unexpected replay"))
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="AABB box evaluation"):
        evaluator.run_eval(_args(timestamped_build, **overrides), setup=False, output_dir=output)
    assert not output.exists()


def test_gta_missing_embeddings_is_rejected_before_replay(tmp_path, monkeypatch) -> None:
    build = dataset_fixtures.materialized_boxes_only_build.__wrapped__(tmp_path)
    monkeypatch.setattr(evaluator, "replay_build", lambda *a, **kw: pytest.fail("unexpected replay"))
    with pytest.raises(ValueError, match="missing required embeddings"):
        evaluator.run_eval(_args(build, postprocessing=("gta",)), setup=False, output_dir=tmp_path / "results")


def test_reporter_adds_postprocessing_step_only_when_requested() -> None:
    ordinary = EvalWorkflowReporter(SimpleNamespace())
    processed = EvalWorkflowReporter(SimpleNamespace(postprocessing=("gta", "gsi")))
    assert [label for label, _state in ordinary.steps] == ["Set up", "Run tracker", "Evaluate results"]
    assert [label for label, _state in processed.steps] == [
        "Set up",
        "Run tracker",
        "Postprocess tracks",
        "Evaluate results",
    ]
    assert processed.EVALUATE == 3
