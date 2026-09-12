"""Exercise process-isolated KITTI sequences with real masks, metrics, and outputs."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import pytest
import torch

from boxmot.datasets.sequence import MultimodalSequence, SensorFrame
from boxmot.engine.eval import eagermot_kitti as replay
from boxmot.engine.eval.kitti_mots_replay import kitti_mots_annotations
from boxmot.engine.eval.replay import ReplayProgressEvent
from boxmot.engine.eval.session import ReplaySession
from tests.unit.engine.eval.test_eagermot_kitti import _fixture
from tests.unit.engine.eval.test_eagermot_visualization import _capture_visualizations


class _ObservedSequence(MultimodalSequence):
    """Record actual reader PIDs and require two readers to run concurrently."""

    barrier: Path | None = None
    fail_at: int | None = None
    stall_after_first: bool = False

    def __getitem__(self, index: int) -> SensorFrame:
        frame = super().__getitem__(index)
        if index == 0 and self.barrier is not None:
            marker = self.barrier / f"{self.sequence_id}-{os.getpid()}"
            marker.write_text(str(torch.get_num_threads()))
            deadline = time.monotonic() + 30.0
            while len(tuple(self.barrier.iterdir())) < 2:
                if time.monotonic() > deadline:
                    raise RuntimeError("Sequence readers did not run concurrently")
                time.sleep(0.02)
        if index == 1 and self.stall_after_first:
            time.sleep(30.0)
        if index == self.fail_at:
            raise ValueError("synthetic sequence read failed")
        return frame


def _inputs(root: Path, count: int = 2, *, fps: float = 10.0) -> replay.KittiReplayInputs:
    """Prepare independent real KITTI sequences, retaining lazy frame decoding."""
    sequences = {}
    annotations = {}
    for index in range(count):
        name = f"{index + 1:04d}"
        data = _fixture(root / name)
        sequence = _ObservedSequence(
            replace(data.sequence_inputs(), sequence_id=name),
            classes={"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}},
            fps=fps,
            split="val",
        )
        sequences[name] = sequence
        annotations[name] = kitti_mots_annotations(name, sequence.frame_paths, sequence.image_size, data.ground_truth)
    return replay.KittiReplayInputs(
        sequences,
        annotations,
        root,
        {"split": "val", "sequences": {name: len(sequence) for name, sequence in sequences.items()}},
        fps=fps,
    )


def _concurrent_readers(inputs: replay.KittiReplayInputs, path: Path) -> None:
    """Give sequence readers a filesystem barrier visible from spawned children."""
    path.mkdir()
    for sequence in inputs.sequences.values():
        sequence.barrier = path


def test_spawned_replay_matches_serial_masks_metrics_and_delivers_parent_progress(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    profiles = replay.load_kitti_profiles()
    serial = tmp_path / "serial"
    expected = replay.evaluate_eagermot_kitti(inputs, profiles, serial, sequence_workers=1)
    _concurrent_readers(inputs, tmp_path / "workers")
    parent_pid = os.getpid()
    original_threads = torch.get_num_threads()
    existing_children = {child.pid for child in multiprocessing.active_children()}
    events = []
    metric_stages = []

    def progress(event: ReplayProgressEvent) -> None:
        assert os.getpid() == parent_pid
        events.append(event)

    def evaluating() -> None:
        terminal = {event.sequence_id for event in events if event.status == "completed"}
        assert terminal == set(inputs.sequences)
        assert {child.pid for child in multiprocessing.active_children()} <= existing_children
        metric_stages.append(True)

    parallel = tmp_path / "parallel"
    actual = replay.evaluate_eagermot_kitti(
        inputs, profiles, parallel, sequence_workers=2, progress_callback=progress, on_evaluate=evaluating
    )

    assert actual == expected
    assert json.loads((parallel / "metrics.json").read_text()) == json.loads((serial / "metrics.json").read_text())
    for name in inputs.sequences:
        assert (parallel / "mots" / f"{name}.txt").read_bytes() == (serial / "mots" / f"{name}.txt").read_bytes()
        sequence_events = [event for event in events if event.sequence_id == name]
        assert sequence_events[0].status == "queued"
        assert sequence_events[-1].status == "completed"
        assert sequence_events[-1].completed == 3
        assert sequence_events[-1].track_rows == 4
        assert len({event.ordinal for event in sequence_events}) == 1
    markers = tuple((tmp_path / "workers").iterdir())
    worker_pids = {int(path.name.split("-")[1]) for path in markers}
    assert len(worker_pids) == 2
    assert parent_pid not in worker_pids
    assert {path.read_text() for path in markers} == {"1"}
    assert metric_stages == [True]
    assert torch.get_num_threads() == original_threads
    assert json.loads((parallel / "run.json").read_text())["sequence_workers"] == 2


def test_sensor_session_reuses_processes_and_isolates_trial_progress(tmp_path: Path) -> None:
    """A retained pool starts fresh identities, filters old progress, and closes after the study."""
    inputs = _inputs(tmp_path / "inputs")
    profiles = replay.load_kitti_profiles()
    expected = replay.evaluate_eagermot_kitti(inputs, profiles, tmp_path / "baseline", sequence_workers=1)
    _concurrent_readers(inputs, tmp_path / "workers")
    existing = {child.pid for child in multiprocessing.active_children()}
    events = []
    with ReplaySession(2) as session:
        first = replay.evaluate_eagermot_kitti(
            inputs,
            profiles,
            tmp_path / "first",
            sequence_workers=2,
            replay_session=session,
            progress_callback=events.append,
        )
        original_executor = session._executor
        markers = {path.name for path in (tmp_path / "workers").iterdir()}
        first_ids = {event.run_id for event in events}
        assert len(first_ids) == 1 and None not in first_ids
        assert len({name.split("-")[1] for name in markers}) == 2
        events.clear()
        repeated = replay.evaluate_eagermot_kitti(
            inputs,
            profiles,
            tmp_path / "repeat",
            sequence_workers=2,
            replay_session=session,
            progress_callback=events.append,
        )
        assert session._executor is original_executor
        assert {path.name.split("-")[1] for path in (tmp_path / "workers").iterdir()} == {
            name.split("-")[1] for name in markers
        }
        assert first_ids.isdisjoint({event.run_id for event in events})
        assert len({event.run_id for event in events}) == 1
    assert session._executor is None
    assert {child.pid for child in multiprocessing.active_children()} <= existing
    assert first == repeated == expected
    for name in inputs.sequences:
        assert (tmp_path / "first/mots" / f"{name}.txt").read_bytes() == (
            tmp_path / "repeat/mots" / f"{name}.txt"
        ).read_bytes()


def test_sensor_session_discards_failed_pool_and_can_restart(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    inputs.sequences["0001"].fail_at = 1
    with ReplaySession(2) as session:
        with pytest.raises(RuntimeError, match="sequence.*0001"):
            replay.evaluate_eagermot_kitti(
                inputs,
                replay.load_kitti_profiles(),
                tmp_path / "failed",
                sequence_workers=2,
                replay_session=session,
            )
        assert session._executor is None
        inputs.sequences["0001"].fail_at = None
        result = replay.evaluate_eagermot_kitti(
            inputs,
            replay.load_kitti_profiles(),
            tmp_path / "recovered",
            sequence_workers=2,
            replay_session=session,
        )
        assert result["cls_comb_cls_av"]["HOTA"] == 100


def test_worker_failure_releases_children_restores_threads_and_skips_metrics(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _concurrent_readers(inputs, tmp_path / "workers")
    inputs.sequences["0001"].fail_at = 1
    existing_children = {child.pid for child in multiprocessing.active_children()}
    original_threads = torch.get_num_threads()
    events = []
    output = tmp_path / "failed"

    with pytest.raises(RuntimeError, match="sequence.*0001") as failure:
        replay.evaluate_eagermot_kitti(
            inputs,
            replay.load_kitti_profiles(),
            output,
            sequence_workers=2,
            progress_callback=events.append,
            on_evaluate=lambda: pytest.fail("Failed tracking must not start metrics"),
        )

    assert "synthetic sequence read failed" in str(failure.value.__cause__)
    assert torch.get_num_threads() == original_threads
    assert {child.pid for child in multiprocessing.active_children()} <= existing_children
    assert any(event.sequence_id == "0001" and event.status == "failed" for event in events)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert not (output / "metrics.json").exists()
    # Worker output handles must have been released even for the failed sequence.
    for path in (output / "mots").iterdir():
        path.rename(path.with_suffix(".closed"))


def test_parallel_save_closes_sequence_videos_and_keeps_3d_mask_outputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs", fps=25.0)
    baseline_output = tmp_path / "baseline"
    baseline = replay.evaluate_eagermot_kitti(inputs, replay.load_kitti_profiles(), baseline_output, sequence_workers=1)
    output = tmp_path / "videos"
    metrics = replay.evaluate_eagermot_kitti(
        inputs, replay.load_kitti_profiles(), output, sequence_workers=2, save=True, show_3d=True
    )

    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sequence_workers"] == 2
    assert manifest["visualization"]["video_fps"] == 25.0
    assert manifest["videos"] == ["videos/0001.mp4", "videos/0002.mp4"]
    assert metrics == baseline
    assert metrics["car"]["HOTA"] == metrics["pedestrian"]["HOTA"] == 100
    for name in inputs.sequences:
        assert (output / "mots" / f"{name}.txt").read_bytes() == (baseline_output / "mots" / f"{name}.txt").read_bytes()
    for relative in manifest["videos"]:
        capture = cv2.VideoCapture(str(output / relative))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0)
        finally:
            capture.release()


@pytest.mark.parametrize(("count", "show"), [(1, False), (2, True)])
def test_single_sequence_and_preview_use_the_calling_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, show: bool
) -> None:
    inputs = _inputs(tmp_path / "inputs", count)
    renderers = _capture_visualizations(monkeypatch)

    def unexpected_pool(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Single-sequence and preview replay must stay in the calling process")

    monkeypatch.setattr(replay, "_run_parallel_sequences", unexpected_pool)
    output = tmp_path / "serial"
    replay.evaluate_eagermot_kitti(inputs, replay.load_kitti_profiles(), output, sequence_workers=8, show=show)

    assert json.loads((output / "run.json").read_text())["sequence_workers"] == 1
    assert len(renderers) == int(show)
    assert all(renderer.closed for renderer in renderers)


def test_automatic_worker_count_is_used_by_python_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from boxmot.engine.config import runtime

    inputs = _inputs(tmp_path / "inputs")
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 4)
    _concurrent_readers(inputs, tmp_path / "workers")
    output = tmp_path / "auto"

    replay.evaluate_eagermot_kitti(inputs, replay.load_kitti_profiles(), output)

    assert json.loads((output / "run.json").read_text())["sequence_workers"] == 2
    assert len({path.name.split("-")[1] for path in (tmp_path / "workers").iterdir()}) == 2


def test_keyboard_interrupt_terminates_active_readers_and_marks_interrupted_run(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _concurrent_readers(inputs, tmp_path / "workers")
    for sequence in inputs.sequences.values():
        sequence.stall_after_first = True
    existing_children = {child.pid for child in multiprocessing.active_children()}
    original_threads = torch.get_num_threads()
    output = tmp_path / "interrupted"
    interrupted_at = None

    def interrupt(event: ReplayProgressEvent) -> None:
        nonlocal interrupted_at
        if event.completed == 1:
            interrupted_at = time.monotonic()
            raise KeyboardInterrupt("stop active replay")

    with pytest.raises(KeyboardInterrupt, match="stop active replay"):
        replay.evaluate_eagermot_kitti(
            inputs, replay.load_kitti_profiles(), output, sequence_workers=2, progress_callback=interrupt
        )

    assert interrupted_at is not None
    assert time.monotonic() - interrupted_at < 10.0
    assert torch.get_num_threads() == original_threads
    assert {child.pid for child in multiprocessing.active_children()} <= existing_children
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"
    assert not (output / "metrics.json").exists()


def test_dismissed_preview_stays_closed_across_serial_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boxmot.engine.eval.visualization import ReplayVisualization

    inputs = _inputs(tmp_path / "inputs")
    windows = []

    def close_preview(self: ReplayVisualization, image: Any, elapsed: float) -> None:
        if self.show:
            windows.append(self._sequence)
            self.show = False

    monkeypatch.setattr(ReplayVisualization, "_display", close_preview)
    output = tmp_path / "preview"

    replay.evaluate_eagermot_kitti(inputs, replay.load_kitti_profiles(), output, sequence_workers=2, show=True)

    assert windows == ["0001"]
    assert (output / "mots/0001.txt").is_file()
    assert (output / "mots/0002.txt").is_file()
