"""Prepared camera motion must retain tracker results without reading pixels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.datasets import cached
from boxmot.engine.calibration import cmc as analysis
from boxmot.engine.eval import cmc, replay
from boxmot.engine.eval.session import ReplaySession
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.motion.cmc.sof import SOF
from tests.unit.engine.eval import test_replay_callback as callback_fixtures


@pytest.fixture
def timestamped_build(tmp_path: Path) -> Path:
    """Retain the existing canonical build fixture with keyed optional payloads."""
    return callback_fixtures.timestamped_build.__wrapped__(tmp_path)


class _TranslationEstimator:
    """Produce deterministic nonidentity motion through the real analysis loop."""

    def __init__(self, **kwargs: object) -> None:
        self.frame = 0

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self.was_initializing = self.frame == 0
        self.inliers_accepted = self.transform_valid = self.frame > 0
        self.match_count = self.inlier_count = 30
        warp = _translation(image)
        self.frame += 1
        return warp


def _translation(image: np.ndarray) -> np.ndarray:
    """Use the fixture's frame-coded pixels for reproducible motion."""
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = 0.0 if image[0, 0, 0] == 30 else 2.0
    return warp


def _inline(tasks, *, workers, progress_callback):
    """Exercise the production worker while exposing image-read instrumentation."""
    return tuple(replay._replay_sequence_task(task, progress_callback=progress_callback) for task in tasks)


def _prepare(build: Path, output: Path, spec: TrackerSpec, monkeypatch: pytest.MonkeyPatch):
    """Analyze the keyed fixture and retain its real source provenance."""
    args = SimpleNamespace(
        build_path=build,
        split="variable",
        seq_info={"a": 3, "b": 1},
        sequence_names=(),
        sequence_workers=1,
        tracker=spec.name,
        tracker_backend="python",
        geometry="aabb",
        per_class=spec.per_class,
        tracker_class_ids=spec.class_ids,
        tracker_class_names=spec.class_names,
        cmc_policy="auto",
    )
    monkeypatch.setattr(analysis, "_DiagnosticSOF", _TranslationEstimator)
    setup = cmc.prepare_eval_cmc(args, spec, output)
    assert setup.policy.sequences == (("a", True), ("b", False))
    assert [item.sequence_id for item in setup.replay_sequences] == ["a"]
    return setup


@pytest.mark.parametrize("tracker", ["bytetrack", "botsort"])
@pytest.mark.parametrize("per_class", [False, True])
@pytest.mark.parametrize("cache_inputs", [False, True])
def test_prepared_replay_matches_live_cmc_without_pixels(
    timestamped_build, tmp_path, monkeypatch, tracker, per_class, cache_inputs
):
    spec = TrackerSpec(name=tracker, class_ids=(1,), per_class=per_class)
    setup = _prepare(timestamped_build, tmp_path / "setup", spec, monkeypatch)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _inline)
    monkeypatch.setattr(SOF, "apply", lambda self, image, dets=None: _translation(image))
    live = replay.replay_build(
        timestamped_build, spec, split="variable", output_dir=tmp_path / "live", cmc_policy=setup.policy
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Prepared CMC must not decode images or estimate camera motion again.")

    monkeypatch.setattr(SOF, "apply", unexpected)
    monkeypatch.setattr(cached, "read_rgb_chw_uint8", unexpected)
    prepared = replay.replay_build(
        timestamped_build,
        spec,
        split="variable",
        output_dir=tmp_path / "prepared",
        cmc_policy=setup.policy,
        cmc_transforms=setup.replay_sequences,
        cache_inputs=cache_inputs,
    )
    assert prepared.frames == live.frames == 4
    assert [path.read_bytes() for path in prepared.sequence_files] == [
        path.read_bytes() for path in live.sequence_files
    ]


def test_prepared_replay_retains_rendering_pixels(timestamped_build, tmp_path, monkeypatch):
    spec = TrackerSpec(name="bytetrack", class_ids=(1,))
    setup = _prepare(timestamped_build, tmp_path / "setup", spec, monkeypatch)
    monkeypatch.setattr(SOF, "apply", lambda *args: pytest.fail("unexpected camera estimation"))
    frames = []
    result = replay.replay_build(
        timestamped_build,
        spec,
        split="variable",
        output_dir=tmp_path / "rendered",
        cmc_policy=setup.policy,
        cmc_transforms=setup.replay_sequences,
        frame_callback=frames.append,
    )
    assert len(frames) == result.frames == 4
    assert all(item.sample.frame is not None for item in frames)


@pytest.mark.parametrize("unknown_sequence", [False, True])
def test_prepared_replay_validates_build_with_subset_counts(
    timestamped_build, tmp_path, monkeypatch, unknown_sequence
):
    spec = TrackerSpec(name="bytetrack", class_ids=(1,))
    setup = _prepare(timestamped_build, tmp_path / "setup", spec, monkeypatch)
    transforms = setup.replay_sequences
    if unknown_sequence:
        transforms = (replace(transforms[0], sequence_id="missing"),)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _inline)
    options = dict(
        split="variable",
        sequence_ids=("b",),
        sequence_frame_counts={"b": 1},
        output_dir=tmp_path / "subset",
        cmc_policy=setup.policy,
        cmc_transforms=transforms,
    )
    if unknown_sequence:
        with pytest.raises(ValueError, match="outside the replay build"):
            replay.replay_build(timestamped_build, spec, **options)
    else:
        result = replay.replay_build(timestamped_build, spec, **options)
        assert result.frames == 1
        assert [path.name for path in result.sequence_files] == ["b.txt"]


@pytest.mark.parametrize("changed", ["image", "order", "threshold", "build", "split"])
def test_prepared_replay_rejects_changed_inputs(timestamped_build, tmp_path, monkeypatch, changed):
    spec = TrackerSpec(name="bytetrack", class_ids=(1,))
    setup = _prepare(timestamped_build, tmp_path / "setup", spec, monkeypatch)
    transforms = setup.replay_sequences
    split = "variable"
    if changed == "image":
        (timestamped_build / "images" / "a-1.png").write_bytes(b"changed source image")
    elif changed == "order":
        transforms = (replace(transforms[0], sample_ids=tuple(reversed(transforms[0].sample_ids))),)
    elif changed == "threshold":
        spec = replace(spec, options=(("cmc_min_conf", 0.91),))
    elif changed == "build":
        timestamped_build = copytree(timestamped_build, tmp_path / "other-build")
    else:
        split = None
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _inline)
    with pytest.raises((RuntimeError, ValueError)):
        replay.replay_build(
            timestamped_build,
            spec,
            split=split,
            output_dir=tmp_path / "rejected",
            cmc_policy=setup.policy,
            cmc_transforms=transforms,
        )
    assert not list((tmp_path / "rejected").glob("*.txt"))


def test_prepared_replay_repeats_in_spawned_session(timestamped_build, tmp_path, monkeypatch):
    spec = TrackerSpec(name="bytetrack", class_ids=(1,))
    setup = _prepare(timestamped_build, tmp_path / "setup", spec, monkeypatch)
    outputs = []
    with ReplaySession(2) as session:
        for trial in range(2):
            result = replay.replay_build(
                timestamped_build,
                spec,
                split="variable",
                output_dir=tmp_path / f"trial-{trial}",
                workers=2,
                session=session,
                cmc_policy=setup.policy,
                cmc_transforms=setup.replay_sequences,
            )
            outputs.append([path.read_bytes() for path in result.sequence_files])
    assert outputs[0] == outputs[1]
