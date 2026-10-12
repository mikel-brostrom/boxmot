"""Sequence policies must preserve tuned options and isolate pixel requirements."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from boxmot.engine.config.cmc import SequenceCMCPolicy
from boxmot.engine.eval import replay
from boxmot.engine.eval.session import ReplaySession
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from tests.unit.engine.eval import test_replay_callback as callback_fixtures


@pytest.fixture
def image_build(tmp_path: Path) -> Path:
    """Use the keyed two-sequence image fixture with genuine decoded pixels."""
    return callback_fixtures.timestamped_build.__wrapped__(tmp_path)


def _run_inline(tasks, *, workers, progress_callback):
    """Keep worker instrumentation visible while exercising real sequence replay."""
    del workers
    return tuple(replay._replay_sequence_task(task, progress_callback=progress_callback) for task in tasks)


@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_policy_only_loads_pixels_for_enabled_sequences(image_build, tmp_path, monkeypatch, cache_inputs, tracker):
    lifetime = "track_buffer" if tracker in {"bytetrack", "botsort"} else "max_age"
    spec = TrackerSpec(name=tracker, options=(("cmc_min_conf", 0.17), ("kalman.type", "xyah"), (lifetime, 17)))
    policy = SequenceCMCPolicy(sequences=(("a", True),))
    opened = []
    original_inputs = replay._worker_inputs

    def worker_inputs(task, **kwargs):
        opened.append((task.sequence_id, task.tracker_spec.option_dict, kwargs))
        return original_inputs(task, **kwargs)

    monkeypatch.setattr(replay, "_worker_inputs", worker_inputs)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_inline)
    result = replay.replay_build(
        image_build,
        spec,
        output_dir=tmp_path / "policy",
        workers=2,
        cache_inputs=cache_inputs,
        cmc_policy=policy,
    )

    assert result.frames == 4
    assert [(sequence, opts["use_cmc"], kwargs["load_images"]) for sequence, opts, kwargs in opened] == [
        ("a", True, True),
        ("b", False, False),
    ]
    assert all(
        opts["cmc_min_conf"] == 0.17 and opts["kalman.type"] == "xyah" and opts[lifetime] == 17
        for _, opts, _ in opened
    )
    assert spec.option_dict == {"cmc_min_conf": 0.17, "kalman.type": "xyah", lifetime: 17}
    recorded = json.loads((result.output_dir / "cmc-policy.json").read_text())
    assert recorded["policy"] == policy.to_dict()
    assert recorded["tracker"]["options"] == spec.option_dict
    assert recorded["sequences"] == {
        "a": {"use_cmc": True, "cmc_method": "sof"},
        "b": {"use_cmc": False, "cmc_method": "sof"},
    }


def test_policy_validates_against_build_even_with_subset_counts(image_build, tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_inline)
    result = replay.replay_build(
        image_build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "subset",
        sequence_ids=("a",),
        sequence_frame_counts={"a": 3},
        cmc_policy=SequenceCMCPolicy(sequences=(("b", True),)),
    )
    assert result.frames == 3
    assert [path.name for path in result.sequence_files] == ["a.txt"]
    with pytest.raises(ValueError, match="Unknown sequence IDs.*missing"):
        replay.replay_build(
            image_build,
            TrackerSpec(name="bytetrack"),
            output_dir=tmp_path / "invalid",
            sequence_ids=("a",),
            sequence_frame_counts={"a": 3},
            cmc_policy=SequenceCMCPolicy(sequences=(("missing", False),)),
        )


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_policy_rendering_loads_all_pixels_and_creates_distinct_trackers(
    image_build, tmp_path, monkeypatch, cache_inputs
):
    created = []
    original_create = replay.create_tracker

    def create_tracker(spec):
        tracker = original_create(spec)
        created.append(tracker)
        return tracker

    frames = []
    monkeypatch.setattr(replay, "create_tracker", create_tracker)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", lambda *_a, **_kw: pytest.fail("unexpected spawn"))
    result = replay.replay_build(
        image_build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "rendered",
        frame_callback=frames.append,
        cmc_policy=SequenceCMCPolicy(sequences=(("a", True),)),
        cache_inputs=cache_inputs,
    )

    assert result.frames == len(frames) == 4
    assert all(item.sample.frame is not None for item in frames)
    assert all((item.sample.frame.image == 30 + item.sample.frame_index).all() for item in frames)
    active = [tracker for tracker in created if tracker.frame_count]
    assert len(active) == 2
    assert [tracker.cmc is not None for tracker in active] == [True, False]
    assert [tracker.frame_count for tracker in active] == [3, 1]


def test_policy_render_failure_does_not_publish_partial_results(image_build, tmp_path):
    destination = tmp_path / "failed"

    def fail_second_sequence(frame):
        if frame.sample.sequence_id == "b":
            raise ValueError("renderer failed")

    with pytest.raises(RuntimeError, match="sequence 'b'"):
        replay.replay_build(
            image_build,
            TrackerSpec(name="bytetrack"),
            output_dir=destination,
            frame_callback=fail_second_sequence,
            cmc_policy=SequenceCMCPolicy(sequences=(("a", True),)),
        )
    assert not list(destination.iterdir())


def test_policy_rejects_caller_owned_tracker_before_loading_build(tmp_path):
    with pytest.raises(ValueError, match="sequence-local, worker-owned"):
        replay.replay_build(
            tmp_path / "not-a-build",
            TrackerSpec(name="bytetrack"),
            output_dir=tmp_path / "output",
            tracker=object(),
            cmc_policy=SequenceCMCPolicy(),
        )


def test_policy_checks_image_references_for_selected_effective_tracker(image_build, tmp_path, monkeypatch):
    manifest = replay.DatasetManifest.load(image_build)
    no_images = replace(manifest, publish=replace(manifest.publish, image_references=False))
    monkeypatch.setattr(replay.DatasetManifest, "load", lambda _path: no_images)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_inline)
    policy = SequenceCMCPolicy(sequences=(("b", True),))
    result = replay.replay_build(
        image_build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "disabled",
        sequence_ids=("a",),
        cmc_policy=policy,
    )
    assert result.frames == 3
    with pytest.raises(replay.BuildCompatibilityError, match="--publish-image-refs"):
        replay.replay_build(
            image_build,
            TrackerSpec(name="bytetrack"),
            output_dir=tmp_path / "enabled",
            sequence_ids=("b",),
            cmc_policy=policy,
        )


@pytest.mark.parametrize("mode", ["ordinary", "rendered", "injected"])
def test_successful_no_policy_replay_removes_previous_policy_metadata(image_build, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_inline)
    destination = tmp_path / mode
    destination.mkdir()
    previous = destination / "cmc-policy.json"
    previous.write_text('{"previous": true}\n')
    spec = TrackerSpec(name="bytetrack")
    kwargs = {}
    if mode == "rendered":
        kwargs["frame_callback"] = lambda _frame: None
    elif mode == "injected":
        kwargs["tracker"] = replay.create_tracker(spec)
    result = replay.replay_build(image_build, spec, output_dir=destination, **kwargs)
    assert result.frames == 4
    assert not previous.exists()


def test_failed_no_policy_rendering_preserves_previous_policy_metadata(image_build, tmp_path):
    destination = tmp_path / "rendered"
    destination.mkdir()
    previous = destination / "cmc-policy.json"
    previous.write_text('{"previous": true}\n')

    def fail(_frame):
        raise ValueError("rendering failed")

    with pytest.raises(ValueError, match="rendering failed"):
        replay.replay_build(
            image_build,
            TrackerSpec(name="bytetrack"),
            output_dir=destination,
            frame_callback=fail,
        )
    assert previous.read_text() == '{"previous": true}\n'


def test_retained_workers_apply_each_runs_cmc_policy(image_build, tmp_path):
    spec = TrackerSpec(name="bytetrack")
    with ReplaySession(1, cache_inputs=True) as session:
        results = []
        for enabled in (False, True, False):
            results.append(
                replay.replay_build(
                    image_build,
                    spec,
                    output_dir=tmp_path / f"run-{len(results)}",
                    workers=1,
                    session=session,
                    cmc_policy=SequenceCMCPolicy(default=enabled),
                )
            )
    assert [result.frames for result in results] == [4, 4, 4]
    # The fixture has no camera motion; enabling CMC produces identity warps.
    assert [[path.read_bytes() for path in result.sequence_files] for result in results] == [
        [path.read_bytes() for path in results[0].sequence_files]
    ] * 3
