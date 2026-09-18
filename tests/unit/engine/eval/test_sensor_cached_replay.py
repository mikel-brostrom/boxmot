"""Mapped sensor inputs preserve scores and identities across fresh replay trials."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import pytest
import yaml

from boxmot.datasets.sequence import MultimodalSequence
from boxmot.engine.eval import eagermot_kitti as evaluation
from boxmot.engine.eval import mots
from boxmot.engine.eval.session import ReplaySession
from tests.unit.engine.eval.test_eagermot_kitti import _fixture


def _arguments(data: SimpleNamespace, *, cache_inputs: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=data.dataset,
        project=data.project,
        split=None,
        sequence_names=("0002",),
        cache_inputs=cache_inputs,
    )


def _worker_pid(_: Any) -> int:
    """Observe actual retained workers without transporting process handles."""
    return os.getpid()


def test_cached_trials_reuse_inputs_before_thresholds_and_preserve_fresh_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No trial rereads masks or raw detections, and high thresholds still remove all tracks."""
    data = _fixture(tmp_path / "data")
    raw = evaluation.prepare_eagermot_kitti(_arguments(data, cache_inputs=False))
    expected = evaluation.evaluate_eagermot_kitti(raw, evaluation.load_kitti_profiles(), tmp_path / "raw")
    inputs = evaluation.prepare_eagermot_kitti(_arguments(data, cache_inputs=True))

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Cached replay must not read raw sensor frames or decode ground-truth PNGs.")

    monkeypatch.setattr(MultimodalSequence, "__getitem__", unexpected)
    monkeypatch.setattr(mots, "_read_gt_frame", unexpected)
    for annotation in data.ground_truth.glob("*.png"):
        annotation.unlink()
    try:
        first = evaluation.evaluate_eagermot_kitti(inputs, evaluation.load_kitti_profiles(), tmp_path / "first")
        filtered = evaluation.load_kitti_profiles()
        for profile in filtered.values():
            profile.update(det_thresh=1.0, det_thresh_3d=1.0)
        second = evaluation.evaluate_eagermot_kitti(inputs, filtered, tmp_path / "filtered")
        repeated = evaluation.evaluate_eagermot_kitti(inputs, evaluation.load_kitti_profiles(), tmp_path / "repeat")
    finally:
        inputs.close()
        raw.close()

    assert first == repeated == expected
    assert second["cls_comb_cls_av"]["HOTA"] == 0
    assert (tmp_path / "filtered/mots/0002.txt").read_bytes() == b""
    assert (tmp_path / "first/mots/0002.txt").read_bytes() == (tmp_path / "repeat/mots/0002.txt").read_bytes()
    assert (tmp_path / "first/mots/0002.txt").read_bytes() == (tmp_path / "raw/mots/0002.txt").read_bytes()


@pytest.mark.parametrize(
    ("ignored_class", "ignore_ids", "label", "valid"),
    [
        (True, [], 10001, True),
        (True, None, 10000, True),
        (False, [10000], 10000, True),
        (False, [], 10000, False),
        (False, None, 10000, False),
        (False, [10000], 10001, False),
    ],
)
def test_cached_and_uncached_sensor_scoring_honor_configured_ignore_labels(
    tmp_path: Path, ignored_class: bool, ignore_ids: list[int] | None, label: int, valid: bool
) -> None:
    """The cache must preserve both configured ignore regions and rejected label errors."""
    data = _fixture(tmp_path / "data")
    config = yaml.safe_load(data.dataset.read_text())
    if not ignored_class:
        config["classes"].pop("ignore")
    if ignore_ids is None:
        config["modalities"]["ground_truth"]["options"].pop("ignore_ids")
    else:
        config["modalities"]["ground_truth"]["options"]["ignore_ids"] = ignore_ids
    data.dataset.write_text(yaml.safe_dump(config))
    for path in data.ground_truth.glob("*.png"):
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        labels[labels == 1001] = label
        assert cv2.imwrite(str(path), labels)
    profiles = evaluation.load_kitti_profiles()

    def evaluate(cache_inputs: bool) -> dict:
        inputs = evaluation.prepare_eagermot_kitti(_arguments(data, cache_inputs=cache_inputs))
        try:
            return evaluation.evaluate_eagermot_kitti(inputs, profiles, tmp_path / str(cache_inputs))
        finally:
            inputs.close()

    if valid:
        raw = evaluate(False)
        assert evaluate(True) == raw
        assert raw["car"]["GT_Dets"] == raw["car"]["Dets"] == 0
    else:
        for cache_inputs in (False, True):
            with pytest.raises(ValueError, match="unsupported labels"):
                evaluate(cache_inputs)


def test_cached_sensor_tasks_reopen_private_views_in_a_retained_worker(tmp_path: Path) -> None:
    """Closing each spawned task's mmap must leave the parent and next trial readable."""
    data = _fixture(tmp_path / "data")
    inputs = evaluation.prepare_eagermot_kitti(_arguments(data, cache_inputs=True))
    events = []
    try:
        with ReplaySession(1, cache_inputs=True) as session:
            first_pid = None
            for name in ("first", "repeat"):
                output = tmp_path / name
                (output / "mots").mkdir(parents=True)
                tasks = (
                    evaluation._KittiSequenceTask(
                        "0002", inputs.sequences["0002"], evaluation.load_kitti_profiles(), output, "val", 0
                    ),
                )
                result = session.run_sensor(tasks, progress_callback=events.append)
                assert result[0].track_rows == 4
                current_pid = session.map(_worker_pid, [None])[0]
                if first_pid is None:
                    first_pid = current_pid
                assert current_pid == first_pid != os.getpid()
                assert len(inputs.sequences["0002"][0].detections) == 2
        assert len({event.run_id for event in events}) == 2
    finally:
        inputs.close()
    assert (tmp_path / "first/mots/0002.txt").read_bytes() == (tmp_path / "repeat/mots/0002.txt").read_bytes()


@pytest.mark.parametrize("fail", (False, True))
def test_sensor_eval_releases_mapped_inputs_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    data = _fixture(tmp_path / "data")
    args = _arguments(data, cache_inputs=True)
    prepare = evaluation.prepare_eagermot_kitti
    captured = []

    def load(value: Any) -> evaluation.KittiReplayInputs:
        inputs = prepare(value)
        captured.extend(inputs.sequences.values())
        return inputs

    monkeypatch.setattr(evaluation, "prepare_eagermot_kitti", load)
    if fail:

        def interrupted(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt("cancelled replay")

        monkeypatch.setattr(evaluation, "evaluate_eagermot_kitti", interrupted)
        with pytest.raises(KeyboardInterrupt, match="cancelled replay"):
            evaluation.run_eagermot_kitti(args)
    else:
        evaluation.run_eagermot_kitti(args)
    assert captured
    for sequence in captured:
        with pytest.raises(ValueError, match="closed"):
            sequence[0]
