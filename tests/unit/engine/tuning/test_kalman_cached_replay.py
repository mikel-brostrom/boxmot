"""Calibrate and reuse seconds-based noise on real cached detections and GT."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval import evaluator, replay
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.engine.tuning.kalman import tune_kalman
from boxmot.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS
from boxmot.trackers.box.bytetrack.tracker import ByteTrack
from boxmot.trackers.config import load_tracker_config


def _timestamped_build(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """Publish eight frames, including two empty detections after 100 ms gaps."""
    timestamps = np.asarray([0, 1, 2, 5, 6, 9, 10, 11], dtype=float) / 30.0
    detect = StagePlan.create("detect", component={"id": "synthetic-detections"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="timestamped-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint({"timestamps": timestamps.tolist()}),
        publish=PublishOptions(image_references=False),
        stages=(detect, finalize),
        metadata={"dataset_id": "timestamped-fixture", "split": "train"},
    )
    plan.staging_root.mkdir(parents=True)
    samples, instances, ground_truth = [], [], []
    for index, timestamp in enumerate(timestamps):
        sample_id = f"seq/{index:04d}"
        x = 10.0 + 30.0 * timestamp
        samples.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "sequence_id": "seq",
                "frame_index": index,
                "timestamp_s": float(timestamp),
                "image_ref": None,
                "height": 80,
                "width": 100,
            }
        )
        ground_truth.append([index + 1, 1, x, 20.0, 20.0, 30.0, 1, 1, 1])
        if index in (3, 5):
            continue
        instances.append(
            {
                "instance_id": f"{plan.build_id}:{sample_id}:0",
                "sample_id": sample_id,
                "detection_index": 0,
                "x1": x,
                "y1": 20.0,
                "x2": x + 20.0,
                "y2": 50.0,
                "score": 0.95,
                "class_id": 1,
            }
        )
    writer = ParquetShardWriter(plan.staging_root, box_type=plan.box_type)
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    build = finalize_build(plan)
    source = tmp_path / "source"
    gt_path = source / "seq" / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True)
    np.savetxt(gt_path, ground_truth, delimiter=",", fmt="%.10g")
    return build, source, timestamps


def test_cached_kalman_calibration_preserves_timestamps_units_and_deployed_scores(monkeypatch, tmp_path: Path) -> None:
    build, source, timestamps = _timestamped_build(tmp_path)
    seen, trial_configs = [], []
    kernel = ByteTrack._track_detections

    def capture_update(self, dets, *args, **kwargs):
        seen.append((self._prediction_dt, len(dets), self.kalman_noise_config))
        return kernel(self, dets, *args, **kwargs)

    def run_tasks_inline(tasks, *, workers, progress_callback):
        # Keep actual cache loading, tracking, serialization, and metrics; process
        # scheduling is covered separately and would dominate this tiny fixture.
        assert workers == 1
        trial_configs.append(dict(tasks[0].tracker_spec.options))
        return tuple(replay._replay_sequence_task(task) for task in tasks)

    monkeypatch.setattr(ByteTrack, "_track_detections", capture_update)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", run_tasks_inline)
    # Inline workers need the same clean progress state as a fresh child process.
    monkeypatch.setattr(replay, "_WORKER_PROGRESS_QUEUE", None)
    args = Namespace(
        _build_validated=True,
        tracker="bytetrack",
        tracker_backend="python",
        tracker_config=None,
        tracker_class_ids=(1,),
        tracker_class_names=((1, "person"),),
        per_class=False,
        variable_dt=True,
        geometry="aabb",
        eval_box_type="aabb",
        dataset_id="timestamped-fixture",
        experiment_id=None,
        build_path=build,
        split="train",
        sequence_names=None,
        seq_info={"seq": len(timestamps)},
        seq_paths=(source / "seq",),
        source=source,
        gt_folder=source,
        evaluation_config={"layout": "mot", "box_type": "aabb", "classes": {"person": {"id": 1}}},
        remapped_class_ids=[1],
        remapped_class_names=["person"],
        compare_trackeval=False,
        n_threads=1,
        kf_trials=2,
    )

    calibration = tune_kalman(args, output_dir=tmp_path / "calibration")
    report = json.loads(calibration.report_path.read_text())
    saved = load_tracker_config("bytetrack", calibration.config_path)
    assert report["status"] == "complete"
    assert len(report["trials"]) == 2
    assert 0.0 < calibration.baseline_hota <= calibration.best_hota <= 100.0
    assert saved["variable_dt"] is True
    assert saved["kf_time_unit"] == "seconds"
    assert saved["kf_reference_dt_s"] == pytest.approx(1.0 / 30.0)
    for key in ("variable_dt", "kf_time_unit", "kf_reference_dt_s"):
        assert report["timing"][key] == saved[key]
    assert set(report["search_bounds"]) == set(KALMAN_NOISE_OPTIONS)
    assert set(report["trials"][0]["scales"]) == set(KALMAN_NOISE_OPTIONS)
    assert trial_configs[0] != trial_configs[1]
    assert {key: value for key, value in trial_configs[0].items() if key not in KALMAN_NOISE_OPTIONS} == {
        key: value for key, value in trial_configs[1].items() if key not in KALMAN_NOISE_OPTIONS
    }

    # A deployment-style replay loads units and timing solely from best.yaml.
    args.tracker_config = calibration.config_path
    args.variable_dt = None
    deployed = evaluator.run_eval(args, setup=False, output_dir=tmp_path / "deployed")
    assert deployed.summary["HOTA"] == pytest.approx(calibration.best_hota)
    assert deployed.timings["frames"] == len(timestamps)
    assert len(seen) == 3 * len(timestamps)
    for offset in range(0, len(seen), len(timestamps)):
        sequence = seen[offset : offset + len(timestamps)]
        assert sequence[0][0] is None
        np.testing.assert_allclose([event[0] for event in sequence[1:]], np.diff(timestamps))
        assert [event[1] for event in sequence] == [1, 1, 1, 0, 1, 0, 1, 1]
        assert all(event[2].time_unit == "seconds" for event in sequence)
        assert all(event[2].reference_dt_s == saved["kf_reference_dt_s"] for event in sequence)
