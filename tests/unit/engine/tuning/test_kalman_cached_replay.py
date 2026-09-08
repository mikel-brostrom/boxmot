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
from boxmot.engine.tuning.kalman import calibrate_kalman
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


def test_cached_kalman_calibration_runs_no_tracker_until_final_evaluation(monkeypatch, tmp_path: Path) -> None:
    build, source, timestamps = _timestamped_build(tmp_path)
    seen, replay_configs = [], []
    kernel = ByteTrack._track_detections

    def capture_update(self, dets, *args, **kwargs):
        seen.append((self._prediction_dt, len(dets), self.kalman_noise_config))
        return kernel(self, dets, *args, **kwargs)

    def run_tasks_inline(tasks, *, workers, progress_callback):
        # Keep actual cache loading, tracking, serialization, and metrics; process
        # scheduling is covered separately and would dominate this tiny fixture.
        assert workers == 1
        replay_configs.append(dict(tasks[0].tracker_spec.options))
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
        sequence_workers=1,
    )

    calibration = calibrate_kalman(args, output_dir=tmp_path / "calibration")
    # Calibration uses cached detections and labelled trajectories directly:
    # no replay workers, tracker association, or tracking metrics run here.
    assert seen == []
    assert replay_configs == []
    assert calibration.config_path.name == "calibrated.yaml"
    assert calibration.report_path.name == "calibration.json"
    assert set(path.name for path in calibration.config_path.parent.iterdir()) == {
        "calibrated.yaml",
        "calibration.json",
    }
    report = json.loads(calibration.report_path.read_text())
    saved = load_tracker_config("bytetrack", calibration.config_path)
    assert report["status"] == "complete"
    assert report["method"] == "supervised_covariance_moments"
    assert "trials" not in report
    assert "final_summary" not in report
    assert calibration.matched_detections == report["statistics"]["matched"] == 6
    assert report["statistics"]["unmatched_ground_truth"] == 2
    assert calibration.gt_transitions == report["statistics"]["gt_transitions"] == len(timestamps) - 2
    assert set(calibration.fitted_parameters) == {
        "kf_process_position_scale",
        "kf_process_velocity_scale",
        "kf_measurement_noise_scale",
    }
    # One labelled birth cannot calibrate population initialization uncertainty.
    for key in ("kf_initial_position_scale", "kf_initial_velocity_scale"):
        assert report["parameters"][key]["status"] == "retained"
        assert saved[key] == report["baseline_config"][key]
    assert saved["variable_dt"] is True
    assert saved["kf_time_unit"] == "seconds"
    assert saved["kf_reference_dt_s"] == pytest.approx(1.0 / 30.0)
    for key in ("variable_dt", "kf_time_unit", "kf_reference_dt_s"):
        assert report["timing"][key] == saved[key]
    assert set(report["parameters"]) == set(KALMAN_NOISE_OPTIONS)
    for key in KALMAN_NOISE_OPTIONS:
        assert saved[key] == report["parameters"][key]["value"]
        assert np.isfinite(saved[key]) and saved[key] > 0

    # A deployment-style replay loads units and timing solely from calibrated.yaml.
    args.tracker_config = calibration.config_path
    args.variable_dt = None
    deployed = evaluator.run_eval(args, setup=False, output_dir=tmp_path / "deployed")
    assert 0 < deployed.summary["HOTA"] <= 100
    assert deployed.timings["frames"] == len(timestamps)
    assert len(replay_configs) == 1
    for key in (*KALMAN_NOISE_OPTIONS, "variable_dt", "kf_time_unit", "kf_reference_dt_s"):
        assert replay_configs[0][key] == saved[key]
    assert len(seen) == len(timestamps)
    assert seen[0][0] is None
    np.testing.assert_allclose([event[0] for event in seen[1:]], np.diff(timestamps))
    assert [event[1] for event in seen] == [1, 1, 1, 0, 1, 0, 1, 1]
    assert all(event[2].time_unit == "seconds" for event in seen)
    assert all(event[2].reference_dt_s == saved["kf_reference_dt_s"] for event in seen)

    # The final score is attached for reporting only, after parameters are fixed.
    calibration.record_final(deployed)
    report = json.loads(calibration.report_path.read_text())
    assert report["final_summary"] == deployed.summary
    assert report["final_output_dir"] == str(deployed.exp_dir)
