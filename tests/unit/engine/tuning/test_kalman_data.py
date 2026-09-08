"""Direct calibration matches canonical detections to GT without replay or perception."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets.cached import CachedVisionDataset
from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.engine.tuning.kalman_data import _match_detections, load_calibration_data
from boxmot.trackers.common.geometry.obb import xywha_to_corners


def _fixture(
    tmp_path: Path,
    *,
    geometry: str = "aabb",
    flat: bool = False,
    missing_timestamps: bool = False,
) -> Namespace:
    """Publish keyed detections with irregular timing, misses and ignored GT rows."""

    detect = StagePlan.create("detect", component={"id": "fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="calibration-fixture",
        box_type=geometry,
        source_fingerprint=fingerprint({"geometry": geometry}),
        publish=PublishOptions(image_references=False),
        stages=(detect, finalize),
        metadata={"dataset_id": "calibration-fixture", "split": "train"},
    )
    plan.staging_root.mkdir(parents=True)
    samples, instances, gt = [], [], []
    # Lexical sample order differs from frame order to detect positional joins.
    for frame, timestamp in enumerate([0.0, 0.1, 0.4, 0.5]):
        sample_id = f"key-{3 - frame}"
        samples.append(
            dict(
                sample_id=sample_id,
                split="train",
                sequence_id="seq",
                frame_index=frame,
                timestamp_s=None if missing_timestamps else timestamp,
                image_ref=None,
                height=100,
                width=100,
            )
        )
        box = [20.0 + timestamp, 20.0, 40.0 + timestamp, 60.0]
        gt_box = [box[0], box[1], 20.0, 40.0]
        if geometry == "obb":
            box = [30.0 + timestamp, 40.0, 20.0, 40.0, 0.2]
            gt_box = xywha_to_corners(np.asarray([box])).reshape(-1).tolist()
        gt.append([frame + 1, 1, *gt_box, 1, 1, 1])
        # Distractor class, unscored GT and invalid identity must be excluded.
        gt.extend(
            [
                [frame + 1, 2, *gt_box, 1, 7, 1],
                [frame + 1, 3, *gt_box, 0, 1, 1],
                [frame + 1, -1, *gt_box, 1, 1, 1],
            ]
        )
        if frame == 2:
            continue
        names = ("x1", "y1", "x2", "y2") if geometry == "aabb" else ("cx", "cy", "w", "h", "angle")
        for index, class_id in enumerate((1, 2)):
            instances.append(
                dict(
                    instance_id=f"{plan.build_id}:{sample_id}:{index}",
                    sample_id=sample_id,
                    detection_index=index,
                    **dict(zip(names, box, strict=True)),
                    score=0.9,
                    class_id=class_id,
                )
            )
    writer = ParquetShardWriter(plan.staging_root, box_type=geometry)
    writer.write(SAMPLES_ARTIFACT, samples[::-1], shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances[::-1], shard_index=0)
    build = finalize_build(plan)
    source = tmp_path / "source"
    gt_path = source / "seq.txt" if flat else source / "seq" / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True)
    np.savetxt(gt_path, gt, delimiter=",", fmt="%.12g")
    return Namespace(
        build_path=build,
        split="train",
        seq_info={"seq": 4},
        sequence_names=None,
        geometry=geometry,
        variable_dt=True,
        source=source,
        gt_folder=source,
        remapped_class_ids=[1],
        remapped_class_names=["person"],
        evaluation_config={
            "layout": "visdrone" if flat else "mot",
            "box_type": geometry,
            "annotation_layout": "flat" if flat else "sequence",
            "classes": {"person": {"id": 1, "evaluation": "target"}, "ignore": {"id": 7, "evaluation": "ignore"}},
        },
    )


@pytest.mark.parametrize("geometry,flat", [("aabb", False), ("aabb", True), ("obb", False), ("obb", True)])
def test_matches_by_sample_class_and_gt_identity_with_explicit_misses(tmp_path, monkeypatch, geometry, flat):
    args = _fixture(tmp_path, geometry=geometry, flat=flat)
    reads = []
    original = CachedVisionDataset._read_sequence_rows

    def record(self, name, **kwargs):
        reads.append(name)
        return original(self, name, **kwargs)

    monkeypatch.setattr(CachedVisionDataset, "_read_sequence_rows", record)
    data = load_calibration_data(args)
    assert reads == [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
    assert len(data.tracks) == 1
    track = data.tracks[0]
    assert (track.sequence_id, track.class_id, track.track_id) == ("seq", 1, 1)
    np.testing.assert_array_equal(track.frame_indices, [0, 1, 2, 3])
    np.testing.assert_allclose(track.timestamps_s, [0.0, 0.1, 0.4, 0.5])
    assert np.isnan(track.detection_boxes[2]).all()
    assert np.isnan(track.scores[2])
    assert data.statistics == {
        "frames": 4,
        "detections": 6,
        "target_detections": 3,
        "ground_truth_rows": 16,
        "ground_truth": 4,
        "filtered_ground_truth": 12,
        "matched": 3,
        "unmatched_ground_truth": 1,
        "unmatched_detections": 0,
        "trajectories": 1,
    }
    if geometry == "aabb":
        np.testing.assert_allclose(track.gt_boxes[[0, 1, 3]], track.detection_boxes[[0, 1, 3]], atol=1e-5)
    else:
        assert track.gt_boxes.shape == (4, 5)
        assert np.all(np.abs(track.gt_boxes[:, 4]) > 0.1)
        assert np.allclose(np.prod(track.gt_boxes[:, 2:4], axis=1), 800, atol=0.01)
    assert len(data.ground_truth_sources) == 1
    assert len(data.ground_truth_sources[0]["sha256"]) == 64
    json.dumps(data.statistics)


def test_obb_matching_uses_oriented_iou_instead_of_enclosing_boxes():
    gt = np.array([[50, 50, 60, 4, np.pi / 4]])
    # Same enclosing AABB; the physical rectangles cross with very low IoU.
    detections = np.array([[50, 50, 60, 4, -np.pi / 4]])
    gt_matches, detection_matches = _match_detections(gt, detections, "obb")
    assert len(gt_matches) == len(detection_matches) == 0


def test_matching_is_one_to_one_and_gated():
    gt = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [30, 30, 40, 40]], dtype=float)
    detections = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    gt_matches, detection_matches = _match_detections(gt, detections, "aabb")
    np.testing.assert_array_equal(gt_matches, [0])
    np.testing.assert_array_equal(detection_matches, [0])


@pytest.mark.parametrize(
    "bad_row,reason",
    [
        ("1,1,bad,20,20,40,1,1,1\n", "Malformed"),
        ("1,1,nan,20,20,40,1,1,1\n", "non-finite"),
        ("1.5,1,20,20,20,40,1,1,1\n", "integer frame"),
        ("5,1,20,20,20,40,1,1,1\n", "outside"),
        ("1,1,20,20,20,40\n", "at least 8"),
    ],
)
def test_invalid_gt_fails_instead_of_becoming_empty(tmp_path, bad_row, reason):
    args = _fixture(tmp_path)
    (args.gt_folder / "seq/gt/gt.txt").write_text(bad_row)
    with pytest.raises(ValueError, match=reason):
        load_calibration_data(args)


def test_duplicate_gt_identity_fails(tmp_path):
    args = _fixture(tmp_path)
    gt_path = args.gt_folder / "seq/gt/gt.txt"
    gt_path.write_text(gt_path.read_text() + "1,1,20,20,20,40,1,1,1\n")
    with pytest.raises(ValueError, match="repeats a class/identity"):
        load_calibration_data(args)


def test_gt_digest_tracks_exact_annotation_bytes(tmp_path):
    args = _fixture(tmp_path)
    first = load_calibration_data(args)
    gt_path = args.gt_folder / "seq/gt/gt.txt"
    gt_path.write_text(gt_path.read_text() + "\n")
    second = load_calibration_data(args)
    assert first.ground_truth_sources[0]["sha256"] != second.ground_truth_sources[0]["sha256"]
    assert first.statistics == second.statistics


def test_variable_dt_requires_measured_timestamps_while_fixed_mode_does_not(tmp_path):
    args = _fixture(tmp_path, missing_timestamps=True)
    with pytest.raises(ValueError, match="requires timestamps for every frame"):
        load_calibration_data(args)
    args.variable_dt = False
    data = load_calibration_data(args)
    assert data.tracks[0].timestamps_s is None
    np.testing.assert_array_equal(data.tracks[0].frame_indices, [0, 1, 2, 3])


def test_gt_annotation_gaps_are_preserved_without_inventing_observations(tmp_path):
    args = _fixture(tmp_path)
    gt_path = args.gt_folder / "seq/gt/gt.txt"
    rows = np.loadtxt(gt_path, delimiter=",")
    np.savetxt(gt_path, rows[rows[:, 0] != 2], delimiter=",")
    data = load_calibration_data(args)
    np.testing.assert_array_equal(data.tracks[0].frame_indices, [0, 2, 3])
    np.testing.assert_allclose(data.tracks[0].timestamps_s, [0, 0.4, 0.5])
    assert data.statistics["unmatched_detections"] == 1


def test_malformed_obb_gt_does_not_fall_back_to_another_annotation(tmp_path):
    args = _fixture(tmp_path, geometry="obb")
    gt_path = args.gt_folder / "seq/gt/gt.txt"
    gt_path.with_name("gt_obb.txt").write_text(gt_path.read_text())
    gt_path.write_text("1,1,20,20,20,40,1,1,1\n")
    with pytest.raises(ValueError, match="13 MMOT corner columns"):
        load_calibration_data(args)
