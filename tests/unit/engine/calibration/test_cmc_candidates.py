"""Camera candidates must use image evidence and runtime-equivalent masking."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets.cached import DatasetSample
from boxmot.engine.calibration import cmc
from boxmot.structures import Boxes, Detections
from boxmot.trackers.bytetrack.config import ByteTrackConfig
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf
from boxmot.trackers.common.motion.cmc.sof import SOF, SOF_DEFAULT_TARGET_WIDTH
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES


def test_analysis_settings_preserve_estimator_identity_through_json(monkeypatch):
    """Saved evidence must distinguish preprocessing, corners, and optical flow."""
    estimator = SOF()
    estimator.target_width = 256
    estimator.resize_before_grayscale = False
    estimator.feature_params["maxCorners"] = 80
    estimator.feature_refresh_interval = 3
    estimator.lk_params["winSize"] = (9, 9)
    estimator.lk_params["maxLevel"] = 1
    estimator.lk_params["criteria"] = (cv2.TERM_CRITERIA_COUNT, 15, 0.02)
    monkeypatch.setattr(cmc, "SOF", lambda **kwargs: estimator)

    settings = cmc.cmc_analysis_settings()

    assert json.loads(json.dumps(settings, allow_nan=False)) == settings
    assert settings["target_width"] == 256
    assert settings["resize_before_grayscale"] is False
    assert settings["feature_params"] == estimator.feature_params
    assert settings["feature_refresh_interval"] == 3
    assert settings["lk_params"] == {
        "winSize": [9, 9],
        "maxLevel": 1,
        "criteria": [cv2.TERM_CRITERIA_COUNT, 15, 0.02],
    }


def _sample(sequence: str, index: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"{sequence}/{index}",
        split="train",
        sequence_id=sequence,
        frame_index=index,
        timestamp_s=None,
        image_size=(100, 100),
        image_ref=f"{sequence}/{index}.png",
        frame=None,
        detections=Detections(
            sample_id=f"{sequence}/{index}",
            geometry=Boxes(torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)),
            scores=torch.tensor([0.2, 0.9], dtype=torch.float32),
            class_ids=torch.tensor([1, 2], dtype=torch.int64),
        ),
    )


class _Dataset(list):
    manifest = SimpleNamespace(box_type="aabb")
    _image_root = "/unused"


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    (build / "manifest.json").write_text("{}")
    datasets = {sequence: _Dataset(_sample(sequence, index) for index in range(4)) for sequence in ("moving", "static")}
    monkeypatch.setattr(
        cmc.CachedVisionDataset, "_for_sequence", lambda build, *, sequence_id, **kwargs: datasets[sequence_id]
    )
    monkeypatch.setattr(
        cmc, "_bgr_image", lambda sample, root: np.full((100, 100, 3), sample.frame_index, dtype=np.uint8)
    )
    args = Namespace(
        tracker="bytetrack",
        tracker_backend="python",
        geometry="aabb",
        build_path=build,
        split="train",
        seq_info={sequence: len(dataset) for sequence, dataset in datasets.items()},
        sequence_names=None,
    )
    return args, datasets


def test_camera_candidates_cache_raw_warps_but_disable_static_sequences(prepared_inputs, monkeypatch, tmp_path):
    args, _datasets = prepared_inputs
    masking_inputs = []
    instances = []

    class Estimator:
        def __init__(self, **kwargs):
            assert kwargs == {"target_width": SOF_DEFAULT_TARGET_WIDTH, "ransac_reproj_threshold": 3.0}
            self.number = len(instances)
            instances.append(self)
            self.frame = 0

        def apply(self, image, boxes):
            masking_inputs.append(boxes.copy())
            self.was_initializing = self.frame == 0
            self.inliers_accepted = self.transform_valid = self.frame > 0
            self.match_count = self.inlier_count = 30
            warp = np.eye(2, 3, dtype=np.float32)
            # Sorted "moving" first; static has measurable subthreshold motion.
            warp[0, 2] = (3.0 if self.number == 0 else 0.01) if self.frame else 0
            self.frame += 1
            return warp

    monkeypatch.setattr(cmc, "_DiagnosticSOF", Estimator)
    previous_threads = cv2.getNumThreads()
    result = cmc.prepare_cmc_candidates(args, {"min_conf": 0.1, "cmc_min_conf": 0.5}, tmp_path / "camera")
    assert cv2.getNumThreads() == previous_threads
    assert dict(result.policy.sequences) == {"moving": True, "static": False}
    assert len(instances) == 2
    assert len(masking_inputs) == 8
    for boxes in masking_inputs:
        np.testing.assert_array_equal(boxes, [[30, 30, 40, 40]])
        assert boxes.dtype == np.float32
    np.testing.assert_array_equal(result.warps["moving"][1:, 0, 2], [3, 3, 3])
    np.testing.assert_array_equal(result.warps["static"], np.broadcast_to(np.eye(2, 3), (4, 2, 3)))
    assert not result.warps["moving"].flags.writeable
    report = json.loads(result.report_path.read_text())
    assert report["ground_truth_used"] is False
    assert report["mask_min_conf"] == result.mask_min_conf == 0.5
    assert report["total_frames"] == 8
    assert report["sequences"]["moving"]["accepted_fraction"] == 1
    assert report["sequences"]["static"]["reason"] == "camera_motion_below_threshold"
    with np.load(result.warps_path, allow_pickle=False) as archive:
        keys = report["sequences"]["static"]["archive_keys"]
        np.testing.assert_array_equal(archive[keys["frame_indices"]], np.arange(4))
        np.testing.assert_allclose(archive[keys["warps"]][1:, 0, 2], 0.01)
    with pytest.raises(ValueError, match="already exist"):
        cmc.prepare_cmc_candidates(args, {}, tmp_path / "camera")


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("explicit_mask", [None, 0.5])
def test_all_kalman_trackers_record_their_identity_and_fixed_mask(
    prepared_inputs, monkeypatch, tmp_path, tracker, explicit_mask
):
    """Analysis derives the tracker-specific mask once and records portable provenance."""
    args, _ = prepared_inputs
    args.tracker = tracker
    inputs = []
    original = cmc._DiagnosticSOF.apply

    def capture(self, image, boxes):
        inputs.append(boxes.copy())
        return original(self, image, boxes)

    monkeypatch.setattr(cmc._DiagnosticSOF, "apply", capture)
    baseline = load_tracker_config(tracker)
    baseline["cmc_min_conf"] = explicit_mask
    threshold = resolve_cmc_min_conf(baseline)
    result = cmc.prepare_cmc_candidates(args, baseline, tmp_path / "camera")
    assert result.report["tracker"] == tracker
    assert result.report["method"] == "sof"
    assert result.mask_min_conf == result.report["mask_min_conf"] == threshold
    sample = _sample("moving", 0)
    expected = sample.detections.geometry.values.numpy()[sample.detections.scores.numpy() >= threshold]
    assert len(inputs) == 8
    assert all(np.array_equal(mask, expected) for mask in inputs)


@pytest.mark.parametrize(
    "overrides,width,threshold",
    [({}, SOF_DEFAULT_TARGET_WIDTH, 3.0), ({"sof_target_width": 320, "sof_ransac_reproj_threshold": 2.0}, 320, 2.0)],
)
def test_occluboost_analysis_uses_resolved_runtime_sof_settings(
    prepared_inputs, monkeypatch, tmp_path, overrides, width, threshold
):
    """Sparse specifications and explicit overrides must share runtime camera geometry."""
    args, _ = prepared_inputs
    args.tracker = "occluboost"
    constructed = []
    original = cmc._DiagnosticSOF.__init__

    def initialize(self, **kwargs):
        constructed.append(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(cmc._DiagnosticSOF, "__init__", initialize)
    result = cmc.prepare_cmc_candidates(args, overrides, tmp_path / "camera")

    assert constructed == [{"target_width": width, "ransac_reproj_threshold": threshold}] * 2
    assert result.report["settings"]["target_width"] == width
    assert result.report["settings"]["ransac_reproj_threshold_resized_px"] == threshold


@pytest.mark.parametrize(
    ("accepted", "motion", "decision", "reason"),
    [
        ([False], [0], False, "no_camera_transitions"),
        ([False, False, False, True], [0, 0, 0, 0.1], False, "insufficient_accepted_transforms"),
        ([False, True, True, True], [0, 0, 0, 0], False, "camera_motion_below_threshold"),
        ([False, True, True, True], [0, 0.001, 0.001, 0.001], True, "reliable_camera_motion"),
        ([False, True, False, True, True, False], [0, 0.01, 0, 0.01, 0.01, 0], True, "reliable_camera_motion"),
    ],
)
def test_automatic_gate_distinguishes_rejected_fits_from_static_motion(accepted, motion, decision, reason):
    assert cmc._automatic_decision(np.asarray(accepted), np.asarray(motion)) == (decision, reason)


def test_diagnostics_preserve_production_warps_for_translated_textured_images():
    random = np.random.default_rng(7)
    image = random.integers(0, 256, size=(320, 400, 3), dtype=np.uint8)
    frames = [cv2.warpAffine(image, np.float32([[1, 0, 3 * index], [0, 1, 0]]), (400, 320)) for index in range(4)]
    expected, diagnostics = SOF(), cmc._DiagnosticSOF()
    accepted = []
    for frame in frames:
        cv2.setRNGSeed(0)
        production_warp = expected.apply(frame, np.empty((0, 4), dtype=np.float32))
        cv2.setRNGSeed(0)
        observed_warp = diagnostics.apply(frame, np.empty((0, 4), dtype=np.float32))
        np.testing.assert_array_equal(observed_warp, production_warp)
        accepted.append(diagnostics.inliers_accepted and diagnostics.transform_valid)
    assert accepted[0] is False
    assert all(accepted[1:])


def test_camera_analysis_restores_threads_after_missing_frame(prepared_inputs, monkeypatch, tmp_path):
    args, datasets = prepared_inputs
    datasets["moving"][1] = _sample("moving", 9)
    original_threads = cv2.getNumThreads()
    with pytest.raises(ValueError, match="contiguous zero-based"):
        cmc.prepare_cmc_candidates(args, {"min_conf": 0.1}, tmp_path / "camera")
    assert cv2.getNumThreads() == original_threads
    assert not (tmp_path / "camera" / "analysis.json").exists()


def test_camera_analysis_validates_published_frame_count(prepared_inputs, tmp_path):
    args, _datasets = prepared_inputs
    args.seq_info["moving"] = 5
    with pytest.raises(ValueError, match="different cached and selected frame count"):
        cmc.prepare_cmc_candidates(args, {}, tmp_path / "camera")


def test_camera_analysis_matches_runtime_class_validation(prepared_inputs, tmp_path):
    args, _datasets = prepared_inputs
    args.tracker_class_ids = (1,)
    with pytest.raises(ValueError, match="not present in the tracker class catalog"):
        cmc.prepare_cmc_candidates(args, {}, tmp_path / "camera")


@pytest.mark.parametrize("per_class", [False, True])
def test_camera_masks_match_runtime_across_all_published_classes(prepared_inputs, monkeypatch, tmp_path, per_class):
    args, _datasets = prepared_inputs
    args.tracker_class_ids = (1, 2)
    args.per_class = per_class
    cached_masks, runtime_masks = [], []
    kernel = cmc._DiagnosticSOF.apply

    def capture(self, image, boxes):
        cached_masks.append(boxes.copy())
        return kernel(self, image, boxes)

    monkeypatch.setattr(cmc._DiagnosticSOF, "apply", capture)
    cmc.prepare_cmc_candidates(args, {"cmc_min_conf": 0.2}, tmp_path / "camera")
    tracker = ByteTrack(
        ByteTrackConfig(use_cmc=True, min_conf=0.8, cmc_min_conf=0.2),
        class_ids=(1, 2),
        per_class=per_class,
    )

    class CaptureCMC:
        def apply(self, image, boxes):
            runtime_masks.append(boxes.copy())
            return np.eye(2, 3, dtype=np.float32)

    tracker.cmc = CaptureCMC()
    sample = _sample("moving", 0)
    tracker.update(sample.detections, frame=np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(runtime_masks) == 1
    np.testing.assert_array_equal(runtime_masks[0], cached_masks[0])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_camera_analysis_rejects_invalid_mask_threshold(prepared_inputs, tmp_path, threshold):
    args, _datasets = prepared_inputs
    with pytest.raises(ValueError, match="confidence"):
        cmc.prepare_cmc_candidates(args, {"cmc_min_conf": threshold}, tmp_path / "camera")


@pytest.mark.parametrize(("tracker", "backend"), [("sfsort", "python"), ("eagermot", "python"), ("bytetrack", "cpp")])
def test_camera_analysis_rejects_unimplemented_runtime_masks(prepared_inputs, tmp_path, tracker, backend):
    args, _datasets = prepared_inputs
    args.tracker, args.tracker_backend = tracker, backend
    with pytest.raises(ValueError, match="Python 2D Kalman tracker"):
        cmc.prepare_cmc_candidates(args, {}, tmp_path / "camera")


def test_image_reader_requires_images_at_published_dimensions(tmp_path: Path):
    sample = _sample("moving", 0)
    path = tmp_path / sample.image_ref
    path.parent.mkdir(parents=True)
    bgr = np.full((100, 100, 3), (10, 40, 80), dtype=np.uint8)
    cv2.imwrite(str(path), bgr)
    np.testing.assert_array_equal(cmc._bgr_image(sample, tmp_path), bgr)
    cv2.imwrite(str(path), bgr[:90])
    with pytest.raises(ValueError, match="published size"):
        cmc._bgr_image(sample, tmp_path)
