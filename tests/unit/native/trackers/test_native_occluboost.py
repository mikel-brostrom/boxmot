from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from boxmot.motion.cmc.sof import SOF
from boxmot.native import _common as native_common
from boxmot.native.trackers import occluboost as native_module
from boxmot.trackers.bbox.occluboost import OccluBoost
from boxmot.trackers.common.geometry.obb import transform_obb_kalman_state


def test_native_occluboost_advertises_inputs_and_requires_image():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            raise AssertionError("Missing images must not reach the native library")

        def get_last_reid_time_ms(self, handle):
            return 0.0

        def destroy(self, handle):
            return None

    tracker = native_module.NativeOccluBoostTracker({"with_reid": False}, library=_FakeLibrary())
    dets = np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float32)
    try:
        assert tracker.supports_masks is False
        assert tracker.uses_img is True
        assert tracker.uses_embs is False
        with pytest.raises(ValueError, match="Native OccluBoost requires img"):
            tracker.update(dets)
    finally:
        tracker.close()


def test_native_occluboost_disabled_reid_model_does_not_require_pixels(monkeypatch):
    calls = []

    def fail_reid_resolution(_weights):
        raise AssertionError("Disabled ReID weights must not be resolved")

    monkeypatch.setattr(native_module, "_ensure_native_reid_model_path", fail_reid_resolution)

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            calls.append((img, embs))
            return np.empty((0, 8), dtype=np.float32)

        def get_last_reid_time_ms(self, handle):
            return 0.0

        def destroy(self, handle):
            return None

    tracker = native_module.NativeOccluBoostTracker(
        {"with_reid": False, "use_cmc": False},
        reid_weights="models/disabled.onnx",
        library=_FakeLibrary(),
    )
    try:
        dets = np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float32)
        output = tracker.update(dets)
        assert tracker.provides_reid is False
        assert tracker.uses_img is False
        assert tracker.uses_embs is False
        assert tracker.requires_image(dets) is False
        assert output.shape == (0, 8)
        assert calls == [(None, None)]
    finally:
        tracker.close()


def test_native_occluboost_c_api_image_requirement_follows_cmc():
    library = native_module._OccluBoostLiveLibrary(native_module.ensure_occluboost_cpp_library())
    active = native_module.NativeOccluBoostTracker(
        {"with_reid": False, "use_cmc": True},
        library=library,
    )
    inactive = native_module.NativeOccluBoostTracker(
        {"with_reid": False, "use_cmc": False},
        library=library,
    )
    detections = np.empty((0, 6), dtype=np.float32)

    try:
        with pytest.raises(RuntimeError, match=r"requires (?:an )?(?:img|image)"):
            library.update(active._handle, detections, None)
        output = library.update(inactive._handle, detections, None)
    finally:
        active.close()
        inactive.close()

    assert output.shape == (0, 8)


def test_native_occluboost_resolves_flattened_defaults():
    cfg = native_module._resolve_tracker_cfg(None)

    assert cfg["use_cmc"] is True
    assert cfg["cmc_method"] == "sof"
    assert "dlo_boost_coef" in cfg
    assert "recovery_appearance_thresh" in cfg
    assert "second_appearance_thresh" in cfg
    assert cfg["obb_det_thresh"] == 0.2
    assert cfg["obb_iou_threshold"] == 0.15
    assert cfg["obb_new_track_thresh"] == 0.3


def test_native_occluboost_live_config_honors_disabled_cmc():
    cfg = native_module._resolve_tracker_cfg({"use_cmc": False})
    cfg["reid_model_path"] = ""
    cfg["reid_preprocess"] = "resize"

    c_cfg = native_module._build_c_config(cfg)

    assert c_cfg.cmc_method == b"none"


def test_native_occluboost_live_config_forwards_obb_operating_point():
    cfg = native_module._resolve_tracker_cfg(
        {
            "obb_det_thresh": 0.11,
            "obb_iou_threshold": 0.12,
            "obb_new_track_thresh": 0.13,
            "obb_instant_confirm_thresh": 0.14,
            "obb_max_age": 17,
            "obb_recovery_max_age": 9,
            "obb_second_iou_thresh": 0.16,
        }
    )
    cfg.update(reid_model_path="", reid_preprocess="resize")
    c_cfg = native_module._build_c_config(cfg)

    assert np.isclose(c_cfg.obb_det_thresh, 0.11)
    assert np.isclose(c_cfg.obb_iou_threshold, 0.12)
    assert np.isclose(c_cfg.obb_new_track_thresh, 0.13)
    assert np.isclose(c_cfg.obb_instant_confirm_thresh, 0.14)
    assert c_cfg.obb_max_age == 17
    assert c_cfg.obb_recovery_max_age == 9
    assert np.isclose(c_cfg.obb_second_iou_thresh, 0.16)


def test_native_occluboost_replay_honors_disabled_cmc(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        native_module,
        "ensure_occluboost_cpp_executable",
        lambda force_rebuild=False: Path("/tmp/occluboost_replay"),
    )
    monkeypatch.setattr(native_module, "_ensure_native_reid_model_path", lambda _weights: None)

    def capture_replay(**kwargs):
        captured.update(kwargs)
        return "MOT17-02-FRCNN", [], {"track_time_ms": 0.0, "num_frames": 0}

    monkeypatch.setattr(native_module._native_trackers, "run_replay_process", capture_replay)

    native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root=str(tmp_path),
        detector_name="yolox_x.pt",
        reid_name="osnet.pt",
        tracker_name="occluboost",
        exp_folder=str(tmp_path),
        target_fps=None,
        cfg_dict={"use_cmc": False, "with_reid": False, "asso_func": "giou"},
        conf_threshold=0.2,
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--cmc-method") + 1] == "none"
    assert cmd[cmd.index("--conf-threshold") + 1] == "0.2"
    assert cmd[cmd.index("--asso-func") + 1] == "giou"


def test_native_occluboost_cmc_masks_match_python_scaling_and_rounding():
    native_module.ensure_occluboost_cpp_library()
    build_dir = native_common.tracker_build_dir("occluboost")
    completed = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "occluboost_cmc_mask_probe",
            "--parallel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    executable_name = "occluboost_cmc_mask_probe.exe" if os.name == "nt" else "occluboost_cmc_mask_probe"
    executable_candidates = (build_dir / executable_name, build_dir / "Release" / executable_name)
    executable = next((path for path in executable_candidates if path.exists()), None)
    assert executable is not None

    image = np.zeros((48, 64), dtype=np.uint8)
    sof = SOF()
    sof._preprocess_scale = (0.13, 0.17)
    cases = {
        "empty": None,
        "aabb": np.array([[12.8, 7.9, 159.7, 111.9]], dtype=np.float32),
        "obb": np.array([[170.25, 95.75, 80.5, 30.25, 0.41]], dtype=np.float32),
    }
    for mode, detections in cases.items():
        expected = sof.generate_mask(image, detections, sof.scale)
        expected_bits = "".join("1" if value else "0" for value in expected.reshape(-1))
        probe = subprocess.run([str(executable), mode], capture_output=True, text=True, check=False)
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == expected_bits


def test_native_occluboost_obb_cmc_similarity_state_matches_python():
    native_module.ensure_occluboost_cpp_library()
    build_dir = native_common.tracker_build_dir("occluboost")
    completed = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "occluboost_cmc_state_probe",
            "--parallel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    executable_name = "occluboost_cmc_state_probe.exe" if os.name == "nt" else "occluboost_cmc_state_probe"
    executable_candidates = (build_dir / executable_name, build_dir / "Release" / executable_name)
    executable = next((path for path in executable_candidates if path.exists()), None)
    assert executable is not None
    probe = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert probe.returncode == 0, probe.stderr
    actual = np.fromstring(probe.stdout, sep=" ")
    assert actual.size == 110

    mean = np.array(
        [120.0, 75.0, 18.0, 50.0 / 18.0, 0.35, 2.0, -1.0, 0.8, -0.4, 0.05],
        dtype=np.float64,
    )
    lower = np.zeros((10, 10), dtype=np.float64)
    for row in range(10):
        lower[row, row] = 1.0 + (0.2 * row)
        for column in range(row):
            lower[row, column] = 0.003 * (row + 1) * (column + 1)
    covariance = lower @ lower.T
    scale = 1.08
    angle = 0.17
    linear = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    transform = np.column_stack((linear, [7.0, -4.0]))

    def measurement_to_box(values):
        return np.array(
            [values[0], values[1], values[2] * values[3], values[2], values[4]],
            dtype=np.float64,
        )

    def box_to_measurement(box):
        return np.array(
            [box[0], box[1], box[3], box[2] / box[3], box[4]],
            dtype=np.float64,
        )

    expected_mean, expected_covariance = transform_obb_kalman_state(
        mean,
        covariance,
        transform,
        measurement_to_box=measurement_to_box,
        box_to_measurement=box_to_measurement,
        velocity_measurement_indices=(0, 1, 2, 3, 4),
    )
    np.testing.assert_allclose(actual[:10], expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual[10:].reshape(10, 10),
        expected_covariance,
        rtol=1e-10,
        atol=1e-10,
    )


def test_native_occluboost_replay_uses_complete_explicit_embedding_cache(monkeypatch, tmp_path):
    sequence_name = "MOT17-02-FRCNN"
    detector_root = tmp_path / "dets_n_embs" / "mot17-mini" / "yolox_x"
    detections_path = detector_root / "dets" / f"{sequence_name}.npy"
    embedding_cache_dir = detector_root / "embs" / "cpp" / "selected-model" / "resize-cropv2"
    detections_path.parent.mkdir(parents=True)
    embedding_cache_dir.mkdir(parents=True)
    np.save(detections_path, np.zeros((3, 7), dtype=np.float32))
    np.save(embedding_cache_dir / f"{sequence_name}.npy", np.zeros((3, 32), dtype=np.float32))

    monkeypatch.setattr(
        native_module,
        "ensure_occluboost_cpp_executable",
        lambda force_rebuild=False: Path("/tmp/occluboost_replay"),
    )
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: (_ for _ in ()).throw(AssertionError("complete cache must not load or export ReID")),
    )
    captured = {}

    def capture_replay(**kwargs):
        captured.update(kwargs)
        return sequence_name, [], {"track_time_ms": 0.0, "num_frames": 0}

    monkeypatch.setattr(native_module._native_trackers, "run_replay_process", capture_replay)

    native_module.process_sequence_cpp(
        seq_name=sequence_name,
        mot_root="/data/train",
        project_root=str(tmp_path),
        detector_name="yolox_x.pt",
        reid_name="/weights/original-model.pt",
        tracker_name="occluboost",
        exp_folder=str(tmp_path / "results"),
        target_fps=None,
        dataset_name="mot17-mini",
        embedding_cache_dir=str(embedding_cache_dir),
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--reid-name") + 1] == "cpp/selected-model"
    assert cmd[cmd.index("--reid-preprocess") + 1] == "resize-cropv2"
    assert cmd[cmd.index("--reid-model") + 1] == ""


def test_native_occluboost_live_obb_damps_angular_velocity():
    library = native_module._OccluBoostLiveLibrary(native_module.ensure_occluboost_cpp_library())
    tracker = native_module.NativeOccluBoostTracker(
        {
            "with_reid": False,
            "use_cmc": False,
            "det_thresh": 0.1,
            "iou_threshold": 0.01,
            "new_track_thresh": 0.1,
            "instant_confirm_thresh": 0.1,
            "confirm_hits": 1,
            "min_hits": 0,
            "aspect_ratio_thresh": 10.0,
            "min_box_area": 1,
        },
        library=library,
    )

    def detection(angle: float):
        return native_module.np.array(
            [[80, 60, 80, 20, angle, 0.95, 0]],
            dtype=native_module.np.float32,
        )

    try:
        first = tracker.update(detection(0.2))
        second = tracker.update(detection(0.7))
        third = tracker.update(detection(0.7))
    finally:
        tracker.close()

    assert first.shape == second.shape == third.shape == (1, 9)
    assert first[0, 5] == second[0, 5] == third[0, 5]
    # Momentum remains visible, but the post-update damping bounds the next
    # prediction's overshoot. With an undamped angular velocity this exceeds
    # the bound for the same deterministic filter sequence.
    overshoot = float(third[0, 4] - 0.7)
    assert 0.0 < overshoot < (0.0009 * (0.7 - 0.2))


def test_native_occluboost_obb_uses_dedicated_threshold_and_dlo_boost():
    library = native_module._OccluBoostLiveLibrary(native_module.ensure_occluboost_cpp_library())
    tracker = native_module.NativeOccluBoostTracker(
        {
            "with_reid": False,
            "use_cmc": False,
            "det_thresh": 0.99,
            "iou_threshold": 0.99,
            "obb_det_thresh": 0.5,
            "obb_iou_threshold": 0.1,
            "obb_new_track_thresh": 0.5,
            "obb_instant_confirm_thresh": 0.5,
            "use_dlo_boost": True,
            "use_duo_boost": False,
            "dlo_boost_coef": 1.0,
            "use_sb": False,
            "use_vt": False,
            "use_second_pass": False,
            "confirm_hits": 1,
            "min_hits": 0,
            "aspect_ratio_thresh": 10.0,
        },
        library=library,
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    first_detection = np.array([[80, 60, 80, 10, 0.0, 0.95, 0]], dtype=np.float32)
    low_same_obb = first_detection.copy()
    low_same_obb[0, 5] = 0.1
    try:
        first = tracker.update(first_detection, image)
        boosted = tracker.update(low_same_obb, image)
    finally:
        tracker.close()

    assert first.shape == boosted.shape == (1, 9)
    assert first[0, 5] == boosted[0, 5]
    assert boosted[0, 6] >= 0.5


@pytest.mark.parametrize(
    "detection",
    (
        np.array([[40, 40, 2, 2, 0.2, 0.95, 0]], dtype=np.float32),
        np.array([[60, 40, 40, 4, 0.0, 0.95, 0]], dtype=np.float32),
    ),
)
def test_native_occluboost_obb_applies_python_geometry_filter(detection):
    library = native_module._OccluBoostLiveLibrary(native_module.ensure_occluboost_cpp_library())
    tracker = native_module.NativeOccluBoostTracker(
        {
            "with_reid": False,
            "use_cmc": False,
            "use_dlo_boost": False,
            "use_duo_boost": False,
            "use_second_pass": False,
            "obb_det_thresh": 0.1,
            "obb_new_track_thresh": 0.1,
            "obb_instant_confirm_thresh": 0.1,
            "confirm_hits": 1,
            "min_hits": 0,
            "min_box_area": 10,
            "aspect_ratio_thresh": 5.0,
        },
        library=library,
    )
    image = np.zeros((96, 128, 3), dtype=np.uint8)

    try:
        output = tracker.update(detection, image)
    finally:
        tracker.close()

    assert output.shape == (0, 9)


def test_native_occluboost_obb_matches_python_lifecycle_and_recovery():
    cfg = native_module._resolve_tracker_cfg(
        {
            "use_cmc": False,
            "min_box_area": 1,
            "aspect_ratio_thresh": 20.0,
            "second_pass_min_hits": 1,
        }
    )
    python_tracker = OccluBoost(reid_model=None, **cfg)
    library = native_module._OccluBoostLiveLibrary(native_module.ensure_occluboost_cpp_library())
    native_tracker = native_module.NativeOccluBoostTracker(cfg, library=library)
    image = np.zeros((180, 240, 3), dtype=np.uint8)

    frames = []
    for frame_id in range(8):
        detections = []
        embeddings = []
        if frame_id != 5:
            detections.append(
                [
                    45 + 8 * frame_id,
                    70 + 2 * frame_id,
                    42,
                    16,
                    0.15 + 0.05 * frame_id,
                    0.1 if frame_id == 4 else 0.92,
                    0,
                ]
            )
            embeddings.append([1.0, 0.0, 0.0])
        if frame_id >= 2 and frame_id != 7:
            detections.append(
                [
                    190 - 7 * frame_id,
                    110 - frame_id,
                    28,
                    12,
                    -1.45 + 0.04 * frame_id,
                    0.18 if frame_id == 6 else 0.88,
                    1,
                ]
            )
            embeddings.append([0.0, 1.0, 0.0])
        frames.append(
            (
                np.asarray(detections, dtype=np.float32).reshape(-1, 7),
                np.asarray(embeddings, dtype=np.float32).reshape(-1, 3),
            )
        )

    try:
        for detections, embeddings in frames:
            python_output = np.asarray(python_tracker.update(detections, image, embeddings))
            native_output = native_tracker.update(detections, image, embeddings)

            assert native_output.shape == python_output.shape
            np.testing.assert_allclose(native_output[:, :5], python_output[:, :5], atol=1e-4)
            # Native IDs are one-based while Python IDs are zero-based; all
            # public geometry and detection metadata must otherwise agree.
            np.testing.assert_allclose(native_output[:, 6:], python_output[:, 6:], atol=1e-6)
    finally:
        native_tracker.close()
