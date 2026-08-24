from __future__ import annotations

from pathlib import Path

import numpy as np

from boxmot.native.trackers import occluboost as native_module


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
        cfg_dict={"use_cmc": False, "with_reid": False},
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--cmc-method") + 1] == "none"


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
    image = native_module.np.zeros((120, 160, 3), dtype=native_module.np.uint8)

    def detection(angle: float):
        return native_module.np.array(
            [[80, 60, 80, 20, angle, 0.95, 0]],
            dtype=native_module.np.float32,
        )

    try:
        first = tracker.update(detection(0.2), image)
        second = tracker.update(detection(0.7), image)
        third = tracker.update(detection(0.7), image)
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
        },
        library=library,
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    # Wide OBBs must not be dropped by the pedestrian-tuned AABB aspect filter.
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
