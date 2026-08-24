from __future__ import annotations

import importlib
import queue
import subprocess
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot.native import _common as native_common
from boxmot.native.trackers import botsort as native_module


def test_cached_embedding_path_uses_versioned_preprocess_directory():
    path = native_common.cached_embedding_path(
        "/runs",
        "yolox_x.pt",
        "/weights/lmbn_n_duke.pt",
        "MOT17-02-FRCNN",
        dataset_name="mot17-mini",
        preprocess_name="resize",
        tracker_backend="cpp",
    )

    assert path == Path(
        "/runs/dets_n_embs/mot17-mini/yolox_x/embs/"
        "cpp/lmbn_n_duke-pt-ort/resize-cropv2/MOT17-02-FRCNN.npy"
    )


def test_process_sequence_cpp_builds_native_command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay")
    )
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: Path("/weights/lmbn_n_duke.onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/botsort_replay"
            assert "--sequence" in cmd
            assert "MOT17-02-FRCNN" in cmd
            assert cmd[cmd.index("--reid-name") + 1] == "cpp/lmbn_n_duke-pt-ort"
            assert cmd[cmd.index("--reid-preprocess") + 1] == "resize-cropv2"
            assert "--track-high-thresh" in cmd
            assert "--cmc-method" in cmd
            assert cmd[cmd.index("--second-match-thresh") + 1] == "0.31"
            assert cmd[cmd.index("--unconfirmed-match-thresh") + 1] == "0.42"
            assert cmd[cmd.index("--unconfirmed-emb-scale") + 1] == "2.75"
            assert "--reid-model" in cmd
            assert cmd[cmd.index("--reid-model") + 1] == "/weights/lmbn_n_duke.onnx"
            assert stdout is native_module.subprocess.PIPE
            assert stderr is native_module.subprocess.PIPE
            assert text is True
            assert bufsize == 1
            self.stdout = StringIO(
                '{"sequence":"MOT17-02-FRCNN","num_frames":2,"track_time_ms":12.5,"kept_frame_ids":[1,2]}\n'
            )
            self.stderr = StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(native_module.subprocess, "Popen", FakePopen)

    seq_name, kept_ids, timing = native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root="/runs",
        detector_name="yolox_x.pt",
        reid_name="/weights/lmbn_n_duke.pt",
        tracker_name="botsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        cfg_dict={
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "second_match_thresh": 0.31,
            "unconfirmed_match_thresh": 0.42,
            "unconfirmed_emb_scale": 2.75,
            "cmc_method": "ecc",
        },
        dataset_name="mot17-mini",
        conf_threshold=0.25,
        preprocess_name="resize",
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}


def test_process_sequence_cpp_passes_onnx_reid_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[cmd.index("--reid-model") + 1] == "/weights/lmbn_n_duke.onnx"
            self.stdout = StringIO(
                '{"sequence":"MOT17-02-FRCNN","num_frames":1,"track_time_ms":1.0,"kept_frame_ids":[1]}\n'
            )
            self.stderr = StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(native_module.subprocess, "Popen", FakePopen)

    seq_name, kept_ids, timing = native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root="/runs",
        detector_name="yolox_x.pt",
        reid_name="/weights/lmbn_n_duke.onnx",
        tracker_name="botsort",
        exp_folder=str(tmp_path),
        target_fps=None,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1]
    assert timing == {"track_time_ms": 1.0, "num_frames": 1}


def test_process_sequence_cpp_keeps_original_reid_cache_key_when_native_model_resolves_to_onnx(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay")
    )
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: Path("/weights/lmbn_n_duke.onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[cmd.index("--reid-name") + 1] == "cpp/lmbn_n_duke-pt-ort"
            assert cmd[cmd.index("--reid-preprocess") + 1] == "resize-cropv2"
            assert cmd[cmd.index("--reid-model") + 1] == "/weights/lmbn_n_duke.onnx"
            self.stdout = StringIO(
                '{"sequence":"MOT17-02-FRCNN","num_frames":1,"track_time_ms":1.0,"kept_frame_ids":[1]}\n'
            )
            self.stderr = StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(native_module.subprocess, "Popen", FakePopen)

    native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root="/runs",
        detector_name="yolox_x.pt",
        reid_name="/weights/lmbn_n_duke.pt",
        tracker_name="botsort",
        exp_folder=str(tmp_path),
        target_fps=None,
    )


def test_process_sequence_cpp_auto_exports_pt_reid_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay")
    )
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda weights: Path(str(weights)).with_suffix(".onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[cmd.index("--reid-name") + 1] == "cpp/lmbn_n_duke-pt-ort"
            assert cmd[cmd.index("--reid-preprocess") + 1] == "resize-cropv2"
            assert cmd[cmd.index("--reid-model") + 1] == "/weights/lmbn_n_duke.onnx"
            self.stdout = StringIO(
                '{"sequence":"MOT17-02-FRCNN","num_frames":1,"track_time_ms":1.0,"kept_frame_ids":[1]}\n'
            )
            self.stderr = StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(native_module.subprocess, "Popen", FakePopen)

    native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root="/runs",
        detector_name="yolox_x.pt",
        reid_name="/weights/lmbn_n_duke.pt",
        tracker_name="botsort",
        exp_folder=str(tmp_path),
        target_fps=None,
    )


def test_process_sequence_cpp_rejects_other_trackers():
    try:
        native_module.process_sequence_cpp(
            seq_name="MOT17-02-FRCNN",
            mot_root="/data/train",
            project_root="/runs",
            detector_name="yolox_x.pt",
            reid_name="/weights/lmbn_n_duke.pt",
            tracker_name="bytetrack",
            exp_folder="/tmp",
            target_fps=None,
        )
    except ValueError as exc:
        assert "tracker='botsort' only" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-BoTSORT tracker")


def test_process_sequence_cpp_uses_complete_explicit_embedding_cache(monkeypatch, tmp_path):
    sequence_name = "MOT17-02-FRCNN"
    detector_root = tmp_path / "dets_n_embs" / "mot17-mini" / "yolox_x"
    detections_path = detector_root / "dets" / f"{sequence_name}.npy"
    embedding_cache_dir = detector_root / "embs" / "cpp" / "selected-model" / "resize-cropv2"
    detections_path.parent.mkdir(parents=True)
    embedding_cache_dir.mkdir(parents=True)
    np.save(detections_path, np.zeros((2, 7), dtype=np.float32))
    np.save(embedding_cache_dir / f"{sequence_name}.npy", np.zeros((2, 32), dtype=np.float32))

    monkeypatch.setattr(
        native_module,
        "ensure_botsort_cpp_executable",
        lambda force_rebuild=False: Path("/tmp/botsort_replay"),
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
        tracker_name="botsort",
        exp_folder=str(tmp_path / "results"),
        target_fps=None,
        dataset_name="mot17-mini",
        embedding_cache_dir=str(embedding_cache_dir),
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--reid-name") + 1] == "cpp/selected-model"
    assert cmd[cmd.index("--reid-preprocess") + 1] == "resize-cropv2"
    assert cmd[cmd.index("--reid-model") + 1] == ""


def test_native_botsort_tracker_uses_live_library_wrapper():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            calls.append(("create", cfg["frame_rate"], cfg["with_reid"]))
            return "handle"

        def reset(self, handle):
            calls.append(("reset", handle))

        def update(self, handle, dets, img, embs):
            calls.append(("update", handle, dets.shape, img.shape, embs.shape))
            return dets

        def get_last_reid_time_ms(self, handle):
            calls.append(("get_last_reid_time_ms", handle))
            return 4.5

        def destroy(self, handle):
            calls.append(("destroy", handle))

    tracker = native_module.NativeBotSortTracker({"frame_rate": 15, "with_reid": True}, library=_FakeLibrary())

    dets = native_module.np.array(
        [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
        dtype=native_module.np.float32,
    )
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)
    embs = native_module.np.ones((2, 4), dtype=native_module.np.float32)

    out = tracker.update(dets, img, embs)
    assert tracker.get_last_reid_time_ms() == 4.5
    tracker.reset()
    assert tracker.get_last_reid_time_ms() == 0.0
    tracker.close()

    assert out.shape == (2, 6)
    assert calls == [
        ("create", 15, True),
        ("update", "handle", (2, 6), (8, 8, 3), (2, 4)),
        ("get_last_reid_time_ms", "handle"),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_botsort_tracker_accepts_obb_rows_and_preserves_empty_mode():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            calls.append(("reset", handle))

        def update(self, handle, dets, img, embs):
            calls.append(("update", handle, dets.shape, img.shape, None if embs is None else embs.shape))
            return dets

        def get_last_reid_time_ms(self, handle):
            return 0.0

        def destroy(self, handle):
            calls.append(("destroy", handle))

    tracker = native_module.NativeBotSortTracker({"with_reid": False}, library=_FakeLibrary())

    dets = native_module.np.ones((2, 7), dtype=native_module.np.float32)
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)

    out = tracker.update(dets, img)
    empty = tracker.update(native_module.np.empty((0, 0), dtype=native_module.np.float32), img)
    tracker.close()

    assert tracker.supports_obb is True
    assert out.shape == (2, 7)
    assert empty.shape == (0, 7)
    assert calls == [
        ("update", "handle", (2, 7), (8, 8, 3), None),
        ("update", "handle", (0, 7), (8, 8, 3), None),
        ("destroy", "handle"),
    ]


def test_native_botsort_empty_obb_frame_latches_layout_before_first_detection():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            calls.append(dets.shape)
            return dets

        def get_last_reid_time_ms(self, handle):
            return 0.0

        def destroy(self, handle):
            return None

    tracker = native_module.NativeBotSortTracker({"with_reid": False}, library=_FakeLibrary())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        first = tracker.update(np.empty((0, 7), dtype=np.float32), image)
        second = tracker.update(np.empty((0, 0), dtype=np.float32), image)
        with pytest.raises(ValueError, match="cannot switch between AABB and OBB inputs"):
            tracker.update(np.array([[0, 0, 2, 2, 0.9, 0]], dtype=np.float32), image)
    finally:
        tracker.close()

    assert first.shape == second.shape == (0, 7)
    assert calls == [(0, 7), (0, 7)]


def test_native_botsort_sof_accepts_bgra_live_images():
    library = native_module._BotSortLiveLibrary(native_module.ensure_botsort_cpp_library())
    tracker = native_module.NativeBotSortTracker(
        {"with_reid": False, "use_cmc": True, "cmc_method": "sof"},
        library=library,
    )
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[8:24, 8:24, :3] = 255
    try:
        first = tracker.update(np.empty((0, 6), dtype=np.float32), image)
        second = tracker.update(np.empty((0, 6), dtype=np.float32), image)
    finally:
        tracker.close()

    assert first.shape == second.shape == (0, 8)


def test_native_botsort_target_fps_keeps_embedding_rows_aligned(tmp_path):
    executable = native_module.ensure_botsort_cpp_executable()
    sequence_name = "FPS-ALIGN"
    mot_root = tmp_path / "mot"
    image_dir = mot_root / sequence_name / "img1"
    image_dir.mkdir(parents=True)
    (mot_root / sequence_name / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=30\n",
        encoding="utf-8",
    )
    for frame_id in range(1, 6):
        assert cv2.imwrite(
            str(image_dir / f"{frame_id:06d}.jpg"),
            np.zeros((48, 64, 3), dtype=np.uint8),
        )

    cache_root = tmp_path / "cache"
    det_dir = cache_root / "detector" / "dets"
    emb_dir = cache_root / "detector" / "embs" / "reid" / "resize"
    det_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    detections = []
    for frame_id in range(1, 6):
        x1 = 10.0 if frame_id == 1 else 18.0
        detections.append([frame_id, x1, 10.0, x1 + 20.0, 30.0, 0.95, 0.0])
    np.save(det_dir / f"{sequence_name}.npy", np.asarray(detections, dtype=np.float32))
    # Embedding caches contain descriptors only, not frame IDs. Keeping the
    # first descriptor component at zero catches the old code that treated it
    # as a frame-id column and silently discarded every row.
    np.save(
        emb_dir / f"{sequence_name}.npy",
        np.repeat(np.array([[0.0, 1.0]], dtype=np.float32), 5, axis=0),
    )

    output = tmp_path / "tracks.txt"
    completed = subprocess.run(
        [
            str(executable),
            "--mot-root",
            str(mot_root),
            "--det-emb-root",
            str(cache_root),
            "--detector-name",
            "detector",
            "--reid-name",
            "reid",
            "--reid-preprocess",
            "resize",
            "--reid-model",
            "",
            "--sequence",
            sequence_name,
            "--output",
            str(output),
            "--target-fps",
            "15",
            "--cmc-method",
            "none",
            "--with-reid",
            "1",
            "--track-high-thresh",
            "0.5",
            "--track-low-thresh",
            "0.1",
            "--new-track-thresh",
            "0.5",
            "--match-thresh",
            "0.1",
            "--proximity-thresh",
            "0.9",
            "--appearance-thresh",
            "0.25",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = np.loadtxt(output, delimiter=",", ndmin=2)
    np.testing.assert_array_equal(rows[:, 0].astype(int), [1, 3, 5])
    np.testing.assert_array_equal(rows[:, 1].astype(int), [1, 1, 1])


def test_native_botsort_rejects_empty_embedding_cache_without_runtime_reid(tmp_path):
    executable = native_module.ensure_botsort_cpp_executable()
    sequence_name = "EMPTY-EMB"
    mot_root = tmp_path / "mot"
    image_dir = mot_root / sequence_name / "img1"
    image_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / "000001.jpg"), np.zeros((16, 16, 3), dtype=np.uint8))

    cache_root = tmp_path / "cache"
    det_dir = cache_root / "detector" / "dets"
    emb_dir = cache_root / "detector" / "embs" / "reid" / "resize"
    det_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    np.save(
        det_dir / f"{sequence_name}.npy",
        np.array([[1, 1, 1, 10, 12, 0.9, 0]], dtype=np.float32),
    )
    np.save(emb_dir / f"{sequence_name}.npy", np.empty((0, 2), dtype=np.float32))

    completed = subprocess.run(
        [
            str(executable),
            "--mot-root",
            str(mot_root),
            "--det-emb-root",
            str(cache_root),
            "--detector-name",
            "detector",
            "--reid-name",
            "reid",
            "--reid-preprocess",
            "resize",
            "--reid-model",
            "",
            "--sequence",
            sequence_name,
            "--output",
            str(tmp_path / "tracks.txt"),
            "--with-reid",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Detection and embedding row counts do not match" in completed.stderr


def test_native_botsort_tracker_rejects_mode_switch_after_initialization():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def destroy(self, handle):
            return None

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            return dets

    tracker = native_module.NativeBotSortTracker({"with_reid": False}, library=_FakeLibrary())
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)

    tracker.update(native_module.np.ones((1, 7), dtype=native_module.np.float32), img)

    try:
        tracker.update(native_module.np.ones((1, 6), dtype=native_module.np.float32), img)
    except ValueError as exc:
        assert "cannot switch between AABB and OBB inputs" in str(exc)
    else:
        raise AssertionError("Expected ValueError when switching native BoTSORT detection layout")
    finally:
        tracker.close()


def test_native_botsort_tracker_marks_native_onnx_reid_provider():
    class _FakeLibrary:
        def create(self, cfg):
            return cfg

        def destroy(self, handle):
            return None

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            return dets

    expected_path = Path("models/lmbn_n_duke.onnx")

    original_resolver = native_module._ensure_native_reid_model_path
    native_module._ensure_native_reid_model_path = lambda _weights: expected_path
    try:
        tracker = native_module.NativeBotSortTracker(
            {"with_reid": True},
            reid_weights="models/lmbn_n_duke.onnx",
            library=_FakeLibrary(),
        )

        assert tracker.provides_reid is True
        assert tracker.cfg["reid_model_path"] == str(expected_path)
        assert tracker.cfg["reid_preprocess"] == "resize"
        tracker.close()
    finally:
        native_module._ensure_native_reid_model_path = original_resolver


def test_native_botsort_tracker_auto_exports_pt_reid_provider(monkeypatch):
    class _FakeLibrary:
        def create(self, cfg):
            return cfg

        def destroy(self, handle):
            return None

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            return dets

    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda weights: Path("models/exported.onnx") if str(weights).endswith(".pt") else Path(weights),
    )

    tracker = native_module.NativeBotSortTracker(
        {"with_reid": True},
        reid_weights="models/lmbn_n_duke.pt",
        library=_FakeLibrary(),
    )

    assert tracker.provides_reid is True
    assert tracker.cfg["reid_model_path"] == "models/exported.onnx"
    tracker.close()


def test_native_botsort_ci_macos_disables_pt_reid_without_cached_onnx(monkeypatch, tmp_path):
    class _FakeLibrary:
        def create(self, cfg):
            return cfg

        def destroy(self, handle):
            return None

        def reset(self, handle):
            return None

        def update(self, handle, dets, img, embs):
            return dets

    pt_weights = tmp_path / "osnet_x0_25_msmt17.pt"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(native_module.sys, "platform", "darwin")
    monkeypatch.setattr(native_module, "_resolve_reid_model_ref", lambda _weights: pt_weights)
    monkeypatch.setattr(native_module, "_native_onnx_cache_path", lambda _weights: tmp_path / "missing.onnx")
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: (_ for _ in ()).throw(AssertionError("should not export PT on macOS CI")),
    )

    tracker = native_module.NativeBotSortTracker(
        {"with_reid": True},
        reid_weights="models/osnet_x0_25_msmt17.pt",
        library=_FakeLibrary(),
    )

    assert tracker.with_reid is False
    assert tracker.provides_reid is False
    assert tracker.cfg["reid_model_path"] == ""
    tracker.close()


def test_ensure_native_reid_model_path_exports_pt_when_onnx_is_missing(monkeypatch, tmp_path):
    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    weights.touch()
    calls = []

    def fake_export(path):
        calls.append(path)
        onnx_path = native_module._native_onnx_cache_path(path)
        onnx_path.touch()
        return onnx_path

    monkeypatch.setattr(native_module, "_export_reid_to_onnx", fake_export)

    resolved = native_module._ensure_native_reid_model_path(weights)

    assert resolved == native_module._native_onnx_cache_path(weights)
    assert calls == [weights]


def test_ensure_native_reid_model_path_reuses_fresh_onnx(monkeypatch, tmp_path):
    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    weights.touch()
    onnx_path = native_module._native_onnx_cache_path(weights)
    onnx_path.touch()
    onnx_path.touch()

    monkeypatch.setattr(
        native_module, "_export_reid_to_onnx", lambda _path: (_ for _ in ()).throw(AssertionError("should not export"))
    )

    resolved = native_module._ensure_native_reid_model_path(weights)

    assert resolved == onnx_path


def test_resolve_reid_model_ref_prefers_native_onnx_cache_for_bare_name(monkeypatch, tmp_path):
    pt_path = tmp_path / "osnet_x0_25_msmt17.pt"
    native_onnx = tmp_path / "osnet_x0_25_msmt17.onnx"
    pt_path.touch()
    native_onnx.touch()

    monkeypatch.setattr(
        native_common,
        "resolve_model_path",
        lambda path: tmp_path / Path(path).name,
    )

    resolved = native_module._resolve_reid_model_ref("osnet_x0_25_msmt17")

    assert resolved == native_onnx


def test_resolve_reid_model_ref_returns_explicit_onnx_as_is(monkeypatch, tmp_path):
    explicit_onnx = tmp_path / "lmbn_n_duke.onnx"
    explicit_onnx.touch()

    monkeypatch.setattr(
        native_common,
        "resolve_model_path",
        lambda path: tmp_path / Path(path).name,
    )

    resolved = native_module._resolve_reid_model_ref(explicit_onnx)

    assert resolved == explicit_onnx


def test_export_reid_to_onnx_uses_native_compatible_export_settings(monkeypatch, tmp_path):
    export_module = importlib.import_module("boxmot.engine.reid.export")
    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    weights.touch()

    class _FakeModel(torch.nn.Module):
        def forward(self, images):
            return images.mean(dim=(2, 3))

    model = _FakeModel()
    dummy_input = torch.randn(1, 3, 256, 128)
    monkeypatch.setattr(export_module, "setup_model", lambda args: (model, dummy_input))

    captured = {}

    def fake_export(model_arg, args_arg, path_arg, **kwargs):
        captured["model"] = model_arg
        captured["args"] = args_arg
        captured["path"] = Path(path_arg)
        captured["kwargs"] = kwargs
        Path(path_arg).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    exported = native_module._export_reid_to_onnx(weights)

    assert exported == tmp_path / "osnet_x0_25_msmt17.onnx"
    assert captured["model"] is model
    assert captured["args"] == (dummy_input,)
    assert captured["path"] == exported
    assert captured["kwargs"]["opset_version"] == 17
    assert captured["kwargs"]["dynamic_axes"] == {
        "images": {0: "batch"},
        "output0": {0: "batch"},
    }


def test_process_sequence_cpp_streams_progress_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay")
    )
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: Path("/weights/lmbn_n_duke.onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/botsort_replay"
            assert stdout is native_module.subprocess.PIPE
            assert stderr is native_module.subprocess.PIPE
            assert text is True
            assert bufsize == 1
            self.stdout = StringIO(
                '{"sequence":"MOT17-02-FRCNN","num_frames":2,"track_time_ms":12.5,"kept_frame_ids":[1,2]}\n'
            )
            self.stderr = StringIO("BOXMOT_PROGRESS\tMOT17-02-FRCNN\t1\t2\nBOXMOT_PROGRESS\tMOT17-02-FRCNN\t2\t2\n")

        def wait(self):
            return 0

    monkeypatch.setattr(native_module.subprocess, "Popen", FakePopen)

    progress_queue = queue.Queue()
    seq_name, kept_ids, timing = native_module.process_sequence_cpp(
        seq_name="MOT17-02-FRCNN",
        mot_root="/data/train",
        project_root="/runs",
        detector_name="yolox_x.pt",
        reid_name="/weights/lmbn_n_duke.pt",
        tracker_name="botsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        progress_queue=progress_queue,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 1, 2)
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 2, 2)
