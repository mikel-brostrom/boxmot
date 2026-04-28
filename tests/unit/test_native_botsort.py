from __future__ import annotations

import importlib
from io import StringIO
from pathlib import Path
import queue

import torch

from boxmot.native import botsort_cpp as native_module


def test_process_sequence_cpp_builds_native_command(monkeypatch, tmp_path):
    monkeypatch.setattr(native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay"))
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
            assert cmd[cmd.index("--reid-name") + 1] == "lmbn_n_duke"
            assert "--track-high-thresh" in cmd
            assert "--cmc-method" in cmd
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
    monkeypatch.setattr(native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay"))

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


def test_process_sequence_cpp_keeps_original_reid_cache_key_when_native_model_resolves_to_opencv_onnx(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay"))
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda _weights: Path("/weights/lmbn_n_duke_opencv.onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[cmd.index("--reid-name") + 1] == "lmbn_n_duke"
            assert cmd[cmd.index("--reid-model") + 1] == "/weights/lmbn_n_duke_opencv.onnx"
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
    monkeypatch.setattr(native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay"))
    monkeypatch.setattr(
        native_module,
        "_ensure_native_reid_model_path",
        lambda weights: Path(str(weights)).with_suffix(".onnx"),
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[cmd.index("--reid-name") + 1] == "lmbn_n_duke"
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

    dets = native_module.np.ones((2, 6), dtype=native_module.np.float32)
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
        assert tracker.cfg["reid_preprocess"] == "resize_pad"
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

    monkeypatch.setattr(native_module, "_export_reid_to_onnx", lambda _path: (_ for _ in ()).throw(AssertionError("should not export")))

    resolved = native_module._ensure_native_reid_model_path(weights)

    assert resolved == onnx_path


def test_resolve_reid_model_ref_prefers_native_opencv_cache_for_bare_name(monkeypatch, tmp_path):
    pt_path = tmp_path / "osnet_x0_25_msmt17.pt"
    generic_onnx = tmp_path / "osnet_x0_25_msmt17.onnx"
    native_onnx = tmp_path / "osnet_x0_25_msmt17_opencv.onnx"
    pt_path.touch()
    generic_onnx.touch()
    native_onnx.touch()

    monkeypatch.setattr(
        native_module,
        "resolve_model_path",
        lambda path: tmp_path / Path(path).name,
    )

    resolved = native_module._resolve_reid_model_ref("osnet_x0_25_msmt17")

    assert resolved == native_onnx


def test_resolve_reid_model_ref_prefers_native_opencv_cache_for_generic_onnx(monkeypatch, tmp_path):
    generic_onnx = tmp_path / "osnet_x0_25_msmt17.onnx"
    native_onnx = tmp_path / "osnet_x0_25_msmt17_opencv.onnx"
    generic_onnx.touch()
    native_onnx.touch()

    monkeypatch.setattr(
        native_module,
        "resolve_model_path",
        lambda path: tmp_path / Path(path).name,
    )

    resolved = native_module._resolve_reid_model_ref(generic_onnx)

    assert resolved == native_onnx


def test_export_reid_to_onnx_uses_native_compatible_export_settings(monkeypatch, tmp_path):
    export_module = importlib.import_module("boxmot.engine.export")
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

    assert exported == tmp_path / "osnet_x0_25_msmt17_opencv.onnx"
    assert captured["model"] is model
    assert captured["args"] == (dummy_input,)
    assert captured["path"] == exported
    assert captured["kwargs"]["opset_version"] == 17
    assert captured["kwargs"]["dynamic_axes"] == {
        "images": {0: "batch"},
        "output0": {0: "batch"},
    }


def test_process_sequence_cpp_streams_progress_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(native_module, "ensure_botsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/botsort_replay"))
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
            self.stderr = StringIO(
                "BOXMOT_PROGRESS\tMOT17-02-FRCNN\t1\t2\n"
                "BOXMOT_PROGRESS\tMOT17-02-FRCNN\t2\t2\n"
            )

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
