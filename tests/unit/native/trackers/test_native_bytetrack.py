from __future__ import annotations

import queue
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from boxmot.native.trackers import bytetrack as native_module


def _empty_tracks_for(dets):
    columns = 9 if dets.shape[1] == 7 else 8
    return np.empty((0, columns), dtype=np.float32)


def test_native_bytetrack_tracker_advertises_obb_support():
    assert native_module.NativeByteTrackTracker.supports_obb is True
    assert native_module.NativeByteTrackTracker.supports_masks is False
    assert native_module.NativeByteTrackTracker.uses_img is False
    assert native_module.NativeByteTrackTracker.uses_embs is False


def test_process_sequence_cpp_builds_native_command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_bytetrack_cpp_executable", lambda force_rebuild=False: Path("/tmp/bytetrack_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/bytetrack_replay"
            assert "--sequence" in cmd
            assert "MOT17-02-FRCNN" in cmd
            assert "--min-conf" in cmd
            assert "--track-thresh" in cmd
            assert "--match-thresh" in cmd
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
        reid_name="/weights/unused.pt",
        tracker_name="bytetrack",
        exp_folder=str(tmp_path),
        target_fps=None,
        cfg_dict={
            "min_conf": 0.1,
            "track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "frame_rate": 30,
        },
        dataset_name="mot17-mini",
        conf_threshold=0.25,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}


def test_process_sequence_cpp_rejects_other_trackers():
    try:
        native_module.process_sequence_cpp(
            seq_name="MOT17-02-FRCNN",
            mot_root="/data/train",
            project_root="/runs",
            detector_name="yolox_x.pt",
            reid_name="/weights/unused.pt",
            tracker_name="botsort",
            exp_folder="/tmp",
            target_fps=None,
        )
    except ValueError as exc:
        assert "tracker='bytetrack' only" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-ByteTrack tracker")


def test_native_bytetrack_tracker_uses_live_library_wrapper():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            calls.append(("create", cfg["frame_rate"], cfg["track_thresh"]))
            return "handle"

        def reset(self, handle):
            calls.append(("reset", handle))

        def update(self, handle, dets, img):
            calls.append(("update", handle, dets.shape, img))
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            calls.append(("destroy", handle))

    tracker = native_module.NativeByteTrackTracker({"frame_rate": 15, "track_thresh": 0.5}, library=_FakeLibrary())

    dets = native_module.np.array(
        [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
        dtype=native_module.np.float32,
    )
    out = tracker.update(dets)
    tracker.reset()
    tracker.close()

    assert out.shape == (0, 8)
    assert out.is_obb is False
    assert calls == [
        ("create", 15, 0.5),
        ("update", "handle", (2, 6), None),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_bytetrack_tracker_accepts_obb_rows():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            calls.append((handle, dets.shape, img))
            return native_module.np.ones((1, 9), dtype=native_module.np.float32)

        def destroy(self, handle):
            return None

    tracker = native_module.NativeByteTrackTracker(library=_FakeLibrary())
    dets = native_module.np.ones((1, 7), dtype=native_module.np.float32)
    out = tracker.update(dets)

    assert out.shape == (1, 9)
    assert out.is_obb is True
    assert calls == [("handle", (1, 7), None)]
    tracker.close()


def test_native_bytetrack_tracker_rejects_mode_switch():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            return None

    tracker = native_module.NativeByteTrackTracker(library=_FakeLibrary())
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)

    tracker.update(native_module.np.array([[1, 1, 4, 5, 0.9, 0]], dtype=native_module.np.float32), img)

    try:
        tracker.update(native_module.np.ones((1, 7), dtype=native_module.np.float32), img)
    except ValueError as exc:
        assert "cannot switch between AABB and OBB inputs" in str(exc)
    else:
        raise AssertionError("Expected ValueError when switching native ByteTrack detection mode")
    finally:
        tracker.close()


def test_native_bytetrack_tracker_rejects_noncanonical_empty_detection_width():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            raise AssertionError("Malformed detections must not reach the native library")

        def destroy(self, handle):
            return None

    tracker = native_module.NativeByteTrackTracker(library=_FakeLibrary())
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    try:
        with pytest.raises(ValueError, match="empty AABB detections with 6 columns"):
            tracker.update(np.empty((0, 5), dtype=np.float32), image)
    finally:
        tracker.close()


def test_native_bytetrack_enforces_configured_class_catalog():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            return np.empty((0, 9), dtype=np.float32)

        def destroy(self, handle):
            return None

    tracker = native_module.NativeByteTrackTracker(library=_FakeLibrary())
    tracker.configure_class_catalog(class_ids=(2,), class_names={2: "car"})
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    allowed = np.array([[4, 4, 3, 2, 0.2, 0.9, 2]], dtype=np.float32)
    tracker.update(allowed, image)

    unknown = allowed.copy()
    unknown[0, 6] = 3
    with pytest.raises(ValueError, match="not present in the tracker class catalog"):
        tracker.update(unknown, image)

    fractional = allowed.copy()
    fractional[0, 6] = 2.5
    with pytest.raises(ValueError, match="class IDs must be integers"):
        tracker.update(fractional, image)

    tracker.close()


def test_native_bytetrack_c_api_rejects_negative_class_ids():
    library = native_module._ByteTrackLiveLibrary(native_module.ensure_bytetrack_cpp_library())
    tracker = native_module.NativeByteTrackTracker(library=library)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    negative_class = np.array([[1, 1, 10, 10, 0.9, -1]], dtype=np.float32)
    try:
        # Bypass NativeTrackerMixin so the C++ ABI validation itself is tested.
        with pytest.raises(RuntimeError, match="class IDs must be non-negative"):
            library.update(tracker._handle, negative_class, image)
    finally:
        tracker.close()


def test_native_bytetrack_target_fps_filters_empty_detection_timeline(tmp_path):
    import cv2

    executable = native_module.ensure_bytetrack_cpp_executable()
    sequence_name = "EMPTY-FPS"
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
            np.zeros((16, 16, 3), dtype=np.uint8),
        )

    cache_root = tmp_path / "cache"
    det_dir = cache_root / "detector" / "dets"
    det_dir.mkdir(parents=True)
    np.save(det_dir / f"{sequence_name}.npy", np.empty((0, 8), dtype=np.float32))
    output = tmp_path / "tracks.txt"
    completed = native_module.subprocess.run(
        [
            str(executable),
            "--mot-root",
            str(mot_root),
            "--det-emb-root",
            str(cache_root),
            "--detector-name",
            "detector",
            "--sequence",
            sequence_name,
            "--output",
            str(output),
            "--target-fps",
            "15",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"kept_frame_ids":[1,3,5]' in completed.stdout
    assert output.read_text(encoding="utf-8") == ""


def test_native_bytetrack_replay_reads_multispectral_npy_frames(tmp_path):
    executable = native_module.ensure_bytetrack_cpp_executable()
    sequence_name = "MMOT-NPY"
    mot_root = tmp_path / "mot"
    image_dir = mot_root / sequence_name
    image_dir.mkdir(parents=True)
    multispectral = np.zeros((16, 20, 8), dtype=np.uint8)
    multispectral[:, :, 1] = 20
    multispectral[:, :, 2] = 40
    multispectral[:, :, 4] = 80
    np.save(image_dir / "000001.npy", multispectral)

    cache_root = tmp_path / "cache"
    det_dir = cache_root / "detector" / "dets"
    det_dir.mkdir(parents=True)
    np.save(
        det_dir / f"{sequence_name}.npy",
        np.array([[1, 10, 8, 8, 4, 0.2, 0.95, 0]], dtype=np.float32),
    )
    output = tmp_path / "tracks.txt"
    completed = native_module.subprocess.run(
        [
            str(executable),
            "--mot-root",
            str(mot_root),
            "--det-emb-root",
            str(cache_root),
            "--detector-name",
            "detector",
            "--sequence",
            sequence_name,
            "--output",
            str(output),
            "--min-conf",
            "0.1",
            "--track-thresh",
            "0.1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"num_frames":1' in completed.stdout
    assert '"kept_frame_ids":[1]' in completed.stdout
    assert output.read_text(encoding="utf-8")


def test_process_sequence_cpp_streams_progress_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_bytetrack_cpp_executable", lambda force_rebuild=False: Path("/tmp/bytetrack_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/bytetrack_replay"
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
        reid_name="/weights/unused.pt",
        tracker_name="bytetrack",
        exp_folder=str(tmp_path),
        target_fps=None,
        progress_queue=progress_queue,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 1, 2)
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 2, 2)
