from __future__ import annotations

import queue
from io import StringIO
from pathlib import Path

import pytest

from boxmot.native.trackers import ocsort as native_module


def _empty_tracks_for(dets):
    columns = 9 if dets.shape[1] == 7 else 8
    return native_module.np.empty((0, columns), dtype=native_module.np.float32)


def test_process_sequence_cpp_builds_native_command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_ocsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/ocsort_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/ocsort_replay"
            assert "--sequence" in cmd
            assert "MOT17-02-FRCNN" in cmd
            assert "--min-conf" in cmd
            assert "--det-thresh" in cmd
            assert "--iou-threshold" in cmd
            assert "--use-byte" in cmd
            assert "--q-xy-scaling" in cmd
            assert "--max-obs" in cmd
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
        tracker_name="ocsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        cfg_dict={
            "min_conf": 0.1,
            "det_thresh": 0.55,
            "iou_threshold": 0.27,
            "max_age": 20,
            "min_hits": 2,
            "delta_t": 3,
            "use_byte": True,
            "inertia": 0.2,
            "Q_xy_scaling": 0.02,
            "Q_s_scaling": 0.0002,
            "max_obs": 25,
        },
        dataset_name="mot17-mini",
        conf_threshold=0.25,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}


def test_process_sequence_cpp_rejects_other_trackers():
    with pytest.raises(ValueError, match="tracker='ocsort' only"):
        native_module.process_sequence_cpp(
            seq_name="MOT17-02-FRCNN",
            mot_root="/data/train",
            project_root="/runs",
            detector_name="yolox_x.pt",
            reid_name="/weights/unused.pt",
            tracker_name="bytetrack",
            exp_folder="/tmp",
            target_fps=None,
        )


def test_native_ocsort_tracker_uses_live_library_wrapper():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            calls.append(("create", cfg["det_thresh"], cfg["iou_threshold"], cfg["use_byte"]))
            return "handle"

        def reset(self, handle):
            calls.append(("reset", handle))

        def update(self, handle, dets, img):
            calls.append(("update", handle, dets.shape, img.shape))
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            calls.append(("destroy", handle))

    tracker = native_module.NativeOCSORTTracker(
        {"det_thresh": 0.55, "iou_threshold": 0.27, "use_byte": True},
        library=_FakeLibrary(),
    )

    dets = native_module.np.array(
        [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
        dtype=native_module.np.float32,
    )
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)

    out = tracker.update(dets, img)
    tracker.reset()
    tracker.close()

    assert out.shape == (0, 8)
    assert calls == [
        ("create", 0.55, 0.27, True),
        ("update", "handle", (2, 6), (8, 8, 3)),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_ocsort_tracker_accepts_obb_rows():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            calls.append((handle, dets.shape, img.shape))
            return native_module.np.ones((1, 9), dtype=native_module.np.float32)

        def destroy(self, handle):
            return None

    tracker = native_module.NativeOCSORTTracker(library=_FakeLibrary())
    dets = native_module.np.ones((1, 7), dtype=native_module.np.float32)
    img = native_module.np.zeros((8, 8, 3), dtype=native_module.np.uint8)

    out = tracker.update(dets, img)

    assert out.shape == (1, 9)
    assert calls == [("handle", (1, 7), (8, 8, 3))]
    tracker.close()


def test_native_ocsort_live_obb_emits_last_observation_like_python():
    library = native_module._OCSORTLiveLibrary(native_module.ensure_ocsort_cpp_library())
    tracker = native_module.NativeOCSORTTracker(
        {
            "min_conf": 0.01,
            "det_thresh": 0.1,
            "iou_threshold": 0.05,
            "min_hits": 0,
        },
        library=library,
    )
    image = native_module.np.zeros((120, 160, 3), dtype=native_module.np.uint8)
    first = native_module.np.array(
        [[80, 60, 80, 20, (4 * native_module.np.pi) + 0.2, 0.95, 0]],
        dtype=native_module.np.float32,
    )
    equivalent = native_module.np.array(
        [[80, 60, 20, 80, 0.2 + (native_module.np.pi / 2), 0.95, 0]],
        dtype=native_module.np.float32,
    )

    try:
        first_output = tracker.update(first, image)
        equivalent_output = tracker.update(equivalent, image)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert equivalent_output.shape == (1, 9)
    native_module.np.testing.assert_allclose(
        first_output[0, :5],
        [80, 60, 80, 20, 0.2],
        atol=1e-4,
    )
    native_module.np.testing.assert_allclose(
        equivalent_output[0, :5],
        equivalent[0, :5],
        atol=1e-4,
    )
    assert equivalent_output[0, 5] == first_output[0, 5]


@pytest.mark.parametrize(
    ("first", "second", "box_columns"),
    [
        (
            [-100, -100, -80, -70, 0.95, 0],
            [-99, -100, -79, -70, 0.95, 0],
            4,
        ),
        (
            [-90, -80, 20, 10, 0.2, 0.95, 0],
            [-89, -80, 20, 10, 0.25, 0.95, 0],
            5,
        ),
    ],
    ids=["aabb", "obb"],
)
def test_native_ocsort_negative_coordinates_emit_latest_observation(first, second, box_columns):
    """Negative geometry is valid; confidence is the observation sentinel."""
    library = native_module._OCSORTLiveLibrary(native_module.ensure_ocsort_cpp_library())
    tracker = native_module.NativeOCSORTTracker(
        {
            "min_conf": 0.01,
            "det_thresh": 0.1,
            "iou_threshold": 0.05,
            "min_hits": 0,
        },
        library=library,
    )
    image = native_module.np.zeros((120, 160, 3), dtype=native_module.np.uint8)
    first_dets = native_module.np.asarray([first], dtype=native_module.np.float32)
    second_dets = native_module.np.asarray([second], dtype=native_module.np.float32)

    try:
        first_output = tracker.update(first_dets, image)
        second_output = tracker.update(second_dets, image)
    finally:
        tracker.close()

    assert first_output.shape[0] == 1
    assert second_output.shape[0] == 1
    native_module.np.testing.assert_allclose(
        second_output[0, :box_columns],
        second_dets[0, :box_columns],
        atol=1e-4,
    )
    assert second_output[0, box_columns] == first_output[0, box_columns]


def test_process_sequence_cpp_streams_progress_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_ocsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/ocsort_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/ocsort_replay"
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
        tracker_name="ocsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        progress_queue=progress_queue,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 1, 2)
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 2, 2)


def test_native_ocsort_rejects_non_iou_association():
    with pytest.raises(NotImplementedError, match="asso_func='iou' only"):
        native_module.NativeOCSORTTracker(
            {"asso_func": "giou"}, library=type("L", (), {"create": lambda self, cfg: None})()
        )
