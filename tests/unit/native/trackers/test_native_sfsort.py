from __future__ import annotations

import queue
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from boxmot.native.trackers import sfsort as native_module
from boxmot.trackers.bbox.sfsort import SFSORT


def _empty_tracks_for(dets):
    columns = 9 if dets.shape[1] == 7 else 8
    return native_module.np.empty((0, columns), dtype=native_module.np.float32)


def test_process_sequence_cpp_builds_native_command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_sfsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/sfsort_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/sfsort_replay"
            assert "--sequence" in cmd
            assert "MOT17-02-FRCNN" in cmd
            assert "--high-th" in cmd
            assert "--match-th-first" in cmd
            assert "--obb-theta-damping" in cmd
            assert cmd[cmd.index("--obb-theta-damping") + 1] == "0.75"
            assert "--dynamic-tuning" in cmd
            assert cmd[cmd.index("--asso-func") + 1] == "giou"
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
        tracker_name="sfsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        cfg_dict={
            "high_th": 0.6,
            "match_th_first": 0.67,
            "new_track_th": 0.7,
            "low_th": 0.1,
            "match_th_second": 0.3,
            "dynamic_tuning": True,
            "cth": 0.5,
            "obb_theta_damping": 0.75,
            "asso_func": "giou",
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
            tracker_name="bytetrack",
            exp_folder="/tmp",
            target_fps=None,
        )
    except ValueError as exc:
        assert "tracker='sfsort' only" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-SFSORT tracker")


def test_native_sfsort_tracker_uses_live_library_wrapper():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            calls.append(("create", cfg["high_th"], cfg["dynamic_tuning"]))
            return "handle"

        def reset(self, handle):
            calls.append(("reset", handle))

        def update(self, handle, dets, img):
            calls.append(("update", handle, dets.shape, img))
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            calls.append(("destroy", handle))

    tracker = native_module.NativeSFSORTTracker({"high_th": 0.55, "dynamic_tuning": True}, library=_FakeLibrary())

    dets = native_module.np.array(
        [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
        dtype=native_module.np.float32,
    )
    out = tracker.update(dets)
    tracker.reset()
    tracker.close()

    assert out.shape == (0, 8)
    assert calls == [
        ("create", 0.55, True),
        ("update", "handle", (2, 6), None),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_sfsort_tracker_accepts_obb_rows():
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

    tracker = native_module.NativeSFSORTTracker(library=_FakeLibrary())
    dets = native_module.np.ones((1, 7), dtype=native_module.np.float32)
    out = tracker.update(dets)

    assert out.shape == (1, 9)
    assert calls == [("handle", (1, 7), None)]
    tracker.close()


def test_native_sfsort_image_capability_follows_margin_configuration():
    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            return None

    no_region_split = native_module.NativeSFSORTTracker(library=_FakeLibrary())
    configured_dimensions = native_module.NativeSFSORTTracker(
        {
            "central_timeout": 5,
            "marginal_timeout": 1,
            "frame_width": 640,
            "frame_height": 480,
        },
        library=_FakeLibrary(),
    )
    inferred_dimensions = native_module.NativeSFSORTTracker(
        {"central_timeout": 5, "marginal_timeout": 1},
        library=_FakeLibrary(),
    )

    try:
        assert no_region_split.supports_masks is False
        assert no_region_split.uses_embs is False
        assert no_region_split.uses_img is False
        assert configured_dimensions.uses_img is False
        assert inferred_dimensions.uses_img is True
    finally:
        no_region_split.close()
        configured_dimensions.close()
        inferred_dimensions.close()


def test_native_sfsort_requires_image_only_for_margin_inference():
    calls = []

    class _FakeLibrary:
        def create(self, cfg):
            return "handle"

        def reset(self, handle):
            return None

        def update(self, handle, dets, img):
            calls.append(None if img is None else img.shape)
            return _empty_tracks_for(dets)

        def destroy(self, handle):
            return None

    tracker = native_module.NativeSFSORTTracker(
        {"central_timeout": 5, "marginal_timeout": 1},
        library=_FakeLibrary(),
    )
    dets = np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float32)

    try:
        with pytest.raises(ValueError, match="requires img to infer frame dimensions"):
            tracker.update(dets)
        output = tracker.update(dets, np.zeros((8, 8, 3), dtype=np.uint8))
        second = tracker.update(dets)
        assert tracker.uses_img is False
        tracker.reset()
        assert tracker.uses_img is True
    finally:
        tracker.close()

    assert output.shape == (0, 8)
    assert second.shape == (0, 8)
    assert calls == [(8, 8, 3), None]


def test_native_sfsort_c_api_rejects_missing_margin_image():
    library = native_module._SFSORTLiveLibrary(native_module.ensure_sfsort_cpp_library())
    tracker = native_module.NativeSFSORTTracker(
        {"central_timeout": 5, "marginal_timeout": 1},
        library=library,
    )
    detections = np.empty((0, 6), dtype=np.float32)

    try:
        with pytest.raises(RuntimeError, match="requires an image to infer frame dimensions"):
            library.update(tracker._handle, detections)
    finally:
        tracker.close()


def test_native_sfsort_live_obb_cost_is_equivalent_form_invariant():
    library = native_module._SFSORTLiveLibrary(native_module.ensure_sfsort_cpp_library())
    tracker = native_module.NativeSFSORTTracker(
        {
            "high_th": 0.5,
            "new_track_th": 0.5,
            "low_th": 0.1,
            # The pre-fix direct-only width/height term produced a cost of
            # 0.5 for this physically identical swapped representation.
            "match_th_first": 0.1,
            "dynamic_tuning": False,
            "frame_width": 160,
            "frame_height": 120,
        },
        library=library,
    )
    first = native_module.np.array(
        [[80, 60, 80, 20, (4 * native_module.np.pi) + 0.2, 0.95, 0]],
        dtype=native_module.np.float32,
    )
    equivalent = native_module.np.array(
        [[80, 60, 20, 80, 0.2 + (native_module.np.pi / 2), 0.95, 0]],
        dtype=native_module.np.float32,
    )

    try:
        first_output = tracker.update(first)
        equivalent_output = tracker.update(equivalent)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert equivalent_output.shape == (1, 9)
    assert -native_module.np.pi <= first_output[0, 4] < native_module.np.pi
    assert equivalent_output[0, 5] == first_output[0, 5]
    native_module.np.testing.assert_allclose(
        equivalent_output[0, :5],
        first_output[0, :5],
        atol=1e-4,
    )


def test_native_sfsort_low_only_obb_frame_keeps_track():
    library = native_module._SFSORTLiveLibrary(native_module.ensure_sfsort_cpp_library())
    tracker = native_module.NativeSFSORTTracker(
        {
            "high_th": 0.6,
            "new_track_th": 0.5,
            "low_th": 0.1,
            "match_th_second": 0.3,
            "dynamic_tuning": False,
            "frame_width": 160,
            "frame_height": 120,
        },
        library=library,
    )
    image = native_module.np.zeros((120, 160, 3), dtype=native_module.np.uint8)
    high = native_module.np.array([[80, 60, 40, 20, 0.2, 0.9, 0]], dtype=native_module.np.float32)
    low = high.copy()
    low[:, 5] = 0.3

    try:
        first = tracker.update(high, image)
        second = tracker.update(low, image)
    finally:
        tracker.close()

    assert first.shape == second.shape == (1, 9)
    assert second[0, 5] == first[0, 5]


@pytest.mark.parametrize(
    ("initial_detections", "ambiguous_detections", "match_threshold"),
    [
        (
            [[0, 0, 10, 10, 0.95, 0], [8, 0, 18, 10, 0.95, 0]],
            [[-4, 0, 2, 10, 0.95, 0], [1, 0, 13, 10, 0.95, 0]],
            0.2,
        ),
        (
            [[5, 5, 10, 10, 0, 0.95, 0], [13, 5, 10, 10, 0, 0.95, 0]],
            [[3, 5, 6, 10, 0, 0.95, 0], [6, 5, 10, 10, 0, 0.95, 0]],
            0.1,
        ),
    ],
    ids=["aabb", "obb"],
)
def test_native_sfsort_threshold_aware_assignment_matches_python(
    initial_detections,
    ambiguous_detections,
    match_threshold,
):
    """Keep the same valid identity that lapjv's cost limit selects."""
    cfg = {
        "high_th": 0.6,
        "new_track_th": 0.7,
        "low_th": 0.1,
        "match_th_first": match_threshold,
        "match_th_second": 0.3,
        "dynamic_tuning": False,
        "frame_width": 100,
        "frame_height": 100,
        "horizontal_margin": 0,
        "vertical_margin": 0,
    }
    python_tracker = SFSORT(**cfg)
    library = native_module._SFSORTLiveLibrary(native_module.ensure_sfsort_cpp_library())
    native_tracker = native_module.NativeSFSORTTracker(cfg, library=library)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        for detections in (initial_detections, ambiguous_detections):
            dets = np.asarray(detections, dtype=np.float32)
            python_output = np.asarray(python_tracker.update(dets, image))
            native_output = np.asarray(native_tracker.update(dets, image))
            np.testing.assert_allclose(native_output, python_output, atol=1e-5)
    finally:
        native_tracker.close()


def test_process_sequence_cpp_streams_progress_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        native_module, "ensure_sfsort_cpp_executable", lambda force_rebuild=False: Path("/tmp/sfsort_replay")
    )

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize):
            assert cmd[0] == "/tmp/sfsort_replay"
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
        tracker_name="sfsort",
        exp_folder=str(tmp_path),
        target_fps=None,
        progress_queue=progress_queue,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1, 2]
    assert timing == {"track_time_ms": 12.5, "num_frames": 2}
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 1, 2)
    assert progress_queue.get_nowait() == ("MOT17-02-FRCNN", 2, 2)
