from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import boxmot.engine.eval.evaluator as evaluator_module
import boxmot.engine.eval.replay as replay_module
from boxmot.trackers.common.geometry.obb import xywha_to_corners


def test_resolve_embedding_cache_dir_reuses_row_aligned_trusted_legacy_cache(tmp_path):
    sequence_name = "MOT17-02-FRCNN"
    detector = Path("models/yolox_x_MOT17_ablation.pt")
    reid = tmp_path / "models" / "lmbn_n_duke.pt"
    reid.parent.mkdir()
    reid.write_bytes(b"weights")

    detector_root = (
        tmp_path
        / "dets_n_embs"
        / "mot17"
        / "ablation"
        / detector.stem
    )
    det_path = detector_root / "dets" / f"{sequence_name}.npy"
    legacy_path = detector_root / "embs" / "lmbn_n_duke" / "resize" / f"{sequence_name}.npy"
    det_path.parent.mkdir(parents=True)
    legacy_path.parent.mkdir(parents=True)
    np.save(det_path, np.ones((2, 7), dtype=np.float32))
    np.save(legacy_path, np.ones((2, 4), dtype=np.float32))

    args = SimpleNamespace(
        detector=[detector],
        reid=[reid],
        benchmark="mot17",
        split="ablation",
        tracker_backend="python",
        reid_preprocess="resize",
        eval_box_type="aabb",
        allow_legacy_reid_cache=True,
    )

    resolved = replay_module._resolve_embedding_cache_dir(args, tmp_path, sequence_name)

    assert Path(resolved) == legacy_path.parent


@pytest.mark.parametrize(
    ("gap_entries", "expected_cols"),
    [
        (
            np.array([[2, 10, 5, 25, 30, 7, 0.8, 2, -1]], dtype=np.float32),
            9,
        ),
        (
            np.array([[2, 50, 40, 20, 10, 0.3, 11, 0.8, 2, -1]], dtype=np.float32),
            13,
        ),
    ],
)
def test_process_sequence_formats_gta_rows_through_canonical_mot_path(
    tmp_path,
    monkeypatch,
    gap_entries,
    expected_cols,
):
    captured = {}

    class FakeTracker:
        cmc = None
        with_reid = False
        embedding_off = False

        def flush_gta(self):
            return gap_entries.copy()

    class FakeTrackerRuntime:
        tracker = FakeTracker()

    monkeypatch.setattr(replay_module.TrackerRuntime, "create", lambda **kwargs: FakeTrackerRuntime())
    monkeypatch.setattr(
        replay_module,
        "MOTDataset",
        lambda **kwargs: SimpleNamespace(get_sequence=lambda *args, **kw: []),
    )
    monkeypatch.setattr(
        replay_module,
        "write_mot_results",
        lambda path, arr: captured.setdefault("rows", arr.copy()),
    )

    replay_module.process_sequence(
        seq_name="sequence",
        mot_root=str(tmp_path / "source"),
        project_root=str(tmp_path),
        detector_name="det.pt",
        reid_name="",
        tracker_name="occluboost",
        exp_folder=str(tmp_path / "results"),
        target_fps=None,
    )

    rows = captured["rows"]
    assert rows.shape == (1, expected_cols)
    assert int(rows[0, 0]) == 2
    assert int(rows[0, 1]) == int(gap_entries[0, -4])
    if expected_cols == 9:
        # AABB export is [frame,id,left,top,width,height,conf,cls+1,det_ind].
        np.testing.assert_allclose(rows[0, 2:6], [10, 5, 15, 25])
        assert int(rows[0, 7]) == 3
    else:
        # OBB export is [frame,id,8 corners,conf,cls,det_ind].
        np.testing.assert_allclose(
            rows[0, 2:10],
            xywha_to_corners(gap_entries[0, 1:6]),
            atol=1e-4,
        )
        assert int(rows[0, 11]) == 2


def test_worker_init_suppresses_worker_logs(monkeypatch):
    calls = []

    monkeypatch.setattr(
        replay_module,
        "_configure_logging",
        lambda **kwargs: calls.append(kwargs),
    )

    replay_module._worker_init()

    assert calls == [{"main_thread_only": True}]


def test_replay_process_backend_uses_spawn_context(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "shared-runs",
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="boosttrack",
        fps=None,
        device="cpu",
        n_threads=2,
        postprocessing="none",
        conf=0.25,
    )

    queue_types = []
    executor_kwargs = {}
    manager_calls = []

    class FakeManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Queue(self):
            progress_queue = queue.Queue()
            queue_types.append(type(progress_queue).__name__)
            return progress_queue

    class FakeSpawnContext:
        def Manager(self):
            manager_calls.append(True)
            return FakeManager()

    spawn_context = FakeSpawnContext()

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            executor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            progress_queue = task_arg[-2]  # second-to-last; last is adaptive_kf
            assert task_arg[2] == str(args.cache_project)
            if progress_queue is not None:
                progress_queue.put_nowait((seq_name, 1, 1))
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(replay_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    replay_module.run_generate_mot_results(args, quiet=True)

    assert args.seq_frame_nums == {
        "MOT17-02-FRCNN": [1],
        "MOT17-04-FRCNN": [1],
    }
    assert manager_calls == []
    assert queue_types == []
    assert executor_kwargs["mp_context"] is spawn_context
    assert executor_kwargs["max_workers"] == 2


def test_replay_nonquiet_uses_manager_queue_for_progress(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="boosttrack",
        fps=None,
        device="cpu",
        n_threads=2,
        postprocessing="none",
        conf=0.25,
    )

    queue_types = []
    manager_calls = []

    class FakeManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Queue(self):
            progress_queue = queue.Queue()
            queue_types.append(type(progress_queue).__name__)
            return progress_queue

    class FakeSpawnContext:
        def Manager(self):
            manager_calls.append(True)
            return FakeManager()

    spawn_context = FakeSpawnContext()

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            progress_queue = task_arg[-2]  # second-to-last; last is adaptive_kf
            progress_queue.put_nowait((seq_name, 1, 1))
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(replay_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    replay_module.run_generate_mot_results(args, quiet=False)

    assert manager_calls == [True]
    assert queue_types == ["Queue"]


def test_process_sequence_reports_separate_reid_and_tracker_rest_time(tmp_path, monkeypatch):
    source = tmp_path / "train"
    exp_dir = tmp_path / "runs"
    exp_dir.mkdir()

    created = {}

    class FakeTrackerRuntime:
        tracker = None

        def update(self, dets, img, embs, masks=None):
            created["timing_stats"].add_reid_time(3.0)
            return np.array([[1, 2, 10, 12, 1, 0.9, 0, 0]], dtype=np.float32), 10.0

    def fake_create(**kwargs):
        created["timing_stats"] = kwargs["timing_stats"]
        return FakeTrackerRuntime()

    monkeypatch.setattr(replay_module.TrackerRuntime, "create", fake_create)
    monkeypatch.setattr(
        replay_module,
        "MOTDataset",
        lambda **kwargs: SimpleNamespace(
            get_sequence=lambda *args, **kw: [
                {
                    "frame_id": 1,
                    "dets": np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
                    "embs": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                    "img": np.zeros((4, 4, 3), dtype=np.uint8),
                }
            ]
        ),
    )
    monkeypatch.setattr(replay_module, "write_mot_results", lambda path, arr: None)

    seq_name, kept_ids, timing = replay_module.process_sequence(
        seq_name="MOT17-02-FRCNN",
        mot_root=str(source),
        project_root=str(tmp_path),
        detector_name="det.pt",
        reid_name="reid.pt",
        tracker_name="deepocsort",
        exp_folder=str(exp_dir),
        target_fps=None,
    )

    assert seq_name == "MOT17-02-FRCNN"
    assert kept_ids == [1]
    assert timing == {"track_time_ms": 7.0, "reid_time_ms": 3.0, "num_frames": 1}


def test_process_sequence_updates_tracker_on_empty_detection_frames(tmp_path, monkeypatch):
    source = tmp_path / "train"
    exp_dir = tmp_path / "runs"
    exp_dir.mkdir()
    update_rows = []

    class FakeTrackerRuntime:
        tracker = SimpleNamespace()

        def update(self, dets, img, embs, masks=None):
            update_rows.append((len(dets), embs, img.copy()))
            return np.empty((0, 8), dtype=np.float32), 1.0

    monkeypatch.setattr(replay_module.TrackerRuntime, "create", lambda **kwargs: FakeTrackerRuntime())
    monkeypatch.setattr(
        replay_module,
        "MOTDataset",
        lambda **kwargs: SimpleNamespace(
            get_sequence=lambda *args, **kw: [
                {
                    "frame_id": 1,
                    "dets": np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
                    "embs": np.empty((1, 0), dtype=np.float32),
                    "img": np.zeros((16, 16, 3), dtype=np.uint8),
                },
                {
                    "frame_id": 2,
                    "dets": np.empty((0, 6), dtype=np.float32),
                    "embs": np.empty((0, 0), dtype=np.float32),
                    "img": np.ones((16, 16, 3), dtype=np.uint8),
                },
                {
                    "frame_id": 3,
                    "dets": np.array([[2, 2, 11, 12, 0.9, 0]], dtype=np.float32),
                    "embs": np.empty((1, 0), dtype=np.float32),
                    "img": np.full((16, 16, 3), 2, dtype=np.uint8),
                },
            ]
        ),
    )
    monkeypatch.setattr(replay_module, "write_mot_results", lambda path, arr: None)

    seq_name, kept_ids, timing = replay_module.process_sequence(
        seq_name="S",
        mot_root=str(source),
        project_root=str(tmp_path),
        detector_name="det.pt",
        reid_name="",
        tracker_name="ocsort",
        exp_folder=str(exp_dir),
        target_fps=2,
    )

    assert seq_name == "S"
    assert kept_ids == [1, 2, 3]
    assert [rows for rows, _, _ in update_rows] == [1, 0, 1]
    assert update_rows[1][1] is None
    np.testing.assert_array_equal(update_rows[1][2], np.ones((16, 16, 3), dtype=np.uint8))
    assert timing == {"track_time_ms": 3.0, "reid_time_ms": 0.0, "num_frames": 3}


def test_run_generate_mot_results_accumulates_worker_reid_timings(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "shared-runs",
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="deepocsort",
        fps=None,
        device="cpu",
        n_threads=2,
        postprocessing="none",
        conf=0.25,
    )

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "reid_time_ms": 2.5, "num_frames": 1}))

    class FakeSpawnContext:
        def Manager(self):
            raise AssertionError("quiet replay should not create a manager")

    monkeypatch.setattr(replay_module.mp, "get_context", lambda method: FakeSpawnContext())
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    timing_stats = replay_module.TimingStats()
    replay_module.run_generate_mot_results(args, timing_stats=timing_stats, quiet=True)

    assert timing_stats.totals["track"] == 10.0
    assert timing_stats.totals["reid"] == 5.0
    assert timing_stats.frames == 2


def test_replay_cpp_backend_uses_native_runner(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "shared-runs",
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="botsort",
        fps=None,
        device="cpu",
        n_threads=2,
        tracker_backend="cpp",
        tracking_backend="thread",
        postprocessing="none",
        conf=0.25,
    )

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    def fake_process_sequence_cpp(*args, **kwargs):
        raise AssertionError("The fake executor should not call the submitted function directly")

    class FakeThreadPoolExecutor:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *task_arg):
            assert fn is fake_process_sequence_cpp
            seq_name = task_arg[0]
            return FakeFuture((seq_name, [1], {"track_time_ms": 7.5, "num_frames": 1}))

    monkeypatch.setattr(
        replay_module,
        "get_native_replay_backend",
        lambda tracker_name: SimpleNamespace(process_sequence=fake_process_sequence_cpp),
    )
    monkeypatch.setattr(replay_module.concurrent.futures, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    replay_module.run_generate_mot_results(args, quiet=True)

    assert args.seq_frame_nums == {
        "MOT17-02-FRCNN": [1],
        "MOT17-04-FRCNN": [1],
    }


def test_replay_cpp_backend_rejects_unsupported_tracker(tmp_path):
    source = tmp_path / "train"
    img_dir = source / "MOT17-02-FRCNN" / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="deepocsort",
        fps=None,
        device="cpu",
        n_threads=1,
        tracker_backend="cpp",
        tracking_backend="thread",
        postprocessing="none",
        conf=0.25,
    )

    try:
        replay_module.run_generate_mot_results(args, quiet=True)
    except ValueError as exc:
        assert "tracker_backend='cpp' is not available" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported native replay tracker")


def test_replay_cpp_tracking_backend_alias_uses_native_runner(tmp_path, monkeypatch):
    source = tmp_path / "train"
    img_dir = source / "MOT17-02-FRCNN" / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="botsort",
        fps=None,
        device="cpu",
        n_threads=1,
        tracking_backend="cpp",
        postprocessing="none",
        conf=0.25,
    )

    monkeypatch.setattr(
        replay_module,
        "get_native_replay_backend",
        lambda tracker_name: SimpleNamespace(
            process_sequence=lambda *task_arg: (
                task_arg[0],
                [1],
                {"track_time_ms": 1.0, "num_frames": 1},
            )
        ),
    )

    replay_module.run_generate_mot_results(args, quiet=True)

    assert args.seq_frame_nums == {"MOT17-02-FRCNN": [1]}


def test_replay_cpp_backend_reports_incremental_progress(tmp_path, monkeypatch):
    source = tmp_path / "train"
    img_dir = source / "MOT17-02-FRCNN" / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("/tmp/reid.pt")],
        tracker="botsort",
        fps=None,
        device="cpu",
        n_threads=1,
        tracker_backend="cpp",
        tracking_backend="thread",
        postprocessing="none",
        conf=0.25,
    )

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    state = {"progress_queue": None, "calls": 0}

    def fake_process_sequence_cpp(*args, **kwargs):
        raise AssertionError("The fake executor should not call the submitted function directly")

    class FakeThreadPoolExecutor:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_workers"] == 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *task_arg):
            assert fn is fake_process_sequence_cpp
            state["progress_queue"] = task_arg[-2]
            return FakeFuture(("MOT17-02-FRCNN", [1, 2, 3], {"track_time_ms": 7.5, "num_frames": 3}))

    messages = []
    monkeypatch.setattr(
        replay_module,
        "get_native_replay_backend",
        lambda tracker_name: SimpleNamespace(process_sequence=fake_process_sequence_cpp),
    )
    monkeypatch.setattr(replay_module.concurrent.futures, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (
            (
                state["progress_queue"].put_nowait(("MOT17-02-FRCNN", 1, 3)),
                state.__setitem__("calls", state["calls"] + 1),
                set(),
                set(pending),
            )[-2:]
            if state["calls"] == 0
            else (
                state["progress_queue"].put_nowait(("MOT17-02-FRCNN", 2, 3)),
                state.__setitem__("calls", state["calls"] + 1),
                set(),
                set(pending),
            )[-2:]
            if state["calls"] == 1
            else (
                state["progress_queue"].put_nowait(("MOT17-02-FRCNN", 3, 3)),
                state.__setitem__("calls", state["calls"] + 1),
                set(pending),
                set(),
            )[-2:]
        ),
    )

    replay_module.run_generate_mot_results(args, quiet=False, progress_callback=messages.append)

    assert any("(2/3)" in message for message in messages)
    assert messages[-1].startswith("Tracking: 1/1 sequences done")


def test_evaluator_reexports_replay_helpers():
    assert evaluator_module.process_sequence is replay_module.process_sequence
    assert evaluator_module.run_generate_mot_results is replay_module.run_generate_mot_results


def test_format_seq_progress_shows_all_sequences_in_order():
    text = replay_module._format_seq_progress(
        ["MOT17-02", "MOT17-04", "MOT17-05"],
        {
            "MOT17-02": (10, 20),
            "MOT17-04": (20, 20),
        },
    )

    lines = text.splitlines()

    assert len(lines) == 3
    assert "MOT17-02" in lines[0]
    assert "MOT17-04" in lines[1]
    assert "MOT17-05" in lines[2]
    assert "(10/20)" in lines[0]
    assert "(done)" in lines[1]
    assert "(pending)" in lines[2]
