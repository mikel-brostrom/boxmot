from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import boxmot.engine.eval.evaluator as evaluator_module
import boxmot.engine.eval.replay as replay_module


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

    class FakeThreadPoolExecutor:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *task_arg):
            assert fn is replay_module.process_sequence_cpp
            seq_name = task_arg[0]
            return FakeFuture((seq_name, [1], {"track_time_ms": 7.5, "num_frames": 1}))

    monkeypatch.setattr(
        replay_module,
        "get_native_replay_backend",
        lambda tracker_name: SimpleNamespace(process_sequence=replay_module.process_sequence_cpp),
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
        lambda tracker_name: SimpleNamespace(process_sequence=lambda *task_arg: (task_arg[0], [1], {"track_time_ms": 1.0, "num_frames": 1})),
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

    class FakeThreadPoolExecutor:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_workers"] == 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *task_arg):
            assert fn is replay_module.process_sequence_cpp
            state["progress_queue"] = task_arg[-1]
            return FakeFuture(("MOT17-02-FRCNN", [1, 2, 3], {"track_time_ms": 7.5, "num_frames": 3}))

    messages = []
    monkeypatch.setattr(
        replay_module,
        "get_native_replay_backend",
        lambda tracker_name: SimpleNamespace(process_sequence=replay_module.process_sequence_cpp),
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
