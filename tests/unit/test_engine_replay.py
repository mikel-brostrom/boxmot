from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import boxmot.engine.evaluator as evaluator_module
import boxmot.engine.replay as replay_module


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
        benchmark="mot17-mini",
        source=source,
        yolo_model=[Path("det.pt")],
        reid_model=[Path("/tmp/reid.pt")],
        tracking_method="boosttrack",
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
            progress_queue = task_arg[-1]
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
        yolo_model=[Path("det.pt")],
        reid_model=[Path("/tmp/reid.pt")],
        tracking_method="boosttrack",
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
            progress_queue = task_arg[-1]
            progress_queue.put_nowait((seq_name, 1, 1))
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(replay_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(
        replay_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )
    monkeypatch.setattr(replay_module.LOGGER, "opt", lambda **kwargs: SimpleNamespace(info=lambda message: None))

    replay_module.run_generate_mot_results(args, quiet=False)

    assert manager_calls == [True]
    assert queue_types == ["Queue"]


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
    assert "(20/20)" in lines[1]
    assert "(pending)" in lines[2]