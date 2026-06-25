from __future__ import annotations

import importlib
from argparse import Namespace

trackeval_module = importlib.import_module("boxmot.engine.eval.trackeval.runner")


_RESOURCE_TRACKER_STDERR = (
    "/Users/test/.local/share/uv/python/cpython-3.12.10/lib/python3.12/multiprocessing/"
    "resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 6 leaked semaphore objects "
    "to clean up at shutdown\n"
    "  warnings.warn('resource_tracker: There appear to be %d '\n"
    "/Users/test/.local/share/uv/python/cpython-3.12.10/lib/python3.12/multiprocessing/"
    "resource_tracker.py:292: UserWarning: resource_tracker: '/mp-abc123': "
    "[Errno 2] No such file or directory\n"
    "  warnings.warn('resource_tracker: %r: %s' % (name, e))\n"
)


class _FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self):
        return self._stdout, self._stderr


def test_should_use_parallel_trackeval_always_true():
    # Parallelism is always enabled; the subprocess env suppresses
    # macOS 3.12+ resource_tracker warnings.
    assert trackeval_module._should_use_parallel_trackeval() is True


def test_filter_trackeval_stderr_drops_resource_tracker_noise():
    assert trackeval_module._filter_trackeval_stderr(_RESOURCE_TRACKER_STDERR) == ""

    mixed = _RESOURCE_TRACKER_STDERR + "real issue\n"
    assert trackeval_module._filter_trackeval_stderr(mixed) == "real issue"


def test_trackeval_enables_parallel_and_suppresses_resource_tracker_stderr(monkeypatch, tmp_path):
    popen_calls = []
    warnings = []

    def fake_popen(*, args, stdout, stderr, text, env):
        popen_calls.append({"args": args, "env": env})
        return _FakeProcess(stdout="summary", stderr=_RESOURCE_TRACKER_STDERR)

    monkeypatch.setattr(trackeval_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(trackeval_module.LOGGER, "warning", lambda message: warnings.append(message))

    args = Namespace(
        exp_dir=tmp_path / "runs" / "exp",
        split="train",
        benchmark="mot17",
        remapped_class_ids=None,
        remapped_class_names=None,
        classes=None,
    )

    output = trackeval_module.trackeval_aabb(
        args,
        seq_paths=[],
        save_dir=tmp_path / "save",
        gt_folder=tmp_path / "gt",
        seq_info={"MOT17-02": 299},
    )

    cmd = [str(part) for part in popen_calls[0]["args"]]
    use_parallel_idx = cmd.index("--USE_PARALLEL")
    cores_idx = cmd.index("--NUM_PARALLEL_CORES")

    assert output == "summary"
    assert "boxmot.engine.eval.trackeval.scripts.mot_challenge" in cmd
    assert cmd[use_parallel_idx + 1] == "True"
    assert int(cmd[cores_idx + 1]) >= 1
    assert "ignore:resource_tracker:UserWarning" in popen_calls[0]["env"]["PYTHONWARNINGS"]
    assert warnings == []


def test_trackeval_reports_non_resource_tracker_stderr(monkeypatch, tmp_path):
    warnings = []

    def fake_popen(*, args, stdout, stderr, text, env):
        return _FakeProcess(stdout="summary", stderr=_RESOURCE_TRACKER_STDERR + "real issue\n")

    monkeypatch.setattr(trackeval_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(trackeval_module.LOGGER, "warning", lambda message: warnings.append(message))

    args = Namespace(
        exp_dir=tmp_path / "runs" / "exp",
        split="train",
        benchmark="mot17",
        remapped_class_ids=None,
        remapped_class_names=None,
        classes=None,
    )

    trackeval_module.trackeval_aabb(
        args,
        seq_paths=[],
        save_dir=tmp_path / "save",
        gt_folder=tmp_path / "gt",
        seq_info={"MOT17-02": 299},
    )

    assert warnings == ["TrackEval stderr:\nreal issue"]
