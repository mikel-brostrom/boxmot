"""Elapsed-time contracts, real native motion, and typed timed C ABI coverage."""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot import SFSORT, KalmanConfig, SFSORTConfig, create_tracker
from boxmot.native import _common as native_common
from boxmot.native.trackers._common import NativeTrackBatch
from boxmot.structures import Frame

TRACKERS = {
    "bytetrack": ("NativeByteTrackTracker", "ByteTrackLibrary", {}),
    "botsort": ("NativeBotSortTracker", "BotSortLibrary", {"use_embeddings": False, "use_cmc": False}),
    "ocsort": ("NativeOcSortTracker", "OcSortLibrary", {"min_hits": 1}),
    "occluboost": ("NativeOccluBoostTracker", "OccluBoostLibrary", {"use_embeddings": False, "use_cmc": False}),
    "sfsort": (
        "NativeSFSORTTracker", "SFSORTLibrary",
        {"frame_width": 640, "frame_height": 480, "central_timeout": 10, "marginal_timeout": 10},
    ),
}
TIMED_TRACKERS = tuple(name for name in TRACKERS if name != "sfsort")


class TimingBinding:
    """Capture native motion intervals without replacing public input validation."""

    def __init__(self) -> None:
        self.intervals: list[float] = []

    def create(self, cfg: dict) -> int:
        return 1

    def destroy(self, handle: int) -> None:
        pass

    def reset(self, handle: int) -> None:
        pass

    def update(self, handle: int, *, geometry: np.ndarray, dt: float, **kwargs) -> NativeTrackBatch:
        self.intervals.append(dt)
        return NativeTrackBatch(
            geometry=np.empty((0, geometry.shape[1]), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
            class_ids=np.empty(0, dtype=np.int64),
            detection_indices=np.empty(0, dtype=np.int64),
        )


def _tracker(name: str, library, *, obb: bool = False, variable_dt: bool = True):
    """Build each domain adapter with model-free inputs and a known reference interval."""
    class_name, _, options = TRACKERS[name]
    cls = getattr(importlib.import_module(f"boxmot.trackers.{name}.native"), class_name)
    timing = {}
    if name in TIMED_TRACKERS:
        timing = {"kalman.variable_dt": variable_dt, "kalman.noise.reference_dt_s": 0.1}
    elif variable_dt:
        timing = {"kalman.variable_dt": True}
    return cls(
        {**options, **timing},
        geometry="obb" if obb else "aabb",
        library=library,
    )


@pytest.mark.parametrize("name", TIMED_TRACKERS)
@pytest.mark.parametrize("obb", [False, True])
def test_native_timestamps_reach_motion_and_reset(name: str, obb: bool) -> None:
    binding = TimingBinding()
    tracker = _tracker(name, binding, obb=obb)
    detections = np.empty((0, 7 if obb else 6), dtype=np.float64)
    assert tracker.supports_variable_dt and tracker.requirements.timestamp
    with pytest.raises(ValueError, match="including the first"):
        tracker.update(detections)
    tracker.update(detections, timestamp_s=1.0)
    assert tracker.validate_timing(timestamp_s=1.05) == pytest.approx(0.05)
    assert binding.intervals == [0.0]
    frame = Frame(image=torch.zeros((3, 4, 4), dtype=torch.uint8), sample_id="frame", timestamp_s=1.05)
    tracker.update(detections, frame)
    tracker.update(detections, timestamp_s=1.3)
    assert binding.intervals == pytest.approx([0.0, 0.5, 2.5])
    for timestamp in (None, 1.3, 1.2, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="timestamp_s"):
            tracker.update(detections, timestamp_s=timestamp)
    with pytest.raises(ValueError, match="not both"):
        tracker.update(detections, frame, timestamp_s=1.4)
    assert len(binding.intervals) == 3
    tracker.reset()
    tracker.update(detections, timestamp_s=0.0)
    assert binding.intervals[-1] == 0.0
    tracker.close()


@pytest.mark.parametrize("name", TRACKERS)
def test_native_fixed_step_ignores_timestamp_metadata(name: str) -> None:
    binding = TimingBinding()
    tracker = _tracker(name, binding, variable_dt=False)
    assert not tracker.requirements.timestamp
    if name == "sfsort":
        assert not tracker.supports_variable_dt
    for timestamp in (None, 0.0, 100.0, -2.0):
        tracker.update(np.empty((0, 6), dtype=np.float64), timestamp_s=timestamp)
    assert binding.intervals == [0.0] * 4
    tracker.close()


@pytest.mark.parametrize("obb", [False, True])
def test_native_sfsort_rejects_variable_timing_configuration(obb: bool) -> None:
    with pytest.raises(ValueError, match="sfsort|SFSORT|kalman"):
        _tracker("sfsort", TimingBinding(), obb=obb)


def test_public_factory_rejects_sfsort_variable_timing_configuration() -> None:
    with pytest.raises(ValueError, match="sfsort|SFSORT|kalman"):
        create_tracker("sfsort", backend="cpp", kalman=KalmanConfig(variable_dt=True))


@pytest.fixture(scope="module")
def native_timing_build() -> Path:
    """Compile all five real tracker kernels and their covariance/recovery probe."""
    build_dir = native_common.repo_root() / "build" / "native" / "timing"
    commands = (
        [
            "cmake", "-S", str(native_common.package_native_root() / "cpp"), "-B", str(build_dir),
            "-DBOXMOT_BUILD_NATIVE_REID=OFF", "-DCMAKE_BUILD_TYPE=Release",
        ],
        [
            "cmake", "--build", str(build_dir), "--config", "Release", "--parallel", "2", "--target",
            "boxmot_native_timing_probe", *(f"{name}_capi" for name in TRACKERS),
        ],
    )
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return build_dir


def test_native_motion_covariance_and_irregular_gap_recovery(native_timing_build: Path) -> None:
    executable = "boxmot_native_timing_probe" + (".exe" if os.name == "nt" else "")
    candidates = (native_timing_build / executable, native_timing_build / "Release" / executable)
    path = next(path for path in candidates if path.is_file())
    completed = subprocess.run([str(path)], capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("name", TIMED_TRACKERS)
@pytest.mark.parametrize("obb", [False, True])
def test_real_native_timed_abi_accepts_empty_frames_and_restarts(
    native_timing_build: Path, name: str, obb: bool
) -> None:
    _, library_name, _ = TRACKERS[name]
    module = importlib.import_module(f"boxmot.native.trackers.{name}")
    path = next((native_timing_build / "trackers" / name).rglob(native_common.library_filename(name)))
    tracker = _tracker(name, getattr(module, library_name)(path), obb=obb)
    row = [100, 100, 40, 80, 0.1, 0.95, 0] if obb else [80, 60, 120, 140, 0.95, 0]
    detections = np.array([row], dtype=np.float64)
    first = tracker.update(detections, timestamp_s=0.0)
    tracker.update(detections, timestamp_s=0.03)
    tracker.update(np.empty((0, len(row))), timestamp_s=0.08)
    recovered = tracker.update(detections, timestamp_s=0.2)
    assert recovered.shape == first.shape == (1, 9 if obb else 8)
    assert recovered[0, 5 if obb else 4] == first[0, 5 if obb else 4]
    assert np.isfinite(recovered).all()
    tracker.reset()
    np.testing.assert_allclose(tracker.update(detections, timestamp_s=0.0), first)
    tracker.close()


def test_native_sfsort_fixed_angular_smoothing_ignores_timestamp_spacing(
    native_timing_build: Path,
) -> None:
    """SFSORT keeps its original per-update angle smoothing, including after gaps."""
    module = importlib.import_module("boxmot.native.trackers.sfsort")
    path = next((native_timing_build / "trackers" / "sfsort").rglob(native_common.library_filename("sfsort")))
    native = _tracker("sfsort", module.SFSORTLibrary(path), obb=True, variable_dt=False)
    python = SFSORT(
        config=SFSORTConfig(frame_width=640, frame_height=480, central_timeout=10, marginal_timeout=10),
        is_obb=True,
    )
    for timestamp, angle in ((0.0, 0.0), (0.05, 0.1), (0.12, None), (0.26, 0.25), (0.3, 0.3)):
        rows = (
            np.empty((0, 7), dtype=np.float64)
            if angle is None else np.array([[100, 100, 40, 20, angle, 0.95, 0]], dtype=np.float64)
        )
        actual = native.update(rows, timestamp_s=timestamp)
        expected = python.update(rows, timestamp_s=timestamp)
        np.testing.assert_allclose(actual[:, :5], expected[:, :5], atol=1.0e-5)
    assert actual[0, 4] == pytest.approx(0.1752, abs=1.0e-6)
    native.close()


@pytest.mark.parametrize("obb", [False, True])
def test_native_sfsort_c_abi_rejects_nonzero_dt_without_advancing_state(
    native_timing_build: Path, obb: bool,
) -> None:
    """The shared v3 entrypoint must not silently accept unsupported elapsed time."""
    module = importlib.import_module("boxmot.native.trackers.sfsort")
    path = next((native_timing_build / "trackers" / "sfsort").rglob(native_common.library_filename("sfsort")))
    library = module.SFSORTLibrary(path)
    tracker = _tracker("sfsort", library, obb=obb, variable_dt=False)
    with pytest.raises(RuntimeError, match="SFSORT does not support variable dt"):
        library.update(
            tracker._handle,
            geometry=np.empty((0, 5 if obb else 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int64),
            detection_indices=np.empty(0, dtype=np.int64),
            embeddings=None,
            image=None,
            dt=0.5,
        )
    rows = np.array(
        [[100, 100, 40, 80, 0.1, 0.95, 0]] if obb else [[80, 60, 120, 140, 0.95, 0]],
        dtype=np.float64,
    )
    actual = tracker.update(rows)
    assert actual.shape == (1, 9 if obb else 8)
    assert actual[0, 5 if obb else 4] == 0
    tracker.close()


@pytest.mark.parametrize("name", TIMED_TRACKERS)
def test_public_factory_typed_timing_config_reaches_real_native_trackers(
    native_timing_build: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The public typed timing configuration reaches each native Kalman tracker."""
    _, library_name, options = TRACKERS[name]
    binding_module = importlib.import_module(f"boxmot.native.trackers.{name}")
    domain_module = importlib.import_module(f"boxmot.trackers.{name}.native")
    path = next((native_timing_build / "trackers" / name).rglob(native_common.library_filename(name)))
    library = getattr(binding_module, library_name)(path)
    monkeypatch.setattr(domain_module, f"get_{name}_library", lambda: library)
    tracker = create_tracker(name, backend="cpp", options=options, kalman=KalmanConfig(variable_dt=True))
    assert tracker.variable_dt and tracker.requirements.timestamp
    for timestamp in (0.0, 0.1):
        result = tracker.update(np.empty((0, 6), dtype=np.float64), timestamp_s=timestamp)
        assert result.shape == (0, 8)
    tracker.close()
