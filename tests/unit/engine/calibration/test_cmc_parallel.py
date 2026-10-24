"""Spawned camera analysis must preserve serial evidence and release its workers."""

from __future__ import annotations

import multiprocessing
import os
import threading
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.calibration import cmc
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint

_COUNTS = {"z-static": 4, "a-moving": 6}


@pytest.fixture
def camera_build(tmp_path: Path) -> Path:
    """Publish independent textured scenes, deliberately stored in reverse key order."""
    detect = StagePlan.create("detect", component={"id": "parallel-cmc-fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="parallel-cmc-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("parallel-cmc-fixture"),
        publish=PublishOptions(image_references=True),
        stages=(detect, finalize),
        metadata={"split": "validation"},
    )
    image_dir = plan.staging_root / "images"
    image_dir.mkdir(parents=True)
    height, width = 128, 192
    samples, instances = [], []
    for seed, (sequence, count) in enumerate(_COUNTS.items()):
        texture = np.random.default_rng(seed + 37).integers(20, 235, (height, width, 3), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
        for index in range(count):
            dx, dy = (3 * index, -index) if sequence == "a-moving" else (0, 0)
            sample_id = f"{sequence}-{index}"
            translation = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            image = cv2.warpAffine(texture, translation, (width, height), borderMode=cv2.BORDER_REFLECT_101)
            assert cv2.imwrite(str(image_dir / f"{sample_id}.png"), image)
            samples.append(
                dict(
                    sample_id=sample_id,
                    split="validation",
                    sequence_id=sequence,
                    frame_index=index,
                    timestamp_s=index / 30.0,
                    image_ref=f"images/{sample_id}.png",
                    height=height,
                    width=width,
                )
            )
            for class_id, (x, y) in enumerate(((40, 50), (120, 85))):
                instances.append(
                    dict(
                        sample_id=sample_id,
                        instance_id=f"{plan.build_id}:{sample_id}:{class_id}",
                        detection_index=class_id,
                        x1=float(x + dx - 10),
                        y1=float(y + dy - 12),
                        x2=float(x + dx + 10),
                        y2=float(y + dy + 12),
                        score=0.95 if class_id == 0 or index % 3 else 0.05,
                        class_id=class_id,
                    )
                )
    writer = ParquetShardWriter(plan.staging_root, box_type=plan.box_type)
    writer.write(SAMPLES_ARTIFACT, list(reversed(samples)), shard_index=0)
    writer.write(INSTANCES_ARTIFACT, list(reversed(instances)), shard_index=0)
    return finalize_build(plan)


def _args(build: Path, workers: int, *, sequences: tuple[str, ...] | None = None) -> Namespace:
    """Keep camera inputs equal while changing execution-only scheduling controls."""
    return Namespace(
        tracker="botsort",
        tracker_backend="python",
        geometry="aabb",
        build_path=build,
        split="validation",
        seq_info=_COUNTS.copy(),
        sequence_names=tuple(_COUNTS) if sequences is None else sequences,
        sequence_workers=workers,
        per_class=True,
        tracker_class_ids=(0, 1),
        tracker_class_names=(),
    )


def _sequence_arrays(result: cmc.PreparedCMCCandidates, sequence: str) -> dict[str, np.ndarray]:
    """Resolve archive names without coupling sequence identity to shard ordinal."""
    with np.load(result.warps_path, allow_pickle=False) as archive:
        return {name: archive[key].copy() for name, key in result.report["sequences"][sequence]["archive_keys"].items()}


def test_spawned_cmc_matches_serial_evidence_and_parent_progress(camera_build, tmp_path) -> None:
    """Execution order must not affect decisions, raw/runtime arrays, or publication."""
    parent = os.getpid(), threading.get_ident()
    events = []

    def progress(message: str) -> None:
        assert (os.getpid(), threading.get_ident()) == parent
        events.append(message)

    previous_threads = cv2.getNumThreads()
    serial = cmc._prepare_cmc_candidates_uncached(_args(camera_build, 1), {}, tmp_path / "serial")
    assert cv2.getNumThreads() == previous_threads
    parallel = cmc._prepare_cmc_candidates_uncached(_args(camera_build, 2), {}, tmp_path / "parallel", progress=progress)
    assert cv2.getNumThreads() == previous_threads
    assert parallel.policy == serial.policy
    assert parallel.policy.sequences == (("a-moving", True), ("z-static", False))
    assert tuple(parallel.report["sequences"]) == ("a-moving", "z-static")
    assert parallel.report == serial.report
    assert parallel.report_path.read_bytes() == serial.report_path.read_bytes()
    assert parallel.warps_path.read_bytes() == serial.warps_path.read_bytes()
    for sequence in sorted(_COUNTS):
        expected, actual = _sequence_arrays(serial, sequence), _sequence_arrays(parallel, sequence)
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(parallel.warps[sequence], serial.warps[sequence])
    assert events
    assert all(any(sequence in message for message in events) for sequence in _COUNTS)


def test_cmc_sequence_estimator_state_is_independent(camera_build, tmp_path) -> None:
    """A static sequence follows a moving sequence without inheriting its optical flow."""
    combined = cmc._prepare_cmc_candidates_uncached(_args(camera_build, 1), {}, tmp_path / "combined")
    for sequence in sorted(_COUNTS):
        isolated = cmc._prepare_cmc_candidates_uncached(_args(camera_build, 1, sequences=(sequence,)), {}, tmp_path / sequence)
        for name, expected in _sequence_arrays(combined, sequence).items():
            np.testing.assert_array_equal(_sequence_arrays(isolated, sequence)[name], expected)
        np.testing.assert_array_equal(combined.warps[sequence], isolated.warps[sequence])


def test_spawned_cmc_failure_releases_workers_without_publishing(camera_build, tmp_path) -> None:
    """A corrupt source image must abort the whole analysis and leave no child processes."""
    initial_children = {child.pid for child in multiprocessing.active_children()}
    (camera_build / "images" / "a-moving-1.png").write_bytes(b"not an image")
    output = tmp_path / "failed"
    previous_threads = cv2.getNumThreads()
    with pytest.raises((ValueError, RuntimeError), match="[Cc]amera|[Cc]annot decode|a-moving"):
        cmc._prepare_cmc_candidates_uncached(_args(camera_build, 2), {}, output)
    assert cv2.getNumThreads() == previous_threads
    assert not (output / "analysis.json").exists()
    assert not (output / "warps.npz").exists()
    assert {child.pid for child in multiprocessing.active_children()} <= initial_children


def test_cmc_worker_limit_is_bounded_and_serial_requests_stay_inline(camera_build, tmp_path, monkeypatch) -> None:
    """A large requested allocation cannot exceed the selected sequence count."""
    original = cmc.analyze_parallel
    scheduled = []

    def observe(tasks, *, workers, progress):
        scheduled.append((workers, len(tasks)))
        return original(tasks, workers=workers, progress=progress)

    monkeypatch.setattr(cmc, "analyze_parallel", observe)
    cmc._prepare_cmc_candidates_uncached(_args(camera_build, 64), {}, tmp_path / "bounded")
    assert scheduled == [(2, 2)]

    def unexpected(*args, **kwargs):
        pytest.fail("One requested worker or one selected sequence must stay inline.")

    monkeypatch.setattr(cmc, "analyze_parallel", unexpected)
    cmc._prepare_cmc_candidates_uncached(_args(camera_build, 1), {}, tmp_path / "one-worker")
    cmc._prepare_cmc_candidates_uncached(_args(camera_build, 64, sequences=("a-moving",)), {}, tmp_path / "one-sequence")


def _native_runtime_state() -> dict:
    """Read native-library settings from inside the initialized spawn worker."""
    import torch

    return {
        "pid": os.getpid(),
        "opencv": cv2.getNumThreads(),
        "torch": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "environment": tuple(
            os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "parent_watchers": sum(thread.name == "boxmot-parent-watcher" for thread in threading.enumerate()),
    }


def test_cmc_worker_initializer_limits_native_threads_without_changing_parent() -> None:
    """Use actual spawn initialization, including its parent-lifetime watcher."""
    from boxmot.engine.calibration.cmc_workers import _initialize_cmc_worker

    previous = _native_runtime_state()
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue()
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=_initialize_cmc_worker,
            initargs=(progress_queue,),
        ) as executor:
            actual = executor.submit(_native_runtime_state).result(timeout=30)
    finally:
        progress_queue.close()
        progress_queue.join_thread()
    assert actual["pid"] != previous["pid"]
    assert (actual["opencv"], actual["torch"], actual["interop"]) == (1, 1, 1)
    assert actual["environment"] == ("1", "1", "1")
    assert actual["parent_watchers"] == 1
    assert _native_runtime_state() == previous


@pytest.mark.parametrize("interrupt", [False, True])
def test_cmc_progress_failures_release_workers(camera_build, tmp_path, interrupt) -> None:
    """Ignore observational errors but propagate interruption and stop every worker."""
    initial_children = {child.pid for child in multiprocessing.active_children()}
    callbacks = []
    observed_workers = set()
    parent = os.getpid(), threading.get_ident()

    def progress(message: str) -> None:
        if not any(sequence in message for sequence in _COUNTS):
            # The startup banner precedes pool creation. Exercise cancellation
            # only once a sequence-specific worker update reaches the parent.
            return
        callbacks.append((os.getpid(), threading.get_ident()))
        observed_workers.update(child.pid for child in multiprocessing.active_children())
        if interrupt:
            raise KeyboardInterrupt("cancel camera analysis")
        raise ValueError("unavailable progress renderer")

    output = tmp_path / "progress-failure"
    if interrupt:
        with pytest.raises(KeyboardInterrupt, match="cancel camera analysis"):
            cmc._prepare_cmc_candidates_uncached(_args(camera_build, 2), {}, output, progress=progress)
        assert not (output / "analysis.json").exists()
        assert not (output / "warps.npz").exists()
    else:
        result = cmc._prepare_cmc_candidates_uncached(_args(camera_build, 2), {}, output, progress=progress)
        assert result.policy.sequences == (("a-moving", True), ("z-static", False))
        assert result.report_path.is_file() and result.warps_path.is_file()
    assert callbacks and set(callbacks) == {parent}
    assert observed_workers - initial_children
    assert {child.pid for child in multiprocessing.active_children()} <= initial_children
