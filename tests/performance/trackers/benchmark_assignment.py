"""Compare historical LAPJV, LAPJV without cost summation, and current LAPCPP.

Install the standalone comparison dependency with ``uv pip install 'lapx==0.11.1'``. Run from the
repository root with ``uv run --no-sync python -m
tests.performance.trackers.benchmark_assignment --help``. Inputs are preloaded
materialized AABB/OBB detections; optionally replay saved camera transforms. No images,
embeddings, detector inference, or HOTA evaluation are needed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import lap
import numpy as np
import pyarrow.parquet as pq
import yaml
from threadpoolctl import threadpool_limits

from boxmot.trackers.bytetrack.config import ByteTrackConfig
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.association import matching
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig


def previous_assignment(
    cost_matrix: np.ndarray, cost_limit: float | None = None, *, return_cost: bool = True
) -> np.ndarray:
    """Use the pre-3a2fdfbe LAPJV wrapper, optionally skipping unused cost summation."""
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    kwargs = {"extend_cost": True}
    finite = np.isfinite(cost_matrix)
    if not finite.any():
        return np.empty((0, 2), dtype=int)
    if not finite.all():
        cost_matrix = np.where(finite, cost_matrix, np.inf)
    if cost_limit is None and not finite.all():
        values = cost_matrix[finite]
        cost_limit = float(values.max() + np.ptp(values) + max(1.0, np.abs(values).max()))
    if cost_limit is not None:
        kwargs["cost_limit"] = cost_limit
    # The final two outputs are always row/column maps, with or without cost.
    row_to_col = lap.lapjv(cost_matrix, return_cost=return_cost, **kwargs)[-2]
    assigned_rows = np.flatnonzero(row_to_col >= 0)
    matches = np.column_stack((assigned_rows, row_to_col[assigned_rows])).astype(int, copy=False)
    return matches[finite[matches[:, 0], matches[:, 1]]]


class SavedCMC:
    """Supply the same precomputed previous-to-current transform on each pass."""

    def __init__(self, warps: list[np.ndarray]) -> None:
        self.warps = iter(warps)

    def apply(self, image: np.ndarray | None, detections: np.ndarray) -> np.ndarray:
        """Advance through the saved transforms without image processing."""
        return next(self.warps)


def load_sequences(options: argparse.Namespace) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
    """Load selected frames in chronological order and preserve detection indices."""
    samples = pq.read_table(options.build / "samples").to_pylist()
    grouped = defaultdict(list)
    for sample in samples:
        if sample["sequence_id"] in options.sequences:
            grouped[sample["sequence_id"]].append(sample)
    for name in options.sequences:
        if name not in grouped:
            raise ValueError(f"Sequence missing from build: {name}")
        grouped[name] = sorted(grouped[name], key=lambda sample: sample["frame_index"])[: options.max_frames]
    selected_ids = {sample["sample_id"] for group in grouped.values() for sample in group}
    box_type = json.loads((options.build / "manifest.json").read_text())["box_type"]
    geometry = {"aabb": ["x1", "y1", "x2", "y2"], "obb": ["cx", "cy", "w", "h", "angle"]}[box_type]
    columns = ["sample_id", "detection_index", *geometry, "score", "class_id"]
    table = pq.read_table(
        options.build / "instances",
        columns=columns,
        filters=[("sample_id", "in", sorted(selected_ids))],
    )
    detections = defaultdict(list)
    for row in table.to_pylist():
        detections[row["sample_id"]].append((row["detection_index"], [row[key] for key in columns[2:]]))
    report = None
    archive = None
    if options.cmc_report:
        report = json.loads(options.cmc_report.read_text())
        archive = np.load(options.cmc_report.parent / "warps.npz")
    result = {}
    try:
        for name in options.sequences:
            warp_by_frame = {}
            if report is not None:
                info = report["sequences"][name]
                keys = info["archive_keys"]
                warp_by_frame = dict(zip(archive[keys["frame_indices"]], archive[keys["warps"]], strict=True))
                if not info["enabled"]:
                    warp_by_frame = dict.fromkeys(warp_by_frame)
            result[name] = []
            for sample in grouped[name]:
                rows = sorted(detections[sample["sample_id"]], key=lambda item: item[0])
                packed = np.asarray([row for _, row in rows], dtype=np.float32).reshape(-1, len(geometry) + 2)
                warp = warp_by_frame[sample["frame_index"]] if archive is not None else None
                result[name].append((packed, warp))
    finally:
        if archive is not None:
            archive.close()
    return result


def replay(
    sequences: dict,
    algorithm: ByteTrackConfig,
    kalman: KalmanConfig,
    solver: Callable,
    *,
    capture: list | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Time only public update calls, collecting matrices only in untimed passes."""
    timings, digests = {}, {}

    def record(cost_matrix: np.ndarray, cost_limit: float | None = None) -> np.ndarray:
        capture.append((cost_matrix.copy(), cost_limit))
        return solver(cost_matrix, cost_limit=cost_limit)

    with patch.object(matching, "solve_assignment", solver if capture is None else record):
        for name, frames in sequences.items():
            tracker = ByteTrack(config=algorithm, kalman=kalman, is_obb=frames[0][0].shape[1] == 7)
            if frames[0][1] is not None:
                # Apply camera transforms in the normal tracking kernel; no pixels required.
                tracker.cmc = SavedCMC([warp for _, warp in frames])
            outputs = []
            gc.collect()
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                started = time.perf_counter()
                for detections, _ in frames:
                    outputs.append(tracker.update(detections))
                timings[name] = time.perf_counter() - started
            finally:
                if gc_enabled:
                    gc.enable()
            digest = hashlib.sha256()
            for output in outputs:
                digest.update(np.asarray(output.shape, dtype=np.int64).tobytes())
                digest.update(output.tobytes())
            digests[name] = digest.hexdigest()
            print(f"  {name}: {len(frames)} frames in {timings[name]:.3f}s", flush=True)
    return timings, digests


def time_assignment(matrices: list, solver: Callable, loops: int = 10) -> float:
    """Replay real association matrices without tracking or input collection."""
    started = time.perf_counter()
    for _ in range(loops):
        for costs, limit in matrices:
            solver(costs, cost_limit=limit)
    return (time.perf_counter() - started) / loops


def main() -> None:
    """Run warmup, rotated solver measurements, and exact-output validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Current ByteTrack algorithm fields plus kalman.")
    parser.add_argument("--cmc-report", type=Path, help="Saved analysis.json with adjacent warps.npz.")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.repeats < 1 or options.max_frames < 1:
        parser.error("--repeats and --max-frames must be positive.")
    settings = yaml.safe_load(options.config.read_text())
    kalman = KalmanConfig.from_mapping(settings.pop("kalman", {}))
    algorithm = ByteTrackConfig.from_mapping(settings)
    if algorithm.use_cmc:
        parser.error("Set use_cmc: false and supply --cmc-report to replay camera motion without pixels.")
    if kalman.variable_dt:
        parser.error("This fixed-step benchmark requires kalman.variable_dt: false.")
    sequences = load_sequences(options)
    solvers = {
        "lapjv": previous_assignment,
        "lapjv_no_cost": partial(previous_assignment, return_cost=False),
        "lapcpp": matching.solve_assignment,
    }
    names = list(solvers)
    orders = [names[repeat % len(names) :] + names[: repeat % len(names)] for repeat in range(options.repeats)]
    results = {
        "python": platform.python_version(),
        "lapx": importlib.metadata.version("lapx"),
        "lapcpp": importlib.metadata.version("lapcpp"),
        "numpy": np.__version__,
        "matching_source_sha256": hashlib.sha256(Path(matching.__file__).read_bytes()).hexdigest(),
        "build": str(options.build),
        "algorithm": algorithm.to_dict(),
        "kalman": kalman.to_dict(),
        "cmc_report": str(options.cmc_report) if options.cmc_report else None,
        "scope": "Preloaded ByteTrack.update, single numerical thread; no pixel processing.",
        "baseline": "Previous LAPJV wrapper from 3a2fdfbe^, plus the same wrapper with return_cost=False.",
        "sequences": {
            name: {"frames": len(frames), "detections": sum(len(dets) for dets, _ in frames)}
            for name, frames in sequences.items()
        },
        "runs": [],
        "output_digests": {},
        "assignment_runs": [],
    }
    matrices = []
    with threadpool_limits(limits=1):
        for name, solver in solvers.items():
            times, digests = replay(sequences, algorithm, kalman, solver, capture=matrices if name == "lapjv" else None)
            results["output_digests"][name] = digests
            print(f"Warmup {name}: {sum(times.values()):.3f}s", flush=True)
        for repeat, order in enumerate(orders):
            paired = {}
            for name in order:
                times, digests = replay(sequences, algorithm, kalman, solvers[name])
                if digests != results["output_digests"][name]:
                    raise RuntimeError(f"Nondeterministic outputs from {name}")
                results["runs"].append({"repeat": repeat, "solver": name, "seconds": times})
                paired[name] = sum(times.values())
            print(f"Round {repeat + 1}: {paired}", flush=True)
        for order in orders:
            results["assignment_runs"].append({name: time_assignment(matrices, solvers[name]) for name in order})
    results["outputs_identical"] = all(
        digests == results["output_digests"]["lapcpp"] for digests in results["output_digests"].values()
    )
    results["matrix_count"] = len(matrices)
    results["matrix_shape_percentiles"] = np.percentile(
        [cost.shape for cost, _ in matrices], [0, 50, 90, 100], axis=0
    ).tolist()
    results["all_ineligible_matrices"] = sum(
        not np.any(np.isfinite(cost) & (cost <= limit)) for cost, limit in matrices
    )
    results["median_tracker_seconds"] = {
        name: statistics.median(sum(run["seconds"].values()) for run in results["runs"] if run["solver"] == name)
        for name in solvers
    }
    results["median_assignment_seconds"] = {
        name: statistics.median(run[name] for run in results["assignment_runs"]) for name in solvers
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: results[key]
                for key in ("outputs_identical", "median_tracker_seconds", "median_assignment_seconds", "matrix_count")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
