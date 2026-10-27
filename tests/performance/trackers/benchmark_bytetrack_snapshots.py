"""Compare ByteTrack package snapshots on identical, preloaded materialized detections.

Install the standalone comparison dependency with ``uv pip install 'lapx==0.11.1'``,
then run from the repository root with ``uv run --no-sync python -m
tests.performance.trackers.benchmark_bytetrack_snapshots --help``. Settings,
sequence selection, and expected output hashes come from benchmark_assignment's
saved results. CMC is disabled. Process startup, loading, warmup, and hashing are
excluded from the replay timings.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, value: dict) -> None:
    """Save measurements outside the timed replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _child(options: argparse.Namespace) -> None:
    """Import the selected snapshot before importing any tracking helpers."""
    sys.path.insert(0, str(options.code_root))
    import torch
    from threadpoolctl import threadpool_limits

    import boxmot
    from boxmot.trackers.bytetrack.config import ByteTrackConfig
    from boxmot.trackers.common.association import matching
    from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
    from tests.performance.trackers.benchmark_assignment import load_sequences, replay

    if Path(boxmot.__file__).resolve().parent != options.code_root / "boxmot":
        raise RuntimeError(f"Imported the wrong package: {boxmot.__file__}")
    source = json.loads(options.reference_results.read_text())
    expected_outputs = source["output_digests"]
    if not expected_outputs or any(not value for value in expected_outputs.values()):
        raise ValueError("The reference must contain nonempty output digests for every solver.")
    algorithm = ByteTrackConfig.from_mapping(source["algorithm"])
    kalman = KalmanConfig.from_mapping(source["kalman"])
    if algorithm.use_cmc or source.get("cmc_report"):
        raise ValueError("The reference must disable CMC, including cached camera transforms.")
    if kalman.variable_dt:
        raise ValueError("This benchmark requires fixed-step Kalman timing.")
    sequences = load_sequences(
        argparse.Namespace(
            build=Path(source["build"]).resolve(),
            sequences=list(source["sequences"]),
            max_frames=max(value["frames"] for value in source["sequences"].values()),
            cmc_report=None,
        )
    )
    counts = {
        name: {"frames": len(frames), "detections": sum(len(dets) for dets, _ in frames)}
        for name, frames in sequences.items()
    }
    if counts != source["sequences"]:
        raise ValueError("Materialized frame/detection counts differ from the reference benchmark.")
    first_sequence = next(iter(sequences))
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        replay(
            {first_sequence: sequences[first_sequence][:100]},
            algorithm,
            kalman,
            matching.solve_assignment,
        )
        seconds, digests = replay(sequences, algorithm, kalman, matching.solve_assignment)
    frames = sum(value["frames"] for value in counts.values())
    total = sum(seconds.values())
    _write_json(
        options.output,
        {
            "boxmot": boxmot.__file__,
            "code_root": str(options.code_root),
            "python": platform.python_version(),
            "lapcpp": importlib.metadata.version("lapcpp"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "sequences": counts,
            "seconds": seconds,
            "total_seconds": total,
            "frames": frames,
            "fps": frames / total,
            "output_digests": digests,
            "matches_saved_outputs": all(digests == expected for expected in expected_outputs.values()),
        },
    )


def _summary(runs: list[dict], sequences: dict) -> dict:
    """Summarize medians while retaining paired timing differences separately."""
    frames = sum(value["frames"] for value in sequences.values())
    medians = {
        version: statistics.median(row["total_seconds"] for row in runs if row["version"] == version)
        for version in ("reference", "candidate")
    }
    by_sequence = {
        name: {
            version: statistics.median(row["seconds"][name] for row in runs if row["version"] == version)
            for version in medians
        }
        for name in sequences
    }
    return {
        "median_seconds": medians,
        "median_fps": {version: frames / seconds for version, seconds in medians.items()},
        "speedup": medians["reference"] / medians["candidate"],
        "runtime_reduction_percent": 100 * (1 - medians["candidate"] / medians["reference"]),
        "median_seconds_by_sequence": by_sequence,
        "outputs_identical": all(row["output_digests"] == runs[0]["output_digests"] for row in runs),
        "matches_saved_outputs": all(row["matches_saved_outputs"] for row in runs),
    }


def main() -> None:
    """Alternate independent snapshot processes and validate all output hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, help="Directory containing the reference boxmot package.")
    parser.add_argument("--candidate-root", type=Path, help="Directory containing the candidate boxmot package.")
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("runs/benchmarks/lapjvr/mot17-ablation.json"),
        help="benchmark_assignment JSON providing the materialization, settings, and expected outputs.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-root", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    for name in ("reference_root", "candidate_root", "reference_results", "output", "code_root"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.expanduser().resolve())
    if options.repeats < 1:
        parser.error("--repeats must be positive.")
    if options.child:
        if options.code_root is None:
            parser.error("--child requires --code-root.")
        _child(options)
        return
    if options.reference_root is None or options.candidate_root is None:
        parser.error("--reference-root and --candidate-root are required.")
    for root in (options.reference_root, options.candidate_root):
        if not (root / "boxmot" / "__init__.py").is_file():
            parser.error(f"No boxmot package found in {root}.")
    source = json.loads(options.reference_results.read_text())
    repository = Path(__file__).resolve().parents[3]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    results = {
        "reference_results": str(options.reference_results),
        "reference_root": str(options.reference_root),
        "candidate_root": str(options.candidate_root),
        "build": source["build"],
        "algorithm": source["algorithm"],
        "kalman": source["kalman"],
        "sequences": source["sequences"],
        "scope": (
            "Preloaded public ByteTrack.update; CMC off; one numerical thread; "
            "100-frame warmup per process; garbage collection disabled during replay."
        ),
        "runs": [],
        "paired_speedups": [],
    }
    with tempfile.TemporaryDirectory(prefix="boxmot-bytetrack-snapshots-") as temporary:
        for repeat in range(options.repeats):
            versions = ["reference", "candidate"]
            if repeat % 2:
                versions.reverse()
            paired = {}
            for version in versions:
                code_root = getattr(options, f"{version}_root")
                output = Path(temporary) / f"{version}-{repeat}.json"
                command = [
                    sys.executable,
                    "-m",
                    "tests.performance.trackers.benchmark_bytetrack_snapshots",
                    "--child",
                    "--code-root",
                    str(code_root),
                    "--reference-results",
                    str(options.reference_results),
                    "--output",
                    str(output),
                ]
                print(f"Pair {repeat + 1}/{options.repeats}: {version}", flush=True)
                subprocess.run(command, cwd=repository, env=environment, check=True)
                row = json.loads(output.read_text())
                row.update(version=version, repeat=repeat)
                results["runs"].append(row)
                paired[version] = row["total_seconds"]
                _write_json(options.output, results)
            results["paired_speedups"].append(paired["reference"] / paired["candidate"])
    results.update(_summary(results["runs"], results["sequences"]))
    _write_json(options.output, results)
    print(json.dumps({key: value for key, value in results.items() if key not in {"runs"}}, indent=2))
    if not results["outputs_identical"] or not results["matches_saved_outputs"]:
        raise AssertionError(f"Tracking outputs differ; inspect {options.output}.")


if __name__ == "__main__":
    main()
