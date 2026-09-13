"""Compare EagerMOT KITTI replay with an independent pre-change package snapshot.

Run from the repository root as ``python -m
tests.performance.trackers.benchmark_kitti_kalman --help``. Both versions use
fresh processes, the same saved detections/preset, and one numerical thread per
sequence worker. Warmups are retained for equality checks but excluded from
reported medians.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _record_worker(progress_queue: Any, log_level: int, logs_disabled: bool) -> None:
    """Keep the normal spawn initializer and record the code each worker loaded."""
    import torch
    from threadpoolctl import threadpool_info

    import boxmot
    from boxmot.engine.eval.eagermot_kitti import _initialize_kitti_worker
    from boxmot.trackers.eagermot.motion import Kalman3D

    _initialize_kitti_worker(progress_queue, log_level, logs_disabled)
    directory = Path(os.environ["BOXMOT_KITTI_BENCHMARK_WORKERS"])
    (directory / f"{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "boxmot": boxmot.__file__,
                "batch_motion": hasattr(Kalman3D, "multi_predict"),
                "torch_threads": torch.get_num_threads(),
                "native_pools": threadpool_info(),
            },
            indent=2,
        )
        + "\n"
    )


def _child(args: argparse.Namespace) -> None:
    """Select the package before any BoxMOT import, including spawned children."""
    sys.path.insert(0, str(args.code_root))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    import boxmot
    from boxmot.engine.eval import eagermot_kitti

    expected_package = args.code_root / "boxmot"
    if Path(boxmot.__file__).resolve().parent != expected_package:
        raise RuntimeError(f"Expected {expected_package}; loaded {boxmot.__file__}")
    output = args.output / args.label
    workers = output / "workers"
    workers.mkdir(parents=True)
    os.environ["BOXMOT_KITTI_BENCHMARK_WORKERS"] = str(workers)
    eagermot_kitti._initialize_kitti_worker = _record_worker
    torch.set_num_threads(1)
    workflow_args = SimpleNamespace(
        dataset=args.dataset,
        split="val",
        sequence_names=(),
        class_config=args.preset,
        sequence_workers=args.workers,
        project=output,
        show=False,
        save=False,
        show_3d=False,
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        result = eagermot_kitti.run_eagermot_kitti(workflow_args, show_progress=False)
    workflow_s = time.perf_counter() - started
    manifest = json.loads((result.exp_dir / "run.json").read_text())
    payload = {
        "label": args.label,
        "code_root": str(args.code_root),
        "loaded_package": boxmot.__file__,
        "workflow_s": workflow_s,
        "timings": result.timings,
        "sequence_workers": manifest["sequence_workers"],
        "sequences": manifest["sequences"],
        "output": str(result.exp_dir),
        "mots_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((result.exp_dir / "mots").glob("*.txt"))
        },
        "metrics": json.loads((result.exp_dir / "metrics.json").read_text()),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "thread_limits": {key: os.environ.get(key) for key in _THREAD_VARIABLES},
        },
        "source_sha256": {
            relative: hashlib.sha256((expected_package / relative).read_bytes()).hexdigest()
            for relative in ("trackers/eagermot/motion.py", "trackers/eagermot/tracker.py")
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    """Alternate snapshot/current runs and compare every output before summarizing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Directory containing the old boxmot package")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("baseline", "dataset", "preset", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.child:
        args.code_root = args.code_root.resolve()
        _child(args)
        return
    if args.repeat < 1 or args.workers < 1:
        parser.error("Repeat and worker counts must be positive.")
    repository = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, **dict.fromkeys(_THREAD_VARIABLES, "1")}
    rows = []
    reference = None
    source_versions = {}
    for repetition in range(args.repeat + 1):
        for version, code_root in (("baseline", args.baseline), ("batched", repository)):
            label = f"{version}-{'warmup' if repetition == 0 else repetition}"
            command = [
                sys.executable,
                "-m",
                "tests.performance.trackers.benchmark_kitti_kalman",
                "--child",
                "--baseline",
                str(args.baseline),
                "--dataset",
                str(args.dataset),
                "--preset",
                str(args.preset),
                "--output",
                str(args.output),
                "--workers",
                str(args.workers),
                "--code-root",
                str(code_root),
                "--label",
                label,
            ]
            print(f"Starting {label}", flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                started = time.perf_counter()
                subprocess.run(
                    command, cwd=repository, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
                )
                wall_s = time.perf_counter() - started
            row = json.loads((args.output / label / "result.json").read_text())
            row.update(version=version, warmup=repetition == 0, process_wall_s=wall_s, command=command)
            assert row["source_sha256"] == source_versions.setdefault(version, row["source_sha256"])
            evidence = [json.loads(path.read_text()) for path in (args.output / label / "workers").glob("*.json")]
            if len(evidence) != args.workers:
                raise AssertionError(f"Expected {args.workers} workers, got {len(evidence)} for {label}")
            for worker in evidence:
                assert Path(worker["boxmot"]).resolve().parent == code_root / "boxmot"
                assert worker["batch_motion"] == (version == "batched")
                assert worker["torch_threads"] == 1
                assert all(pool["num_threads"] == 1 for pool in worker["native_pools"])
            row["worker_evidence"] = evidence
            if reference is None:
                reference = row
            assert row["mots_sha256"] == reference["mots_sha256"], f"MOTS output differs: {label}"
            assert row["metrics"] == reference["metrics"], f"Metrics differ: {label}"
            assert row["sequences"] == reference["sequences"]
            assert row["sequence_workers"] == args.workers
            assert len(row["mots_sha256"]) == len(row["sequences"])
            rows.append(row)
            (args.output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                f"Finished {label}: wall={wall_s:.3f}s replay={row['timings']['totals_ms']['track'] / 1000:.3f}s",
                flush=True,
            )
    summary = {
        "frames": reference["timings"]["frames"],
        "sequences": reference["sequences"],
        "sequence_workers": args.workers,
        "repetitions": args.repeat,
        "all_outputs_and_metrics_equal": True,
        "medians_s": {},
    }
    for version in ("baseline", "batched"):
        measured = [row for row in rows if row["version"] == version and not row["warmup"]]
        medians = {"process_wall": statistics.median(row["process_wall_s"] for row in measured)}
        medians["workflow"] = statistics.median(row["workflow_s"] for row in measured)
        medians.update(
            {
                key: statistics.median(row["timings"]["totals_ms"][key] / 1000 for row in measured)
                for key in ("track", "eval", "total")
            }
        )
        summary["medians_s"][version] = medians
    summary["speedups"] = {
        key: summary["medians_s"]["baseline"][key] / summary["medians_s"]["batched"][key]
        for key in summary["medians_s"]["baseline"]
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
