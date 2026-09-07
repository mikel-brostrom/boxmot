"""Benchmark the real evaluation CLI and optionally profile spawned replay workers.

Run as ``python -m tests.performance.benchmark_eval --help`` from the repository root.
Artifacts and profiling output stay in the selected output directory.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _profile_sequence(task: Any) -> Any:
    """Profile the complete worker, including lazy dataset iteration."""
    from boxmot.engine.eval import replay

    profiler = cProfile.Profile()
    try:
        return profiler.runcall(replay._benchmark_original_sequence_task, task)
    finally:
        profiler.dump_stats(str(Path(os.environ["BOXMOT_BENCHMARK_PROFILE"]) / f"{task.sequence_id}.prof"))


def _install_worker_profile() -> None:
    """Install the same pickle-safe wrapper in the parent and spawn children."""
    from boxmot.engine.eval import replay

    replay._benchmark_original_sequence_task = replay._replay_sequence_task
    replay._replay_sequence_task = _profile_sequence


if os.environ.get("BOXMOT_BENCHMARK_PROFILE"):
    _install_worker_profile()


def _child(args: argparse.Namespace) -> None:
    """Time stage boundaries without changing the CLI or worker scheduling."""
    from boxmot.engine.cli import main as cli
    from boxmot.engine.eval import evaluator

    timings: dict[str, float] = {}
    captured: dict[str, Any] = {}

    def timed(name: str, function: Any) -> Any:
        def invoke(*positional: Any, **keywords: Any) -> Any:
            started = time.perf_counter()
            result = function(*positional, **keywords)
            timings[name] = time.perf_counter() - started
            if name == "replay":
                captured["replay"] = result
            elif name == "metrics":
                captured["metrics"] = result
            return result

        return invoke

    for name, attribute in (("setup", "eval_setup"), ("replay", "replay_build"), ("metrics", "run_motmetrics")):
        setattr(evaluator, attribute, timed(name, getattr(evaluator, attribute)))
    if args.reuse_build:
        from boxmot.engine.materialization import workflow

        workflow.main = timed("build_reuse", workflow.main)
    selection = (
        [
            "--dataset",
            "mot17",
            "--split",
            "ablation",
            "--detector",
            "yolox-x-mot17",
            "--reid",
            "lmbn-n-duke",
            "--device",
            "mps",
        ]
        if args.reuse_build
        else ["--experiment", args.experiment, "--build", args.build]
    )
    command = [
        "eval",
        *selection,
        "--tracker",
        args.tracker,
        "--tracker-backend",
        "python",
        "--n-threads",
        str(args.workers),
        "--project",
        str(args.output),
        "--name",
        args.label,
        "--exist-ok",
        "--show-timing",
    ]
    if args.sequence:
        command += ["--sequence", args.sequence]
    cli(command, standalone_mode=False)
    replay = captured["replay"]
    if replay.build.resolve() != Path(args.build).resolve():
        raise RuntimeError(f"Expected build {args.build}, replay selected {replay.build}.")
    payload = {
        "timings_s": timings,
        "frames": replay.frames,
        "track_rows": replay.track_rows,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in replay.sequence_files},
        "metrics": captured["metrics"],
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": sys.modules["torch"].__version__,
            "numpy": sys.modules["numpy"].__version__,
            "workers": args.workers,
            "thread_settings": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "build": str(replay.build.resolve()),
            "command": command,
        },
    }
    (args.output / f"{args.label}.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    """Measure fresh CLI invocations and retain raw timings and output checksums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--experiment", default="mot17/ablation-yolox-lmbn.yaml")
    parser.add_argument("--tracker", default="occluboost")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/replay-benchmark"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--sequence")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--reuse-build", action="store_true", help="Include the MOT17 component-selection/build-reuse workflow"
    )
    parser.add_argument(
        "--compare-to", type=Path, help="Require identical tracking files and metrics to a previous JSON result"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.workers < 1:
        parser.error("--repeat and --workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.child:
        _child(args)
        return
    rows = []
    for index in range(args.repeat):
        label = f"{args.label}-{index + 1}"
        command = [
            sys.executable,
            "-m",
            "tests.performance.benchmark_eval",
            "--child",
            "--build",
            args.build,
            "--experiment",
            args.experiment,
            "--tracker",
            args.tracker,
            "--workers",
            str(args.workers),
            "--output",
            str(args.output),
            "--label",
            label,
        ]
        if args.sequence:
            command += ["--sequence", args.sequence]
        if args.reuse_build:
            command += ["--reuse-build"]
        env = os.environ.copy()
        if args.profile:
            profile_dir = args.output / f"{label}-profiles"
            profile_dir.mkdir(exist_ok=True)
            env["BOXMOT_BENCHMARK_PROFILE"] = str(profile_dir.resolve())
        started = time.perf_counter()
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - started
        result_path = args.output / f"{label}.json"
        result = json.loads(result_path.read_text())
        result["timings_s"]["invocation"] = elapsed
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        rows.append(result)
        if args.compare_to:
            reference = json.loads(args.compare_to.read_text())
            if result["sha256"] != reference["sha256"] or result["metrics"] != reference["metrics"]:
                raise RuntimeError(f"Tracking files or metrics differ from {args.compare_to}.")
        print(label, json.dumps(result["timings_s"], sort_keys=True), flush=True)
    print(
        "median_s",
        json.dumps(
            {key: statistics.median(row["timings_s"][key] for row in rows) for key in rows[0]["timings_s"]},
            sort_keys=True,
        ),
    )
    if any(row["sha256"] != rows[0]["sha256"] or row["metrics"] != rows[0]["metrics"] for row in rows[1:]):
        raise RuntimeError("Repeated runs produced different tracking files or metrics.")


if __name__ == "__main__":
    main()
