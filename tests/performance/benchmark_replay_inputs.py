"""Compare direct, mapped and persistent replay against an existing build.

Run as ``python -m tests.performance.benchmark_replay_inputs --help``. This
never runs perception or edits source Parquet. A baseline package snapshot is
optional; each implementation is imported in an independent subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _worker_init(queue: object) -> None:
    """Record actual worker imports while preserving production thread limits."""
    import boxmot
    from boxmot.engine.eval.replay import _initialize_replay_worker

    _initialize_replay_worker(queue)
    _write(
        Path(os.environ["BOXMOT_INPUT_BENCHMARK_WORKERS"]) / f"{os.getpid()}.json",
        {"pid": os.getpid(), "package": boxmot.__file__},
    )


def _child(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.code_root))
    import boxmot
    from boxmot.engine.config.runtime import build_mode_namespace
    from boxmot.engine.eval import catalog_cache, evaluator, replay
    from boxmot.engine.materialization import metadata_cache
    from boxmot.engine.ui.logging import suppress_boxmot_logs

    if Path(boxmot.__file__).resolve().parent != args.code_root / "boxmot":
        raise AssertionError(f"Incorrect package imported: {boxmot.__file__}")
    output = args.output / args.label
    output.mkdir()
    metadata = output / "metadata"
    metadata.mkdir()
    worker_records = output / "workers"
    worker_records.mkdir()
    os.environ["BOXMOT_INPUT_BENCHMARK_WORKERS"] = str(worker_records)
    replay._initialize_replay_worker = _worker_init
    options = {
        "experiment": args.experiment,
        "build": str(args.build),
        "split": args.split,
        "tracker": args.tracker,
        "tracker_backend": "python",
        "device": args.device,
        "sequence_workers": args.workers,
        "sequence_names": (),
        "project": output,
        "name": args.tracker,
        "show": False,
        "save": False,
        "eval_masks": False,
        "calibrate_kf": False,
        "compare_trackeval": False,
        "verbose": False,
    }
    runtime = build_mode_namespace("eval", options, explicit_keys=options.keys())
    persistent = args.mode.startswith("persistent-")
    mapped = args.mode.endswith("mapped")
    prepare_times: list[float] = []
    cache_paths: set[Path] = set()
    cache_context = nullcontext()
    if mapped:
        from boxmot.datasets import replay_cache

        prepare = replay_cache.prepare_replay_sequence

        def timed_prepare(*positional: object, **keywords: object) -> Path:
            started = time.perf_counter()
            path = prepare(*positional, **keywords)
            prepare_times.append(time.perf_counter() - started)
            cache_paths.add(path)
            return path

        cache_context = patch.object(replay_cache, "prepare_replay_sequence", timed_prepare)
    with (
        patch.object(catalog_cache, "user_cache_path", return_value=metadata),
        patch.object(metadata_cache, "user_cache_path", return_value=metadata),
        suppress_boxmot_logs(True, level="WARNING"),
        cache_context,
    ):
        started = time.perf_counter()
        evaluator.eval_setup(runtime)
        setup_s = time.perf_counter() - started
        spec = evaluator._tracker_spec(runtime)
        context = nullcontext(None)
        if persistent:
            from boxmot.engine.eval.session import ReplaySession

            context = ReplaySession(runtime.sequence_workers, cache_inputs=mapped)
        rows = []
        with context as session:
            for trial in range(args.repeat + 1):
                prepare_times.clear()
                callbacks = {}
                if session is not None:
                    callbacks["session"] = session
                if mapped:
                    callbacks["cache_inputs"] = True
                started = time.perf_counter()
                result = replay.replay_build(
                    runtime.build_path,
                    spec,
                    split=runtime.split,
                    output_dir=output / f"tracks-{trial}",
                    sequence_ids=runtime.sequence_names,
                    sequence_frame_counts=runtime.sequence_frame_counts,
                    workers=runtime.sequence_workers,
                    **callbacks,
                )
                replay_s = time.perf_counter() - started
                runtime.exp_dir = result.output_dir
                started = time.perf_counter()
                with session.metric_execution() if session is not None else nullcontext():
                    metrics = evaluator.run_motmetrics(runtime, verbose=False)
                metrics_s = time.perf_counter() - started
                workers = [json.loads(path.read_text()) for path in worker_records.glob("*.json")]
                for worker in workers:
                    if Path(worker["package"]).resolve().parent != args.code_root / "boxmot":
                        raise AssertionError(f"Incorrect worker package: {worker}")
                rows.append(
                    {
                        "trial": trial,
                        "warmup": trial == 0,
                        "replay_s": replay_s,
                        "metrics_s": metrics_s,
                        "total_s": replay_s + metrics_s,
                        "cache_prepare_s": sum(prepare_times),
                        "cache_bytes": sum(
                            file.stat().st_size for path in cache_paths for file in path.rglob("*") if file.is_file()
                        ),
                        "frames": result.frames,
                        "metrics": metrics,
                        "worker_pids_seen": sorted(worker["pid"] for worker in workers),
                        "sha256": {path.name: _sha256(path) for path in result.sequence_files},
                    }
                )
                _write(output / "result.json", {"setup_s": setup_s, "runs": rows})
        if persistent and any(row["worker_pids_seen"] != rows[0]["worker_pids_seen"] for row in rows):
            raise AssertionError("Persistent session replaced workers between successful trials")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baseline", type=Path, help="Snapshot directory containing the old boxmot package")
    parser.add_argument("--split", default="ablation")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1, help="Reverse mode order on alternate rounds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("build", "baseline", "output", "code_root"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if min(args.repeat, args.workers, args.rounds) < 1:
        parser.error("repeat, workers and rounds must be positive")
    if args.child:
        _child(args)
        return
    repository = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    # Bind parity claims to immutable source bytes before and after all runs.
    source_files = sorted(path for path in args.build.rglob("*") if path.is_file())
    source_hashes = {str(path): _sha256(path) for path in source_files}
    modes = [
        ("indexed", repository),
        ("mapped", repository),
        ("persistent-indexed", repository),
        ("persistent-mapped", repository),
    ]
    if args.baseline is not None:
        modes.insert(0, ("baseline", args.baseline))
    rows = []
    reference = None
    for round_id in range(args.rounds):
        for mode, code_root in modes if round_id % 2 == 0 else reversed(modes):
            label = f"{mode}-{round_id}"
            command = [sys.executable, "-m", "tests.performance.benchmark_replay_inputs"]
            for name in ("build", "experiment", "split", "tracker", "device", "workers", "repeat", "output"):
                command.extend([f"--{name}", str(getattr(args, name))])
            command.extend(["--child", "--code-root", str(code_root), "--label", label, "--mode", mode])
            print(f"Running {label}", flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((args.output / label / "result.json").read_text())
            for row in result["runs"]:
                signature = {key: row[key] for key in ("frames", "metrics", "sha256")}
                if reference is None:
                    reference = signature
                if signature != reference:
                    raise AssertionError(f"Tracking bytes or metrics differ in {label}, trial {row['trial']}")
                rows.append({"mode": mode, "round": round_id, **row})
            _write(args.output / "runs.json", rows)
    for path in source_files:
        if _sha256(path) != source_hashes[str(path)]:
            raise AssertionError(f"Source build changed: {path}")
    medians = {
        mode: {
            timing: statistics.median(row[timing] for row in rows if row["mode"] == mode and not row["warmup"])
            for timing in ("replay_s", "metrics_s", "total_s")
        }
        for mode, _ in modes
    }
    _write(
        args.output / "summary.json", {"medians": medians, "exact_outputs_and_metrics": True, "source_unchanged": True}
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
