"""Compare cached tracker replay against an independent package snapshot.

Run from the repository root with ``python -m
tests.performance.trackers.benchmark_cached_cmc --help``. No perception models
are run: both versions consume the same explicit materialized build. The
preloaded mode excludes decoding; replay mode includes sequence workers and I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _write_json(path: Path, value: Any) -> None:
    """Write benchmark metadata after the measured work has finished."""
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _record_worker(progress_queue: Any) -> None:
    """Record source provenance without changing normal worker thread policy."""
    import cv2
    import torch

    import boxmot
    from boxmot.engine.eval.replay import _initialize_replay_worker

    _initialize_replay_worker(progress_queue)
    _write_json(
        Path(os.environ["BOXMOT_CMC_BENCHMARK_WORKERS"]) / f"{os.getpid()}.json",
        {
            "boxmot": boxmot.__file__,
            "torch_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
        },
    )


def _child(options: argparse.Namespace) -> None:
    """Import the selected package before setup or spawned worker creation."""
    sys.path.insert(0, str(options.code_root))
    import cv2
    import torch

    import boxmot
    from boxmot.datasets import CachedVisionDataset
    from boxmot.engine.config.runtime import build_mode_namespace
    from boxmot.engine.eval import catalog_cache, evaluator, replay
    from boxmot.engine.materialization import metadata_cache
    from boxmot.engine.ui.logging import suppress_boxmot_logs
    from boxmot.pipelines import PipelineOutputs, TrackingPipeline
    from boxmot.trackers import create_tracker

    if Path(boxmot.__file__).resolve().parent != options.code_root / "boxmot":
        raise RuntimeError(f"Loaded the wrong package: {boxmot.__file__}")
    output = options.output / options.label
    output.mkdir()
    metadata_root = output / "metadata-cache"
    metadata_root.mkdir()
    workers = output / "workers"
    workers.mkdir()
    os.environ["BOXMOT_CMC_BENCHMARK_WORKERS"] = str(workers)
    replay._initialize_replay_worker = _record_worker
    payload = {
        "experiment": options.experiment,
        "build": str(options.build),
        "tracker": options.tracker,
        "tracker_backend": "python",
        "device": options.device,
        "split": options.split,
        "sequence_workers": options.workers,
        "sequence_names": (),
        "project": output,
        "name": options.tracker,
        "show": False,
        "save": False,
        "eval_masks": False,
        "calibrate_kf": False,
        "compare_trackeval": False,
        "verbose": False,
    }
    args = build_mode_namespace("eval", payload, explicit_keys=payload.keys())
    with (
        patch.object(catalog_cache, "user_cache_path", return_value=metadata_root),
        patch.object(metadata_cache, "user_cache_path", return_value=metadata_root),
        suppress_boxmot_logs(True, level="WARNING"),
    ):
        started = time.perf_counter()
        evaluator.eval_setup(args)
        setup_s = time.perf_counter() - started
        spec = evaluator._tracker_spec(args)
        if options.mode == "preloaded":
            dataset = CachedVisionDataset._stream_sequence(
                options.build,
                sequence_id=options.sequence,
                split=args.split,
                load_images=True,
                load_embeddings=True,
            )
            samples = list(itertools.islice(dataset, options.frames))
            if len(samples) != options.frames:
                raise ValueError(f"Requested {options.frames} frames, got {len(samples)}")
            pipeline = TrackingPipeline(
                detector=None,
                tracker=create_tracker(spec),
                outputs=PipelineOutputs(embeddings=True),
            )
            tracks = []
            started = time.perf_counter()
            for sample in samples:
                tracks.append(pipeline.step_detections(sample.frame, sample.detections).tracks)
            elapsed_s = time.perf_counter() - started
            sequence_file = output / f"{options.sequence}.txt"
            with sequence_file.open("w") as handle:
                for sample, result in zip(samples, tracks):
                    replay._write_rows(handle, replay.tracks_to_mot_rows(result, sample.frame_index))
            sequence_files = [sequence_file]
            frames = len(samples)
            metrics, metrics_s = None, None
        else:
            started = time.perf_counter()
            result = evaluator.replay_build(
                args.build_path,
                spec,
                split=args.split,
                output_dir=output / "tracks",
                sequence_ids=args.sequence_names,
                sequence_frame_counts=args.sequence_frame_counts,
                workers=args.sequence_workers,
            )
            elapsed_s = time.perf_counter() - started
            sequence_files, frames = result.sequence_files, result.frames
            args.exp_dir = result.output_dir
            started = time.perf_counter()
            metrics = evaluator.run_motmetrics(args, verbose=False)
            metrics_s = time.perf_counter() - started
    worker_evidence = [json.loads(path.read_text()) for path in workers.glob("*.json")]
    expected_workers = min(options.workers, len(sequence_files)) if options.mode == "replay" else 0
    # Single-worker replay executes locally and does not create a pool.
    if expected_workers <= 1:
        expected_workers = 0
    if len(worker_evidence) != expected_workers:
        raise AssertionError(f"Expected {expected_workers} workers, got {len(worker_evidence)}")
    for worker in worker_evidence:
        if Path(worker["boxmot"]).resolve().parent != options.code_root / "boxmot":
            raise AssertionError(f"Worker imported the wrong source: {worker}")
    _write_json(
        output / "result.json",
        {
            "label": options.label,
            "boxmot": boxmot.__file__,
            "elapsed_s": elapsed_s,
            "setup_s": setup_s,
            "metrics_s": metrics_s,
            "frames": frames,
            "metrics": metrics,
            "sequence_files": {path.name: str(path) for path in sequence_files},
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sequence_files},
            "worker_evidence": worker_evidence,
            "torch_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
        },
    )


def _check_outputs(reference: dict, candidate: dict) -> bool:
    """Check IDs and metrics exactly, allowing only tiny coordinate differences."""
    import numpy as np

    if reference["metrics"] != candidate["metrics"] or reference["frames"] != candidate["frames"]:
        raise AssertionError(f"Evaluation results changed in {candidate['label']}")
    if reference["sha256"].keys() != candidate["sha256"].keys():
        raise AssertionError("Sequence files changed")
    exact = reference["sha256"] == candidate["sha256"]
    for name, digest in reference["sha256"].items():
        if digest == candidate["sha256"][name]:
            continue
        before = np.loadtxt(reference["sequence_files"][name], delimiter=",", ndmin=2)
        after = np.loadtxt(candidate["sequence_files"][name], delimiter=",", ndmin=2)
        np.testing.assert_array_equal(before[:, :2], after[:, :2], err_msg=name)
        np.testing.assert_allclose(before[:, 2:6], after[:, 2:6], rtol=1e-6, atol=1e-4, err_msg=name)
        np.testing.assert_array_equal(before[:, 6:], after[:, 6:], err_msg=name)
    return exact


def main() -> None:
    """Alternate implementations and retain all runs, including the warmups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Directory containing the before boxmot package")
    parser.add_argument("--build", type=Path, required=True, help="Existing materialization; never regenerated")
    parser.add_argument("--experiment", required=True, help="Catalog reference, e.g. mot17/ablation-yolox-lmbn.yaml")
    parser.add_argument("--split", default="ablation")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("preloaded", "replay"), default="replay")
    parser.add_argument("--sequence", default="MOT17-04-FRCNN", help="Preloaded mode sequence")
    parser.add_argument("--frames", type=int, default=100, help="Preloaded mode frame count")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("baseline", "build", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if min(args.repeat, args.workers, args.frames) < 1:
        parser.error("Repeat, worker, and frame counts must be positive")
    if args.child:
        args.code_root = args.code_root.resolve()
        _child(args)
        return
    repository = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    rows, reference = [], None
    all_exact = True
    for repetition in range(args.repeat + 1):
        versions = [("before", args.baseline), ("after", repository)]
        if repetition % 2:
            versions.reverse()
        for version, code_root in versions:
            label = f"{version}-{repetition}"
            command = [sys.executable, "-m", "tests.performance.trackers.benchmark_cached_cmc"]
            for name in (
                "baseline",
                "build",
                "experiment",
                "split",
                "tracker",
                "device",
                "mode",
                "sequence",
                "frames",
                "workers",
                "output",
            ):
                command.extend([f"--{name}", str(getattr(args, name))])
            command.extend(["--child", "--code-root", str(code_root), "--label", label])
            print(f"Starting {args.mode} {label}", flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((args.output / label / "result.json").read_text())
            row.update(version=version, warmup=repetition == 0, command=command)
            reference = row if reference is None else reference
            all_exact = _check_outputs(reference, row) and all_exact
            rows.append(row)
            _write_json(args.output / "runs.json", rows)
            print(f"Finished {label}: {row['elapsed_s']:.4f}s", flush=True)
    timings = {
        version: [row["elapsed_s"] for row in rows if row["version"] == version and not row["warmup"]]
        for version in ("before", "after")
    }
    medians = {version: statistics.median(values) for version, values in timings.items()}
    summary = {
        "mode": args.mode,
        "tracker": args.tracker,
        "frames": reference["frames"],
        "times_s": timings,
        "medians_s": medians,
        "speedup": medians["before"] / medians["after"],
        "byte_identical_outputs": all_exact,
        "equal_metrics_and_track_ids": True,
        "repetitions": args.repeat,
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
