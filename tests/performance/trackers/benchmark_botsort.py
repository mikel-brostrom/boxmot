"""Measure BoT-SORT with real cached inputs fully loaded before tracking.

Run from the repository root with ``python -m
tests.performance.trackers.benchmark_botsort --help``. Loading, output hashing,
component instrumentation, and optional cProfile runs are timed separately.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import itertools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from boxmot import BotSort
from boxmot.datasets import CachedVisionDataset
from boxmot.engine.eval.replay import tracks_to_mot_rows
from boxmot.engine.eval.workers import initialize_worker_threads
from boxmot.pipelines import PipelineOutputs, TrackingPipeline

_COMPONENTS = (
    "update", "_track_detections", "_frame_to_bgr", "_split_detections",
    "_create_detections", "_first_association", "_apply_camera_motion_compensation",
    "_first_association_cost", "_second_association", "_handle_unconfirmed_tracks",
    "_initialize_new_tracks", "_update_track_states", "_prepare_output",
)


def _pipeline() -> TrackingPipeline:
    """Construct fresh default BoT-SORT state for every replay."""
    return TrackingPipeline(
        detector=None, tracker=BotSort(), outputs=PipelineOutputs(embeddings=True),
    )


def _instrument(tracker: BotSort, timings: dict[str, float]) -> None:
    """Measure nested components without changing the uninstrumented runs."""
    for name in _COMPONENTS:
        original = getattr(tracker, name)

        def timed(*args: Any, _name=name, _original=original, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                timings[_name] += time.perf_counter() - started

        setattr(tracker, name, timed)


def _measure(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    """Load one bounded frame prefix, then replay it with independent trackers."""
    started = time.perf_counter()
    source = iter(CachedVisionDataset._stream_sequence(
        args.build, sequence_id=sequence, split=args.split,
        load_images=True, load_embeddings=True,
    ))
    try:
        samples = list(itertools.islice(source, args.frames))
    finally:
        source.close()
    preload_s = time.perf_counter() - started
    if len(samples) != args.frames:
        raise ValueError(f"Requested {args.frames} frames, got {len(samples)} for {sequence}.")

    runs, hashes = [], []
    for _ in range(args.repeat + 1):
        pipeline = _pipeline()
        results = []
        try:
            started = time.perf_counter()
            for sample in samples:
                results.append(pipeline.step_detections(sample.frame, sample.detections).tracks)
            runs.append(time.perf_counter() - started)
            rows = [tracks_to_mot_rows(result, sample.frame_index) for result, sample in zip(results, samples)]
            hashes.append(hashlib.sha256(repr(rows).encode()).hexdigest())
        finally:
            pipeline.reset()
    if len(set(hashes)) != 1:
        raise AssertionError(f"Repeated tracking outputs differ for {sequence}.")

    pipeline = _pipeline()
    timings: dict[str, float] = defaultdict(float)
    _instrument(pipeline.tracker, timings)
    try:
        started = time.perf_counter()
        for sample in samples:
            pipeline.step_detections(sample.frame, sample.detections)
        timings["pipeline_total"] = time.perf_counter() - started
    finally:
        pipeline.reset()

    if args.profile:
        pipeline = _pipeline()
        profiler = cProfile.Profile()
        try:
            with profiler:
                for sample in samples:
                    pipeline.step_detections(sample.frame, sample.detections)
            profiler.dump_stats(str(args.output / f"{sequence}-{args.label}.prof"))
        finally:
            pipeline.reset()

    return {
        "preload_s": preload_s, "warmup_s": runs[0], "runs_s": runs[1:],
        "median_s": statistics.median(runs[1:]), "frames": len(samples),
        "detections": sum(len(sample.detections) for sample in samples),
        "embedding_dim": samples[0].detections.embeddings.shape[1],
        "timings_s": dict(timings), "sha256": hashes[0],
    }


def main() -> None:
    """Measure sequentially with the same native thread budget as replay workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--split", default="ablation")
    parser.add_argument("--sequences", nargs="+", default=["MOT17-04-FRCNN", "MOT17-05-FRCNN"])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    if min(args.frames, args.repeat) < 1:
        parser.error("frames and repeat must be positive")
    args.build = args.build.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # This standalone benchmark owns its process; match a production worker.
    initialize_worker_threads()
    from threadpoolctl import threadpool_info

    reference = None if args.compare_to is None else json.loads(args.compare_to.read_text())
    report: dict[str, Any] = {
        "build": str(args.build), "native_pools": threadpool_info(), "sequences": {},
    }
    for sequence in args.sequences:
        result = _measure(args, sequence)
        if reference is not None and result["sha256"] != reference["sequences"][sequence]["sha256"]:
            raise AssertionError(f"Tracking output differs from baseline for {sequence}.")
        report["sequences"][sequence] = result
        (args.output / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(sequence, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
