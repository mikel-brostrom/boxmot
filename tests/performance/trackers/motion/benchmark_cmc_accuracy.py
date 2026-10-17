"""Controlled tracking accuracy study for fast, intermediate and historical CMC.

Run as a module from the repository root. Uses an existing MOT17 materialization,
never runs detector/ReID inference, and leaves production defaults unchanged.
CMC latency is measured serially before tracker workers start. All trackers then
consume identical cached warps and detections for each profile. CMC profiles
receive identical mask input boxes; their masking implementations can differ.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.dataset as pads
import torch
from threadpoolctl import threadpool_limits

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.engine.calibration.cmc import _bgr_image
from boxmot.engine.eval import motmetrics as mm
from boxmot.engine.eval.replay import _write_rows, tracks_to_mot_rows
from boxmot.engine.ui.logging import suppress_boxmot_logs
from boxmot.trackers import TrackerSpec, create_tracker
from boxmot.trackers.common.motion.cmc.base import BaseCMC
from boxmot.utils import ROOT
from tests.performance.trackers.motion.cmc_study_profiles import StudyProfile, build_profiles

DEFAULT_BUILD = ROOT / "runs/materializations/1e76b6d339353a1d38db5a177c5f32d5a835074a294eb5144839b135565bd8c2"
MOVING_SEQUENCES = tuple(f"MOT17-{number}-FRCNN" for number in ("05", "10", "11", "13"))
METRICS = ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDSW", "IDs", "Frames", "GT_Dets")


def _write_json(path: Path, value: Any) -> None:
    """Save reviewable artifacts, including NumPy metric scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n"
    )


def _latency(values: np.ndarray) -> dict[str, float]:
    """Summarize measured apply calls in milliseconds."""
    return {
        name: float(value)
        for name, value in {
            "mean_ms": values.mean(),
            "median_ms": np.median(values),
            "p95_ms": np.percentile(values, 95),
            "max_ms": values.max(),
        }.items()
    }


def _mask_boxes(sample: Any, min_conf: float) -> np.ndarray:
    """Use one fixed background mask independent of tracker-specific filtering."""
    detections = sample.detections
    selected = (detections.scores >= min_conf) & (detections.class_ids == 1)
    return detections.geometry.values[selected].numpy()


class CachedWarp:
    """Serve exactly one precomputed warp for every tracker update."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.index = 0

    def reset(self) -> None:
        """Restart sequence-local replay."""
        self.index = 0

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        """Ignore placeholder pixels and return the shared camera estimate."""
        del img, dets
        if self.index >= len(self.values):
            raise RuntimeError("Tracker requested more camera transforms than available frames.")
        value = self.values[self.index].copy()
        self.index += 1
        return value


def _compute_warps(args: argparse.Namespace, profiles: dict[str, StudyProfile], sequences: dict[str, int]) -> dict:
    """Decode each source image once and time independently stateful estimators."""
    results = {name: {"metadata": profile.metadata, "per_sequence": {}} for name, profile in profiles.items()}
    all_timings: dict[str, list[np.ndarray]] = {name: [] for name in profiles}
    for sequence, count in sequences.items():
        data = CachedVisionDataset._for_sequence(args.build, sequence_id=sequence, split="ablation")
        estimators = {name: profile.create() for name, profile in profiles.items()}
        warps = {name: np.empty((count, 2, 3), dtype=np.float32) for name in profiles}
        times = {name: np.zeros(count, dtype=np.float64) for name in profiles}
        names = list(profiles)
        for index, sample in enumerate(data):
            if sample.frame_index != index:
                raise ValueError("Study requires contiguous zero-based frame indices.")
            image = _bgr_image(sample, data._image_root)
            boxes = _mask_boxes(sample, args.mask_min_conf)
            # Rotate first execution position to balance shared-image cache effects.
            order = names[index % len(names) :] + names[: index % len(names)]
            for name in order:
                estimator = estimators[name]
                if estimator is None:
                    value = np.eye(2, 3, dtype=np.float32)
                else:
                    cv2.setRNGSeed(0)
                    started = time.perf_counter_ns()
                    value = estimator.apply(image, boxes)
                    times[name][index] = (time.perf_counter_ns() - started) / 1e6
                if not BaseCMC.is_valid_transform(value):
                    raise RuntimeError(f"Invalid {name} warp at {sequence}/{index}.")
                warps[name][index] = value
            if (index + 1) % 100 == 0:
                print(f"CMC {sequence}: {index + 1}/{count} frames", flush=True)
        for name in names:
            directory = args.output / "warps" / name
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / f"{sequence}.npy", warps[name])
            np.save(directory / f"{sequence}-latency.npy", times[name])
            measured = times[name][min(args.warmup, count - 1) :]
            all_timings[name].append(measured)
            results[name]["per_sequence"][sequence] = {
                "frames": count,
                "timing_frames": len(measured),
                "timing": _latency(measured),
                "identity_fraction": float(np.isclose(warps[name], np.eye(2, 3), atol=1e-6).all(axis=(1, 2)).mean()),
                "warp_sha256": hashlib.sha256(warps[name].tobytes()).hexdigest(),
            }
        print(f"CMC {sequence}: complete", flush=True)
        _write_json(args.output / "cmc-progress.json", results)
    for name in names:
        results[name]["timing"] = _latency(np.concatenate(all_timings[name]))
    _write_json(args.output / "cmc_profiles.json", results)
    return results


def _tracker(name: str, profile: str, mask_min_conf: float) -> Any:
    """Freeze built-in tracker/Kalman defaults, changing only CMC selection."""
    options = {
        "use_cmc": profile != "off",
        "cmc_method": "sof" if profile == "off" else profile.split("-")[0],
        "cmc_min_conf": mask_min_conf,
    }
    return create_tracker(
        TrackerSpec(name=name, class_ids=(1,), class_names=((1, "pedestrian"),), options=tuple(sorted(options.items())))
    )


def _replay_sequence(task: dict[str, Any]) -> tuple[str, list[dict], dict]:
    """Replay all fixed tracker/profile combinations over one cached sequence."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    sequence = task["sequence"]
    output, build = Path(task["output"]), Path(task["build"])
    with threadpool_limits(limits=1), suppress_boxmot_logs(True, level="CRITICAL"):
        stream = CachedVisionDataset._stream_sequence(
            build, sequence_id=sequence, split="ablation", load_embeddings=True
        )
        samples = list(stream)
        placeholders = {sample.image_size: None for sample in samples}
        placeholders = {shape: np.zeros((*shape, 3), dtype=np.uint8) for shape in placeholders}
        rows, bundles = [], {}
        for tracker_name in task["trackers"]:
            for profile in task["profiles"]:
                cv2.setRNGSeed(0)
                tracker = _tracker(tracker_name, profile, task["mask_min_conf"])
                detections = [
                    sample.detections
                    if tracker.requirements.embeddings
                    else replace(sample.detections, embeddings=None)
                    for sample in samples
                ]
                cached = None
                if profile != "off":
                    cached = CachedWarp(np.load(output / "warps" / profile / f"{sequence}.npy"))
                    tracker.cmc = cached
                directory = output / "tracks" / tracker_name / profile
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{sequence}.txt"
                started = time.perf_counter()
                with path.open("w") as handle:
                    for sample, detection in zip(samples, detections):
                        tracks = tracker.update(
                            detection, frame=placeholders[sample.image_size], timestamp_s=sample.timestamp_s
                        )
                        _write_rows(handle, tracks_to_mot_rows(tracks, sample.frame_index))
                if cached is not None and cached.index != len(samples):
                    raise RuntimeError(f"{tracker_name} consumed {cached.index}/{len(samples)} camera transforms.")
                elapsed = time.perf_counter() - started
                _name, class_bundles = mm._evaluate_aabb_sequence_task(
                    mm.AABBSequenceEvaluationTask(
                        seq_name=sequence,
                        gt_path=Path(task["gt_root"]) / sequence / "gt/gt.txt",
                        tracker_path=path,
                        class_pairs=(("pedestrian", 1),),
                        num_timesteps=len(samples),
                        distractor_ids=frozenset({2, 7, 8, 12}),
                        cache_root=None,
                    )
                )
                bundle = class_bundles["pedestrian"]
                bundles[(tracker_name, profile)] = bundle
                metrics = mm._summary_from_bundle(bundle)
                rows.append(
                    {
                        "tracker": tracker_name,
                        "profile": profile,
                        "sequence": sequence,
                        "metrics": metrics,
                        "replay_wall_s": elapsed,
                        "track_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                if sequence == task["first_sequence"]:
                    _write_json(
                        output / "configs" / f"{tracker_name}-{profile}.json",
                        {
                            "algorithm": asdict(tracker.config),
                            "kalman": asdict(tracker.kalman_config),
                            "shared_mask_min_conf": task["mask_min_conf"],
                        },
                    )
        _write_json(output / "sequence-results" / f"{sequence}.json", rows)
        return sequence, rows, bundles


def main(argv: list[str] | None = None) -> int:
    """Run an explicit local ablation with separate timing and replay stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--gt-root", type=Path, default=ROOT / "datasets/mot/MOT17/ablation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trackers", nargs="+", choices=("bytetrack", "botsort", "ocsort"), default=["bytetrack", "botsort", "ocsort"]
    )
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mask-min-conf", type=float, default=0.1)
    parser.add_argument("--stage", choices=("all", "warps", "replay", "analyze"), default="all")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.warmup < 0 or not 0 <= args.mask_min_conf <= 1:
        parser.error("workers must be positive; warmup nonnegative; mask confidence in [0,1].")
    args.build, args.output, args.gt_root = args.build.resolve(), args.output.resolve(), args.gt_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "analyze":
        from tests.performance.trackers.motion.cmc_study_analysis import analyze

        analyze(args.output)
        return 0
    manifest = DatasetManifest.load(args.build)
    if manifest.box_type != "aabb" or manifest.metadata.get("dataset_id") != "mot17":
        parser.error("This study evaluates an AABB MOT17 materialization.")
    if args.stage == "replay":
        # Validate current settings without overwriting the measured source archive.
        with tempfile.TemporaryDirectory(prefix="boxmot-cmc-replay-") as temporary:
            profiles = build_profiles(Path(temporary))
    else:
        profiles = build_profiles(args.output / "sources")
    if args.profiles:
        unknown = set(args.profiles) - profiles.keys()
        if unknown:
            parser.error(f"Unknown profiles: {sorted(unknown)}")
        profiles = {name: profiles[name] for name in args.profiles}
    sample_table = pads.dataset(args.build / "samples", format="parquet").to_table(columns=["sequence_id"])
    names, counts = np.unique(sample_table["sequence_id"].to_numpy(), return_counts=True)
    sequences = {str(name): int(count) for name, count in zip(names, counts)}
    if args.sequences:
        sequences = {name: sequences[name] for name in args.sequences}
    environment = {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv_threads": 1,
        "replay_workers": args.workers,
        "cpu": subprocess.check_output(["lscpu"], text=True),
    }
    provenance = {
        "build": str(args.build),
        "gt_root": str(args.gt_root),
        "build_id": manifest.build_id,
        "manifest_sha256": hashlib.sha256((args.build / "manifest.json").read_bytes()).hexdigest(),
        "data_components": manifest.metadata.get("components", {}),
        "data_counts": manifest.counts,
        "gt_sha256": {
            sequence: hashlib.sha256((args.gt_root / sequence / "gt/gt.txt").read_bytes()).hexdigest()
            for sequence in sequences
        },
        "sequences": sequences,
        "moving_sequences": sorted(set(sequences) & set(MOVING_SEQUENCES)),
        "motion_strata_source": "Pre-existing GT-free SOF camera analysis; fixed across variants.",
        "mask_min_conf": args.mask_min_conf,
        "warmup_frames_excluded_from_latency_per_sequence": args.warmup,
        "profile_names": list(profiles),
        "trackers": args.trackers,
        "practical_hota_threshold_pp": 1.0,
        "opencv_threads": 1,
        "rng_seed_before_each_apply": 0,
        "opencv_version": cv2.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": environment,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mask_policy": "Identical input boxes: class-1 cached detections >= fixed confidence. Mask code may differ.",
        "replay_policy": "CMC always on for active profiles; off control. No automatic gate or refitting.",
        "latency_scope": "Serial CMC.apply includes preprocessing, masking and state; excludes decode and tracking.",
        "limits": [
            "Seven MOT17 tuning sequences, not held-out validation.",
            "Tracker defaults frozen; no per-profile tuning or KF recalibration.",
            "Replay reuses cached warps and placeholder frames; replay wall time is not live tracker FPS.",
        ],
    }
    old_path = args.output / "study.json"
    if args.stage == "replay":
        previous = json.loads(old_path.read_text())
        for key in ("build_id", "profile_names", "sequences", "mask_min_conf"):
            if previous[key] != provenance[key]:
                raise ValueError(f"Replay {key} does not match computed camera warps.")
        measured_profiles = json.loads((args.output / "cmc_profiles.json").read_text())
        for name, profile in profiles.items():
            if measured_profiles[name]["metadata"] != profile.metadata:
                raise ValueError(f"Replay {name} source/settings differ from computed camera warps.")
        provenance = {**previous, "trackers": args.trackers, "replay_environment": environment}
    _write_json(old_path, provenance)
    shutil.copyfile(args.build / "manifest.json", args.output / "data-manifest.json")
    patch_name = "replay-working-tree.patch" if args.stage == "replay" else "working-tree.patch"
    (args.output / patch_name).write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", "boxmot"]))
    source_directory = args.output / "study_sources"
    source_directory.mkdir(exist_ok=True)
    for name in ("benchmark_cmc_accuracy.py", "cmc_study_profiles.py", "cmc_study_analysis.py"):
        shutil.copyfile(Path(__file__).with_name(name), source_directory / name)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), suppress_boxmot_logs(True, level="CRITICAL"):
        cmc_profiles = (
            _compute_warps(args, profiles, sequences)
            if args.stage in ("all", "warps")
            else json.loads((args.output / "cmc_profiles.json").read_text())
        )
    if args.stage == "warps":
        return 0
    tasks = [
        dict(
            sequence=sequence,
            first_sequence=next(iter(sequences)),
            build=str(args.build),
            output=str(args.output),
            gt_root=str(args.gt_root),
            trackers=args.trackers,
            profiles=list(profiles),
            mask_min_conf=args.mask_min_conf,
        )
        for sequence in sequences
    ]
    all_bundles: dict[tuple[str, str], dict] = {
        (tracker, profile): {} for tracker in args.trackers for profile in profiles
    }
    sequence_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_replay_sequence, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            sequence, rows, bundles = future.result()
            sequence_rows.extend(rows)
            for key, bundle in bundles.items():
                all_bundles[key][sequence] = bundle
            print(f"Replay {sequence}: all {len(rows)} tracker/profile combinations complete", flush=True)
    runs = [
        {
            "tracker": tracker,
            "profile": profile,
            "metrics": mm._summary_from_bundle(mm._combine_bundles(bundles)),
            "per_sequence": {sequence: mm._summary_from_bundle(bundles[sequence]) for sequence in sorted(bundles)},
        }
        for (tracker, profile), bundles in all_bundles.items()
    ]
    tracker_settings = {
        path.stem: json.loads(path.read_text()) for path in sorted((args.output / "configs").glob("*.json"))
    }
    _write_json(
        args.output / "summary.json",
        {**provenance, "tracker_settings": tracker_settings, "cmc_profiles": cmc_profiles, "runs": runs},
    )
    _write_json(args.output / "replay-details.json", sequence_rows)
    with (args.output / "metric_bundles.pkl").open("wb") as handle:
        pickle.dump(all_bundles, handle)
    for run in runs:
        print(run["tracker"], run["profile"], {key: round(run["metrics"][key], 3) for key in METRICS}, flush=True)
    if not args.skip_analysis:
        from tests.performance.trackers.motion.cmc_study_analysis import analyze

        analyze(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
