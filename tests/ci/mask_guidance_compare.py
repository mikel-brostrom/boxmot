"""Compare ordinary and guided ByteTrack on the same bounded cached sequence.

Each variant runs in a fresh process. Results include MOT metrics, synchronized
tracker latency, and process/device memory; detector inference is excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from boxmot import ByteTrack
from boxmot.datasets import CachedVisionDataset
from boxmot.engine.eval.motmetrics import _build_aabb_sequence_data, _eval_bundle, _summary_from_bundle
from boxmot.engine.eval.provenance import write_mask_guidance_provenance
from boxmot.engine.eval.replay import tracks_to_mot_rows
from boxmot.segmentors.propagation.weights import resolve_edgetam_checkpoint
from boxmot.trackers import MaskGuidanceConfig, TrackerSpec
from boxmot.utils.devices import resolve_device
from tests.ci.mask_guidance_memory import _sample
from tests.ci.mask_guidance_smoke import _synchronize


def _run_variant(args: argparse.Namespace, guided: bool) -> None:
    """Stream one variant without retaining decoded frames or model outputs."""
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    name = "guided" if guided else "baseline"
    output = args.output / name
    output.mkdir(parents=True)
    dataset = CachedVisionDataset._stream_sequence(
        args.build, sequence_id=args.sequence, split=args.split, load_images=True,
    )
    config = MaskGuidanceConfig(args.checkpoint, device=str(device), max_objects=args.max_objects) if guided else None
    tracker = ByteTrack(mask_guidance=config)
    track_path, memory_path = output / f"{args.sequence}.txt", output / "memory.jsonl"
    frames, latencies = [], []
    _sample(memory_path, stage="before_sequence", frame=0, device=device)
    with track_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for index, sample in enumerate(islice(dataset, args.frames)):
            _synchronize(device)
            started = perf_counter()
            tracks = tracker.update(sample.detections, sample.frame)
            _synchronize(device)
            latencies.append(1000 * (perf_counter() - started))
            frames.append(sample.frame_index + 1)
            writer.writerows(tracks_to_mot_rows(tracks, sample.frame_index))
            _sample(memory_path, stage="frame", frame=index + 1, device=device, tracker=tracker if guided else None)
            if (index + 1) % 10 == 0:
                print(f"{name}: {index + 1}/{args.frames}", flush=True)
    if not frames:
        raise ValueError("No frames selected")
    gt = np.loadtxt(args.ground_truth, delimiter=",", ndmin=2)
    gt_path = output / "selected-gt.txt"
    np.savetxt(gt_path, gt[np.isin(gt[:, 0], frames)], delimiter=",", fmt="%g")
    data = _build_aabb_sequence_data(
        seq_name=args.sequence, gt_path=gt_path, tracker_path=track_path,
        class_id=1, distractor_ids={2, 7, 8, 12}, seq_info={args.sequence: None},
    )
    metrics = _summary_from_bundle(_eval_bundle(data))
    samples = [json.loads(line) for line in memory_path.read_text().splitlines()]
    warmed = latencies[min(8, len(latencies) - 1):]
    report = {
        "variant": name, "build": str(args.build.resolve()), "sequence": args.sequence,
        "device": str(device), "frames": len(frames), "source_frames": frames,
        "max_objects": args.max_objects if guided else None,
        "initial_update_ms": latencies[0], "warmup_frames": len(latencies) - len(warmed),
        "mean_update_ms": float(np.mean(warmed)), "p95_update_ms": float(np.percentile(warmed, 95)),
        "metrics": {key: metrics[key] for key in ("HOTA", "IDF1", "MOTA", "IDSW")},
        "peak_sampled_bytes": {
            key: max(sample.get(key, 0) for sample in samples)
            for key in ("rss_bytes", "live_device_bytes", "driver_device_bytes", "mask_state_storage_bytes")
        },
        "notes": "MOTChallenge pedestrian slice; latency excludes decoding, detector inference, and memory sampling.",
    }
    if guided:
        write_mask_guidance_provenance(
            output, checkpoint=args.checkpoint, device=str(device), build=args.build,
            tracker_spec=TrackerSpec("bytetrack"), max_objects=args.max_objects,
            sequence_names=(args.sequence,),
        )
    tracker.reset()
    del tracker
    del dataset
    _sample(memory_path, stage="after_sequence", frame=len(frames), device=device)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    """Run both variants separately and save a reproducible comparison artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--split", default="ablation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max-objects", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.max_objects < 1:
        parser.error("--frames and --max-objects must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty")
    args.checkpoint = resolve_edgetam_checkpoint(args.checkpoint)
    args.build = args.build.resolve()
    for guided in (False, True):
        process = multiprocessing.get_context("spawn").Process(target=_run_variant, args=(args, guided))
        process.start()
        process.join()
        if process.exitcode:
            raise RuntimeError(f"{'Guided' if guided else 'Baseline'} comparison failed (exit {process.exitcode})")
    reports = [json.loads((args.output / name / "summary.json").read_text()) for name in ("baseline", "guided")]
    (args.output / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
