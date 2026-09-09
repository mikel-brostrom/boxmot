"""Replay TrackR-CNN predictions with image trackers and evaluate KITTI MOTS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from boxmot import __version__, create_tracker
from boxmot.datasets.readers.images import read_rgb_chw_uint8
from boxmot.datasets.trackrcnn import TrackRcnnSequence
from boxmot.engine.eval.kitti_mots_replay import evaluate_kitti_mots, kitti_mots_annotations, kitti_mots_sequences
from boxmot.engine.eval.mots_io import prepare_mots_tracks, tracks_to_mots_rows, write_mots_rows
from boxmot.engine.eval.output import increment_path
from boxmot.engine.tracker_config import resolve_tracker_options
from boxmot.pipelines import PipelineResult
from boxmot.structures import Frame
from boxmot.trackers import TrackerSpec
from boxmot.utils import logger as LOGGER


def _run(args: Any) -> Path:
    """Replay each complete sequence using native class IDs and current RGB frames."""
    names = kitti_mots_sequences(args.split, tuple(args.sequence_names))
    options = resolve_tracker_options(args, include_defaults=True)
    spec = TrackerSpec(args.tracker, per_class=True, options=tuple(sorted(options.items())))
    tracker = create_tracker(spec)
    if tracker.requirements.embeddings:
        raise ValueError(
            f"Tracker {args.tracker!r} requires embeddings. This replay supplies image frames, boxes, and masks; "
            "select a configuration that does not require an appearance encoder."
        )
    detections_root = Path(args.detections).expanduser().resolve()
    image_root = Path(args.images).expanduser().resolve()
    instances_root = Path(args.instances).expanduser().resolve()
    sequences = {}
    annotations = {}
    for name in names:
        sequence = TrackRcnnSequence(detections_root, image_root, name)
        sequences[name] = sequence
        annotations[name] = kitti_mots_annotations(name, sequence.frame_paths, sequence.image_size, instances_root)

    output = increment_path(Path(args.project).expanduser().resolve() / args.split, mkdir=True)
    prediction_dir = output / "mots"
    prediction_dir.mkdir()
    manifest = {
        "status": "running",
        "boxmot_version": __version__,
        "tracker": args.tracker,
        "tracker_backend": "python",
        "tracker_options": options,
        "per_class": True,
        "evaluation": "KITTI MOTS; mask IoU HOTA, CLEAR, and Identity metrics",
        "split": args.split,
        "sequences": {name: len(sequence) for name, sequence in sequences.items()},
        "detections": str(detections_root),
        "images": str(image_root),
        "instances": str(instances_root),
        "dropped_empty_masks": {},
    }
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, sequence in sequences.items():
        tracker.reset()
        dropped = 0
        LOGGER.info(f"{args.tracker} {name}: tracking {len(sequence)} frames")
        with (prediction_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for sample in sequence:
                detections = sample.detections
                # Empty TrackR-CNN RLEs have no segmentation observation; MAF requires foreground.
                valid = torch.nonzero(detections.masks.values.flatten(1).any(dim=1), as_tuple=False).flatten()
                dropped += len(detections) - len(valid)
                detections = detections.select(valid)
                frame = Frame(
                    image=read_rgb_chw_uint8(sample.image_path.as_uri(), image_root),
                    sample_id=detections.sample_id,
                    sequence_id=name,
                    frame_index=sample.frame_index,
                    timestamp_s=sample.frame_index / 10.0,
                    source_uri=sample.image_path.as_uri(),
                )
                tracks = tracker.update(detections, frame)
                prepared = prepare_mots_tracks(PipelineResult(detections, tracks), sample.image_size)
                write_mots_rows(handle, tracks_to_mots_rows(prepared, sample.frame_index))
                if (sample.frame_index + 1) % 100 == 0:
                    LOGGER.info(f"{args.tracker} {name}: {sample.frame_index + 1}/{len(sequence)} frames")
        manifest["dropped_empty_masks"][name] = dropped
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    evaluate_kitti_mots(prediction_dir, output, instances_root, annotations)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def run_trackrcnn(args: Any) -> Path:
    """Run CPU replay with bounded mask-operation threading and restore caller settings."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            return _run(args)
    finally:
        torch.set_num_threads(previous_threads)


__all__ = ("run_trackrcnn",)
