"""KITTI sensor replay and mask-based evaluation of the EagerMOT Python tracker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from boxmot import EagerMot, __version__
from boxmot.datasets.kitti_fusion import KittiFusionFrame, KittiFusionSequence
from boxmot.engine.eval.kitti_mots_replay import evaluate_kitti_mots, kitti_mots_annotations, kitti_mots_sequences
from boxmot.engine.eval.mots_io import prepare_mots_tracks, tracks_to_mots_rows, write_mots_rows
from boxmot.engine.eval.output import increment_path
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, MaskBatch, Tracks
from boxmot.utils import logger as LOGGER

_KITTI_SHARED = {
    "det_thresh_3d": 0.0,
    "max_age": 3,
    "max_age_2d": 3,
    "fusion_iou_threshold": 0.01,
    "iou_threshold": 0.3,
    "first_matching_method": "dist_2d_full",
    "iou_3d_threshold": 0.01,
    "is_angular": False,
    "per_class": False,
    "asso_func": "iou",
}
KITTI_PROFILES = {
    1: {**_KITTI_SHARED, "det_thresh": 0.0, "min_hits": 1, "distance_threshold": 3.5},
    2: {**_KITTI_SHARED, "det_thresh": 0.9, "min_hits": 2, "distance_threshold": 0.3},
}


def _track_frame(frame: KittiFusionFrame, trackers: dict[int, EagerMot]) -> Tracks:
    """Track each class with its preset, restoring global row indices and IDs."""
    outputs: list[Tracks] = []
    for class_id, tracker in trackers.items():
        indices = torch.nonzero(
            (frame.detections.class_ids == class_id) & (frame.detections.scores > tracker.det_thresh),
            as_tuple=False,
        ).flatten()
        # The released KITTI loader visits image observations in ascending score order.
        order = torch.argsort(frame.detections.scores[indices], stable=True)
        indices = indices[order]
        spatial_indices = torch.nonzero(
            (frame.detections_3d.class_ids == class_id) & (frame.detections_3d.scores > tracker.det_thresh_3d),
            as_tuple=False,
        ).flatten()
        result = tracker.update(
            frame.detections.select(indices),
            detections_3d=frame.detections_3d.select(spatial_indices),
            camera=frame.camera,
        ).image_tracks
        outputs.append(
            Tracks(
                geometry=result.geometry,
                track_ids=2 * result.track_ids + class_id - 1,
                scores=result.scores,
                class_ids=result.class_ids,
                detection_indices=indices[result.detection_indices],
                sample_id=result.sample_id,
                masks=result.masks,
            )
        )
    if any(output.masks is None for output in outputs):
        raise ValueError("KITTI segmentation evaluation requires TrackR-CNN masks on image tracks.")
    return Tracks(
        geometry=Boxes(torch.cat([output.geometry.values for output in outputs])),
        track_ids=torch.cat([output.track_ids for output in outputs]),
        scores=torch.cat([output.scores for output in outputs]),
        class_ids=torch.cat([output.class_ids for output in outputs]),
        detection_indices=torch.cat([output.detection_indices for output in outputs]),
        sample_id=frame.detections.sample_id,
        masks=MaskBatch(torch.cat([output.masks.values for output in outputs])),
    )


def _run(args: Any) -> Path:
    """Validate sequence alignment, replay saved predictions, and persist metrics."""
    names = kitti_mots_sequences(args.split, tuple(args.sequence_names))
    image_root = Path(args.images).expanduser().resolve()
    instances_root = Path(args.instances).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sequences: dict[str, KittiFusionSequence] = {}
    annotations: dict[str, list[tuple[int, str, int, int]]] = {}
    for name in names:
        sequence = KittiFusionSequence(data_root, image_root, name, car_variant=args.pointgnn_car)
        sequences[name] = sequence
        annotations[name] = kitti_mots_annotations(name, sequence.frame_paths, sequence.image_size, instances_root)

    output = increment_path(Path(args.project).expanduser().resolve() / args.split, mkdir=True)
    prediction_dir = output / "mots"
    prediction_dir.mkdir()
    manifest = {
        "status": "running",
        "boxmot_version": __version__,
        "tracker": "eagermot",
        "evaluation": "KITTI MOTS; mask IoU HOTA, CLEAR, and Identity metrics",
        "split": args.split,
        "sequences": {name: len(sequence) for name, sequence in sequences.items()},
        "missing_pointgnn_frames": {name: sequence.missing_3d_frames for name, sequence in sequences.items()},
        "data_root": str(data_root),
        "images": str(image_root),
        "instances": str(instances_root),
        "pointgnn_car": args.pointgnn_car,
        "pointgnn_pedestrian": "results_tracking_ped_cyl_auto_trainval",
        "image_detector": "trackrcnn_detections",
        "pointgnn_score_mapping": "s / (1 + s); bounded ranking score, not a calibrated probability",
        "tracker_profiles": KITTI_PROFILES,
        "limitations": [
            "Detector checkpoint training provenance is not independently verified.",
            "This evaluates segmentation tracking, not 3D boxes or published EagerMOT benchmark parity.",
            "Full rigid poses transform centers; box orientation remains yaw-only.",
        ],
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, sequence in sequences.items():
        trackers = {class_id: EagerMot(**profile) for class_id, profile in KITTI_PROFILES.items()}
        LOGGER.info(f"EagerMOT {name}: tracking {len(sequence)} frames")
        with (prediction_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for frame in sequence:
                tracks = _track_frame(frame, trackers)
                prepared = prepare_mots_tracks(PipelineResult(frame.detections, tracks), frame.image_size)
                write_mots_rows(handle, tracks_to_mots_rows(prepared, frame.frame_index))
                if (frame.frame_index + 1) % 200 == 0:
                    LOGGER.info(f"EagerMOT {name}: {frame.frame_index + 1}/{len(sequence)} frames")

    evaluate_kitti_mots(prediction_dir, output, instances_root, annotations)
    manifest["status"] = "complete"
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def run_eagermot_kitti(args: Any) -> Path:
    """Replay saved predictions on CPU; avoid thread overhead on small mask batches."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            return _run(args)
    finally:
        torch.set_num_threads(previous_threads)


__all__ = ("run_eagermot_kitti",)
