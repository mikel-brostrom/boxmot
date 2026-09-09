"""KITTI sensor replay and mask-based evaluation of the EagerMOT Python tracker."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from boxmot import EagerMot, __version__
from boxmot.datasets.kitti_fusion import KittiFusionFrame, KittiFusionSequence
from boxmot.engine.eval.kitti_mots_replay import (
    GroundTruthEntry,
    evaluate_kitti_mots,
    kitti_mots_annotations,
    kitti_mots_sequences,
)
from boxmot.engine.eval.mots_io import prepare_mots_tracks, tracks_to_mots_rows, write_mots_rows
from boxmot.engine.eval.output import increment_path
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Boxes3D, Frame, MaskBatch, MultimodalTracks, Tracks, Tracks3D
from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.engine.eval.visualization import ReplayVisualization

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
KITTI_CLASSES = {1: "car", 2: "pedestrian"}


def load_kitti_profiles(path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Load car/pedestrian overrides, validating every option before replay."""
    profiles = {class_id: dict(profile) for class_id, profile in KITTI_PROFILES.items()}
    if path is None:
        return profiles
    try:
        values = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid EagerMOT class configuration in {path}: {exc}") from exc
    if not isinstance(values, dict) or set(values) != set(KITTI_CLASSES.values()):
        raise ValueError("EagerMOT class configuration must contain exactly 'car' and 'pedestrian' mappings.")
    for class_id, name in KITTI_CLASSES.items():
        overrides = values[name]
        if not isinstance(overrides, dict):
            raise ValueError(f"EagerMOT {name} configuration must be a mapping of tracker options.")
        unknown = set(overrides) - set(profiles[class_id])
        if unknown:
            raise ValueError(f"Unknown EagerMOT {name} options: {', '.join(sorted(map(str, unknown)))}")
        profile = {**profiles[class_id], **overrides}
        if profile["per_class"] is not False:
            raise ValueError("EagerMOT class profiles require per_class=false; replay already separates classes.")
        try:
            EagerMot(**profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EagerMOT {name} configuration: {exc}") from exc
        profiles[class_id] = profile
    return profiles


def write_kitti_profiles(path: Path, profiles: dict[int, dict[str, Any]]) -> None:
    """Persist complete class profiles in the format accepted by evaluation."""
    path.write_text(
        yaml.safe_dump({name: profiles[class_id] for class_id, name in KITTI_CLASSES.items()}, sort_keys=False),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class KittiReplayInputs:
    """Indexed sensor inputs reused across trials; full masks remain lazy."""

    sequences: dict[str, KittiFusionSequence]
    annotations: dict[str, list[GroundTruthEntry]]
    instances_root: Path
    manifest: dict[str, Any]


def _track_frame(frame: KittiFusionFrame, trackers: dict[int, EagerMot]) -> MultimodalTracks:
    """Merge class trackers while preserving independent image and spatial rows."""
    outputs: list[Tracks] = []
    spatial_outputs: list[Tracks3D] = []
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
        )
        image = result.image_tracks
        outputs.append(
            Tracks(
                geometry=image.geometry,
                track_ids=2 * image.track_ids + class_id - 1,
                scores=image.scores,
                class_ids=image.class_ids,
                detection_indices=indices[image.detection_indices],
                sample_id=image.sample_id,
                masks=image.masks,
            )
        )
        spatial = result.spatial_tracks
        original_indices = spatial.detection_indices.clone()
        observed = original_indices >= 0
        original_indices[observed] = spatial_indices[original_indices[observed]]
        spatial_outputs.append(
            Tracks3D(
                geometry=spatial.geometry,
                track_ids=2 * spatial.track_ids + class_id - 1,
                scores=spatial.scores,
                class_ids=spatial.class_ids,
                detection_indices=original_indices,
                sample_id=spatial.sample_id,
            )
        )
    if any(output.masks is None for output in outputs):
        raise ValueError("KITTI segmentation evaluation requires TrackR-CNN masks on image tracks.")
    image_tracks = Tracks(
        geometry=Boxes(torch.cat([output.geometry.values for output in outputs])),
        track_ids=torch.cat([output.track_ids for output in outputs]),
        scores=torch.cat([output.scores for output in outputs]),
        class_ids=torch.cat([output.class_ids for output in outputs]),
        detection_indices=torch.cat([output.detection_indices for output in outputs]),
        sample_id=frame.detections.sample_id,
        masks=MaskBatch(torch.cat([output.masks.values for output in outputs])),
    )
    spatial_tracks = Tracks3D(
        geometry=Boxes3D(torch.cat([output.geometry.values for output in spatial_outputs])),
        track_ids=torch.cat([output.track_ids for output in spatial_outputs]),
        scores=torch.cat([output.scores for output in spatial_outputs]),
        class_ids=torch.cat([output.class_ids for output in spatial_outputs]),
        detection_indices=torch.cat([output.detection_indices for output in spatial_outputs]),
        sample_id=frame.detections.sample_id,
    )
    return MultimodalTracks(image_tracks, spatial_tracks)


def prepare_eagermot_kitti(args: Any) -> KittiReplayInputs:
    """Validate alignment and index sensor inputs once for evaluation or tuning."""
    names = kitti_mots_sequences(args.split, tuple(args.sequence_names))
    image_root = Path(args.images).expanduser().resolve()
    instances_root = Path(args.instances).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sequences: dict[str, KittiFusionSequence] = {}
    annotations: dict[str, list[GroundTruthEntry]] = {}
    for name in names:
        sequence = KittiFusionSequence(data_root, image_root, name, car_variant=args.pointgnn_car)
        sequences[name] = sequence
        annotations[name] = kitti_mots_annotations(name, sequence.frame_paths, sequence.image_size, instances_root)

    manifest = {
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
        "limitations": [
            "Detector checkpoint training provenance is not independently verified.",
            "This evaluates segmentation tracking, not 3D boxes or published EagerMOT benchmark parity.",
            "Full rigid poses transform centers; box orientation remains yaw-only.",
        ],
    }
    return KittiReplayInputs(sequences, annotations, instances_root, manifest)


def _visualize_frame(
    visualization: ReplayVisualization,
    frame: KittiFusionFrame,
    tracks: Tracks,
    sequence: KittiFusionSequence,
    split: str,
    *,
    spatial_tracks: Tracks3D | None = None,
) -> None:
    """Decode RGB only for requested visualization of the evaluated track masks."""
    from boxmot.datasets import DatasetSample
    from boxmot.datasets.readers.images import read_rgb_chw_uint8
    from boxmot.engine.eval.replay import ReplayFrame

    image_path = sequence.frame_paths[frame.frame_index]
    image = Frame(
        image=read_rgb_chw_uint8(image_path.as_uri(), sequence.image_root),
        sample_id=frame.detections.sample_id,
        sequence_id=sequence.sequence_id,
        frame_index=frame.frame_index,
        timestamp_s=frame.frame_index / 10.0,
        source_uri=image_path.as_uri(),
    )
    if image.image_size != frame.image_size:
        raise ValueError(f"KITTI visualization image dimensions changed: {image_path}")
    sample = DatasetSample(
        sample_id=image.sample_id,
        split=split,
        sequence_id=sequence.sequence_id,
        frame_index=frame.frame_index,
        timestamp_s=image.timestamp_s,
        image_size=frame.image_size,
        image_ref=image.source_uri,
        frame=image,
        detections=frame.detections,
    )
    replayed = ReplayFrame(sample, PipelineResult(frame.detections, tracks))
    if spatial_tracks is None:
        visualization(replayed)
    else:
        visualization(replayed, spatial_tracks=spatial_tracks, camera=frame.camera)


def _replay(
    inputs: KittiReplayInputs,
    profiles: dict[int, dict[str, Any]],
    output: Path,
    *,
    visualization: ReplayVisualization | None = None,
    show_3d: bool = False,
) -> dict[str, dict[str, Any]]:
    """Replay a fresh tracker per class and sequence, then evaluate mask identities."""
    prediction_dir = output / "mots"
    prediction_dir.mkdir()
    for name, sequence in inputs.sequences.items():
        trackers = {class_id: EagerMot(**profile) for class_id, profile in profiles.items()}
        LOGGER.info(f"EagerMOT {name}: tracking {len(sequence)} frames")
        with (prediction_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for frame in sequence:
                tracks = _track_frame(frame, trackers)
                prepared = prepare_mots_tracks(PipelineResult(frame.detections, tracks.image_tracks), frame.image_size)
                write_mots_rows(handle, tracks_to_mots_rows(prepared, frame.frame_index))
                if visualization is not None:
                    _visualize_frame(
                        visualization,
                        frame,
                        prepared,
                        sequence,
                        inputs.manifest["split"],
                        spatial_tracks=tracks.spatial_tracks if show_3d else None,
                    )
                if (frame.frame_index + 1) % 200 == 0:
                    LOGGER.info(f"EagerMOT {name}: {frame.frame_index + 1}/{len(sequence)} frames")
    return evaluate_kitti_mots(prediction_dir, output, inputs.instances_root, inputs.annotations)


def evaluate_eagermot_kitti(
    inputs: KittiReplayInputs,
    profiles: dict[int, dict[str, Any]],
    output: Path,
    *,
    show: bool = False,
    save: bool = False,
    show_3d: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate profiles on CPU into a new directory, restoring thread settings."""
    if set(profiles) != set(KITTI_CLASSES):
        raise ValueError("EagerMOT KITTI replay requires profiles for both car and pedestrian.")
    if show_3d and not (show or save):
        raise ValueError("--show-3d requires --show or --save.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **inputs.manifest,
        "status": "running",
        "tracker_profiles": profiles,
        "visualization": {"show": show, "save": save, "show_3d": show_3d, "video_fps": 10.0},
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode(), ExitStack() as stack:
            visualization = None
            if show or save:
                from boxmot.engine.eval.visualization import ReplayVisualization

                visualization = stack.enter_context(
                    ReplayVisualization(output, show=show, save=save, class_names=KITTI_CLASSES, video_fps=10.0)
                )
            results = _replay(inputs, profiles, output, visualization=visualization, show_3d=show_3d)
            if visualization is not None:
                manifest["videos"] = [str(path.relative_to(output)) for path in visualization.video_paths]
        manifest["status"] = "complete"
        return results
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_eagermot_kitti(args: Any) -> Path:
    """Replay saved sensor predictions with default or saved per-class profiles."""
    profiles = load_kitti_profiles(getattr(args, "class_config", None))
    inputs = prepare_eagermot_kitti(args)
    output = increment_path(Path(args.project).expanduser().resolve() / args.split)
    evaluate_eagermot_kitti(
        inputs,
        profiles,
        output,
        show=bool(getattr(args, "show", False)),
        save=bool(getattr(args, "save", False)),
        show_3d=bool(getattr(args, "show_3d", False)),
    )
    return output


__all__ = (
    "evaluate_eagermot_kitti",
    "load_kitti_profiles",
    "prepare_eagermot_kitti",
    "run_eagermot_kitti",
    "write_kitti_profiles",
)
