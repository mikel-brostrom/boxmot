"""Evaluate image trackers on configured saved boxes without detector inference."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from boxmot import __version__, create_tracker
from boxmot.datasets.cached import DatasetSample
from boxmot.datasets.inputs import DatasetInputs
from boxmot.datasets.readers.boxes2d import read_kitti_tracking_labels_2d
from boxmot.datasets.readers.images import read_rgb_chw_uint8
from boxmot.datasets.sequence import DetectionSequence
from boxmot.engine.config.datasets import load_saved_2d_evaluation_inputs
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.eval.kitti_tracking import run_kitti_tracking_metrics
from boxmot.engine.eval.output import increment_path
from boxmot.engine.eval.replay import ReplayFrame, _write_rows, tracks_to_mot_rows
from boxmot.engine.eval.results import ValidationResult, _select_plot_metrics_data
from boxmot.engine.eval.visualization import ReplayVisualization
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, EvalWorkflowReporter
from boxmot.pipelines import TrackingPipeline
from boxmot.reid import create_reid_encoder
from boxmot.reid.config import resolve_reid_spec
from boxmot.structures import Frame
from boxmot.trackers import TrackerSpec
from boxmot.utils import logger


def _tracking_inputs(dataset: DatasetInputs) -> DatasetInputs:
    """Keep annotations in scoring and cache only the configured tracking inputs."""
    return replace(
        dataset,
        sequences=tuple(
            replace(
                sequence,
                modalities={
                    role: modality
                    for role, modality in sequence.modalities.items()
                    if role in {"images", "detections_2d"}
                },
            )
            for sequence in dataset.sequences
        ),
    )


def run_saved_detections(args: Any, *, pipeline: Any | None = None) -> ValidationResult:
    """Replay native 2D predictions, enrich their appearance, and score KITTI tracks."""
    if pipeline is not None:
        pipeline.update("Loading saved 2D predictions and KITTI ground truth…")
    dataset = load_saved_2d_evaluation_inputs(
        args.dataset,
        split=getattr(args, "split", None) or None,
        sequence_names=tuple(getattr(args, "sequence_names", ())),
        data_root=getattr(args, "data_root", None),
    )
    options = resolve_tracker_options(args, include_defaults=True, factory_options=True)
    spec = TrackerSpec(
        args.tracker,
        backend=getattr(args, "tracker_backend", "python"),
        per_class=bool(getattr(args, "per_class", False)),
        options=tuple(sorted(options.items())),
    )
    with ExitStack() as resources:
        tracker = create_tracker(spec)
        if callable(getattr(tracker, "close", None)):
            resources.callback(tracker.close)
        cache_inputs = bool(getattr(args, "cache_inputs", False))
        reference = getattr(args, "reid", None)
        if tracker.requirements.embeddings and not reference:
            raise ValueError(f"'{args.tracker}' requires appearance embeddings. Select --reid <profile>.")
        if reference and not tracker.requirements.embeddings:
            raise ValueError(f"'{args.tracker}' does not use ReID in this configuration; omit --reid.")
        encoder = None
        reid_provenance = None
        if reference:
            encoder_spec, reid_provenance = resolve_reid_spec(reference)
            if getattr(args, "device", None):
                encoder_spec = replace(encoder_spec, device=str(args.device))
            reid_provenance = {**reid_provenance, "spec": asdict(encoder_spec)}
            encoder = create_reid_encoder(encoder_spec)
            if encoder.requirements.masks:
                raise ValueError("The selected ReID encoder requires masks; this dataset declares only 2D boxes.")
            if cache_inputs:
                from boxmot.engine.eval.saved_input_cache import CachedAppearanceEncoder

                encoder = CachedAppearanceEncoder(
                    encoder, encoder_spec, dataset.root / ".boxmot" / "replay_cache" / "embeddings"
                )

        tracking_dataset = _tracking_inputs(dataset)
        show, save = bool(getattr(args, "show", False)), bool(getattr(args, "save", False))
        load_images = tracker.requirements.frame_pixels or encoder is not None or show or save
        output = increment_path(Path(args.project).expanduser().resolve() / dataset.split, mkdir=True)
        args.exp_dir = output
        args.dataset_id = dataset.id
        args.benchmark = dataset.id
        args.split = dataset.split
        args.per_class = spec.per_class
        args.sequence_workers = 1
        args.tracker_class_names = {
            metadata["id"]: name for name, metadata in dataset.classes.items() if metadata["evaluation"] == "target"
        }
        args.remapped_class_names = list(args.tracker_class_names.values())
        args.remapped_class_ids = list(args.tracker_class_names)
        manifest = {
            "status": "running",
            "boxmot_version": __version__,
            "experiment_id": getattr(args, "experiment_id", None),
            "experiment_config": str(args.experiment) if getattr(args, "experiment", None) else None,
            "dataset_id": dataset.id,
            "dataset_config": str(dataset.config_path),
            "split": dataset.split,
            "tracker": args.tracker,
            "tracker_backend": spec.backend,
            "tracker_options": options,
            "per_class": spec.per_class,
            "reid": reid_provenance,
            "cache_inputs": cache_inputs,
            "image_cache": {},
            "evaluation": "KITTI 2D tracking; BoxMOT built-in image-box IoU HOTA, CLEAR, and Identity metrics",
            "fps": dataset.fps,
            "inputs": {
                sequence.sequence_id: {
                    role: {
                        "format": modality.format,
                        "paths": list(map(str, modality.paths)),
                        "options": modality.options,
                    }
                    for role, modality in sequence.modalities.items()
                }
                for sequence in dataset.sequences
            },
        }
        manifest_path = output / "run.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        frames = 0
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ExitStack() as contexts, torch.inference_mode():
                sequences = {}
                selections = {}
                for source in dataset.sequences:
                    name = source.sequence_id
                    if cache_inputs:
                        from boxmot.datasets.sensor_cache import (
                            SensorReplayCacheStorageError,
                            open_sensor_sequence,
                            prepare_sensor_sequence,
                        )

                        try:
                            prepared = prepare_sensor_sequence(tracking_dataset, name, load_images=load_images)
                        except SensorReplayCacheStorageError:
                            if not load_images:
                                raise
                            logger.warning(
                                "Input cache: insufficient disk space for sequence %s images; "
                                "reading original images and retaining detection/ReID caching.",
                                name,
                            )
                            prepared = prepare_sensor_sequence(tracking_dataset, name, load_images=False)
                        sequence = contexts.enter_context(closing(open_sensor_sequence(prepared)))
                    else:
                        sequence = DetectionSequence(
                            source, classes=dataset.classes, fps=dataset.fps, split=dataset.split
                        )
                    sequences[name] = sequence
                    manifest["image_cache"][name] = (
                        "cached" if cache_inputs and sequence.load_images else "source" if load_images else "unused"
                    )
                    ground_truth = source.modalities["ground_truth"].paths[0]
                    read_kitti_tracking_labels_2d(ground_truth, frame_count=len(sequence), cache_inputs=cache_inputs)
                    selections[name] = {
                        "path": str(ground_truth),
                        "frame_count": len(sequence),
                        "frames": [(index, index) for index in range(len(sequence))],
                    }
                args.seq_info = {name: len(sequence) for name, sequence in sequences.items()}
                args.evaluation_config = {"classes": dataset.classes, "kitti_gt_sequences": selections}
                manifest["sequences"] = args.seq_info
                tracking = TrackingPipeline(detector=None, tracker=tracker, reid=encoder)
                visualizer = contexts.enter_context(
                    ReplayVisualization(
                        output, show=show, save=save, class_names=args.tracker_class_names, video_fps=dataset.fps
                    )
                )
                presenter = None
                if pipeline is not None:
                    pipeline.advance("Tracking saved 2D predictions…")
                    presenter = contexts.enter_context(
                        EvalSequenceProgressPresenter(pipeline.callback(), args.seq_info)
                    )
                for name, sequence in sequences.items():
                    tracking.reset()
                    with (output / f"{name}.txt").open("w", encoding="utf-8") as handle:
                        for index, sample in enumerate(sequence):
                            path = sequence.frame_paths[index]
                            if load_images:
                                image = (
                                    sequence.read_image(index)
                                    if cache_inputs and sequence.load_images
                                    else read_rgb_chw_uint8(path.as_uri(), dataset.root)
                                )
                            else:
                                # Only geometry and capture timing can consume this frame.
                                image = torch.zeros((3, *sample.image_size), dtype=torch.uint8)
                            frame = Frame(
                                image=image,
                                sample_id=sample.detections.sample_id,
                                sequence_id=name,
                                frame_index=sample.frame_index,
                                timestamp_s=sample.timestamp_s,
                                source_uri=path.as_uri(),
                            )
                            result = tracking.step_detections(frame, sample.detections)
                            _write_rows(handle, tracks_to_mot_rows(result.tracks, sample.frame_index))
                            frames += 1
                            if show or save:
                                visualizer(
                                    ReplayFrame(
                                        DatasetSample(
                                            frame.sample_id,
                                            dataset.split,
                                            name,
                                            sample.frame_index,
                                            sample.timestamp_s,
                                            sample.image_size,
                                            path.as_uri(),
                                            frame,
                                            result.detections,
                                        ),
                                        result,
                                    )
                                )
                            if presenter is not None:
                                presenter.update(name, index + 1, len(sequence), status="running")
                    if presenter is not None:
                        presenter.update(name, len(sequence), len(sequence), status="completed")
                if presenter is not None:
                    presenter.flush()
                    pipeline.store_step_info(presenter.renderable)
                args.video_paths = visualizer.video_paths
            tracking_finished = time.perf_counter()
            if pipeline is not None:
                pipeline.advance("Computing KITTI 2D tracking metrics…")
            metrics = run_kitti_tracking_metrics(args, (), output, dataset.root, seq_info=args.seq_info)
            summary_label, summary = _select_plot_metrics_data(metrics)
            total_ms = (time.perf_counter() - started) * 1000
            tracking_ms = (tracking_finished - started) * 1000
            manifest.update(status="complete", videos=[str(path.relative_to(output)) for path in args.video_paths])
            return ValidationResult(
                benchmark=dataset.id,
                raw=metrics,
                summary_label=summary_label,
                summary=dict(summary),
                exp_dir=output,
                timings={
                    "frames": frames,
                    "totals_ms": {"track": tracking_ms, "eval": total_ms - tracking_ms, "total": total_ms},
                    "avg_ms": {"track": tracking_ms / frames, "total": total_ms / frames},
                    "fps": 1000 * frames / total_ms if total_ms else 0.0,
                },
                args=args,
                workflow_rendered=pipeline is not None,
            )
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            torch.set_num_threads(previous_threads)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(args: Any) -> ValidationResult:
    """Present saved-detection evaluation through the standard evaluation panel."""
    from rich.console import Group
    from rich.text import Text

    from boxmot.engine.ui.logging import suppress_boxmot_logs

    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        with suppress_boxmot_logs(True, level="WARNING"):
            result = run_saved_detections(args, pipeline=pipeline)
        details = [
            result.renderable(include_sequences=False, include_timings=bool(getattr(args, "show_timing", False))),
            Text(f"Results: {result.exp_dir}"),
        ]
        if getattr(args, "video_paths", ()):
            details.append(Text("Saved tracking videos:\n" + "\n".join(map(str, args.video_paths))))
        pipeline.finish(Group(*details), exp_dir=result.exp_dir)
        return result


__all__ = ("main", "run_saved_detections")
