"""Evaluate EdgeTAM with one ground-truth box prompt per eligible person."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from boxmot import EdgeTAMModelConfig, __version__, create_tracker
from boxmot.components.resolution import component_options
from boxmot.datasets.cached import DatasetSample
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.eval.catalog_cache import catalog_mot_dataset_for_evaluation
from boxmot.engine.eval.evaluator import _resolve_selection, _summary
from boxmot.engine.eval.motmetrics import run_motmetrics
from boxmot.engine.eval.output import increment_path
from boxmot.engine.eval.prompts import PersonPrompt, load_first_visible_prompts
from boxmot.engine.eval.replay import ReplayFrame, _write_rows, tracks_to_mot_rows
from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.eval.visualization import ReplayVisualization
from boxmot.engine.materialization.catalog import resolve_dataset_annotation_root, resolve_dataset_split_root
from boxmot.engine.materialization.resources import ensure_dataset_split_available
from boxmot.engine.materialization.source import SourceSample, decode_source_sample
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, EvalWorkflowReporter
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Detections, Frame
from boxmot.trackers import TrackerSpec


def _empty_detections(frame: Frame) -> Detections:
    """Pass frame identity through the canonical API without detector observations."""
    return Detections(
        Boxes(torch.empty((0, 4), dtype=torch.float32)),
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.int64),
        sample_id=frame.sample_id,
    )


def _validate_options(args: Any) -> None:
    """Keep direct workflow calls subject to the prompt protocol as well."""
    if getattr(args, "tracker", None) != "edgetam" or getattr(args, "prompt_source", None) != "first-visible":
        raise ValueError("Prompted evaluation requires --tracker edgetam --prompt-source first-visible.")
    if getattr(args, "tracker_backend", "python") != "python" or getattr(args, "per_class", False):
        raise ValueError("Prompted evaluation requires the Python backend without --per-class.")
    if getattr(args, "experiment", None) or not getattr(args, "dataset", None):
        raise ValueError("Prompted evaluation requires --dataset, without --experiment.")
    if getattr(args, "sequence_workers", 1) not in (None, 1):
        raise ValueError("Prompted evaluation currently requires --sequence-workers 1.")
    visibility = getattr(args, "prompt_min_visibility", 1.0)
    if not math.isfinite(visibility) or not 0 <= visibility <= 1:
        raise ValueError("--prompt-min-visibility must be finite and within [0, 1].")
    unsupported = (
        "build",
        "build_ref",
        "build_root",
        "detector",
        "reid",
        "fps",
        "eval_masks",
        "eval_3d",
        "eval_ap",
        "show_3d",
        "postprocessing",
        "calibrate_kf",
        "variable_dt",
        "asso_func",
        "cmc_policy",
        "class_config",
        "compare_trackeval",
        "allow_noncanonical_build",
        "edgetam",
        "mask_guidance_weights",
        "mask_guidance_max_objects",
    )
    for name in unsupported:
        if getattr(args, name, None):
            raise ValueError(f"Prompted evaluation does not support --{name.replace('_', '-')}.")


def run_prompted_evaluation(args: Any, *, pipeline: Any | None = None) -> ValidationResult:
    """Seed each person once, propagate every frame, and score the complete sequence.

    Ground truth only supplies the initial box. It never corrects or removes
    objects. Standard MOT scoring retains observations before the prompt and
    identities which never satisfy the prompting criteria.
    """
    _validate_options(args)
    started = time.perf_counter()
    dataset, _ = _resolve_selection(args)
    if dataset["layout"] != "mot" or dataset["box_type"] != "aabb":
        raise ValueError("First-visible prompting requires a MOT-format AABB dataset with visibility annotations.")
    people = [
        (int(metadata["id"]), name)
        for name, metadata in dataset["classes"].items()
        if metadata["evaluation"] == "target" and name.lower() in {"pedestrian", "person"}
    ]
    if len(people) != 1:
        raise ValueError("First-visible prompting requires exactly one target class named pedestrian or person.")
    person_class_id, person_class_name = people[0]
    split = dataset["split"]
    data_root = getattr(args, "data_root", None)
    status = pipeline.update if pipeline is not None else None
    ensure_dataset_split_available(dataset, split=split, data_root=data_root, status_callback=status)
    catalog = catalog_mot_dataset_for_evaluation(dataset, split=split, data_root=data_root, status_callback=status)
    sequences: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in catalog.samples:
        sequences[sample.sequence_id].append(sample)
    requested = tuple(getattr(args, "sequence_names", ()) or ())
    missing = set(requested).difference(sequences)
    if missing:
        raise ValueError(f"Unknown evaluation sequence(s): {', '.join(sorted(missing))}")
    sequences = {
        name: sorted(sequences[name], key=lambda sample: sample.frame_index) for name in requested or sequences
    }
    args.geometry = "aabb"
    options = resolve_tracker_options(args, include_defaults=True, factory_options=True, allow_prompted=True)
    spec = TrackerSpec(
        "edgetam",
        geometry="aabb",
        class_ids=(person_class_id,),
        class_names=((person_class_id, person_class_name),),
        options=component_options(options),
    )
    gt_root = resolve_dataset_annotation_root(dataset, split, data_root)
    flat = dataset["splits"][split].get("annotations") is not None
    min_visibility = getattr(args, "prompt_min_visibility", 1.0)
    prompts: dict[str, tuple[PersonPrompt, ...]] = {}
    for name, samples in sequences.items():
        if len({sample.image_size for sample in samples}) != 1:
            raise ValueError(f"EdgeTAM requires constant frame dimensions within sequence {name!r}.")
        path = gt_root / f"{name}.txt" if flat else gt_root / name / "gt" / "gt.txt"
        prompts[name] = load_first_visible_prompts(
            path,
            image_sizes={sample.frame_index: sample.image_size for sample in samples},
            person_class_id=person_class_id,
            min_visibility=min_visibility,
        )
        if len(prompts[name]) > options["max_objects"]:
            raise ValueError(
                f"Sequence {name!r} needs {len(prompts[name])} prompted identities, but max_objects is "
                f"{options['max_objects']}. Set max_objects >= {len(prompts[name])} in --tracker-config. "
                "Ground truth is never used to retire objects."
            )

    output = increment_path(Path(args.project).expanduser().resolve() / dataset["id"] / "edgetam", mkdir=True)
    args.exp_dir = output
    args.dataset_id = args.benchmark = dataset["id"]
    args.split = split
    args.source = resolve_dataset_split_root(dataset, split, data_root)
    args.eval_box_type = "aabb"
    args.sequence_workers = 1
    args.seq_info = {name: samples[-1].frame_index + 1 for name, samples in sequences.items()}
    args.remapped_class_ids = [person_class_id]
    args.remapped_class_names = [person_class_name]
    args.evaluation_config = {
        "id": dataset["id"],
        "layout": "mot",
        "box_type": "aabb",
        "classes": dataset["classes"],
        "annotation_layout": "flat" if flat else "sequence",
    }
    prompt_records = {
        name: [
            {"frame": prompt.frame_index + 1, "object_id": prompt.object_id, "box_xyxy": prompt.box}
            for prompt in selected
        ]
        for name, selected in prompts.items()
    }
    (output / "prompts.json").write_text(json.dumps(prompt_records, indent=2) + "\n", encoding="utf-8")
    model = EdgeTAMModelConfig(
        checkpoint=getattr(args, "prompt_weights", None) or "edgetam.pt",
        device=str(getattr(args, "device", "cpu")),
    )
    manifest = {
        "status": "running",
        "boxmot_version": __version__,
        "dataset": dataset["id"],
        "split": split,
        "dataset_config": str(dataset["config_path"]),
        "source_catalog_digest": catalog.fingerprint,
        "tracker": "edgetam",
        "tracker_options": options,
        "checkpoint": str(model.checkpoint),
        "device": model.device,
        "prompt_source": "first-visible",
        "prompt_min_visibility": min_visibility,
        "prompt_box": "Full ground-truth XYXY box inside image bounds; no clipping",
        "prompt_frames": "One-based MOT frame numbers in prompts.json",
        "sequence_lengths": args.seq_info,
        "scoring": "Full sequences, including observations before prompting and people never eligible for a prompt",
        "ground_truth_use": "One initialization per identity; no corrections, reinitialization, or object retirement",
        "prompt_counts": {name: len(selected) for name, selected in prompts.items()},
    }
    manifest_path = output / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    frames = 0
    try:
        with ExitStack() as resources, torch.inference_mode():
            tracker = create_tracker(spec, prompt_model=model)
            resources.callback(tracker.reset)
            visualizer = resources.enter_context(
                ReplayVisualization(
                    output,
                    show=bool(getattr(args, "show", False)),
                    save=bool(getattr(args, "save", False)),
                    class_names={person_class_id: person_class_name},
                )
            )
            presenter = None
            if pipeline is not None:
                pipeline.advance("Tracking from first-visible ground-truth prompts…")
                presenter = resources.enter_context(
                    EvalSequenceProgressPresenter(
                        pipeline.callback(), {name: len(rows) for name, rows in sequences.items()}
                    )
                )
            tracking_started = time.perf_counter()
            for name, samples in sequences.items():
                tracker.reset()
                by_frame: dict[int, list[PersonPrompt]] = defaultdict(list)
                for prompt in prompts[name]:
                    by_frame[prompt.frame_index].append(prompt)
                try:
                    with (output / f"{name}.txt").open("w", encoding="utf-8") as handle:
                        for completed, sample in enumerate(samples, start=1):
                            frame = decode_source_sample(sample)
                            for prompt in by_frame.get(sample.frame_index, ()):
                                tracker.add_box(prompt.box, object_id=prompt.object_id, class_id=person_class_id)
                            detections = _empty_detections(frame)
                            tracks = tracker.update(detections, frame)
                            _write_rows(handle, tracks_to_mot_rows(tracks, sample.frame_index))
                            frames += 1
                            if visualizer.show or visualizer.save:
                                visualizer(
                                    ReplayFrame(
                                        DatasetSample(
                                            sample.sample_id,
                                            split,
                                            name,
                                            sample.frame_index,
                                            sample.timestamp_s,
                                            sample.image_size,
                                            sample.image_ref,
                                            frame,
                                            detections,
                                        ),
                                        PipelineResult(detections, tracks),
                                    )
                                )
                            if presenter is not None:
                                presenter.update(name, completed, len(samples), status="running")
                except BaseException as exc:
                    if presenter is not None:
                        presenter.fail(name, exc)
                    raise
                if presenter is not None:
                    presenter.complete(name)
            tracking_ms = (time.perf_counter() - tracking_started) * 1000
            args.video_paths = visualizer.video_paths
        if pipeline is not None:
            pipeline.advance("Scoring all frames against unchanged MOT ground truth…")
        raw = run_motmetrics(args, [args.source / name for name in sequences], output, gt_root, seq_info=args.seq_info)
        (output / "metrics.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        label, summary = _summary(raw)
        total_ms = (time.perf_counter() - started) * 1000
        manifest.update(status="completed", frames=frames, videos=list(map(str, args.video_paths)))
        return ValidationResult(
            benchmark=dataset["id"],
            raw=raw,
            summary_label=label,
            summary=summary,
            exp_dir=output,
            timings={
                "frames": frames,
                "totals_ms": {"track": tracking_ms, "total": total_ms},
                "avg_ms": {"track": tracking_ms / frames, "total": total_ms / frames},
                "fps": 1000 * frames / total_ms,
            },
            args=args,
            workflow_rendered=pipeline is not None,
        )
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(args: Any) -> ValidationResult:
    """Render prompted evaluation through the normal evaluation reporter."""
    from rich.console import Group
    from rich.text import Text

    from boxmot.engine.ui.logging import suppress_boxmot_logs

    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        with suppress_boxmot_logs(True, level="WARNING"):
            result = run_prompted_evaluation(args, pipeline=pipeline)
        pipeline.finish(
            Group(
                result.renderable(include_timings=bool(getattr(args, "show_timing", False))),
                Text("Ground-truth initialized; full-sequence MOT metrics."),
                Text(f"Results: {result.exp_dir}"),
            ),
            exp_dir=result.exp_dir,
        )
        return result
