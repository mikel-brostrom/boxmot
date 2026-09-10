"""Build-backed tracking evaluation.

Evaluation is intentionally a consumer of an explicit immutable dataset build.
It never creates detections, masks, or embeddings and never selects a "latest"
cache. Ground truth follows the dataset adapter's frame selection.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from boxmot.components.resolution import ArtifactResolver, freeze_json
from boxmot.datasets import DatasetManifest
from boxmot.datasets.config import load_dataset_config
from boxmot.detectors.config import resolve_detector_spec
from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.engine.config.trackers import resolve_tracker_options, validate_image_tracker
from boxmot.engine.dataset_variants.fps import materialize_fps_ground_truth
from boxmot.engine.eval.catalog_cache import (
    EvaluationArtifactResolver,
    catalog_mot_dataset_for_evaluation,
)
from boxmot.engine.eval.motmetrics import run_motmetrics as _run_motmetrics
from boxmot.engine.eval.output import increment_path
from boxmot.engine.eval.replay import replay_build
from boxmot.engine.eval.results import SUMMARY_COLUMNS, ValidationResult
from boxmot.engine.materialization import fingerprint
from boxmot.engine.materialization.builds import resolve_build_path, validate_build_compatibility
from boxmot.engine.materialization.catalog import (
    resolve_dataset_annotation_root,
    resolve_dataset_split_root,
)
from boxmot.engine.materialization.resources import ensure_dataset_split_available
from boxmot.engine.ui.logging import suppress_boxmot_logs
from boxmot.engine.ui.reporters.eval import (
    EvalSequenceProgressPresenter,
    EvalWorkflowReporter,
    _refresh_eval_pipeline_intro,
)
from boxmot.reid.config import resolve_reid_spec
from boxmot.segmentors.config import resolve_segmentor_spec
from boxmot.trackers import TrackerSpec
from boxmot.utils import logger as LOGGER


def _detector_reference(resolved: Mapping[str, Any]) -> str:
    detector = resolved.get("detector") or {}
    if not detector.get("ref") or not detector.get("checkpoint"):
        raise ValueError("Experiment detector configuration is incomplete.")
    return f"{detector['ref']}/{detector['checkpoint']}"


def _reid_reference(resolved: Mapping[str, Any]) -> Mapping[str, Any] | None:
    config = resolved.get("reid")
    if not config:
        return None
    return {
        "artifact": {"path": config["model"], "uri": config.get("uri")},
        "device": "cpu" if config.get("device") in {None, "", "auto"} else config["device"],
        "precision": config.get("precision") or "fp32",
        "preprocessing": config.get("preprocess") or "default",
        "options": {"image_size": tuple(config.get("image_size") or (256, 128))},
    }


def _provenance_at_build_device(
    provenance: Mapping[str, Any],
    manifest: DatasetManifest,
    component_name: str,
) -> dict[str, Any]:
    """Recompute experiment provenance using the build's execution device only.

    Device selection is part of a build fingerprint but is an execution choice,
    not an authored experiment override. All other spec and artifact fields stay
    resolved from the selected experiment, so manifest data cannot substitute a
    different model or preprocessing policy.
    """

    components = manifest.metadata.get("components")
    component = components.get(component_name) if isinstance(components, Mapping) else None
    recorded_spec = component.get("spec") if isinstance(component, Mapping) else None
    device = recorded_spec.get("device") if isinstance(recorded_spec, Mapping) else None
    if not isinstance(device, str) or not device:
        raise ValueError(f"Build manifest is missing components.{component_name}.spec.device provenance.")

    resolved_spec = provenance.get("spec")
    if not isinstance(resolved_spec, Mapping):
        raise ValueError(f"Resolved {component_name} provenance is missing its canonical spec.")
    return {
        **dict(provenance),
        "spec": {**dict(resolved_spec), "device": device},
    }


def _experiment_component_fingerprints(
    resolved: Mapping[str, Any],
    manifest: DatasetManifest,
    *,
    artifact_resolver: ArtifactResolver | None = None,
) -> dict[str, str]:
    """Resolve only components represented by the selected experiment build."""

    geometry = str(resolved["dataset"]["box_type"])
    resolver_options = {} if artifact_resolver is None else {"artifact_resolver": artifact_resolver}
    _, detector_provenance = resolve_detector_spec(
        _detector_reference(resolved),
        geometry=geometry,
        allow_download=False,
        **resolver_options,
    )
    detector_provenance = _provenance_at_build_device(detector_provenance, manifest, "detector")
    expected = {"detector": fingerprint(detector_provenance)}
    actual = manifest.metadata.get("component_fingerprints")
    actual = actual if isinstance(actual, Mapping) else {}

    if actual.get("segmentor") is not None:
        reference = resolved.get("segmentor")
        if reference is None:
            raise ValueError("The build contains masks from a segmentor absent from the experiment configuration.")
        if isinstance(reference, Mapping) and set(reference) == {"ref"}:
            reference = reference["ref"]
        _, provenance = resolve_segmentor_spec(
            reference,
            geometry=geometry,
            allow_download=False,
            **resolver_options,
        )
        provenance = _provenance_at_build_device(provenance, manifest, "segmentor")
        expected["segmentor"] = fingerprint(provenance)

    if actual.get("reid") is not None:
        reference = _reid_reference(resolved)
        if reference is None:
            raise ValueError("The build contains embeddings from a ReID encoder absent from the experiment.")
        _, provenance = resolve_reid_spec(
            reference,
            allow_download=False,
            **resolver_options,
        )
        provenance = _provenance_at_build_device(provenance, manifest, "reid")
        expected["reid"] = fingerprint(provenance)
    return expected


def _resolve_selection(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    experiment_ref = getattr(args, "experiment", None)
    dataset_ref = getattr(args, "dataset", None)
    if bool(experiment_ref) == bool(dataset_ref):
        raise ValueError("Evaluation requires exactly one of --experiment or --dataset.")
    if experiment_ref:
        resolved = resolve_experiment_config(
            experiment_ref,
            split=getattr(args, "split", None) or None,
            mode="eval",
        )
        return dict(resolved["dataset"]), resolved

    dataset = load_dataset_config(dataset_ref)
    split = str(getattr(args, "split", None) or dataset["default_split"])
    try:
        split_config = dataset["splits"][split]
    except KeyError as exc:
        available = ", ".join(sorted(dataset["splits"]))
        raise ValueError(f"Unknown split {split!r}; available splits: {available}.") from exc
    if not split_config["has_ground_truth"]:
        raise ValueError(f"Dataset {dataset['id']!r} split {split!r} has no evaluation ground truth.")
    dataset["split"] = split
    return dataset, None


def _target_classes(
    dataset: Mapping[str, Any],
    experiment: Mapping[str, Any] | None,
) -> tuple[tuple[int, ...], tuple[tuple[int, str], ...]]:
    if experiment is not None:
        entries = tuple((int(item["dataset_id"]), str(item["name"])) for item in experiment["evaluation"]["classes"])
    else:
        entries = tuple(
            (int(metadata["id"]), str(name))
            for name, metadata in dataset["classes"].items()
            if metadata.get("evaluation") == "target"
        )
    entries = tuple(sorted(entries))
    if not entries:
        raise ValueError("Evaluation configuration contains no target classes.")
    return tuple(class_id for class_id, _ in entries), entries


def _split_root(dataset: Mapping[str, Any], data_root: str | Path | None) -> Path:
    return resolve_dataset_split_root(dataset, str(dataset["split"]), data_root)


def eval_setup(args: argparse.Namespace, pipeline: Any | None = None) -> None:
    """Resolve and validate raw ground truth plus one explicit immutable build."""

    if getattr(args, "tracker", None) is not None:
        validate_image_tracker(str(args.tracker))
    dataset, experiment = _resolve_selection(args)
    is_mots = dataset["layout"] == "kitti-mots"
    eval_masks = bool(getattr(args, "eval_masks", False))
    if eval_masks and not is_mots:
        raise ValueError("--eval-masks requires a KITTI MOTS dataset with instance PNG ground truth.")
    if is_mots and getattr(args, "calibrate_kf", False):
        raise ValueError(
            "Kalman calibration does not support KITTI MOTS mask ground truth. Run evaluation without --calibrate-kf."
        )
    split = str(dataset["split"])
    status_callback = pipeline.update if pipeline is not None and callable(getattr(pipeline, "update", None)) else None
    build_path = resolve_build_path(args.build, build_root=getattr(args, "build_root", None))
    manifest = DatasetManifest.load(build_path)
    recorded_fps = manifest.metadata.get("fps")
    requested_fps = getattr(args, "fps", None)
    if requested_fps is not None and requested_fps != recorded_fps:
        recorded_label = "native FPS" if recorded_fps is None else f"{recorded_fps:g} FPS"
        raise ValueError(
            f"Build {manifest.build_id!r} uses {recorded_label}, but --fps {requested_fps:g} was requested. "
            "Omit --build for automatic materialization, or create a matching build with "
            f"boxmot materialize --experiment <experiment-yaml> --fps {requested_fps:g}."
        )
    args.fps = recorded_fps if requested_fps is None else requested_fps
    ensure_dataset_split_available(
        dataset,
        split=split,
        data_root=getattr(args, "data_root", None),
        status_callback=status_callback,
    )
    catalog = catalog_mot_dataset_for_evaluation(
        dataset,
        split=split,
        data_root=getattr(args, "data_root", None),
        status_callback=status_callback,
        fps=args.fps,
    )
    component_fingerprints = None
    if experiment is not None:
        if status_callback is not None:
            status_callback("Validating experiment component artifacts…")
        with EvaluationArtifactResolver() as artifact_resolver:
            component_fingerprints = _experiment_component_fingerprints(
                experiment,
                manifest,
                artifact_resolver=artifact_resolver,
            )
    validate_build_compatibility(
        manifest,
        dataset_id=str(dataset["id"]),
        split=split,
        geometry=str(dataset["box_type"]),
        source_catalog_digest=catalog.fingerprint,
        class_taxonomy_digest=str(catalog.metadata["class_taxonomy_digest"]),
        component_fingerprints=component_fingerprints,
        require_masks=eval_masks,
    )
    if experiment is not None and manifest.metadata.get("experiment_id") != experiment["id"]:
        raise ValueError(
            f"Build {manifest.build_id!r} belongs to experiment "
            f"{manifest.metadata.get('experiment_id')!r}, not {experiment['id']!r}."
        )

    class_ids, class_names = _target_classes(dataset, experiment)
    sequence_lengths: dict[str, int] = {}
    sequence_frame_counts: dict[str, int] = {}
    for sample in catalog.samples:
        sequence = sample.sequence_id
        sequence_frame_counts[sequence] = sequence_frame_counts.get(sequence, 0) + 1
        sequence_lengths[sequence] = max(
            sequence_lengths.get(sequence, 0),
            sample.frame_index + 1,
        )
    requested_sequences = getattr(args, "sequence_names", None)
    if requested_sequences:
        requested = tuple(dict.fromkeys(str(name) for name in requested_sequences))
        missing = sorted(set(requested).difference(sequence_lengths))
        if missing:
            raise ValueError(f"Unknown evaluation sequence(s): {', '.join(missing)}")
        sequence_lengths = {name: sequence_lengths[name] for name in requested}
        sequence_frame_counts = {name: sequence_frame_counts[name] for name in requested}
        args.sequence_names = requested
    else:
        args.sequence_names = None
    split_root = _split_root(dataset, getattr(args, "data_root", None))
    gt_folder = resolve_dataset_annotation_root(dataset, split, getattr(args, "data_root", None))
    if args.fps is not None and not is_mots:
        if status_callback is not None:
            status_callback(f"Aligning ground truth to {args.fps:g} FPS…")
        variant_key = fingerprint({"catalog": catalog.fingerprint, "root": catalog.source_root.as_uri()})
        split_root, gt_folder = materialize_fps_ground_truth(
            dataset,
            catalog,
            Path(getattr(args, "project", None) or "runs") / ".fps-ground-truth" / variant_key,
            data_root=getattr(args, "data_root", None),
        )
    sequence_paths = []
    for name in sorted(sequence_lengths):
        path = split_root / name
        sequence_paths.append(path / "img1" if (path / "img1").is_dir() else path)

    args.build_path = build_path
    args.dataset_id = str(dataset["id"])
    args.experiment_id = None if experiment is None else str(experiment["id"])
    args.split = split
    args.geometry = str(dataset["box_type"])
    args.eval_box_type = args.geometry
    args.source = split_root
    split_config = dataset["splits"][split]
    args.gt_folder = gt_folder
    args.seq_paths = tuple(sequence_paths)
    args.seq_info = sequence_lengths
    args.sequence_frame_counts = sequence_frame_counts
    args.evaluation_config = {
        "id": dataset["id"],
        "layout": dataset["layout"],
        "box_type": dataset["box_type"],
        "classes": dataset["classes"],
        "annotation_layout": (
            "mots_png"
            if is_mots
            else "flat"
            if split_config.get("annotations") is not None or dataset["layout"] == "visdrone"
            else "sequence"
        ),
    }
    if is_mots:
        # Use the catalog's exact image references: FPS selection renumbers
        # evaluation frames while original PNG filenames remain unchanged.
        gt_frames: dict[str, list[tuple[int, str, int, int]]] = {name: [] for name in sequence_lengths}
        for sample in catalog.samples:
            if sample.sequence_id in gt_frames:
                if sample.image_ref is None:
                    raise ValueError("KITTI MOTS catalog samples require an image reference.")
                annotation = gt_folder / sample.sequence_id / Path(sample.image_ref).name
                gt_frames[sample.sequence_id].append((sample.frame_index, str(annotation), *sample.image_size))
        args.evaluation_config["mots_gt_frames"] = gt_frames
    args.remapped_class_ids = list(class_ids)
    args.remapped_class_names = [name.lower() for _, name in class_names]
    args.tracker_class_ids = class_ids
    args.tracker_class_names = class_names
    args._build_validated = True
    if pipeline is not None and callable(getattr(pipeline, "update", None)):
        pipeline.update(f"Validated build {manifest.build_id[:12]} for {dataset['id']}:{split}")


def _ensure_setup(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "_build_validated", False)):
        eval_setup(args)


def _tracker_options(
    args: argparse.Namespace,
    overrides: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...]:
    options = resolve_tracker_options(args, overrides)
    return tuple((str(key), freeze_json(value, location=f"tracker.{key}")) for key, value in sorted(options.items()))


def _tracker_spec(args: argparse.Namespace, overrides: Mapping[str, Any] | None = None) -> TrackerSpec:
    return TrackerSpec(
        name=str(args.tracker),
        backend=str(getattr(args, "tracker_backend", "python")),
        geometry=str(args.geometry),
        per_class=bool(getattr(args, "per_class", False)),
        class_ids=tuple(args.tracker_class_ids),
        class_names=tuple(args.tracker_class_names),
        options=_tracker_options(args, overrides),
    )


def _output_directory(args: argparse.Namespace, overrides: Mapping[str, Any] | None) -> Path:
    base = Path(getattr(args, "project", "runs")) / str(args.dataset_id) / str(getattr(args, "name", "exp"))
    if overrides:
        base = base / "trials" / fingerprint(dict(_tracker_options(args, overrides)))[:16]
        base.mkdir(parents=True, exist_ok=True)
        return base
    return increment_path(base, exist_ok=bool(getattr(args, "exist_ok", False)), mkdir=True)


def run_motmetrics(args: argparse.Namespace, verbose: bool = True) -> dict[str, Any]:
    """Evaluate replayed MOT or MOTS files against adapter-owned ground truth."""

    _ensure_setup(args)
    evaluate = _run_motmetrics
    if getattr(args, "evaluation_config", {}).get("layout") == "kitti-mots":
        if bool(getattr(args, "eval_masks", False)):
            from boxmot.engine.eval.mots import run_mots_metrics

            evaluate = run_mots_metrics
        else:
            from boxmot.engine.eval.kitti_boxes import run_kitti_box_metrics

            evaluate = run_kitti_box_metrics
    results = evaluate(
        args,
        tuple(Path(value) for value in args.seq_paths),
        Path(args.exp_dir),
        Path(args.gt_folder),
        seq_info=args.seq_info,
    )
    if verbose:
        LOGGER.info("Evaluation metrics: %s", json.dumps(results, sort_keys=True))
    return results


def _summary(results: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not results:
        return "", {}
    if any(column in results for column in SUMMARY_COLUMNS):
        metrics = results
        label = "single_class"
    else:
        label = next(
            (name for name in ("cls_comb_det_av", "cls_comb_cls_av", "all") if name in results),
            next(iter(results)) if len(results) == 1 else "",
        )
        metrics = results.get(label, {}) if label else {}
    return label, {
        column: metrics[column] for column in SUMMARY_COLUMNS if isinstance(metrics, Mapping) and column in metrics
    }


def run_eval(
    args: argparse.Namespace,
    *,
    evolve_config: Mapping[str, Any] | None = None,
    setup: bool = True,
    prepare_cache: bool = False,
    verbose: bool | None = None,
    show_progress: bool | None = None,
    pipeline: Any | None = None,
    per_class_configs: Mapping[int, Mapping[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> ValidationResult:
    """Replay one explicit build and evaluate it; perception is never run here."""

    if prepare_cache:
        raise ValueError("Evaluation never materializes implicitly. Run `boxmot materialize ...` and pass --build.")
    if per_class_configs:
        raise ValueError("Per-class tracker configurations are not supported by the canonical TrackerSpec yet.")
    if setup:
        eval_setup(args, pipeline=pipeline)
    else:
        _ensure_setup(args)
    if pipeline is not None:
        _refresh_eval_pipeline_intro(getattr(pipeline, "workflow", None), args)
        pipeline.advance("Replaying materialized detections through the tracker…")
    spec = _tracker_spec(args, evolve_config)

    output_dir = _output_directory(args, evolve_config) if output_dir is None else Path(output_dir)
    presenter = None
    if pipeline is not None and show_progress is not False and getattr(args, "seq_info", None):
        presenter = EvalSequenceProgressPresenter(
            pipeline.callback(),
            getattr(args, "sequence_frame_counts", args.seq_info),
        )
    started = time.perf_counter()
    visualization = None
    with ExitStack() as contexts:
        replay_callbacks = {}
        if bool(getattr(args, "eval_masks", False)):
            replay_callbacks["output_format"] = "mots"
        if presenter is not None:
            replay_callbacks["progress_callback"] = contexts.enter_context(presenter)
        if bool(getattr(args, "show", False)) or bool(getattr(args, "save", False)):
            from boxmot.engine.eval.visualization import ReplayVisualization

            visualization = contexts.enter_context(
                ReplayVisualization(
                    output_dir,
                    show=bool(getattr(args, "show", False)),
                    save=bool(getattr(args, "save", False)),
                    class_names=dict(args.tracker_class_names),
                )
            )
            replay_callbacks["frame_callback"] = visualization
        replay = replay_build(
            args.build_path,
            spec,
            split=args.split,
            output_dir=output_dir,
            sequence_ids=args.sequence_names,
            sequence_frame_counts=getattr(args, "sequence_frame_counts", args.seq_info),
            workers=int(getattr(args, "sequence_workers", 1)),
            **replay_callbacks,
        )
    args.video_paths = () if visualization is None else tuple(visualization.video_paths)
    if presenter is not None:
        pipeline.store_step_info(presenter.renderable, step=EvalWorkflowReporter.TRACK)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if pipeline is not None:
        pipeline.advance("Computing evaluation metrics…")
    args.exp_dir = replay.output_dir
    raw = run_motmetrics(args, verbose=bool(verbose))
    summary_label, summary = _summary(raw)
    timings = {
        "frames": replay.frames,
        "totals_ms": {"track": elapsed_ms, "total": elapsed_ms},
        "avg_ms": {
            "track": elapsed_ms / replay.frames if replay.frames else 0.0,
            "total": elapsed_ms / replay.frames if replay.frames else 0.0,
        },
        "fps": (1000.0 * replay.frames / elapsed_ms) if elapsed_ms else 0.0,
    }
    return ValidationResult(
        benchmark=str(args.experiment_id or args.dataset_id),
        raw=raw,
        summary_label=summary_label,
        summary=summary,
        exp_dir=replay.output_dir,
        timings=timings,
        args=args,
        workflow_rendered=pipeline is not None,
    )


def main(args: argparse.Namespace) -> ValidationResult:
    """CLI entry point for explicit-build evaluation."""

    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        calibration = None
        with suppress_boxmot_logs(True, level="WARNING"):
            if getattr(args, "calibrate_kf", False):
                from boxmot.engine.calibration.kalman import calibrate_kalman

                eval_setup(args, pipeline=pipeline)
                output_dir = _output_directory(args, None)
                calibration = calibrate_kalman(args, output_dir=output_dir, progress=pipeline.update)
                args.tracker_config = str(calibration.config_path)
                result = run_eval(args, setup=False, verbose=False, pipeline=pipeline, output_dir=output_dir)
                calibration.record_final(result)
            else:
                result = run_eval(args, prepare_cache=False, verbose=False, pipeline=pipeline)
        rendered = result.renderable(
            include_sequences=result.summary_label == "single_class",
            include_timings=bool(getattr(args, "show_timing", False)),
        )
        if calibration is not None or getattr(args, "video_paths", ()):
            from rich.console import Group
            from rich.text import Text

            details = []
            if calibration is not None:
                details.append(Text(calibration.description))
            if getattr(args, "video_paths", ()):
                details.append(Text("Saved tracking videos:\n" + "\n".join(map(str, args.video_paths))))
            rendered = Group(*details, rendered)
        pipeline.finish(
            rendered,
            exp_dir=result.exp_dir,
        )
        return result


__all__ = ("eval_setup", "main", "run_eval", "run_motmetrics")
