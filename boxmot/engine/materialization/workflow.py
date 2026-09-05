"""CLI workflow for reproducible, resumable perception dataset builds."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

import torch

from boxmot import __version__
from boxmot.detectors import DetectorSpec
from boxmot.detectors.config import resolve_detector_spec
from boxmot.detectors.factory import detector_capabilities
from boxmot.engine.experiment_config import resolve_experiment_config
from boxmot.engine.logging import suppress_boxmot_logs
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    DetectStage,
    EmbedStage,
    FileMetadataCache,
    FinalizeStage,
    PublishOptions,
    SegmentStage,
    StagePlan,
    default_source_metadata_cache_path,
    fingerprint,
)
from boxmot.engine.materialization.catalog import (
    SourceCatalog,
    catalog_mot_dataset,
    resolve_dataset_root,
)
from boxmot.engine.materialization.progress import MaterializationProgress, MaterializationProgressReporter
from boxmot.engine.materialization.settings import load_executor_settings
from boxmot.engine.ui.core.ui import get_console
from boxmot.engine.ui.reporters.materialize import MaterializeWorkflowReporter
from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.reid.config import resolve_reid_spec
from boxmot.segmentors import SegmentorSpec
from boxmot.segmentors.config import resolve_segmentor_spec

_ComponentSpec = TypeVar("_ComponentSpec", DetectorSpec, SegmentorSpec, ReIDEncoderSpec)


def _normalize_device(value: object) -> str:
    """Return one canonical single-device selector for component specs."""

    device = str(value).strip().lower()
    if not device:
        raise ValueError("--device must be a non-empty device selector.")
    if device.isdecimal():
        return f"cuda:{int(device)}"
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda:"):
        index = device.removeprefix("cuda:")
        if index.isdecimal():
            return f"cuda:{int(index)}"
    if device in {"cpu", "mps"}:
        return device
    raise ValueError(f"Unsupported materialization device {value!r}; expected cpu, mps, cuda, cuda:N, or N.")


def _device_override(args: Any) -> str | None:
    """Resolve an explicit CLI device without replacing authored spec defaults."""

    explicit_keys = getattr(args, "materialize_explicit_keys", None)
    if explicit_keys is not None and "device" not in explicit_keys:
        return None
    value = getattr(args, "device", None)
    return None if value is None else _normalize_device(value)


def _require_available_device(device: str) -> None:
    """Fail before artifact loading when an explicit accelerator is unavailable."""

    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_built() or not mps.is_available():
            raise RuntimeError(
                "--device mps is unavailable in this PyTorch runtime. "
                "Use --device cpu or install an MPS-enabled PyTorch build on a supported macOS host."
            )
        return
    if not device.startswith("cuda:"):
        return
    index = int(device.removeprefix("cuda:"))
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if index >= count:
        raise RuntimeError(
            f"--device {device} is unavailable; PyTorch reports {count} CUDA device(s). "
            "Use --device cpu or select an available CUDA index."
        )


def _with_device(
    spec: _ComponentSpec,
    provenance: Mapping[str, Any],
    device: str | None,
    *,
    auto_device: str = "cpu",
) -> tuple[_ComponentSpec, Mapping[str, Any]]:
    """Apply an execution-device override to runtime and fingerprint provenance."""

    if device is None:
        if spec.device.strip().lower() != "auto":
            return spec, provenance
        device = auto_device
    resolved = replace(spec, device=device)
    resolved_provenance = dict(provenance)
    resolved_provenance["spec"] = asdict(resolved)
    return resolved, resolved_provenance


def _detector_reference(resolved: Mapping[str, Any]) -> str:
    detections = resolved.get("detections") or {}
    if detections.get("source") != "model":
        raise ValueError('Experiment detections.source must be "model" for canonical materialization.')
    model = detections.get("model") or {}
    reference = model.get("ref")
    checkpoint = model.get("checkpoint")
    if not reference or not checkpoint:
        raise ValueError("Experiment model detections must resolve a detector ref and checkpoint.")
    return f"{reference}/{checkpoint}"


def _reid_reference(resolved: Mapping[str, Any]) -> Mapping[str, Any] | None:
    config = resolved.get("reid")
    if not config:
        return None
    return {
        "artifact": {"path": config["model"], "uri": config.get("uri")},
        "device": "cpu" if config.get("device") in {None, "", "auto"} else config["device"],
        "precision": config.get("precision") or "fp32",
        "preprocessing": config.get("preprocess") or "default",
        "crop_strategy": config.get("crop_strategy") or "aabb",
        "options": {"image_size": tuple(config.get("image_size") or (256, 128))},
    }


def _resolved_inputs(
    args: Any,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[
    str,
    str,
    SourceCatalog,
    object,
    object | None,
    object | None,
    Mapping[str, Any],
]:
    """Resolve the one canonical experiment-backed materialization input."""

    experiment = getattr(args, "experiment", None)
    if not isinstance(experiment, (str, Path)) or not str(experiment).strip():
        raise ValueError("Materialization requires an authored experiment ID or YAML path.")
    resolved = resolve_experiment_config(experiment, mode="materialize")
    dataset = resolved["dataset"]
    data_root = getattr(args, "data_root", None)
    source_root = resolve_dataset_root(dataset, data_root)
    with FileMetadataCache(
        default_source_metadata_cache_path(source_root),
        status_callback=status_callback,
    ) as metadata_cache:
        catalog = catalog_mot_dataset(
            dataset,
            split=dataset["split"],
            data_root=data_root,
            metadata_resolver=metadata_cache.resolve,
        )
    metadata = {
        "experiment_id": resolved["id"],
        "dataset_id": dataset["id"],
        "split": dataset["split"],
        "class_taxonomy": dataset["classes"],
        "class_bridge": tuple(
            {
                "name": str(entry["name"]),
                "dataset_id": int(entry["dataset_id"]),
                "detector_name": str(entry["detector_name"]),
                "detector_id": int(entry["detector_id"]),
            }
            for entry in resolved["evaluation"]["classes"]
        ),
        "experiment_config": str(resolved["source_path"]),
    }
    return (
        dataset["id"],
        dataset["box_type"],
        catalog,
        _detector_reference(resolved),
        resolved.get("segmentor"),
        _reid_reference(resolved),
        metadata,
    )


def _stage_plan(
    name: str,
    *,
    settings: Mapping[str, Mapping[str, Any]],
    component: Mapping[str, Any],
    upstream: tuple[str, ...],
    depends_on: tuple[str, ...],
    semantic_config: Mapping[str, Any] | None = None,
) -> StagePlan:
    values = settings[name]
    return StagePlan.create(
        name,
        config={} if semantic_config is None else semantic_config,
        component=component,
        upstream_fingerprints=upstream,
        depends_on=depends_on,
        batch_size=int(values["batch_size"]),
        workers=int(values["workers"]),
        executor=str(values["executor"]),
        max_attempts=int(values["retries"]) + 1,
        retry_backoff_s=float(values["retry_backoff_s"]),
    )


def materialize(
    args: Any,
    *,
    progress: MaterializationProgressReporter | None = None,
) -> Path:
    """Resolve components and execute one immutable local build."""

    progress = progress or MaterializationProgress()
    device_override = _device_override(args)
    command_device = _normalize_device(getattr(args, "device", None) or "cpu")
    if device_override is not None:
        _require_available_device(device_override)
    progress.setup_status("Cataloging source samples…")

    (
        dataset_name,
        geometry,
        catalog,
        detector_reference,
        segmentor_reference,
        reid_reference,
        source_metadata,
    ) = _resolved_inputs(args, status_callback=progress.setup_status)
    progress.setup_status(f"Cataloged {len(catalog.samples):,} source samples; resolving executor settings…")
    settings = load_executor_settings(
        getattr(args, "plan_path", None),
        tuple(getattr(args, "plan_overrides", ()) or ()),
    )

    progress.setup_status("Resolving detector artifact and fingerprint…")
    detector_spec, detector_provenance = resolve_detector_spec(detector_reference, geometry=geometry)
    detector_spec, detector_provenance = _with_device(
        detector_spec,
        detector_provenance,
        device_override,
        auto_device=command_device,
    )
    class_id_map = {
        int(entry["detector_id"]): int(entry["dataset_id"]) for entry in source_metadata.get("class_bridge", ())
    }
    capabilities = detector_capabilities(detector_spec)

    publish = PublishOptions(
        image_references=bool(getattr(args, "publish_image_refs", True)),
        masks=bool(getattr(args, "publish_masks", False)),
        embeddings=bool(getattr(args, "publish_embeddings", True)),
    )
    encoder_spec: ReIDEncoderSpec | None = None
    encoder_provenance: Mapping[str, Any] | None = None
    encoder_requirements: EncoderRequirements | None = None
    encoder_dimension: int | None = None
    if publish.embeddings and not capabilities.provides_embeddings:
        if reid_reference is None:
            raise ValueError(
                "Published embeddings require the selected experiment to define ReID because its detector does not "
                "provide embeddings. Update the experiment or use --no-publish-embeddings."
            )
        encoder_spec, encoder_provenance = resolve_reid_spec(reid_reference)
        encoder_spec, encoder_provenance = _with_device(
            encoder_spec,
            encoder_provenance,
            device_override,
            auto_device=command_device,
        )
        encoder_requirements = EncoderRequirements(masks=encoder_spec.crop_strategy == "mask_aware")
        candidate_dimension = encoder_spec.option_values().get("embedding_dim")
        if candidate_dimension is not None:
            if (
                isinstance(candidate_dimension, bool)
                or not isinstance(candidate_dimension, int)
                or candidate_dimension <= 0
            ):
                raise ValueError("ReID embedding_dim must be a positive integer.")
            encoder_dimension = candidate_dimension

    needs_masks = publish.masks or bool(encoder_requirements is not None and encoder_requirements.masks)
    segmentor_spec: SegmentorSpec | None = None
    segmentor_provenance: Mapping[str, Any] | None = None
    if needs_masks and not capabilities.provides_masks:
        if segmentor_reference is None:
            reason = "the appearance crop policy" if not publish.masks else "published masks"
            raise ValueError(
                f"{reason.capitalize()} require the selected experiment to define a segmentor because its detector "
                "does not provide masks."
            )
        if isinstance(segmentor_reference, Mapping) and set(segmentor_reference) == {"ref"}:
            segmentor_reference = segmentor_reference["ref"]
        segmentor_spec, segmentor_provenance = resolve_segmentor_spec(segmentor_reference, geometry=geometry)
        segmentor_spec, segmentor_provenance = _with_device(
            segmentor_spec,
            segmentor_provenance,
            device_override,
            auto_device=command_device,
        )

    stage_plans: list[StagePlan] = []
    detect_plan = _stage_plan(
        "detect",
        settings=settings,
        component=detector_provenance,
        upstream=(),
        depends_on=(),
        semantic_config={
            "geometry": geometry,
            "class_id_map": {str(key): value for key, value in sorted(class_id_map.items())},
        },
    )
    stage_plans.append(detect_plan)

    samples = catalog.samples
    if not publish.image_references:
        samples = tuple(replace(sample, image_ref=None) for sample in samples)
    native_encoder_fingerprint = None
    native_embedding_dim = None
    if publish.embeddings and capabilities.provides_embeddings:
        native_encoder_fingerprint = fingerprint({"provider": "detector", "detector": detector_provenance})
        candidate_dimension = detector_spec.option_values().get("embedding_dim")
        if candidate_dimension is not None:
            if isinstance(candidate_dimension, bool) or not isinstance(candidate_dimension, int):
                raise ValueError("Detector embedding_dim must be a positive integer.")
            native_embedding_dim = candidate_dimension
    last_dependencies = ["detect"]
    last_fingerprints = [detect_plan.fingerprint]
    if segmentor_spec is not None:
        segment_plan = _stage_plan(
            "segment",
            settings=settings,
            component=segmentor_provenance or {},
            upstream=(detect_plan.fingerprint,),
            depends_on=("detect",),
        )
        stage_plans.append(segment_plan)
        last_dependencies.append("segment")
        last_fingerprints.append(segment_plan.fingerprint)

    embedding_metadata = None
    encoder_fingerprint: str | None = None
    if encoder_spec is not None:
        encoder_fingerprint = fingerprint(encoder_provenance)
        assert encoder_requirements is not None
        embed_dependencies = ("detect", "segment") if encoder_requirements.masks else ("detect",)
        embed_upstream = tuple(stage.fingerprint for stage in stage_plans if stage.name in set(embed_dependencies))
        embed_plan = _stage_plan(
            "embed",
            settings=settings,
            component=encoder_provenance or {},
            upstream=embed_upstream,
            depends_on=embed_dependencies,
        )
        stage_plans.append(embed_plan)
        last_dependencies.append("embed")
        last_fingerprints.append(embed_plan.fingerprint)
        embedding_metadata = {
            "encoder_fingerprint": encoder_fingerprint,
            "dim": encoder_dimension,
        }
        if encoder_dimension is None:
            embedding_metadata = None
    elif publish.embeddings:
        embedding_metadata = (
            None
            if native_embedding_dim is None
            else {
                "encoder_fingerprint": native_encoder_fingerprint,
                "dim": native_embedding_dim,
            }
        )

    finalize_plan = _stage_plan(
        "finalize",
        settings=settings,
        component={},
        upstream=tuple(last_fingerprints),
        depends_on=tuple(last_dependencies),
        semantic_config={"schema": "boxmot.dataset/v1", "publish": publish},
    )
    stage_plans.append(finalize_plan)

    components = {
        "detector": detector_provenance,
        "segmentor": segmentor_provenance,
        "reid": encoder_provenance,
    }
    plan = BuildPlan.create(
        build_root=getattr(args, "build_root", None),
        dataset_name=dataset_name,
        box_type=geometry,
        source_fingerprint=catalog.fingerprint,
        publish=publish,
        stages=tuple(stage_plans),
        metadata={
            **catalog.metadata,
            **source_metadata,
            "boxmot_version": __version__,
            "geometry": geometry,
            "components": components,
            "component_fingerprints": {
                name: None if value is None else fingerprint(value) for name, value in components.items()
            },
        },
    )
    if not bool(getattr(args, "resume", True)) and plan.staging_root.exists():
        if plan.staging_root.parent != plan.build_root / ".staging":
            raise RuntimeError("Refusing to discard staging outside the selected build root.")
        shutil.rmtree(plan.staging_root)

    # Keep every runtime as its immutable specification until the first pending
    # shard reaches that stage. The stage worker then constructs and caches the
    # model. Completed stages and published builds therefore never initialize
    # unused accelerator runtimes merely to validate resumable state.
    source_metadata_cache = FileMetadataCache(
        default_source_metadata_cache_path(catalog.source_root),
    )
    stage_objects: list[object] = [
        DetectStage(
            detector_spec,
            samples,
            class_id_map=class_id_map or None,
            native_encoder_fingerprint=native_encoder_fingerprint,
            native_embedding_dim=native_embedding_dim,
            decode_workers=int(settings["decode"]["workers"]),
            source_digest_resolver=source_metadata_cache.resolve_digest,
        )
    ]
    if segmentor_spec is not None:
        stage_objects.append(
            SegmentStage(
                segmentor_spec,
                samples,
                decode_workers=int(settings["decode"]["workers"]),
                source_digest_resolver=source_metadata_cache.resolve_digest,
            )
        )
    if encoder_spec is not None:
        assert encoder_requirements is not None
        assert encoder_fingerprint is not None
        stage_objects.append(
            EmbedStage(
                encoder_spec,
                samples,
                encoder_fingerprint=encoder_fingerprint,
                use_masks=encoder_requirements.masks,
                decode_workers=int(settings["decode"]["workers"]),
                source_digest_resolver=source_metadata_cache.resolve_digest,
            )
        )
    stage_objects.append(
        FinalizeStage(
            embedding_metadata=embedding_metadata,
            target_shard_rows=int(settings["writer"]["instance_rows_per_shard"]),
        )
    )

    progress.setup_status("Build plan resolved; preparing resumable stage state…")
    with source_metadata_cache:
        result = DatasetMaterializer(plan, stage_objects, progress=progress).run()
    if result != plan.output_root:
        raise RuntimeError(f"Materialization stopped before publication; resumable state remains at {result}.")
    return result


def main(args: Any) -> None:
    """CLI entry point."""

    interactive = bool(get_console(stderr=True).is_terminal)
    progress: MaterializationProgressReporter
    progress = MaterializeWorkflowReporter(args) if interactive else MaterializationProgress()
    progress.start()
    try:
        with suppress_boxmot_logs(interactive, level="WARNING"):
            materialize(args, progress=progress)
    except BaseException as exc:
        progress.unhandled_failure(exc)
        raise
    finally:
        progress.stop()


__all__ = ("main", "materialize")
