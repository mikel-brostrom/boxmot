"""Detection stage writing stable sample and instance keys."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import torch

from boxmot.datasets.schema import (
    ARTIFACT_PATHS,
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
)
from boxmot.datasets.storage import (
    ParquetShardWriter,
    embedding_records,
    instance_records,
    mask_records,
    sample_record,
)
from boxmot.detectors.factory import create_detector
from boxmot.detectors.protocols import Detector
from boxmot.detectors.specs import DetectorSpec
from boxmot.structures import Boxes, Detections, OrientedBoxes

from ..source import BoundedFrameDecoder, SourceDigestResolver, SourceSample
from ._runtime import release_accelerator_memory
from .base import MaterializationContext, StageOutcome

_WORKER_DETECTORS: dict[DetectorSpec, Any] = {}


def _predict_from_spec(item):
    spec, shard_index, frames = item
    detector = _WORKER_DETECTORS.get(spec)
    if detector is None:
        detector = create_detector(spec)
        _WORKER_DETECTORS[spec] = detector
    return shard_index, detector.predict(frames)


def assign_instance_ids(detections: Detections, build_id: str) -> Detections:
    """Attach deterministic ``build:sample:index`` keys to detector output."""

    instance_ids = tuple(f"{build_id}:{detections.sample_id}:{index}" for index in range(len(detections)))
    return detections.with_instance_ids(instance_ids)


def _batches(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield start // size, items[start : start + size]


class DetectStage:
    """Run a reusable detector and persist geometry without positional joins."""

    name = "detect"

    def __init__(
        self,
        detector: Detector | DetectorSpec,
        samples: Sequence[SourceSample],
        *,
        class_id_map: Mapping[int, int] | None = None,
        native_encoder_fingerprint: str | None = None,
        native_embedding_dim: int | None = None,
        decode_workers: int = 1,
        source_digest_resolver: SourceDigestResolver | None = None,
    ) -> None:
        samples = tuple(samples)
        if not samples:
            raise ValueError("Detection materialization requires at least one source sample.")
        if any(not isinstance(sample, SourceSample) for sample in samples):
            raise TypeError("Detection materialization samples must be SourceSample records.")
        ids = [sample.sample_id for sample in samples]
        if len(set(ids)) != len(ids):
            raise ValueError("Source sample IDs must be unique.")
        if isinstance(decode_workers, bool) or not isinstance(decode_workers, int) or decode_workers <= 0:
            raise ValueError("decode_workers must be a positive integer.")
        self.detector = detector
        self.samples = samples
        self.decode_workers = decode_workers
        self.source_digest_resolver = source_digest_resolver
        if native_encoder_fingerprint is not None and not re.fullmatch(r"[0-9a-f]{64}", native_encoder_fingerprint):
            raise ValueError("native_encoder_fingerprint must be a full SHA-256 digest.")
        if native_embedding_dim is not None and (
            isinstance(native_embedding_dim, bool)
            or not isinstance(native_embedding_dim, int)
            or native_embedding_dim <= 0
        ):
            raise ValueError("native_embedding_dim must be a positive integer.")
        if class_id_map is not None and not isinstance(class_id_map, Mapping):
            raise TypeError("class_id_map must be a mapping from detector class IDs to dataset class IDs.")
        normalized_class_id_map: dict[int, int] | None = None
        if class_id_map is not None:
            normalized_class_id_map = {}
            for detector_id, dataset_id in class_id_map.items():
                if isinstance(detector_id, bool) or not isinstance(detector_id, int) or detector_id < 0:
                    raise ValueError("class_id_map detector IDs must be non-negative integers.")
                if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id < 0:
                    raise ValueError("class_id_map dataset IDs must be non-negative integers.")
                normalized_class_id_map[detector_id] = dataset_id
            normalized_class_id_map = dict(sorted(normalized_class_id_map.items()))
        self.class_id_map = None if normalized_class_id_map is None else MappingProxyType(normalized_class_id_map)
        self.native_encoder_fingerprint = native_encoder_fingerprint
        self.native_embedding_dim = native_embedding_dim

    def _map_class_ids(self, detections: Detections) -> Detections:
        """Filter and remap detector classes while preserving row-aligned payloads."""

        if self.class_id_map is None:
            return detections
        keep = torch.zeros(len(detections), dtype=torch.bool)
        mapped = torch.empty_like(detections.class_ids)
        for detector_id, dataset_id in self.class_id_map.items():
            matches = detections.class_ids == detector_id
            keep.logical_or_(matches)
            mapped.masked_fill_(matches, dataset_id)
        indices = torch.nonzero(keep, as_tuple=True)[0].contiguous()
        selected = detections.select(indices)
        return replace(
            selected,
            class_ids=mapped.index_select(0, indices).contiguous(),
        )

    def _predict(self, item):
        shard_index, batch, frames = item
        return shard_index, batch, self.detector.predict(frames)

    def _infer(self, context: MaterializationContext, pending):
        if context.stage_plan.workers > 1:
            if context.stage_plan.executor != "process":
                raise ValueError("Multiple detector workers require the spawned process executor.")
            if not isinstance(self.detector, DetectorSpec):
                raise TypeError("Multiple detector workers require a serializable DetectorSpec, not a model handle.")
        if context.stage_plan.executor == "process" and not isinstance(self.detector, DetectorSpec):
            raise TypeError("Process detector execution requires a serializable DetectorSpec, not a model handle.")
        if isinstance(self.detector, DetectorSpec):
            tasks = [(self.detector, shard_index, frames) for shard_index, _, frames in pending]
            outputs = context.executor.map(_predict_from_spec, tasks)
            batches = {shard_index: batch for shard_index, batch, _ in pending}
            return [(shard_index, batches[shard_index], predictions) for shard_index, predictions in outputs]
        return context.executor.map(self._predict, pending)

    def run(self, context: MaterializationContext) -> StageOutcome:
        if context.stage_plan.name != self.name:
            raise ValueError(f"DetectStage cannot execute plan stage {context.stage_plan.name!r}.")
        planned_stages = context.build_plan.stage_by_name
        segment_planned = "segment" in planned_stages
        embed_planned = "embed" in planned_stages
        capture_native_masks = not segment_planned
        capture_native_embeddings = not embed_planned
        publish_native_masks = context.build_plan.publish.masks and not segment_planned
        publish_native_embeddings = context.build_plan.publish.embeddings and not embed_planned
        stage_state = context.state.state.by_name[self.name]
        resume_artifacts = set(stage_state.artifacts)
        recovering_incomplete_stage = stage_state.status != "completed"
        resumable = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
        if (
            MASKS_ARTIFACT in resume_artifacts
            or publish_native_masks
            or (recovering_incomplete_stage and (context.staging_root / ARTIFACT_PATHS[MASKS_ARTIFACT]).is_dir())
        ):
            resumable.append(MASKS_ARTIFACT)
        if (
            EMBEDDINGS_ARTIFACT in resume_artifacts
            or publish_native_embeddings
            or (recovering_incomplete_stage and (context.staging_root / ARTIFACT_PATHS[EMBEDDINGS_ARTIFACT]).is_dir())
        ):
            resumable.append(EMBEDDINGS_ARTIFACT)
        context.validate_completed_shards(*resumable)
        writer = ParquetShardWriter(context.staging_root, box_type=context.build_plan.box_type)
        encoder_fingerprint = self.native_encoder_fingerprint or context.stage_plan.fingerprint
        resolved_embedding_dim = self.native_embedding_dim
        if resolved_embedding_dim is None:
            detector_dim = getattr(self.detector, "embedding_dim", None)
            if isinstance(detector_dim, int) and not isinstance(detector_dim, bool) and detector_dim > 0:
                resolved_embedding_dim = detector_dim
        native_masks_written = False
        native_embeddings_written = False
        pending = [
            (shard_index, batch)
            for shard_index, batch in _batches(self.samples, context.stage_plan.batch_size)
            if f"{shard_index:05d}" not in context.completed_shards
        ]
        with BoundedFrameDecoder(
            self.decode_workers,
            digest_resolver=self.source_digest_resolver,
        ) as decoder:
            for start in range(0, len(pending), context.stage_plan.workers):
                decoded = [
                    (shard_index, batch, decoder.decode(batch))
                    for shard_index, batch in pending[start : start + context.stage_plan.workers]
                ]
                inferred = self._infer(context, decoded)
                frames_by_shard = {shard_index: frames for shard_index, _, frames in decoded}
                for shard_index, batch, outputs in inferred:
                    shard_id = f"{shard_index:05d}"
                    frames = frames_by_shard[shard_index]
                    if len(outputs) != len(frames):
                        raise RuntimeError(f"Detector returned {len(outputs)} results for {len(frames)} frames.")

                    sample_rows: list[dict[str, Any]] = []
                    detection_rows: list[dict[str, Any]] = []
                    native_mask_rows: list[dict[str, Any]] = []
                    native_embedding_rows: list[dict[str, Any]] = []
                    native_masks_available = False
                    native_embeddings_available = False
                    for source, frame, detections in zip(batch, frames, outputs, strict=True):
                        if not isinstance(detections, Detections):
                            raise TypeError(
                                f"Detector returned {type(detections).__name__}; "
                                "materialization requires canonical Detections."
                            )
                        if detections.sample_id != source.sample_id:
                            raise ValueError("Detector result sample_id does not match its input frame.")
                        detections = self._map_class_ids(detections)
                        is_obb = isinstance(detections.geometry, OrientedBoxes)
                        if is_obb != (context.build_plan.box_type == "obb"):
                            raise ValueError("Detector geometry does not match the build plan box_type.")
                        if not isinstance(detections.geometry, (Boxes, OrientedBoxes)):
                            raise TypeError("Detector result has unsupported geometry.")
                        keyed = assign_instance_ids(detections, context.build_plan.build_id)
                        image_ref = source.image_ref if context.build_plan.publish.image_references else None
                        sample_rows.append(sample_record(frame, split=source.split, image_ref=image_ref))
                        detection_rows.extend(instance_records(keyed, build_id=context.build_plan.build_id))
                        if capture_native_masks:
                            if keyed.masks is None:
                                if publish_native_masks and len(keyed):
                                    raise ValueError(
                                        "Published masks require detector-native masks when no segment stage "
                                        "is planned."
                                    )
                            else:
                                native_masks_available = True
                                if keyed.masks.image_size != source.image_size:
                                    raise ValueError(
                                        "Detector-native masks must be full-frame at the source resolution."
                                    )
                                native_mask_rows.extend(mask_records(keyed))
                        if capture_native_embeddings:
                            if keyed.embeddings is None:
                                if publish_native_embeddings and len(keyed):
                                    raise ValueError(
                                        "Published embeddings require detector-native embeddings when no embed stage "
                                        "is planned."
                                    )
                            else:
                                native_embeddings_available = True
                                output_dim = int(keyed.embeddings.shape[1])
                                if resolved_embedding_dim is None:
                                    resolved_embedding_dim = output_dim
                                elif resolved_embedding_dim != output_dim:
                                    raise ValueError("Detector-native embedding dimensions must be consistent.")
                                native_embedding_rows.extend(
                                    embedding_records(keyed, encoder_fingerprint=encoder_fingerprint)
                                )

                    writer.write(SAMPLES_ARTIFACT, sample_rows, shard_index=shard_index)
                    writer.write(INSTANCES_ARTIFACT, detection_rows, shard_index=shard_index)
                    written_artifacts = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
                    if native_mask_rows or native_masks_available or publish_native_masks:
                        writer.write(MASKS_ARTIFACT, native_mask_rows, shard_index=shard_index)
                        native_masks_written = True
                        written_artifacts.append(MASKS_ARTIFACT)
                    if native_embedding_rows or native_embeddings_available or publish_native_embeddings:
                        if resolved_embedding_dim is None:
                            raise ValueError(
                                "An empty detector-native embedding artifact requires native_embedding_dim "
                                "or detector.embedding_dim."
                            )
                        writer.write(
                            EMBEDDINGS_ARTIFACT,
                            native_embedding_rows,
                            shard_index=shard_index,
                            embedding_dim=resolved_embedding_dim,
                            encoder_fingerprint=encoder_fingerprint,
                        )
                        native_embeddings_written = True
                        written_artifacts.append(EMBEDDINGS_ARTIFACT)
                    context.record_shard(
                        shard_id,
                        *written_artifacts,
                        items=len(batch),
                        rows=len(detection_rows),
                    )

        artifacts = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
        if native_masks_written or MASKS_ARTIFACT in resume_artifacts:
            artifacts.append(MASKS_ARTIFACT)
        if native_embeddings_written or EMBEDDINGS_ARTIFACT in resume_artifacts:
            artifacts.append(EMBEDDINGS_ARTIFACT)
        return StageOutcome(artifacts=tuple(artifacts), metrics={"samples": len(self.samples)})

    def release(self) -> None:
        """Release a lazily constructed detector after all retries finish."""

        if isinstance(self.detector, DetectorSpec):
            runtime = _WORKER_DETECTORS.pop(self.detector, None)
            if runtime is not None:
                del runtime
                release_accelerator_memory(self.detector.device)


__all__ = ("DetectStage", "SourceSample", "assign_instance_ids")
