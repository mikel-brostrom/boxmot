"""Segmentation stage producing keyed, full-frame boolean masks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from boxmot.datasets.readers import read_detection_batches
from boxmot.datasets.schema import MASKS_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, mask_records
from boxmot.segmentors.factory import create_segmentor
from boxmot.segmentors.protocols import Segmentor
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import MaskBatch

from ..source import BoundedFrameDecoder, SourceDigestResolver, SourceSample
from ._runtime import release_accelerator_memory
from .base import MaterializationContext, StageOutcome
from .detect import _batches

_WORKER_SEGMENTORS: dict[SegmentorSpec, Any] = {}


def _segment_from_spec(item):
    spec, shard_index, frames, detections = item
    segmentor = _WORKER_SEGMENTORS.get(spec)
    if segmentor is None:
        segmentor = create_segmentor(spec)
        _WORKER_SEGMENTORS[spec] = segmentor
    return shard_index, segmentor.segment(frames, detections)


class SegmentStage:
    """Run a reusable segmentor against persisted detector geometry."""

    name = "segment"

    def __init__(
        self,
        segmentor: Segmentor | SegmentorSpec,
        samples: Sequence[SourceSample],
        *,
        decode_workers: int = 1,
        source_digest_resolver: SourceDigestResolver | None = None,
    ) -> None:
        samples = tuple(samples)
        if not samples:
            raise ValueError("Segmentation materialization requires at least one source sample.")
        if any(not isinstance(sample, SourceSample) for sample in samples):
            raise TypeError("Segmentation materialization samples must be SourceSample records.")
        if isinstance(decode_workers, bool) or not isinstance(decode_workers, int) or decode_workers <= 0:
            raise ValueError("decode_workers must be a positive integer.")
        self.segmentor = segmentor
        self.samples = samples
        self.decode_workers = decode_workers
        self.source_digest_resolver = source_digest_resolver

    def _segment(self, item):
        shard_index, frames, detections = item
        return shard_index, frames, detections, self.segmentor.segment(frames, detections)

    def _infer(self, context: MaterializationContext, pending):
        if context.stage_plan.workers > 1:
            if context.stage_plan.executor != "process":
                raise ValueError("Multiple segmentor workers require the spawned process executor.")
            if not isinstance(self.segmentor, SegmentorSpec):
                raise TypeError("Multiple segmentor workers require a serializable SegmentorSpec, not a model handle.")
        if context.stage_plan.executor == "process" and not isinstance(self.segmentor, SegmentorSpec):
            raise TypeError("Process segmentor execution requires a serializable SegmentorSpec, not a model handle.")
        if isinstance(self.segmentor, SegmentorSpec):
            tasks = [(self.segmentor, shard_index, frames, detections) for shard_index, frames, detections in pending]
            outputs = context.executor.map(_segment_from_spec, tasks)
            inputs = {shard_index: (frames, detections) for shard_index, frames, detections in pending}
            return [(shard_index, *inputs[shard_index], masks) for shard_index, masks in outputs]
        return context.executor.map(self._segment, pending)

    def run(self, context: MaterializationContext) -> StageOutcome:
        if context.stage_plan.name != self.name:
            raise ValueError(f"SegmentStage cannot execute plan stage {context.stage_plan.name!r}.")
        context.validate_completed_shards(MASKS_ARTIFACT)
        staged = read_detection_batches(context.staging_root, context.build_plan.box_type)
        writer = ParquetShardWriter(context.staging_root, box_type=context.build_plan.box_type)
        mask_count = 0
        pending = [
            (
                shard_index,
                batch,
                [staged[sample.sample_id] for sample in batch],
            )
            for shard_index, batch in _batches(self.samples, context.stage_plan.batch_size)
            if f"{shard_index:05d}" not in context.completed_shards
        ]
        with BoundedFrameDecoder(
            self.decode_workers,
            digest_resolver=self.source_digest_resolver,
        ) as decoder:
            for start in range(0, len(pending), context.stage_plan.workers):
                decoded = [
                    (shard_index, decoder.decode(batch), detections)
                    for shard_index, batch, detections in pending[start : start + context.stage_plan.workers]
                ]
                inferred = self._infer(context, decoded)
                for shard_index, frames, detections, outputs in inferred:
                    shard_id = f"{shard_index:05d}"
                    if len(outputs) != len(frames):
                        raise RuntimeError(f"Segmentor returned {len(outputs)} mask batches for {len(frames)} frames.")

                    enriched = []
                    for frame, detection_batch, masks in zip(frames, detections, outputs, strict=True):
                        if not isinstance(masks, MaskBatch):
                            raise TypeError(f"Segmentor returned {type(masks).__name__}, expected MaskBatch.")
                        if len(masks) != len(detection_batch):
                            raise ValueError("Segmentor masks are not aligned with detections.")
                        if masks.image_size != frame.image_size:
                            raise ValueError("Segmentor masks must be full-frame at the original frame resolution.")
                        enriched.append(detection_batch.with_masks(masks))
                        mask_count += len(masks)

                    rows = [row for detections_with_masks in enriched for row in mask_records(detections_with_masks)]
                    writer.write(MASKS_ARTIFACT, rows, shard_index=shard_index)
                    context.record_shard(
                        shard_id,
                        MASKS_ARTIFACT,
                        items=len(frames),
                        rows=len(rows),
                    )
        return StageOutcome(artifacts=(MASKS_ARTIFACT,), metrics={"masks": mask_count})

    def release(self) -> None:
        """Release a lazily constructed segmentor after all retries finish."""

        if isinstance(self.segmentor, SegmentorSpec):
            runtime = _WORKER_SEGMENTORS.pop(self.segmentor, None)
            if runtime is not None:
                del runtime
                release_accelerator_memory(self.segmentor.device)


__all__ = ("SegmentStage",)
