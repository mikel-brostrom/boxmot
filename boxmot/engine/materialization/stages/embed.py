"""Appearance embedding stage keyed to immutable detections."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch

from boxmot.datasets.readers import attach_masks, read_detection_batches
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, embedding_records
from boxmot.reid.factory import create_reid_encoder
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Detections, Frame

from ..source import BoundedFrameDecoder, SourceDigestResolver, SourceSample
from ._runtime import release_accelerator_memory
from .base import MaterializationContext, StageOutcome
from .detect import _batches

_WORKER_ENCODERS: dict[ReIDEncoderSpec, Any] = {}


def _encode_bounded(
    encoder: AppearanceEncoder,
    frames: Sequence[Frame],
    detections: Sequence[Detections],
    *,
    max_rows: int,
) -> list[torch.Tensor]:
    """Encode at most ``max_rows`` detection crops in one model call.

    Materialization shards are grouped by source samples, whose detection
    cardinality is data-dependent. Treating the stage batch size as a frame
    count can therefore turn a nominal batch of 64 into several thousand ReID
    crops. Keep checkpoint shards sample-aligned while independently bounding
    accelerator inference by detection rows.
    """

    if len(frames) != len(detections):
        raise ValueError(f"frames and detections must be aligned; received {len(frames)} and {len(detections)}.")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer.")

    dimension = int(encoder.embedding_dim)
    if dimension <= 0:
        raise ValueError("Appearance encoder embedding_dim must be positive.")
    parts: list[list[torch.Tensor]] = [[] for _ in detections]
    chunk_frames: list[Frame] = []
    chunk_detections: list[Detections] = []
    destinations: list[int] = []
    chunk_rows = 0

    def flush() -> None:
        nonlocal chunk_rows
        if not chunk_detections:
            return
        outputs = encoder.encode(chunk_frames, chunk_detections)
        if len(outputs) != len(chunk_detections):
            raise RuntimeError(
                f"Encoder returned {len(outputs)} tensors for {len(chunk_detections)} detection batches."
            )
        for destination, batch, embeddings in zip(
            destinations,
            chunk_detections,
            outputs,
            strict=True,
        ):
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"Encoder returned {type(embeddings).__name__}, expected torch.Tensor.")
            if (
                embeddings.device.type != "cpu"
                or embeddings.dtype is not torch.float32
                or embeddings.ndim != 2
                or not embeddings.is_contiguous()
            ):
                raise ValueError("Encoder output must be a contiguous CPU float32 tensor with shape [N,D].")
            if embeddings.shape != (len(batch), dimension):
                raise ValueError(
                    f"Encoder output has shape {tuple(embeddings.shape)}, expected ({len(batch)}, {dimension})."
                )
            parts[destination].append(embeddings)
        chunk_frames.clear()
        chunk_detections.clear()
        destinations.clear()
        chunk_rows = 0

    for destination, (frame, batch) in enumerate(zip(frames, detections, strict=True)):
        offset = 0
        while offset < len(batch):
            available = max_rows - chunk_rows
            take = min(available, len(batch) - offset)
            indices = torch.arange(offset, offset + take, dtype=torch.int64)
            chunk_frames.append(frame)
            chunk_detections.append(batch.select(indices))
            destinations.append(destination)
            chunk_rows += take
            offset += take
            if chunk_rows == max_rows:
                flush()
    flush()

    return [
        torch.empty((0, dimension), dtype=torch.float32)
        if not batch_parts
        else torch.cat(batch_parts, dim=0).contiguous()
        for batch_parts in parts
    ]


def _encode_from_spec(item):
    spec, shard_index, frames, detections, max_rows = item
    encoder = _WORKER_ENCODERS.get(spec)
    if encoder is None:
        encoder = create_reid_encoder(spec)
        _WORKER_ENCODERS[spec] = encoder
    return (
        shard_index,
        int(encoder.embedding_dim),
        _encode_bounded(
            encoder,
            frames,
            detections,
            max_rows=max_rows,
        ),
    )


class EmbedStage:
    """Run a reusable appearance encoder and persist fixed-size vectors."""

    name = "embed"

    def __init__(
        self,
        encoder: AppearanceEncoder | ReIDEncoderSpec,
        samples: Sequence[SourceSample],
        *,
        encoder_fingerprint: str,
        use_masks: bool = False,
        decode_workers: int = 1,
        source_digest_resolver: SourceDigestResolver | None = None,
    ) -> None:
        samples = tuple(samples)
        if not samples:
            raise ValueError("Embedding materialization requires at least one source sample.")
        if any(not isinstance(sample, SourceSample) for sample in samples):
            raise TypeError("Embedding materialization samples must be SourceSample records.")
        if not isinstance(encoder_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", encoder_fingerprint):
            raise ValueError("encoder_fingerprint must be a full SHA-256 digest.")
        if not isinstance(use_masks, bool):
            raise TypeError("use_masks must be a boolean.")
        if isinstance(decode_workers, bool) or not isinstance(decode_workers, int) or decode_workers <= 0:
            raise ValueError("decode_workers must be a positive integer.")
        self.encoder = encoder
        self.samples = samples
        self.encoder_fingerprint = encoder_fingerprint
        self.use_masks = use_masks
        self.decode_workers = decode_workers
        self.source_digest_resolver = source_digest_resolver

    def _encode(self, item):
        shard_index, frames, detections, max_rows = item
        return (
            shard_index,
            detections,
            _encode_bounded(
                self.encoder,
                frames,
                detections,
                max_rows=max_rows,
            ),
        )

    def _infer(self, context: MaterializationContext, pending):
        if context.stage_plan.workers > 1:
            if context.stage_plan.executor != "process":
                raise ValueError("Multiple encoder workers require the spawned process executor.")
            if not isinstance(self.encoder, ReIDEncoderSpec):
                raise TypeError("Multiple encoder workers require a serializable ReIDEncoderSpec, not a model handle.")
        if context.stage_plan.executor == "process" and not isinstance(self.encoder, ReIDEncoderSpec):
            raise TypeError("Process encoder execution requires a serializable ReIDEncoderSpec, not a model handle.")
        if isinstance(self.encoder, ReIDEncoderSpec):
            tasks = [
                (
                    self.encoder,
                    shard_index,
                    frames,
                    detections,
                    context.stage_plan.batch_size,
                )
                for shard_index, frames, detections in pending
            ]
            outputs = context.executor.map(_encode_from_spec, tasks)
            inputs = {shard_index: detections for shard_index, _, detections in pending}
            return [
                (shard_index, inputs[shard_index], dimension, embeddings)
                for shard_index, dimension, embeddings in outputs
            ]
        tasks = [
            (shard_index, frames, detections, context.stage_plan.batch_size)
            for shard_index, frames, detections in pending
        ]
        return [
            (shard_index, detections, int(self.encoder.embedding_dim), outputs)
            for shard_index, detections, outputs in context.executor.map(self._encode, tasks)
        ]

    def run(self, context: MaterializationContext) -> StageOutcome:
        if context.stage_plan.name != self.name:
            raise ValueError(f"EmbedStage cannot execute plan stage {context.stage_plan.name!r}.")
        context.validate_completed_shards(EMBEDDINGS_ARTIFACT)
        staged = read_detection_batches(context.staging_root, context.build_plan.box_type)
        if self.use_masks:
            staged = attach_masks(context.staging_root, staged)
        writer = ParquetShardWriter(context.staging_root, box_type=context.build_plan.box_type)
        expected_dimension = None if isinstance(self.encoder, ReIDEncoderSpec) else int(self.encoder.embedding_dim)
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("Appearance encoder embedding_dim must be positive.")

        embedding_count = 0
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
                for shard_index, detections, worker_dimension, outputs in inferred:
                    shard_id = f"{shard_index:05d}"
                    if expected_dimension is None:
                        expected_dimension = worker_dimension
                    elif worker_dimension != expected_dimension:
                        raise ValueError("Encoder workers reported inconsistent embedding dimensions.")
                    if len(outputs) != len(detections):
                        raise RuntimeError(
                            f"Encoder returned {len(outputs)} tensors for {len(detections)} detection batches."
                        )

                    enriched: list[Detections] = []
                    for detection_batch, embeddings in zip(detections, outputs, strict=True):
                        if not isinstance(embeddings, torch.Tensor):
                            raise TypeError(f"Encoder returned {type(embeddings).__name__}, expected torch.Tensor.")
                        if (
                            embeddings.device.type != "cpu"
                            or embeddings.dtype is not torch.float32
                            or embeddings.ndim != 2
                            or not embeddings.is_contiguous()
                        ):
                            raise ValueError("Encoder output must be a contiguous CPU float32 tensor with shape [N,D].")
                        if embeddings.shape != (len(detection_batch), expected_dimension):
                            raise ValueError(
                                f"Encoder output has shape {tuple(embeddings.shape)}, expected "
                                f"({len(detection_batch)}, {expected_dimension})."
                            )
                        enriched.append(detection_batch.with_embeddings(embeddings))
                        embedding_count += len(embeddings)

                    rows = [
                        row
                        for detections_with_embeddings in enriched
                        for row in embedding_records(
                            detections_with_embeddings,
                            encoder_fingerprint=self.encoder_fingerprint,
                        )
                    ]
                    writer.write(
                        EMBEDDINGS_ARTIFACT,
                        rows,
                        shard_index=shard_index,
                        embedding_dim=expected_dimension,
                        encoder_fingerprint=self.encoder_fingerprint,
                    )
                    context.record_shard(
                        shard_id,
                        EMBEDDINGS_ARTIFACT,
                        items=len(detections),
                        rows=len(rows),
                    )
        return StageOutcome(
            artifacts=(EMBEDDINGS_ARTIFACT,),
            metrics={"embeddings": embedding_count, "dim": expected_dimension},
        )

    def release(self) -> None:
        """Release a lazily constructed encoder after all retries finish."""

        if isinstance(self.encoder, ReIDEncoderSpec):
            runtime = _WORKER_ENCODERS.pop(self.encoder, None)
            if runtime is not None:
                del runtime
                release_accelerator_memory(self.encoder.device)


__all__ = ("EmbedStage",)
