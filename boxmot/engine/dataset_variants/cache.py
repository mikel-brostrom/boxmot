"""Derive immutable caches by selecting frames without rerunning perception."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from boxmot.datasets import DatasetManifest
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, read_parquet_artifact
from boxmot.datasets.validation import validate_published_build
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    FinalizeStage,
    MaterializationContext,
    PublishOptions,
    StageOutcome,
    StagePlan,
    fingerprint,
)
from boxmot.engine.materialization.builds import validate_build_compatibility
from boxmot.engine.materialization.catalog import SourceCatalog
from boxmot.engine.materialization.progress import MaterializationProgressReporter


def _validate_selection(
    parent_build: Path,
    manifest: DatasetManifest,
    parent_catalog: SourceCatalog,
    catalog: SourceCatalog,
    sample_map: Mapping[str, str],
) -> None:
    """Require an exact image/timestamp subset of the validated parent source."""
    validate_build_compatibility(
        manifest,
        dataset_id=str(parent_catalog.metadata["dataset_id"]),
        split=str(parent_catalog.metadata["split"]),
        source_catalog_digest=parent_catalog.fingerprint,
        class_taxonomy_digest=str(parent_catalog.metadata["class_taxonomy_digest"]),
    )
    if catalog.metadata["class_taxonomy_digest"] != parent_catalog.metadata["class_taxonomy_digest"]:
        raise ValueError("Cached frame variants must preserve the parent's class taxonomy.")
    targets = {sample.sample_id: sample for sample in catalog.samples}
    originals = {sample.sample_id: sample for sample in parent_catalog.samples}
    if set(sample_map) != set(targets) or len(set(sample_map.values())) != len(sample_map):
        raise ValueError("sample_map must map every derived sample to a distinct parent sample.")
    if not set(sample_map.values()).issubset(originals):
        raise ValueError("sample_map refers to a sample outside the parent source catalog.")
    cached = {
        row["sample_id"]: row
        for row in read_parquet_artifact(
            parent_build / SAMPLES_ARTIFACT,
            artifact_name=SAMPLES_ARTIFACT,
            filters=[("sample_id", "in", list(sample_map.values()))],
        ).to_pylist()
    }
    if set(cached) != set(sample_map.values()):
        raise ValueError("The parent build is missing selected source samples.")
    for target_id, original_id in sample_map.items():
        target, original, row = targets[target_id], originals[original_id], cached[original_id]
        if target.source_sha256 != original.source_sha256 or target.image_size != original.image_size:
            raise ValueError(f"Derived sample {target_id!r} changes its parent's image bytes or dimensions.")
        if target.timestamp_s != original.timestamp_s:
            raise ValueError(f"Derived sample {target_id!r} must preserve its parent's timestamp.")
        expected = (
            original.split,
            original.sequence_id,
            original.frame_index,
            original.timestamp_s,
            *original.image_size,
        )
        actual = tuple(row[key] for key in ("split", "sequence_id", "frame_index", "timestamp_s", "height", "width"))
        if actual != expected:
            raise ValueError(f"Cached sample {original_id!r} does not match the parent source catalog.")


class _ReuseCachedFrames:
    """Copy selected Arrow batches and retain perception payloads unchanged."""

    name = "reuse-cached-frames"

    def __init__(
        self,
        parent_build: Path,
        manifest: DatasetManifest,
        catalog: SourceCatalog,
        sample_map: Mapping[str, str],
        *,
        artifacts: tuple[str, ...] | None = None,
    ) -> None:
        self.parent_build = parent_build
        self.manifest = manifest
        self.catalog = catalog
        self.target_ids = {original: target for target, original in sample_map.items()}
        self.artifacts = tuple(artifact.name for artifact in manifest.artifacts) if artifacts is None else artifacts

    def _copy_artifact(self, context: MaterializationContext, name: str) -> None:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        source = ds.dataset(str(self.parent_build / name), format="parquet")
        scanner = source.scanner(
            filter=ds.field("sample_id").isin(list(self.target_ids)),
            batch_size=4096,
            batch_readahead=1,
            fragment_readahead=1,
        )
        destination = context.staging_root / name
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for batch in scanner.to_batches():
            if not batch.num_rows:
                continue
            table = pa.Table.from_batches([batch])
            old_samples = table.column("sample_id").to_pylist()
            old_instances = table.column("instance_id").to_pylist()
            new_samples, new_instances = [], []
            for sample_id, instance_id in zip(old_samples, old_instances, strict=True):
                prefix = f"{self.manifest.build_id}:{sample_id}:"
                if not instance_id.startswith(prefix):
                    raise ValueError("Parent cache contains a non-canonical instance ID.")
                target_id = self.target_ids[sample_id]
                new_samples.append(target_id)
                new_instances.append(f"{context.build_plan.build_id}:{target_id}:{instance_id[len(prefix) :]}")
            for field, values in (("sample_id", new_samples), ("instance_id", new_instances)):
                index = table.schema.get_field_index(field)
                table = table.set_column(index, table.schema.field(index), pa.array(values, type=pa.string()))
            # All non-key Arrow arrays, including fixed-size embedding vectors
            # and encoded masks, are retained without Python payload conversion.
            pq.write_table(table, destination / f"part-{written:05d}.parquet", compression="zstd")
            written += 1
        if not written:
            pq.write_table(pa.Table.from_batches([], schema=source.schema), destination / "part-00000.parquet")
            written = 1
        # A retry of this same deterministic stage must not retain old tail shards.
        for stale in destination.glob("part-*.parquet"):
            if int(stale.stem.removeprefix("part-")) >= written:
                stale.unlink()

    def run(self, context: MaterializationContext) -> StageOutcome:
        """Write complete sample rows, including frames with no detections."""
        for name in self.artifacts:
            destination = context.staging_root / name
            if destination.exists():
                shutil.rmtree(destination)
        if SAMPLES_ARTIFACT in self.artifacts:
            writer = ParquetShardWriter(context.staging_root, box_type=self.manifest.box_type)
            writer.write(
                SAMPLES_ARTIFACT,
                (
                    {
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "sequence_id": sample.sequence_id,
                        "frame_index": sample.frame_index,
                        "timestamp_s": sample.timestamp_s,
                        "image_ref": sample.source_uri if context.build_plan.publish.image_references else None,
                        "height": sample.image_size[0],
                        "width": sample.image_size[1],
                    }
                    for sample in self.catalog.samples
                ),
                shard_index=0,
            )
        for name in self.artifacts:
            if name != SAMPLES_ARTIFACT:
                self._copy_artifact(context, name)
        return StageOutcome(artifacts=self.artifacts, metrics={"samples": len(self.catalog.samples)})


class _ReusePlannedStage(_ReuseCachedFrames):
    """Restore one canonical perception stage under its original plan identity."""

    def __init__(
        self,
        name: str,
        parent_build: Path,
        manifest: DatasetManifest,
        catalog: SourceCatalog,
        sample_map: Mapping[str, str],
        *,
        artifacts: tuple[str, ...],
    ) -> None:
        super().__init__(parent_build, manifest, catalog, sample_map, artifacts=artifacts)
        self.name = name

    def _repartition_shards(self, context: MaterializationContext, name: str) -> None:
        """Keep restored checkpoints aligned with normal inference sample batches."""
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        destination = context.staging_root / name
        fragments = context.staging_root / f".{name}.repartition"
        if fragments.exists():
            shutil.rmtree(fragments)
        fragments.mkdir()
        batch_size = context.stage_plan.batch_size
        shard_by_sample = {
            sample.sample_id: index // batch_size for index, sample in enumerate(self.catalog.samples)
        }
        source = ds.dataset(str(destination), format="parquet")
        schema = source.schema
        scanner = source.scanner(batch_size=4096, batch_readahead=1, fragment_readahead=1)
        for batch_index, batch in enumerate(scanner.to_batches()):
            if not batch.num_rows:
                continue
            rows_by_shard: dict[int, list[int]] = {}
            for index, sample_id in enumerate(batch.column("sample_id").to_pylist()):
                rows_by_shard.setdefault(shard_by_sample[sample_id], []).append(index)
            table = pa.Table.from_batches([batch])
            for shard_index, indices in rows_by_shard.items():
                fragment_root = fragments / f"{shard_index:05d}"
                fragment_root.mkdir(exist_ok=True)
                pq.write_table(table.take(indices), fragment_root / f"{batch_index:05d}.parquet", compression="zstd")
        shutil.rmtree(destination)
        destination.mkdir()
        shard_count = (len(self.catalog.samples) + batch_size - 1) // batch_size
        for shard_index in range(shard_count):
            path = destination / f"part-{shard_index:05d}.parquet"
            with pq.ParquetWriter(path, schema, compression="zstd") as writer:
                for path in sorted((fragments / f"{shard_index:05d}").glob("*.parquet")):
                    writer.write_table(pq.read_table(path))
        shutil.rmtree(fragments)

    def run(self, context: MaterializationContext) -> StageOutcome:
        """Replace partial inference shards and checkpoint the copied artifacts."""
        current = context.state.state.by_name[self.name]
        had_local_artifacts = bool(current.completed_shards or current.artifacts) or any(
            (context.staging_root / name).exists() for name in self.artifacts
        )
        if current.status == "completed":
            context.state.invalidate((self.name,))
            context.state.begin(self.name)
        temporary = context.build_plan.build_root / ".staging" / f".reuse-{context.build_plan.build_id}-{self.name}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        prepared = replace(context, staging_root=temporary)
        try:
            super().run(prepared)
            for name in self.artifacts:
                self._repartition_shards(prepared, name)
            context.state.replace_shards(self.name, {})
            for name in self.artifacts:
                destination = context.staging_root / name
                if destination.exists():
                    shutil.rmtree(destination)
                (temporary / name).replace(destination)
        finally:
            shutil.rmtree(temporary)
        shards: dict[str, dict[str, str]] = {}
        for name in self.artifacts:
            for path in sorted((context.staging_root / name).glob("part-*.parquet")):
                shard_id = path.stem.removeprefix("part-")
                shards.setdefault(shard_id, {})[name] = sha256_file(path)
        context.state.replace_shards(self.name, shards)
        context.repaired_shards = had_local_artifacts
        return StageOutcome(
            artifacts=self.artifacts,
            metrics={"cache": "hit", "parent_build_id": self.manifest.build_id, "samples": len(self.catalog.samples)},
        )


def reuse_cached_build(
    parent_build: Path,
    *,
    parent_catalog: SourceCatalog,
    catalog: SourceCatalog,
    sample_map: Mapping[str, str],
    plan: BuildPlan,
    progress: MaterializationProgressReporter | None = None,
    embedding_metadata: Mapping[str, Any] | None = None,
    target_shard_rows: int = 50_000,
) -> Path:
    """Fill a compatible canonical build plan from cached selected frame outputs.

    Callers must establish matching perception configuration before reusing a
    parent. This helper validates source selection and payload availability, then
    preserves the requested build identity and metadata while executing copied
    detect/segment/embed outputs through the regular locked materializer.
    """
    parent_build = Path(parent_build).expanduser().resolve()
    manifest = DatasetManifest.load(parent_build)
    validate_published_build(parent_build, manifest=manifest)
    if plan.source_fingerprint != catalog.fingerprint:
        raise ValueError("The requested build plan does not match the selected source catalog.")
    if plan.box_type != manifest.box_type:
        raise ValueError("Cached frame reuse requires matching geometry.")
    if plan.publish.masks and not manifest.publish.masks:
        raise ValueError("The parent build does not contain the requested masks.")
    if plan.publish.embeddings and not manifest.publish.embeddings:
        raise ValueError("The parent build does not contain the requested embeddings.")
    names = set(plan.stage_by_name)
    if not {"detect", "finalize"} <= names or names - {"detect", "segment", "embed", "finalize"}:
        raise ValueError("Cached frame reuse requires a detect/segment/embed/finalize build plan.")
    sample_map = dict(sample_map)
    _validate_selection(parent_build, manifest, parent_catalog, catalog, sample_map)
    artifacts = {"detect": [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]}
    for name, requested, stage in (
        (MASKS_ARTIFACT, plan.publish.masks, "segment"),
        (EMBEDDINGS_ARTIFACT, plan.publish.embeddings, "embed"),
    ):
        if requested:
            artifacts.setdefault(stage if stage in names else "detect", []).append(name)
    stages = [
        _ReusePlannedStage(
            stage.name, parent_build, manifest, catalog, sample_map, artifacts=tuple(artifacts.get(stage.name, ()))
        )
        for stage in plan.ordered_stages()
        if stage.name != "finalize"
    ]
    if plan.publish.embeddings and embedding_metadata is None:
        embedding_metadata = dict(manifest.artifact(EMBEDDINGS_ARTIFACT).metadata)
    return DatasetMaterializer(
        plan,
        [*stages, FinalizeStage(embedding_metadata=embedding_metadata, target_shard_rows=target_shard_rows)],
        progress=progress,
    ).run()


def derive_cached_build(
    parent_build: Path,
    *,
    parent_catalog: SourceCatalog,
    catalog: SourceCatalog,
    sample_map: Mapping[str, str],
    build_root: Path,
) -> Path:
    """Publish a validated frame subset with original detections and embeddings.

    ``sample_map`` maps each derived sample ID to its original sample ID. Both
    catalogs must describe identical selected image bytes, dimensions, capture
    timestamps, and taxonomy. The derived catalog owns sequence/frame numbering
    and ground truth; perception component identities remain those of the parent.
    """
    parent_build = Path(parent_build).expanduser().resolve()
    manifest = DatasetManifest.load(parent_build)
    validate_published_build(parent_build, manifest=manifest)
    sample_map = dict(sample_map)
    _validate_selection(parent_build, manifest, parent_catalog, catalog, sample_map)
    lineage = {
        "kind": "frame-subset",
        "parent_build_id": manifest.build_id,
        "parent_source_catalog_digest": parent_catalog.fingerprint,
        "sample_mapping_digest": fingerprint(sample_map),
    }
    reuse = StagePlan.create(
        _ReuseCachedFrames.name,
        config=lineage,
        component={"parent_artifacts": {artifact.name: artifact.sha256 for artifact in manifest.artifacts}},
    )
    finalize = StagePlan.create("finalize", depends_on=(reuse.name,), upstream_fingerprints=(reuse.fingerprint,))
    metadata = {
        key: manifest.metadata[key]
        for key in ("components", "component_fingerprints", "class_bridge")
        if key in manifest.metadata
    }
    metadata.update(catalog.metadata)
    metadata["geometry"] = manifest.box_type
    metadata["derivation"] = {
        **lineage,
        "parent_build": str(parent_build),
        "parent_experiment_id": manifest.metadata.get("experiment_id"),
        "parent_stages": [stage.to_dict() for stage in manifest.stages],
    }
    plan = BuildPlan.create(
        build_root=build_root,
        dataset_name=str(catalog.metadata["dataset_id"]),
        box_type=manifest.box_type,
        source_fingerprint=catalog.fingerprint,
        publish=PublishOptions(
            image_references=True, masks=manifest.publish.masks, embeddings=manifest.publish.embeddings
        ),
        stages=(reuse, finalize),
        metadata=metadata,
    )
    embedding_metadata = dict(manifest.artifact(EMBEDDINGS_ARTIFACT).metadata) if manifest.publish.embeddings else None
    return DatasetMaterializer(
        plan,
        [
            _ReuseCachedFrames(parent_build, manifest, catalog, sample_map),
            FinalizeStage(embedding_metadata=embedding_metadata),
        ],
    ).run()


__all__ = ("derive_cached_build", "reuse_cached_build")
