from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import boxmot.datasets.validation as validation_module
from boxmot.datasets import DatasetManifest
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, MASKS_ARTIFACT
from boxmot.datasets.storage import describe_parquet_artifact
from boxmot.datasets.validation import DatasetValidationError, validate_dataset, validate_published_build


def _rewrite_last_artifact_shard(
    root: Path,
    manifest: DatasetManifest,
    artifact_name: str,
    update,
) -> DatasetManifest:
    artifact = manifest.artifact(artifact_name)
    shard_path = root / artifact.shards[-1].path
    table = pq.read_table(shard_path)
    rows = table.to_pylist()
    update(rows[-1])
    rewritten = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(rewritten, shard_path, compression="zstd", row_group_size=1)
    replacement = describe_parquet_artifact(
        root,
        name=artifact_name,
        relative_path=artifact_name,
        metadata=artifact.metadata,
    )
    return replace(
        manifest,
        artifacts=tuple(
            replacement if candidate.name == artifact_name else candidate for candidate in manifest.artifacts
        ),
    )


def test_embedding_validation_streams_non_finite_vectors_across_shards(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    modified = _rewrite_last_artifact_shard(
        root,
        manifest,
        EMBEDDINGS_ARTIFACT,
        lambda row: row["values"].__setitem__(1, float("nan")),
    )

    with pytest.raises(DatasetValidationError, match="contains non-finite values"):
        validate_dataset(root, manifest=modified)


def test_embedding_validation_checks_dimensions_across_shards(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    modified = _rewrite_last_artifact_shard(
        root,
        manifest,
        EMBEDDINGS_ARTIFACT,
        lambda row: row.__setitem__("dim", 2),
    )

    with pytest.raises(DatasetValidationError, match="Embedding dimensions must be one consistent"):
        validate_dataset(root, manifest=modified)


def test_embedding_validation_checks_keys_across_shards(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    modified = _rewrite_last_artifact_shard(
        root,
        manifest,
        EMBEDDINGS_ARTIFACT,
        lambda row: row.__setitem__("instance_id", "foreign-instance"),
    )

    with pytest.raises(DatasetValidationError, match="Embedding keys must match"):
        validate_dataset(root, manifest=modified)


def test_embedding_validation_checks_sample_alignment_across_shards(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    modified = _rewrite_last_artifact_shard(
        root,
        manifest,
        EMBEDDINGS_ARTIFACT,
        lambda row: row.__setitem__("sample_id", "sample-a"),
    )

    with pytest.raises(DatasetValidationError, match="Embedding .* has the wrong sample key"):
        validate_dataset(root, manifest=modified)


def test_mask_validation_streams_payload_errors_across_shards(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    modified = _rewrite_last_artifact_shard(
        root,
        manifest,
        MASKS_ARTIFACT,
        lambda row: row.__setitem__("data", b""),
    )

    with pytest.raises(DatasetValidationError, match="Mask .* has an invalid payload"):
        validate_dataset(root, manifest=modified)


def test_optional_payload_validation_is_row_group_bounded(materialized_build, monkeypatch) -> None:
    root = materialized_build["root"]
    observed_batches: dict[str, list[int]] = {}
    embedding_projections: list[tuple[str, ...]] = []
    finite_input_sizes: list[int] = []

    original_iter_batches = validation_module._iter_artifact_batches
    original_read = validation_module.read_parquet_artifact
    original_isfinite = validation_module.np.isfinite

    def recording_batches(*args, **kwargs):
        name = args[2]
        for batch in original_iter_batches(*args, **kwargs):
            observed_batches.setdefault(name, []).append(batch.num_rows)
            yield batch

    def recording_read(*args, **kwargs):
        if kwargs.get("artifact_name") == EMBEDDINGS_ARTIFACT:
            columns = tuple(kwargs.get("columns") or ())
            embedding_projections.append(columns)
            assert "values" not in columns
        return original_read(*args, **kwargs)

    def recording_isfinite(values):
        finite_input_sizes.append(values.size)
        return original_isfinite(values)

    monkeypatch.setattr(validation_module, "PARQUET_ROW_GROUP_ROWS", 1)
    monkeypatch.setattr(validation_module, "_iter_artifact_batches", recording_batches)
    monkeypatch.setattr(validation_module, "read_parquet_artifact", recording_read)
    monkeypatch.setattr(validation_module.np, "isfinite", recording_isfinite)

    report = validate_dataset(root)

    assert report.embeddings == 3
    assert embedding_projections == [("sample_id", "instance_id", "encoder_fingerprint", "dim")]
    assert observed_batches["masks"] == [1, 1, 1]
    assert observed_batches["embeddings"] == [1, 1, 1]
    assert max(finite_input_sizes) == 3


def test_validation_hashes_each_physical_shard_once(materialized_build, monkeypatch) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    hashed: list[Path] = []
    original_sha256_file = validation_module.sha256_file

    def recording_sha256_file(path, *args, **kwargs):
        hashed.append(Path(path))
        return original_sha256_file(path, *args, **kwargs)

    monkeypatch.setattr(validation_module, "sha256_file", recording_sha256_file)

    validate_dataset(root, manifest=manifest)

    expected = [root / shard.path for artifact in manifest.artifacts for shard in artifact.shards]
    assert hashed == expected


def test_published_validation_checks_embedding_schema_metadata_without_payload_reads(
    materialized_build,
    monkeypatch,
) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)

    def reject_payload_read(*_args, **_kwargs):
        raise AssertionError("published validation must use only hashes and Parquet metadata")

    monkeypatch.setattr(validation_module, "read_parquet_artifact", reject_payload_read)
    monkeypatch.setattr(validation_module, "_iter_artifact_batches", reject_payload_read)

    report = validate_published_build(root, manifest=manifest)

    assert report.samples == 2
    assert report.instances == report.masks == report.embeddings == 3


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"dim": 4}, "dimension differs from its manifest"),
        ({"encoder_fingerprint": "f" * 64}, "encoder fingerprint differs from its manifest"),
    ],
)
def test_published_validation_binds_embedding_shard_metadata_to_manifest(
    materialized_build,
    metadata,
    message,
) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    embeddings = manifest.artifact(EMBEDDINGS_ARTIFACT)
    changed_embeddings = replace(embeddings, metadata={**dict(embeddings.metadata), **metadata})
    modified = replace(
        manifest,
        artifacts=tuple(
            changed_embeddings if artifact.name == EMBEDDINGS_ARTIFACT else artifact for artifact in manifest.artifacts
        ),
    )

    with pytest.raises(DatasetValidationError, match=message):
        validate_published_build(root, manifest=modified)


def test_published_validation_rejects_escaping_shard_symlink_before_hashing(
    materialized_build,
    monkeypatch,
) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    shard = manifest.artifact(EMBEDDINGS_ARTIFACT).shards[0]
    shard_path = root / shard.path
    outside = root.parent.parent / "outside.parquet"
    shutil.copyfile(shard_path, outside)
    shard_path.unlink()
    try:
        shard_path.symlink_to(outside)
    except OSError:
        pytest.skip("This filesystem does not permit symlinks")

    hashed_targets: list[Path] = []
    original_sha256_file = validation_module.sha256_file

    def recording_sha256_file(path, *args, **kwargs):
        hashed_targets.append(Path(path).resolve())
        return original_sha256_file(path, *args, **kwargs)

    monkeypatch.setattr(validation_module, "sha256_file", recording_sha256_file)

    with pytest.raises(DatasetValidationError, match="escapes the dataset root"):
        validate_published_build(root, manifest=manifest)

    assert outside.resolve() not in hashed_targets
