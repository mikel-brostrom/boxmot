"""Versioned Parquet schemas for materialized BoxMOT datasets.

The public dataset representation is deliberately key based.  Rows in optional
tables are joined through ``instance_id`` and never through their physical row
position in a Parquet file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pyarrow as pa

SCHEMA_ID = "boxmot.dataset/v1"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "_SUCCESS"

SAMPLES_ARTIFACT = "samples"
INSTANCES_ARTIFACT = "instances"
MASKS_ARTIFACT = "masks"
EMBEDDINGS_ARTIFACT = "embeddings"

SAMPLES_PATH = "samples"
INSTANCES_PATH = "instances"
MASKS_PATH = "masks"
EMBEDDINGS_PATH = "embeddings"

SAMPLES_FILENAME = f"{SAMPLES_PATH}/part-00000.parquet"
INSTANCES_FILENAME = f"{INSTANCES_PATH}/part-00000.parquet"
MASKS_FILENAME = f"{MASKS_PATH}/part-00000.parquet"
EMBEDDINGS_FILENAME = f"{EMBEDDINGS_PATH}/part-00000.parquet"

BoxType = Literal["aabb", "obb"]


class ParquetDependencyError(ImportError):
    """Raised when the optional Parquet runtime is unavailable."""


def require_pyarrow():
    """Import and return :mod:`pyarrow` with an actionable failure message."""

    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised without the optional runtime
        raise ParquetDependencyError(
            "Materialized BoxMOT datasets require pyarrow. Install a BoxMOT "
            "environment that includes the Parquet dependencies."
        ) from exc
    return pa


def samples_schema() -> "pa.Schema":
    """Return the schema for one row per source sample."""

    pa = require_pyarrow()
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("sequence_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("timestamp_s", pa.float64()),
            pa.field("image_ref", pa.string()),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
        ]
    )


def instances_schema(box_type: BoxType) -> "pa.Schema":
    """Return the geometry-specific detection table schema."""

    pa = require_pyarrow()
    if box_type == "aabb":
        geometry = [
            pa.field("x1", pa.float32(), nullable=False),
            pa.field("y1", pa.float32(), nullable=False),
            pa.field("x2", pa.float32(), nullable=False),
            pa.field("y2", pa.float32(), nullable=False),
        ]
    elif box_type == "obb":
        geometry = [
            pa.field("cx", pa.float32(), nullable=False),
            pa.field("cy", pa.float32(), nullable=False),
            pa.field("w", pa.float32(), nullable=False),
            pa.field("h", pa.float32(), nullable=False),
            pa.field("angle", pa.float32(), nullable=False),
        ]
    else:
        raise ValueError(f"Unknown box type {box_type!r}.")
    return pa.schema(
        [
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("detection_index", pa.int32(), nullable=False),
            *geometry,
            pa.field("score", pa.float32(), nullable=False),
            pa.field("class_id", pa.int64(), nullable=False),
        ]
    )


def masks_schema() -> "pa.Schema":
    """Return the key-aligned, bit-packed mask table schema."""

    pa = require_pyarrow()
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("codec", pa.string(), nullable=False),
            pa.field("data", pa.binary(), nullable=False),
        ]
    )


def embeddings_schema(dim: int) -> "pa.Schema":
    """Return the key-aligned embedding table schema."""

    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    pa = require_pyarrow()
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("encoder_fingerprint", pa.string(), nullable=False),
            pa.field("dim", pa.int32(), nullable=False),
            pa.field("values", pa.list_(pa.float32(), dim), nullable=False),
        ]
    )


ARTIFACT_FILENAMES = {
    SAMPLES_ARTIFACT: SAMPLES_FILENAME,
    INSTANCES_ARTIFACT: INSTANCES_FILENAME,
    MASKS_ARTIFACT: MASKS_FILENAME,
    EMBEDDINGS_ARTIFACT: EMBEDDINGS_FILENAME,
}

ARTIFACT_PATHS = {
    SAMPLES_ARTIFACT: SAMPLES_PATH,
    INSTANCES_ARTIFACT: INSTANCES_PATH,
    MASKS_ARTIFACT: MASKS_PATH,
    EMBEDDINGS_ARTIFACT: EMBEDDINGS_PATH,
}

ARTIFACT_SCHEMAS = {
    SAMPLES_ARTIFACT: samples_schema,
    MASKS_ARTIFACT: masks_schema,
}


__all__ = (
    "ARTIFACT_FILENAMES",
    "ARTIFACT_PATHS",
    "ARTIFACT_SCHEMAS",
    "BoxType",
    "EMBEDDINGS_ARTIFACT",
    "EMBEDDINGS_FILENAME",
    "EMBEDDINGS_PATH",
    "INSTANCES_ARTIFACT",
    "INSTANCES_FILENAME",
    "INSTANCES_PATH",
    "MANIFEST_FILENAME",
    "MASKS_ARTIFACT",
    "MASKS_FILENAME",
    "MASKS_PATH",
    "ParquetDependencyError",
    "SAMPLES_ARTIFACT",
    "SAMPLES_FILENAME",
    "SAMPLES_PATH",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SUCCESS_FILENAME",
    "embeddings_schema",
    "instances_schema",
    "masks_schema",
    "require_pyarrow",
    "samples_schema",
)
