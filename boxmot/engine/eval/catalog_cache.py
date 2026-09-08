"""Evaluation reuse of shared source metadata and eval-only artifact metadata.

Evaluation verifies that a materialized build still belongs to the selected
raw dataset. Re-hashing every image on every invocation is unnecessary when a
file's complete local identity is unchanged. Source catalog metadata is shared
with materialization; this module owns only the additional artifact-resolution
cache needed by evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from boxmot.components.artifacts import ResolvedArtifact, resolve_artifact
from boxmot.datasets.manifest import canonical_json_bytes
from boxmot.engine.materialization.catalog import (
    SourceCatalog,
    catalog_mot_dataset,
    inspect_catalog_file,
    resolve_dataset_root,
)
from boxmot.engine.materialization.metadata_cache import (
    FileMetadataCache,
    StatusCallback,
    default_source_metadata_cache_path,
)

_CACHE_SCHEMA = "boxmot.eval-file-metadata/v1"


class EvaluationFileMetadataCache(FileMetadataCache):
    """Evaluation artifact cache using the shared stat-safe implementation."""

    def __init__(
        self,
        path: str | Path,
        *,
        status_callback: StatusCallback | None = None,
    ) -> None:
        super().__init__(
            path,
            status_callback=status_callback,
            metadata_resolver=inspect_catalog_file,
            progress_label="Validating source catalog",
            write_schema=_CACHE_SCHEMA,
            discover_legacy_evaluation=False,
        )

    def __enter__(self) -> EvaluationFileMetadataCache:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.save()


class EvaluationArtifactResolver:
    """Resolve model artifacts with safe file-digest reuse during evaluation.

    Directory artifacts retain the canonical resolver's fresh recursive hash.
    A directory identity includes its complete relative file set, so reducing
    it to the directory inode timestamps would weaken that contract.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        status_callback: StatusCallback | None = None,
    ) -> None:
        self._cache = EvaluationFileMetadataCache(
            default_evaluation_artifact_cache_path() if cache_path is None else cache_path,
            status_callback=status_callback,
        )

    def __call__(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        expected_sha256: str | None = None,
        allow_download: bool = False,
    ) -> ResolvedArtifact:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            return resolve_artifact(
                resolved,
                source_uri=source_uri,
                expected_sha256=expected_sha256,
                allow_download=allow_download,
            )

        metadata = self._cache.resolve(resolved, False)
        if expected_sha256 is not None and metadata.sha256 != expected_sha256:
            raise ValueError(
                f"Artifact SHA-256 mismatch for {resolved}: expected {expected_sha256}, got {metadata.sha256}."
            )
        return ResolvedArtifact(
            path=resolved,
            sha256=metadata.sha256,
            source_uri=source_uri,
        )

    def __enter__(self) -> EvaluationArtifactResolver:
        self._cache.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._cache.__exit__(exc_type, exc, traceback)


def default_evaluation_catalog_cache_path(
    config: Mapping[str, Any],
    *,
    split: str,
    data_root: str | Path | None,
) -> Path:
    """Return the canonical source-metadata cache shared with materialization."""

    dataset_root = resolve_dataset_root(config, data_root)
    return default_source_metadata_cache_path(dataset_root)


def _legacy_evaluation_catalog_cache_path(
    config: Mapping[str, Any],
    *,
    split: str,
    data_root: str | Path | None,
) -> Path:
    """Return the pre-shared-cache path for backward-compatible reads."""

    dataset_root = resolve_dataset_root(config, data_root)
    namespace = {
        "dataset_root": str(dataset_root),
        "dataset_id": str(config.get("id") or ""),
        "layout": str(config.get("layout") or ""),
        "split": split,
    }
    key = hashlib.sha256(canonical_json_bytes(namespace)).hexdigest()
    return user_cache_path("boxmot") / "evaluation" / "source-catalogs" / f"{key}.json"


def default_evaluation_artifact_cache_path() -> Path:
    """Return the platform-cache path for eval-only model file identities."""

    return user_cache_path("boxmot") / "evaluation" / "artifact-metadata.json"


def catalog_mot_dataset_for_evaluation(
    config: Mapping[str, Any],
    *,
    split: str,
    data_root: str | Path | None = None,
    cache_path: str | Path | None = None,
    status_callback: StatusCallback | None = None,
    fps: float | None = None,
) -> SourceCatalog:
    """Catalog MOT data with safe, eval-only reuse of file metadata."""

    if cache_path is None:
        resolved_cache_path = default_evaluation_catalog_cache_path(
            config,
            split=split,
            data_root=data_root,
        )
        cache: FileMetadataCache = FileMetadataCache(
            resolved_cache_path,
            status_callback=status_callback,
            metadata_resolver=inspect_catalog_file,
            fallback_paths=(
                _legacy_evaluation_catalog_cache_path(
                    config,
                    split=split,
                    data_root=data_root,
                ),
            ),
            progress_label="Validating source catalog",
            discover_legacy_evaluation=False,
        )
    else:
        resolved_cache_path = Path(cache_path).expanduser()
        cache = EvaluationFileMetadataCache(
            resolved_cache_path,
            status_callback=status_callback,
        )
    with cache:
        catalog = catalog_mot_dataset(
            config,
            split=split,
            data_root=data_root,
            metadata_resolver=cache.resolve,
            fps=fps,
        )
    return catalog


__all__ = (
    "EvaluationArtifactResolver",
    "EvaluationFileMetadataCache",
    "catalog_mot_dataset_for_evaluation",
    "default_evaluation_artifact_cache_path",
    "default_evaluation_catalog_cache_path",
)
