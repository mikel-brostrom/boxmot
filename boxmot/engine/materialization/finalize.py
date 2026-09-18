"""Validation and atomic publication of immutable dataset builds."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from boxmot.datasets.manifest import DatasetManifest, ManifestError, PublishedContent, StageProvenance
from boxmot.datasets.schema import (
    ARTIFACT_PATHS,
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MANIFEST_FILENAME,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SUCCESS_FILENAME,
    BoxType,
)
from boxmot.datasets.storage import (
    describe_parquet_artifact,
    read_embedding_metadata,
    write_compacted_parquet_artifact,
)
from boxmot.datasets.validation import validate_dataset

from .plan import BuildPlan


class FinalizeError(RuntimeError):
    """Raised when an immutable dataset cannot be safely published."""


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some platforms/filesystems do not permit directory fsync.
        pass


def compact_artifact(
    root: Path,
    *,
    artifact_name: str,
    box_type: BoxType,
    target_rows: int = 50_000,
) -> None:
    """Globally sort one artifact and rewrite deterministic bounded shards."""

    directory = root / ARTIFACT_PATHS[artifact_name]
    temporary = root / f".{artifact_name}.compacting"
    backup = root / f".{artifact_name}.previous"
    if directory.exists():
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            shutil.rmtree(backup)
    elif backup.exists():
        os.replace(backup, directory)
        _fsync_directory(root)
        if temporary.exists():
            shutil.rmtree(temporary)
    elif temporary.exists():
        shutil.rmtree(temporary)
    try:
        shards = write_compacted_parquet_artifact(
            directory,
            temporary,
            artifact_name=artifact_name,
            box_type=box_type,
            target_rows=target_rows,
        )
        for path in shards:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(temporary)
        os.replace(directory, backup)
        _fsync_directory(root)
        try:
            os.replace(temporary, directory)
            _fsync_directory(root)
        except BaseException:
            os.replace(backup, directory)
            _fsync_directory(root)
            raise
        shutil.rmtree(backup)
        _fsync_directory(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def compact_build(plan: BuildPlan, *, target_rows: int = 50_000, root: Path | None = None) -> None:
    """Compact every selected artifact before checksums and publication."""

    selected = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
    if plan.publish.masks:
        selected.append(MASKS_ARTIFACT)
    if plan.publish.embeddings:
        selected.append(EMBEDDINGS_ARTIFACT)
    for artifact_name in selected:
        compact_artifact(
            plan.staging_root if root is None else root,
            artifact_name=artifact_name,
            box_type=plan.box_type,
            target_rows=target_rows,
        )


def _remove_unpublished_artifacts(plan: BuildPlan, *, root: Path) -> None:
    for artifact_name, published in (
        (MASKS_ARTIFACT, plan.publish.masks),
        (EMBEDDINGS_ARTIFACT, plan.publish.embeddings),
    ):
        path = root / ARTIFACT_PATHS[artifact_name]
        if not published and path.exists():
            if not path.is_dir():
                raise FinalizeError(f"Unpublished artifact path is not a directory: {path}")
            shutil.rmtree(path)


def _remove_stale_atomic_temps(root: Path) -> None:
    """Remove only materializer-owned temp files left by a terminated process."""

    prefixes = (".manifest-", ".state-", ".success-", ".part-")
    for path in root.rglob(".*"):
        if path.is_file() and path.name.startswith(prefixes):
            path.unlink()


def build_manifest(
    plan: BuildPlan,
    *,
    embedding_metadata: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> DatasetManifest:
    """Describe selected artifacts in staging or a publication candidate."""

    root = plan.staging_root if root is None else root

    selected = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
    if plan.publish.masks:
        selected.append(MASKS_ARTIFACT)
    if plan.publish.embeddings:
        embedding_path = root / ARTIFACT_PATHS[EMBEDDINGS_ARTIFACT]
        try:
            embedding_metadata = read_embedding_metadata(embedding_path, declared=embedding_metadata)
        except ValueError as exc:
            raise FinalizeError(str(exc)) from exc
        selected.append(EMBEDDINGS_ARTIFACT)

    artifacts = []
    for name in selected:
        metadata = embedding_metadata if name == EMBEDDINGS_ARTIFACT else None
        artifacts.append(
            describe_parquet_artifact(
                root,
                name=name,
                relative_path=name,
                metadata=metadata,
            )
        )

    provenance = tuple(
        StageProvenance(
            name=stage.name,
            fingerprint=stage.fingerprint,
            batch_size=stage.batch_size,
            inputs=stage.depends_on,
            component=stage.component,
            config=stage.config,
        )
        for stage in plan.ordered_stages()
    )
    metadata = {
        **dict(plan.metadata),
        "dataset_name": plan.dataset_name,
        "source_fingerprint": plan.source_fingerprint,
        "publish": {
            "image_references": plan.publish.image_references,
            "masks": plan.publish.masks,
            "embeddings": plan.publish.embeddings,
        },
    }
    return DatasetManifest(
        build_id=plan.build_id,
        box_type=plan.box_type,
        artifacts=tuple(artifacts),
        publish=PublishedContent(
            image_references=plan.publish.image_references,
            masks=plan.publish.masks,
            embeddings=plan.publish.embeddings,
        ),
        stages=provenance,
        metadata=metadata,
    )


def _write_success(root: Path, build_id: str) -> None:
    payload = json.dumps({"schema": SCHEMA_ID, "build_id": build_id}, sort_keys=True) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=root, prefix=".success-", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, root / SUCCESS_FILENAME)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def finalize_build(
    plan: BuildPlan,
    *,
    embedding_metadata: Mapping[str, Any] | None = None,
    verify_hashes: bool = True,
    before_publish: Callable[[], None] | None = None,
    target_shard_rows: int = 50_000,
) -> Path:
    """Validate staging contents and atomically publish them as an immutable build."""

    if plan.output_root.exists():
        try:
            existing = DatasetManifest.load(plan.output_root)
        except ManifestError as exc:
            raise FinalizeError(f"Output path already exists but is not a valid build: {plan.output_root}") from exc
        if existing.build_id != plan.build_id:
            raise FinalizeError(f"Output path belongs to a different build: {plan.output_root}")
        validate_dataset(plan.output_root, manifest=existing, verify_hashes=verify_hashes)
        return plan.output_root

    if not plan.staging_root.is_dir():
        raise FinalizeError(f"Staging directory does not exist: {plan.staging_root}")
    (plan.staging_root / SUCCESS_FILENAME).unlink(missing_ok=True)
    _remove_stale_atomic_temps(plan.staging_root)
    # Compaction changes filenames and hashes. Keep inference checkpoints intact
    # until a fully validated candidate has been atomically published.
    candidate = plan.staging_root / ".publishing"
    if candidate.exists():
        shutil.rmtree(candidate)

    def ignore_checkpoint_metadata(directory: str, _names: list[str]) -> set[str]:
        """Exclude publication scratch and metadata owned by this finalization."""
        if Path(directory) == plan.staging_root:
            return {candidate.name, MANIFEST_FILENAME, SUCCESS_FILENAME, plan.state_path.name}
        return set()

    shutil.copytree(
        plan.staging_root, candidate, copy_function=os.link, symlinks=True,
        ignore=ignore_checkpoint_metadata,
    )
    compact_build(plan, target_rows=target_shard_rows, root=candidate)
    manifest = build_manifest(plan, embedding_metadata=embedding_metadata, root=candidate)
    manifest.write(candidate / MANIFEST_FILENAME)
    validate_dataset(
        candidate,
        manifest=manifest,
        verify_hashes=verify_hashes,
        require_success=False,
    )
    _remove_unpublished_artifacts(plan, root=candidate)
    if before_publish is not None:
        before_publish()
    if plan.state_path.is_file():
        os.link(plan.state_path, candidate / plan.state_path.name)
    _write_success(candidate, plan.build_id)
    _fsync_directory(candidate)

    plan.output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(candidate, plan.output_root)
    except OSError as exc:
        (candidate / SUCCESS_FILENAME).unlink(missing_ok=True)
        _fsync_directory(candidate)
        raise FinalizeError(f"Unable to atomically publish dataset build to {plan.output_root}") from exc
    _fsync_directory(plan.output_root.parent)
    shutil.rmtree(plan.staging_root)
    return plan.output_root


__all__ = ("FinalizeError", "build_manifest", "compact_artifact", "compact_build", "finalize_build")
