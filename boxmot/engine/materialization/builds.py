"""Resolution and compatibility checks for immutable materialized builds."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from filelock import FileLock
from platformdirs import user_cache_path

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.manifest import PublishedContent, StageProvenance
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, MANIFEST_FILENAME, MASKS_ARTIFACT, SCHEMA_ID, SUCCESS_FILENAME
from boxmot.datasets.storage import resolve_artifact_path
from boxmot.datasets.validation import validate_published_build

from .plan import BuildPlan, default_build_root

_BUILD_ID = re.compile(r"[0-9a-f]{64}")


class BuildCompatibilityError(ValueError):
    """The selected build cannot satisfy a requested workflow."""


def former_default_build_root() -> Path:
    """Return the external build root used before repository-local materializations."""

    return user_cache_path("boxmot") / "builds"


def _uses_repository_default(plan: BuildPlan) -> bool:
    return not os.environ.get("BOXMOT_BUILDS_DIR") and plan.build_root == (Path("runs") / "materializations").resolve()


def _manifest_matches_plan(manifest: DatasetManifest, plan: BuildPlan) -> bool:
    expected_stages = tuple(
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
    return (
        manifest.build_id == plan.build_id
        and manifest.box_type == plan.box_type
        and manifest.publish
        == PublishedContent(
            image_references=plan.publish.image_references,
            masks=plan.publish.masks,
            embeddings=plan.publish.embeddings,
        )
        and manifest.stages == expected_stages
        and manifest.metadata.get("dataset_name") == plan.dataset_name
        and manifest.metadata.get("source_fingerprint") == plan.source_fingerprint
        and manifest.metadata.get("experiment_id") == plan.metadata.get("experiment_id")
    )


def _remove_import_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def import_former_default_build(
    plan: BuildPlan,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> bool:
    """Atomically import an identical v1 build from BoxMOT's former default root."""

    if plan.output_root.exists() or not _uses_repository_default(plan):
        return False
    source = former_default_build_root().expanduser().resolve() / plan.build_id
    if source == plan.output_root or source.is_symlink() or not source.is_dir():
        return False
    try:
        manifest = DatasetManifest.load(source)
        marker = json.loads((source / SUCCESS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if marker != {"schema": SCHEMA_ID, "build_id": plan.build_id} or not _manifest_matches_plan(manifest, plan):
        return False

    lock_path = plan.build_root / ".locks" / f"{plan.build_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if plan.output_root.exists():
            return False
        if status_callback is not None:
            status_callback("Importing identical materialization from the former build root…")
        import_parent = plan.build_root / ".imports"
        import_parent.mkdir(parents=True, exist_ok=True)
        for stale in import_parent.glob(f"{plan.build_id}-*"):
            _remove_import_path(stale)
        temporary = Path(tempfile.mkdtemp(prefix=f"{plan.build_id}-", dir=import_parent))
        try:
            for artifact in manifest.artifacts:
                for shard in artifact.shards:
                    source_file = resolve_artifact_path(source, shard.path)
                    destination = temporary / shard.path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
                    with destination.open("rb") as stream:
                        os.fsync(stream.fileno())
            for name in (MANIFEST_FILENAME, SUCCESS_FILENAME):
                destination = temporary / name
                shutil.copy2(source / name, destination)
                with destination.open("rb") as stream:
                    os.fsync(stream.fileno())
            _fsync_directory(temporary)
            copied_manifest = DatasetManifest.load(temporary)
            validate_published_build(temporary, manifest=copied_manifest)
            os.replace(temporary, plan.output_root)
            _fsync_directory(plan.build_root)
            return True
        except (FileNotFoundError, PermissionError, ValueError):
            return False
        finally:
            if temporary.exists():
                _remove_import_path(temporary)


def resolve_build_path(build: str | Path, *, build_root: str | Path | None = None) -> Path:
    """Resolve an existing path directly, otherwise resolve a build ID under its root."""

    requested = Path(build).expanduser()
    if requested.exists():
        resolved = requested.resolve()
    else:
        build_id = str(build)
        if "/" in build_id or "\\" in build_id:
            raise FileNotFoundError(
                f"Build path does not exist: {requested}. Build IDs cannot contain path separators."
            )
        if _BUILD_ID.fullmatch(build_id) is None:
            raise FileNotFoundError(
                f"Build path does not exist: {requested}. Build IDs must be full lowercase SHA-256 digests."
            )
        root = default_build_root() if build_root is None else Path(build_root).expanduser()
        resolved = (root / build_id).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Materialized build {build_id!r} does not exist under {root.resolve()}. "
                "Run `boxmot materialize ...` first."
            )
    DatasetManifest.load(resolved)
    return resolved


def _metadata_value(metadata: Mapping[str, Any], *path: str) -> Any:
    value: Any = metadata
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def validate_build_compatibility(
    manifest: DatasetManifest,
    *,
    dataset_id: str | None = None,
    split: str | None = None,
    geometry: str | None = None,
    source_catalog_digest: str | None = None,
    class_taxonomy_digest: str | None = None,
    component_fingerprints: Mapping[str, str] | None = None,
    require_masks: bool = False,
    require_embeddings: bool = False,
) -> None:
    """Validate semantic build identity before evaluation/tuning/research."""

    metadata = manifest.metadata
    checks = (
        ("dataset", dataset_id, _metadata_value(metadata, "dataset_id")),
        ("split", split, _metadata_value(metadata, "split")),
        ("geometry", geometry, manifest.box_type),
        ("source catalog", source_catalog_digest, _metadata_value(metadata, "source_catalog_digest")),
        ("class taxonomy", class_taxonomy_digest, _metadata_value(metadata, "class_taxonomy_digest")),
    )
    for label, expected, actual in checks:
        if expected is not None and actual != expected:
            raise BuildCompatibilityError(
                f"Build {manifest.build_id!r} {label} mismatch: expected {expected!r}, got {actual!r}."
            )

    if component_fingerprints:
        actual_fingerprints = _metadata_value(metadata, "component_fingerprints")
        actual_fingerprints = actual_fingerprints if isinstance(actual_fingerprints, Mapping) else {}
        for name, expected in component_fingerprints.items():
            actual = actual_fingerprints.get(name)
            if actual != expected:
                raise BuildCompatibilityError(
                    f"Build {manifest.build_id!r} component {name!r} mismatch: expected {expected!r}, got {actual!r}."
                )

    artifacts = manifest.artifacts_by_name
    missing: list[str] = []
    if require_masks and MASKS_ARTIFACT not in artifacts:
        missing.append("masks")
    if require_embeddings and EMBEDDINGS_ARTIFACT not in artifacts:
        missing.append("embeddings")
    if missing:
        flags = " ".join(f"--publish-{name}" for name in missing)
        raise BuildCompatibilityError(
            f"Build {manifest.build_id!r} is missing required {', '.join(missing)}. "
            f"Run `boxmot materialize ... {flags}` and pass the resulting --build."
        )


def load_cached_build(
    build: str | Path,
    *,
    build_root: str | Path | None = None,
    split: str | None = None,
    load_images: bool = False,
    load_masks: bool = False,
    load_embeddings: bool = False,
) -> CachedVisionDataset:
    """Resolve and open one explicit build with requested artifacts validated eagerly."""

    path = resolve_build_path(build, build_root=build_root)
    return CachedVisionDataset(
        path,
        split=split,
        load_images=load_images,
        load_masks=load_masks,
        load_embeddings=load_embeddings,
    )


__all__ = (
    "BuildCompatibilityError",
    "default_build_root",
    "former_default_build_root",
    "import_former_default_build",
    "load_cached_build",
    "resolve_build_path",
    "validate_build_compatibility",
)
