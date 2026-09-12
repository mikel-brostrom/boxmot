"""Resolution and compatibility checks for immutable materialized builds."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock
from platformdirs import user_cache_path

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.manifest import PublishedContent, StageProvenance
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, MANIFEST_FILENAME, MASKS_ARTIFACT, SCHEMA_ID, SUCCESS_FILENAME
from boxmot.datasets.storage import resolve_artifact_path
from boxmot.datasets.validation import validate_published_build

from .ids import component_content, fingerprint, stage_content
from .plan import BuildPlan, default_build_root

if TYPE_CHECKING:
    from .catalog import SourceCatalog

_BUILD_ID = re.compile(r"[0-9a-f]{64}")


class BuildCompatibilityError(ValueError):
    """The selected build cannot satisfy a requested workflow."""


def former_default_build_root() -> Path:
    """Return the external build root used before repository-local materializations."""

    return user_cache_path("boxmot") / "builds"


def _uses_repository_default(plan: BuildPlan) -> bool:
    return not os.environ.get("BOXMOT_BUILDS_DIR") and plan.build_root == (Path("runs") / "materializations").resolve()


def _manifest_matches_plan(manifest: DatasetManifest, plan: BuildPlan) -> bool:
    return manifest.build_id == plan.build_id and _manifest_matches_content(manifest, plan)


def _component_content_fingerprints(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Compare component settings without erasing inconsistent recorded hashes."""

    fingerprints = metadata.get("component_fingerprints") or {}
    components = metadata.get("components") or {}
    if not isinstance(fingerprints, Mapping) or not isinstance(components, Mapping):
        raise ValueError("Build component provenance must contain mappings.")
    recorded = dict(fingerprints)
    for name, provenance in components.items():
        if isinstance(provenance, Mapping) and recorded.get(name) == fingerprint(provenance):
            recorded[name] = fingerprint(component_content(provenance))
    return recorded


def _plan_stages(plan: BuildPlan) -> tuple[StageProvenance, ...]:
    """Describe the planned stages in their canonical execution order."""

    return tuple(
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


def _manifest_matches_content(manifest: DatasetManifest, plan: BuildPlan) -> bool:
    """Match perception semantics independently of the producing release and device."""

    return (
        manifest.box_type == plan.box_type
        and manifest.publish
        == PublishedContent(
            image_references=plan.publish.image_references,
            masks=plan.publish.masks,
            embeddings=plan.publish.embeddings,
        )
        and stage_content(manifest.stages) == stage_content(_plan_stages(plan))
        and _component_content_fingerprints(manifest.metadata) == _component_content_fingerprints(plan.metadata)
        and manifest.metadata.get("dataset_name") == plan.dataset_name
        and manifest.metadata.get("source_fingerprint") == plan.source_fingerprint
        and manifest.metadata.get("experiment_id") == plan.metadata.get("experiment_id")
        and all(
            manifest.metadata.get(key) == plan.metadata.get(key)
            for key in (
                "dataset_id",
                "split",
                "source_catalog_digest",
                "source_count",
                "class_taxonomy_digest",
                "class_bridge",
                "fps",
            )
        )
    )


def _published_build_candidates(plan: BuildPlan) -> Iterator[Path]:
    """Visit immutable build directories in deterministic order within configured roots."""

    roots = [plan.build_root]
    if _uses_repository_default(plan):
        roots.append(former_default_build_root())
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            candidates = sorted(root.iterdir())
        except OSError:
            if root == plan.build_root:
                raise
            continue
        for path in candidates:
            if _BUILD_ID.fullmatch(path.name) and not path.is_symlink() and path.is_dir():
                if (path / SUCCESS_FILENAME).is_file():
                    yield path


def find_matching_build(
    plan: BuildPlan,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Reuse a fully validated build with identical source and perception semantics.

    The creation release and device remain part of a build's immutable identity
    and provenance, but do not invalidate otherwise identical perception outputs.
    The exact requested output, when present, remains authoritative and is
    validated by the materializer rather than replaced by another candidate.
    """

    if plan.output_root.exists():
        return None
    for path in _published_build_candidates(plan):
        try:
            manifest = DatasetManifest.load(path)
            if manifest.build_id != path.name or not _manifest_matches_content(manifest, plan):
                continue
            if status_callback is not None:
                status_callback(f"Validating matching perception build {path.name[:12]}…")
            validate_published_build(path, manifest=manifest)
        except (OSError, ValueError):
            continue
        return path
    return None


def _remove_import_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _matches_fps_parent(manifest: DatasetManifest, plan: BuildPlan) -> bool:
    """Match full-rate perception semantics before checking expensive source data."""

    metadata = manifest.metadata
    if metadata.get("fps") is not None or manifest.box_type != plan.box_type:
        return False
    for key in ("dataset_id", "split", "class_taxonomy_digest", "class_bridge"):
        if metadata.get(key) != plan.metadata.get(key):
            return False
    if plan.publish.embeddings and not manifest.publish.embeddings:
        return False
    if plan.publish.masks and not manifest.publish.masks:
        return False
    actual_components = _component_content_fingerprints(metadata)
    expected_components = _component_content_fingerprints(plan.metadata)
    if any(actual_components.get(name) != value for name, value in expected_components.items() if value is not None):
        return False
    stages = {stage.name: stage for stage in stage_content(manifest.stages)}
    for expected in stage_content(_plan_stages(plan)):
        if expected.name == "finalize":
            continue
        actual = stages.get(expected.name)
        if actual != expected:
            return False
    return True


def find_fps_parent_build(
    plan: BuildPlan,
    *,
    load_catalog: Callable[[], SourceCatalog],
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Path, SourceCatalog] | None:
    """Find a validated native-rate build with the requested perception outputs.

    Discovery uses a deterministic path order, never modification time. Catalog
    construction is deferred until a candidate's models and output contract
    match. Its complete source identity then verifies pixels, GT, and timing.
    """

    if plan.metadata.get("fps") is None or plan.output_root.exists():
        return None
    catalog = None
    for path in _published_build_candidates(plan):
        try:
            manifest = DatasetManifest.load(path)
            if manifest.build_id != path.name or not _matches_fps_parent(manifest, plan):
                continue
        except (OSError, ValueError):
            continue
        if catalog is None:
            if status_callback is not None:
                status_callback("Validating full-rate source data for cached frame reuse…")
            catalog = load_catalog()
        try:
            validate_build_compatibility(
                manifest,
                dataset_id=plan.dataset_name,
                split=str(plan.metadata["split"]),
                geometry=plan.box_type,
                source_catalog_digest=catalog.fingerprint,
                class_taxonomy_digest=str(catalog.metadata["class_taxonomy_digest"]),
            )
            validate_published_build(path, manifest=manifest)
        except (OSError, ValueError):
            continue
        return path, catalog
    return None


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
    "find_fps_parent_build",
    "find_matching_build",
    "former_default_build_root",
    "import_former_default_build",
    "load_cached_build",
    "resolve_build_path",
    "validate_build_compatibility",
)
