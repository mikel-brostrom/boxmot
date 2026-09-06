"""Content-addressed reuse of detector outputs across derived builds."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from boxmot import __version__
from boxmot.datasets.manifest import DatasetManifest, ManifestError, PublishedContent, StageProvenance, sha256_file
from boxmot.datasets.schema import (
    ARTIFACT_PATHS,
    INSTANCES_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SCHEMA_VERSION,
    SUCCESS_FILENAME,
    BoxType,
)
from boxmot.datasets.storage import (
    artifact_files,
    describe_parquet_artifact,
    resolve_artifact_path,
    write_parquet_records,
    write_rekeyed_instance_artifact,
    write_repartitioned_instance_artifact,
)
from boxmot.datasets.validation import DatasetValidationError, validate_dataset, validate_published_build

from .builds import former_default_build_root
from .ids import fingerprint
from .plan import BuildPlan, StagePlan
from .source import SourceSample
from .stages.base import StageOutcome

if TYPE_CHECKING:
    from .stages.base import MaterializationContext


DETECTION_CACHE_SCHEMA = "boxmot.materialization/detect-cache/v1"
_BUILD_ID = re.compile(r"^[0-9a-f]{64}$")


def make_detection_cache_id(plan: BuildPlan, detect_plan: StagePlan) -> str:
    """Return detector-output identity without final-build or ReID identity."""

    if detect_plan.name != "detect":
        raise ValueError("Detection cache identity requires the detect stage plan.")
    return fingerprint(
        {
            "cache_schema": DETECTION_CACHE_SCHEMA,
            "dataset_schema": SCHEMA_ID,
            "dataset_schema_version": SCHEMA_VERSION,
            "boxmot_version": __version__,
            "dataset_name": plan.dataset_name,
            "source_fingerprint": plan.source_fingerprint,
            "box_type": plan.box_type,
            "detect_stage_fingerprint": detect_plan.fingerprint,
        }
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Directory fsync is unavailable on some supported filesystems.
        pass


def _remove_owned_path(path: Path) -> None:
    """Remove one exact cache-owned path without following directory symlinks."""

    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _path_exists(path: Path) -> bool:
    """Return whether a path entry exists, including a dangling symlink."""

    return os.path.lexists(path)


def _reject_symlinked_components(path: Path, *, root: Path) -> None:
    """Refuse cache paths that escape through a symlinked descendant."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Detection cache path escapes its build root: {path}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"Detection cache path contains a symlink: {current}")


def _write_success(root: Path, build_id: str) -> None:
    payload = json.dumps({"schema": SCHEMA_ID, "build_id": build_id}, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=root,
            prefix=".success-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / SUCCESS_FILENAME)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _sample_rows(samples: tuple[SourceSample, ...], *, image_references: bool) -> list[dict[str, Any]]:
    rows = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "sequence_id": sample.sequence_id,
            "frame_index": sample.frame_index,
            "timestamp_s": sample.timestamp_s,
            "image_ref": sample.image_ref if image_references else None,
            "height": sample.image_size[0],
            "width": sample.image_size[1],
        }
        for sample in samples
    ]
    return sorted(rows, key=lambda row: (row["split"], row["sequence_id"], row["frame_index"], row["sample_id"]))


def _write_sample_shards(
    root: Path,
    samples: tuple[SourceSample, ...],
    *,
    image_references: bool,
    shard_count: int,
) -> tuple[Path, ...]:
    """Write the sample universe across the detector artifact's shard IDs."""

    if shard_count <= 0:
        raise ValueError("Detection cache shard_count must be positive.")
    rows = _sample_rows(samples, image_references=image_references)
    paths = []
    for shard_index in range(shard_count):
        start = shard_index * len(rows) // shard_count
        end = (shard_index + 1) * len(rows) // shard_count
        path = root / ARTIFACT_PATHS[SAMPLES_ARTIFACT] / f"part-{shard_index:05d}.parquet"
        write_parquet_records(path, rows[start:end], artifact_name=SAMPLES_ARTIFACT)
        paths.append(path)
    return tuple(paths)


def _write_restored_detection_shards(
    source: Path,
    destination: Path,
    samples: tuple[SourceSample, ...],
    *,
    box_type: BoxType,
    source_build_id: str,
    target_build_id: str,
    batch_size: int,
    image_references: bool,
) -> tuple[Path, ...]:
    """Stream cache rows into the target detector's sample-batch checkpoints."""

    if batch_size <= 0:
        raise ValueError("Detection cache restore batch_size must be positive.")
    shard_count = (len(samples) + batch_size - 1) // batch_size
    shard_by_sample: dict[str, int] = {}
    for shard_index, start in enumerate(range(0, len(samples), batch_size)):
        batch = samples[start : start + batch_size]
        shard_by_sample.update({sample.sample_id: shard_index for sample in batch})
        sample_path = destination / ARTIFACT_PATHS[SAMPLES_ARTIFACT] / f"part-{shard_index:05d}.parquet"
        write_parquet_records(
            sample_path,
            _sample_rows(batch, image_references=image_references),
            artifact_name=SAMPLES_ARTIFACT,
        )
    return write_repartitioned_instance_artifact(
        source,
        destination / ARTIFACT_PATHS[INSTANCES_ARTIFACT],
        box_type=box_type,
        source_build_id=source_build_id,
        target_build_id=target_build_id,
        shard_by_sample=shard_by_sample,
        shard_count=shard_count,
    )


def _install_artifact_directories(
    temporary: Path,
    target: Path,
    artifact_names: tuple[str, ...],
) -> None:
    """Install a prepared artifact generation with rollback on ordinary errors."""

    backup = temporary / ".previous"
    backup.mkdir()
    moved_old: list[str] = []
    installed: list[str] = []
    try:
        for name in artifact_names:
            destination = target / ARTIFACT_PATHS[name]
            if not _path_exists(destination):
                continue
            if destination.is_symlink() or not destination.is_dir():
                raise RuntimeError(f"Materialization artifact path is not a directory: {destination}")
            os.replace(destination, backup / name)
            moved_old.append(name)
        for name in artifact_names:
            os.replace(temporary / ARTIFACT_PATHS[name], target / ARTIFACT_PATHS[name])
            installed.append(name)
        _fsync_directory(target)
    except BaseException:
        for name in reversed(installed):
            _remove_owned_path(target / ARTIFACT_PATHS[name])
        for name in reversed(moved_old):
            os.replace(backup / name, target / ARTIFACT_PATHS[name])
        _fsync_directory(target)
        raise


@dataclass(frozen=True, slots=True)
class DetectionCache:
    """Immutable detector artifacts shared by ReID-derived materializations."""

    plan: BuildPlan
    detect_plan: StagePlan
    samples: tuple[SourceSample, ...]
    cache_id: str

    @classmethod
    def from_plan(cls, plan: BuildPlan, samples: tuple[SourceSample, ...]) -> DetectionCache:
        """Create the cache contract for one build's detect stage."""

        detect_plan = plan.stage_by_name.get("detect")
        if detect_plan is None:
            raise ValueError("Detection caching requires a detect stage.")
        resolved_samples = tuple(samples)
        if not resolved_samples:
            raise ValueError("Detection caching requires source samples.")
        if len({sample.sample_id for sample in resolved_samples}) != len(resolved_samples):
            raise ValueError("Detection cache source sample IDs must be unique.")
        return cls(
            plan=plan,
            detect_plan=detect_plan,
            samples=resolved_samples,
            cache_id=make_detection_cache_id(plan, detect_plan),
        )

    @property
    def root(self) -> Path:
        return self.plan.build_root / ".cache" / "detect" / self.cache_id

    @property
    def lock_path(self) -> Path:
        return self.plan.build_root / ".locks" / "detect" / f"{self.cache_id}.lock"

    @property
    def _staging_parent(self) -> Path:
        return self.plan.build_root / ".cache" / ".staging" / "detect"

    @property
    def _expected_stage(self) -> StageProvenance:
        return StageProvenance(
            name=self.detect_plan.name,
            fingerprint=self.detect_plan.fingerprint,
            batch_size=self.detect_plan.batch_size,
            inputs=self.detect_plan.depends_on,
            component=self.detect_plan.component,
            config=self.detect_plan.config,
        )

    def lock(self) -> FileLock:
        """Return the cross-build lock guarding this cache identity."""

        _reject_symlinked_components(self.lock_path.parent, root=self.plan.build_root)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(self.lock_path)

    def restore(self, context: MaterializationContext) -> StageOutcome | None:
        """Hydrate the target build from a valid cache, bootstrapping if needed."""

        self._validate_context(context)
        manifest = self._load_valid_manifest()
        if manifest is None:
            self._bootstrap_from_published_builds()
            manifest = self._load_valid_manifest()
        if manifest is None:
            return None

        had_local_artifacts = bool(context.completed_shards) or any(
            (context.staging_root / ARTIFACT_PATHS[name]).exists() for name in (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT)
        )
        for stale in context.staging_root.glob(".detect-cache-*"):
            _remove_owned_path(stale)
        temporary = Path(tempfile.mkdtemp(prefix=".detect-cache-", dir=context.staging_root))
        try:
            instance_paths = _write_restored_detection_shards(
                self.root / ARTIFACT_PATHS[INSTANCES_ARTIFACT],
                temporary,
                self.samples,
                box_type=context.build_plan.box_type,
                source_build_id=self.cache_id,
                target_build_id=context.build_plan.build_id,
                batch_size=context.stage_plan.batch_size,
                image_references=context.build_plan.publish.image_references,
            )
            _install_artifact_directories(
                temporary,
                context.staging_root,
                (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT),
            )
        finally:
            if _path_exists(temporary):
                _remove_owned_path(temporary)

        shard_hashes = {
            f"{shard_index:05d}": {
                name: sha256_file(context.staging_root / ARTIFACT_PATHS[name] / f"part-{shard_index:05d}.parquet")
                for name in (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT)
            }
            for shard_index in range(len(instance_paths))
        }
        context.state.replace_shards(context.stage_plan.name, shard_hashes)
        context.repaired_shards = context.repaired_shards or had_local_artifacts
        return StageOutcome(
            artifacts=(SAMPLES_ARTIFACT, INSTANCES_ARTIFACT),
            metrics={"cache": "hit", "samples": len(self.samples)},
        )

    def publish(self, context: MaterializationContext) -> None:
        """Publish the just-completed detector artifacts into the shared cache."""

        self._validate_context(context)
        if self._load_valid_manifest() is not None:
            return
        self._publish_from_source(context.staging_root, source_build_id=context.build_plan.build_id)

    def _validate_context(self, context: MaterializationContext) -> None:
        plan = context.build_plan
        if (
            plan.build_root != self.plan.build_root
            or plan.dataset_name != self.plan.dataset_name
            or plan.source_fingerprint != self.plan.source_fingerprint
            or plan.box_type != self.plan.box_type
            or make_detection_cache_id(plan, context.stage_plan) != self.cache_id
        ):
            raise ValueError("Detection cache does not belong to the active materialization context.")

    def _load_valid_manifest(self) -> DatasetManifest | None:
        _reject_symlinked_components(self.root.parent, root=self.plan.build_root)
        if self.root.is_symlink():
            self.root.unlink()
            return None
        if not self.root.exists():
            return None
        try:
            manifest = DatasetManifest.load(self.root)
            if not self._manifest_matches(manifest):
                raise DatasetValidationError("Detection cache provenance does not match its content identity.")
            validate_published_build(self.root, manifest=manifest)
            return manifest
        except (FileNotFoundError, ValueError, ManifestError):
            _remove_owned_path(self.root)
            return None

    def _manifest_matches(self, manifest: DatasetManifest) -> bool:
        metadata = manifest.metadata
        return (
            manifest.build_id == self.cache_id
            and manifest.box_type == self.plan.box_type
            and not manifest.publish.masks
            and not manifest.publish.embeddings
            and set(manifest.artifacts_by_name) == {SAMPLES_ARTIFACT, INSTANCES_ARTIFACT}
            and manifest.stages == (self._expected_stage,)
            and metadata.get("cache_schema") == DETECTION_CACHE_SCHEMA
            and metadata.get("boxmot_version") == __version__
            and metadata.get("dataset_name") == self.plan.dataset_name
            and metadata.get("source_fingerprint") == self.plan.source_fingerprint
            and metadata.get("source_count") == len(self.samples)
        )

    def _bootstrap_from_published_builds(self) -> bool:
        roots = [self.plan.build_root]
        local_default = (Path("runs") / "materializations").resolve()
        if not os.environ.get("BOXMOT_BUILDS_DIR") and self.plan.build_root == local_default:
            former_default = former_default_build_root().expanduser().resolve()
            if former_default != self.plan.build_root:
                roots.append(former_default)
        for build_root in roots:
            if build_root.is_symlink() or not build_root.is_dir():
                continue
            try:
                candidates = sorted(build_root.iterdir(), key=lambda path: path.name)
            except OSError:
                if build_root == self.plan.build_root:
                    raise
                continue
            for candidate in candidates:
                if candidate.is_symlink() or not candidate.is_dir() or not _BUILD_ID.fullmatch(candidate.name):
                    continue
                try:
                    manifest = DatasetManifest.load(candidate)
                    if not self._candidate_matches(candidate, manifest):
                        continue
                    self._verify_candidate_instances(candidate, manifest)
                    self._publish_from_source(candidate, source_build_id=manifest.build_id)
                    return True
                except (OSError, ValueError, ManifestError):
                    continue
        return False

    def _candidate_matches(self, root: Path, manifest: DatasetManifest) -> bool:
        try:
            success = json.loads((root / SUCCESS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        detect_stages = tuple(stage for stage in manifest.stages if stage.name == "detect")
        metadata = manifest.metadata
        return (
            success == {"schema": SCHEMA_ID, "build_id": manifest.build_id}
            and manifest.box_type == self.plan.box_type
            and detect_stages == (self._expected_stage,)
            and metadata.get("boxmot_version") == __version__
            and metadata.get("dataset_name") == self.plan.dataset_name
            and metadata.get("source_fingerprint") == self.plan.source_fingerprint
            and metadata.get("source_count") == len(self.samples)
            and {SAMPLES_ARTIFACT, INSTANCES_ARTIFACT} <= set(manifest.artifacts_by_name)
        )

    @staticmethod
    def _verify_candidate_instances(root: Path, manifest: DatasetManifest) -> None:
        artifact = manifest.artifact(INSTANCES_ARTIFACT)
        path = resolve_artifact_path(root, artifact.path)
        files = artifact_files(path)
        declared = tuple(shard.path for shard in artifact.shards)
        actual = tuple(file.relative_to(root.resolve()).as_posix() for file in files)
        if actual != declared:
            raise DatasetValidationError("Published candidate instance shards differ from its manifest.")
        for file, shard in zip(files, artifact.shards, strict=True):
            if file.stat().st_size != shard.size_bytes or sha256_file(file) != shard.sha256:
                raise DatasetValidationError(f"Published candidate shard {shard.path!r} failed validation.")

    def _publish_from_source(self, source_root: Path, *, source_build_id: str) -> None:
        _reject_symlinked_components(self._staging_parent, root=self.plan.build_root)
        self._staging_parent.mkdir(parents=True, exist_ok=True)
        for stale in self._staging_parent.glob(f"{self.cache_id}-*"):
            _remove_owned_path(stale)
        temporary = Path(tempfile.mkdtemp(prefix=f"{self.cache_id}-", dir=self._staging_parent))
        try:
            publish_image_references = all(sample.image_ref is not None for sample in self.samples)
            instance_paths = write_rekeyed_instance_artifact(
                source_root / ARTIFACT_PATHS[INSTANCES_ARTIFACT],
                temporary / ARTIFACT_PATHS[INSTANCES_ARTIFACT],
                box_type=self.plan.box_type,
                source_build_id=source_build_id,
                target_build_id=self.cache_id,
            )
            _write_sample_shards(
                temporary,
                self.samples,
                image_references=publish_image_references,
                shard_count=len(instance_paths),
            )
            artifacts = tuple(
                describe_parquet_artifact(temporary, name=name, relative_path=ARTIFACT_PATHS[name])
                for name in (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT)
            )
            manifest = DatasetManifest(
                build_id=self.cache_id,
                box_type=self.plan.box_type,
                artifacts=artifacts,
                publish=PublishedContent(
                    image_references=publish_image_references,
                    masks=False,
                    embeddings=False,
                ),
                stages=(self._expected_stage,),
                metadata={
                    "cache_schema": DETECTION_CACHE_SCHEMA,
                    "boxmot_version": __version__,
                    "dataset_name": self.plan.dataset_name,
                    "source_fingerprint": self.plan.source_fingerprint,
                    "source_count": len(self.samples),
                },
            )
            manifest.write(temporary)
            validate_dataset(temporary, manifest=manifest, require_success=False)
            _write_success(temporary, self.cache_id)
            _fsync_directory(temporary)
            self.root.parent.mkdir(parents=True, exist_ok=True)
            if _path_exists(self.root):
                if self._load_valid_manifest() is not None:
                    return
            os.replace(temporary, self.root)
            _fsync_directory(self.root.parent)
        finally:
            if _path_exists(temporary):
                _remove_owned_path(temporary)


__all__ = ("DETECTION_CACHE_SCHEMA", "DetectionCache", "make_detection_cache_id")
