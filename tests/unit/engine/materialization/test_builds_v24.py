from __future__ import annotations

from pathlib import Path

import pytest

from boxmot.datasets import ArtifactRecord, DatasetManifest, PublishedContent, ShardRecord, StageProvenance
from boxmot.engine.materialization.builds import (
    BuildCompatibilityError,
    default_build_root,
    resolve_build_path,
    validate_build_compatibility,
)


def _artifact(name: str) -> ArtifactRecord:
    shard = ShardRecord(f"{name}/part-00000.parquet", 0, "0" * 64, 0)
    metadata = {"encoder_fingerprint": "1" * 64, "dim": 4} if name == "embeddings" else {}
    return ArtifactRecord(name, name, 0, "0" * 64, 0, (shard,), metadata)


def _manifest(*, embeddings: bool = False, metadata: dict | None = None) -> DatasetManifest:
    artifacts = [_artifact("samples"), _artifact("instances")]
    if embeddings:
        artifacts.append(_artifact("embeddings"))
    resolved_metadata = {"dataset_id": "mot17", "split": "val", "source_fingerprint": "3" * 64}
    resolved_metadata.update(metadata or {})
    return DatasetManifest(
        build_id="0" * 64,
        box_type="aabb",
        artifacts=tuple(artifacts),
        publish=PublishedContent(image_references=False, masks=False, embeddings=embeddings),
        stages=(StageProvenance(name="detect", fingerprint="2" * 64),),
        metadata=resolved_metadata,
    )


def test_build_compatibility_validates_semantics_and_required_artifacts() -> None:
    manifest = _manifest()
    validate_build_compatibility(manifest, dataset_id="mot17", split="val", geometry="aabb")

    with pytest.raises(BuildCompatibilityError, match="split mismatch"):
        validate_build_compatibility(manifest, split="test")
    with pytest.raises(BuildCompatibilityError, match="materialize.*publish-embeddings"):
        validate_build_compatibility(manifest, require_embeddings=True)


def test_build_compatibility_rejects_changed_reid_crop_provenance() -> None:
    rotated_fingerprint = "4" * 64
    manifest = _manifest(
        embeddings=True,
        metadata={"component_fingerprints": {"reid": rotated_fingerprint}},
    )

    validate_build_compatibility(
        manifest,
        component_fingerprints={"reid": rotated_fingerprint},
    )
    with pytest.raises(BuildCompatibilityError, match="component 'reid' mismatch"):
        validate_build_compatibility(
            manifest,
            component_fingerprints={"reid": "5" * 64},
        )


def test_build_id_resolves_only_below_explicit_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="full lowercase SHA-256"):
        resolve_build_path("build-deadbeef", build_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist under"):
        resolve_build_path("a" * 64, build_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="cannot contain path separators"):
        resolve_build_path("missing/build", build_root=tmp_path)


def test_default_build_root_is_run_scoped_and_distinct_from_native_builds(monkeypatch) -> None:
    monkeypatch.delenv("BOXMOT_BUILDS_DIR", raising=False)

    assert default_build_root() == Path("runs/materializations")


def test_build_root_environment_override_remains_authoritative(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "shared-materialized"
    monkeypatch.setenv("BOXMOT_BUILDS_DIR", str(configured))

    assert default_build_root() == configured
