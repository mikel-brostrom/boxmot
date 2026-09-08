from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

import boxmot.engine.eval.catalog_cache as cache_module
import boxmot.engine.materialization.catalog as catalog_module
import boxmot.engine.materialization.metadata_cache as metadata_cache_module
from boxmot.components.artifacts import ResolvedArtifact
from boxmot.detectors.config import resolve_detector_spec
from boxmot.engine.eval.catalog_cache import (
    EvaluationArtifactResolver,
    catalog_mot_dataset_for_evaluation,
    default_evaluation_artifact_cache_path,
    default_evaluation_catalog_cache_path,
)
from boxmot.engine.materialization.catalog import catalog_mot_dataset


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    data_root = tmp_path / "raw"
    sequence_root = data_root / "Fixture" / "train" / "SEQ-01"
    image_root = sequence_root / "img1"
    gt_root = sequence_root / "gt"
    image_root.mkdir(parents=True)
    gt_root.mkdir()
    image_path = image_root / "000001.bmp"
    assert cv2.imwrite(str(image_path), np.zeros((3, 5, 3), dtype=np.uint8))
    (sequence_root / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=30\n",
        encoding="utf-8",
    )
    (gt_root / "gt.txt").write_text("1,1,0,0,1,1,1,1,1\n", encoding="utf-8")
    config: dict[str, object] = {
        "id": "fixture",
        "layout": "mot",
        "root": "Fixture",
        "default_split": "train",
        "splits": {"train": {"path": "train", "has_ground_truth": True}},
        "classes": {"person": {"id": 1, "evaluation": "target"}},
    }
    return config, data_root, image_path


@pytest.mark.parametrize("fps", (None, 5.0))
def test_eval_catalog_cache_hits_preserve_exact_uncached_catalog(tmp_path, monkeypatch, fps) -> None:
    config, data_root, _ = _fixture(tmp_path)
    uncached = catalog_mot_dataset(config, split="train", data_root=data_root, fps=fps)
    cache_path = tmp_path / "cache" / "catalog.json"
    real_inspect = cache_module.inspect_catalog_file
    inspected: list[Path] = []

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(cache_module, "inspect_catalog_file", recording_inspect)
    cold = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
        fps=fps,
    )
    cold_reads = tuple(inspected)
    inspected.clear()
    warm = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
        fps=fps,
    )

    assert len(cold_reads) == 3  # image, seqinfo.ini, and ground truth
    assert inspected == []
    assert cold == warm == uncached
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "boxmot.eval-file-metadata/v1"
    assert all(
        set(entry["identity"]) == {"path", "device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
        for entry in payload["entries"].values()
    )
    assert list(cache_path.parent.glob(f".{cache_path.name}.*.tmp")) == []


def test_eval_catalog_cache_invalidates_changed_file_even_with_restored_mtime(tmp_path, monkeypatch) -> None:
    config, data_root, image_path = _fixture(tmp_path)
    cache_path = tmp_path / "catalog.json"
    initial = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )
    before = image_path.stat()
    time.sleep(0.002)
    assert cv2.imwrite(str(image_path), np.full((3, 5, 3), 255, dtype=np.uint8))
    os.utime(image_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert image_path.stat().st_ctime_ns != before.st_ctime_ns

    real_inspect = cache_module.inspect_catalog_file
    inspected: list[Path] = []

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(cache_module, "inspect_catalog_file", recording_inspect)
    changed = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )

    assert inspected == [image_path.resolve()]
    assert changed.fingerprint != initial.fingerprint


def test_eval_catalog_cache_detects_added_and_removed_files(tmp_path, monkeypatch) -> None:
    config, data_root, image_path = _fixture(tmp_path)
    cache_path = tmp_path / "catalog.json"
    initial = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )
    added_path = image_path.with_name("000002.bmp")
    assert cv2.imwrite(str(added_path), np.ones((3, 5, 3), dtype=np.uint8))

    real_inspect = cache_module.inspect_catalog_file
    inspected: list[Path] = []

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(cache_module, "inspect_catalog_file", recording_inspect)
    added = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )
    assert inspected == [added_path.resolve()]
    assert len(added.samples) == 2
    assert added.fingerprint != initial.fingerprint

    added_path.unlink()
    inspected.clear()
    removed = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )
    assert inspected == []
    assert removed == initial


def test_eval_catalog_cache_recovers_from_corruption_and_write_failure(tmp_path) -> None:
    config, data_root, _ = _fixture(tmp_path)
    cache_path = tmp_path / "corrupt.json"
    cache_path.write_text("not-json", encoding="utf-8")

    catalog = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=cache_path,
    )
    assert len(catalog.samples) == 1
    assert json.loads(cache_path.read_text(encoding="utf-8"))["schema"] == ("boxmot.eval-file-metadata/v1")

    unwritable_target = tmp_path / "target-is-a-directory"
    unwritable_target.mkdir()
    repeated = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
        cache_path=unwritable_target,
    )
    assert repeated.fingerprint == catalog.fingerprint


def test_materialization_catalog_remains_fresh_by_default(tmp_path, monkeypatch) -> None:
    config, data_root, _ = _fixture(tmp_path)
    real_inspect = catalog_module.inspect_catalog_file
    inspected: list[Path] = []

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(catalog_module, "inspect_catalog_file", recording_inspect)
    catalog_mot_dataset(config, split="train", data_root=data_root)
    first_count = len(inspected)
    catalog_mot_dataset(config, split="train", data_root=data_root)

    assert first_count == 3
    assert len(inspected) == first_count * 2


def test_default_eval_catalog_cache_uses_platform_cache(tmp_path, monkeypatch) -> None:
    config, data_root, _ = _fixture(tmp_path)
    platform_cache = tmp_path / "platform-cache"
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)
    monkeypatch.setattr(metadata_cache_module, "user_cache_path", lambda _name: platform_cache)

    path = default_evaluation_catalog_cache_path(
        config,
        split="train",
        data_root=data_root,
    )

    dataset_root = catalog_module.resolve_dataset_root(config, data_root)
    assert path == metadata_cache_module.default_source_metadata_cache_path(dataset_root)
    assert path.parent == platform_cache / "materialization" / "source-metadata"
    assert path.suffix == ".json"
    assert data_root not in path.parents


def test_default_eval_catalog_migrates_legacy_cache_into_shared_materialization_path(tmp_path, monkeypatch) -> None:
    config, data_root, _ = _fixture(tmp_path)
    platform_cache = tmp_path / "platform-cache"
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)
    monkeypatch.setattr(metadata_cache_module, "user_cache_path", lambda _name: platform_cache)
    legacy_path = cache_module._legacy_evaluation_catalog_cache_path(
        config,
        split="train",
        data_root=data_root,
    )
    with cache_module.EvaluationFileMetadataCache(legacy_path) as cache:
        expected = catalog_mot_dataset(
            config,
            split="train",
            data_root=data_root,
            metadata_resolver=cache.resolve,
        )

    canonical_path = default_evaluation_catalog_cache_path(
        config,
        split="train",
        data_root=data_root,
    )
    assert not canonical_path.exists()
    inspected: list[Path] = []
    real_inspect = cache_module.inspect_catalog_file

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(cache_module, "inspect_catalog_file", recording_inspect)
    actual = catalog_mot_dataset_for_evaluation(
        config,
        split="train",
        data_root=data_root,
    )

    assert inspected == []
    assert actual == expected
    assert json.loads(canonical_path.read_text(encoding="utf-8"))["schema"] == "boxmot.file-metadata/v1"
    assert list(canonical_path.parent.glob(f".{canonical_path.name}.*.tmp")) == []


def test_eval_artifact_resolver_caches_files_and_revalidates_changes(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "model.pt"
    original = b"resolved-model-v1"
    changed = b"resolved-model-v2"
    assert len(original) == len(changed)
    artifact.write_bytes(original)
    cache_path = tmp_path / "artifact-cache.json"

    real_inspect = cache_module.inspect_catalog_file
    inspected: list[Path] = []

    def recording_inspect(path: Path, include_image_size: bool):
        inspected.append(path)
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(cache_module, "inspect_catalog_file", recording_inspect)
    with EvaluationArtifactResolver(cache_path) as resolver:
        cold = resolver(artifact, allow_download=False)
    assert cold.sha256 == sha256(original).hexdigest()
    assert inspected == [artifact.resolve()]

    inspected.clear()
    with EvaluationArtifactResolver(cache_path) as resolver:
        warm = resolver(artifact, expected_sha256=cold.sha256, allow_download=False)
    assert warm == cold
    assert inspected == []

    before = artifact.stat()
    time.sleep(0.002)
    artifact.write_bytes(changed)
    os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
    inspected.clear()
    with EvaluationArtifactResolver(cache_path) as resolver:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            resolver(artifact, expected_sha256=cold.sha256, allow_download=False)
    assert inspected == [artifact.resolve()]


def test_eval_artifact_resolver_hashes_directories_fresh(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"weights")
    calls: list[Path] = []
    real_resolve = cache_module.resolve_artifact

    def recording_resolve(path, **kwargs):
        calls.append(Path(path))
        return real_resolve(path, **kwargs)

    monkeypatch.setattr(cache_module, "resolve_artifact", recording_resolve)
    cache_path = tmp_path / "artifact-cache.json"
    for _ in range(2):
        with EvaluationArtifactResolver(cache_path) as resolver:
            resolver(snapshot, allow_download=False)

    assert calls == [snapshot.resolve(), snapshot.resolve()]
    assert not cache_path.exists()


def test_default_eval_artifact_cache_uses_platform_cache(tmp_path, monkeypatch) -> None:
    platform_cache = tmp_path / "platform-cache"
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)

    assert default_evaluation_artifact_cache_path() == (platform_cache / "evaluation" / "artifact-metadata.json")


def test_component_resolution_accepts_explicit_eval_artifact_resolver(tmp_path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"model")
    calls: list[tuple[Path, bool]] = []

    def resolver(
        path,
        *,
        source_uri=None,
        expected_sha256=None,
        allow_download=False,
    ):
        assert source_uri is None
        assert expected_sha256 is None
        calls.append((Path(path), allow_download))
        return ResolvedArtifact(path=Path(path), sha256="a" * 64)

    spec, provenance = resolve_detector_spec(
        {
            "backend": "ultralytics",
            "artifact": str(artifact),
            "geometry_mode": "aabb",
        },
        geometry="aabb",
        allow_download=False,
        artifact_resolver=resolver,
    )

    assert calls == [(artifact, False)]
    assert spec.artifact_sha256 == "a" * 64
    assert provenance["artifact"]["sha256"] == "a" * 64
