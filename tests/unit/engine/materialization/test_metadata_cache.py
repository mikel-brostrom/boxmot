from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

import boxmot.engine.materialization.catalog as catalog_module
import boxmot.engine.materialization.metadata_cache as cache_module
import boxmot.engine.materialization.source as source_module
from boxmot.datasets.manifest import sha256_file
from boxmot.engine.materialization import BoundedFrameDecoder
from boxmot.engine.materialization.catalog import (
    CatalogFileMetadata,
    catalog_local_source,
    inspect_catalog_file,
)
from boxmot.engine.materialization.metadata_cache import (
    FileMetadataCache,
    default_source_metadata_cache_path,
)


def _save_frame(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((3, 5, 8), value, dtype=np.uint8))


def test_local_catalog_resolves_each_still_image_metadata_once(tmp_path, monkeypatch) -> None:
    source = tmp_path / "images"
    first = source / "seq-a" / "000001.npy"
    second = source / "seq-b" / "000001.npy"
    _save_frame(first, 1)
    _save_frame(second, 2)
    calls: list[tuple[Path, bool]] = []

    def resolve(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        calls.append((path, include_image_size))
        return CatalogFileMetadata(
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            image_size=(3, 5),
        )

    monkeypatch.setattr(
        catalog_module,
        "_probe_image_size",
        lambda _path: (_ for _ in ()).throw(AssertionError("image metadata resolved twice")),
    )
    catalog = catalog_local_source(source, split="test", metadata_resolver=resolve)

    assert calls == [(first, True), (second, True)]
    assert [sample.image_size for sample in catalog.samples] == [(3, 5), (3, 5)]


def test_metadata_cache_warm_hits_preserve_exact_source_fingerprint(tmp_path) -> None:
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 7)
    uncached = catalog_local_source(source, split="test")
    cache_path = tmp_path / "cache" / "metadata.json"
    inspected: list[Path] = []

    def inspect(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        inspected.append(path)
        return inspect_catalog_file(path, include_image_size)

    with FileMetadataCache(cache_path, metadata_resolver=inspect) as cache:
        cold = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (0, 1)
    with FileMetadataCache(cache_path, metadata_resolver=inspect) as cache:
        warm = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (1, 0)

    assert inspected == [frame.resolve()]
    assert warm.fingerprint == cold.fingerprint == uncached.fingerprint
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "boxmot.file-metadata/v1"
    assert all(
        set(entry["identity"]) == {"path", "device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
        for entry in payload["entries"].values()
    )
    assert list(cache_path.parent.glob(f".{cache_path.name}.*.tmp")) == []


def test_metadata_cache_invalidates_mutation_with_restored_mtime(tmp_path) -> None:
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 3)
    cache_path = tmp_path / "metadata.json"
    inspected: list[Path] = []

    def inspect(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        inspected.append(path)
        return inspect_catalog_file(path, include_image_size)

    with FileMetadataCache(cache_path, metadata_resolver=inspect) as cache:
        initial = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    before = frame.stat()
    time.sleep(0.002)
    _save_frame(frame, 9)
    os.utime(frame, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert frame.stat().st_ctime_ns != before.st_ctime_ns

    with FileMetadataCache(cache_path, metadata_resolver=inspect) as cache:
        changed = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (0, 1)

    assert inspected == [frame.resolve(), frame.resolve()]
    assert changed.fingerprint != initial.fingerprint


def test_metadata_cache_catalog_tracks_added_and_removed_sources(tmp_path) -> None:
    source = tmp_path / "images"
    removed = source / "seq-a" / "000001.npy"
    retained = source / "seq-a" / "000002.npy"
    added = source / "seq-a" / "000003.npy"
    _save_frame(removed, 1)
    _save_frame(retained, 2)
    cache_path = tmp_path / "metadata.json"

    with FileMetadataCache(cache_path) as cache:
        initial = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    removed.unlink()
    _save_frame(added, 3)
    with FileMetadataCache(cache_path) as cache:
        changed = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (1, 1)

    assert [sample.image_ref for sample in changed.samples] == ["seq-a/000002.npy", "seq-a/000003.npy"]
    assert changed.fingerprint != initial.fingerprint


def test_cached_pending_decode_does_not_hash_source_again(tmp_path, monkeypatch) -> None:
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 4)
    cache_path = tmp_path / "metadata.json"

    with FileMetadataCache(cache_path) as cache:
        catalog = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    def unexpected_hash(_path):
        pytest.fail("a stat-valid cached source must not be SHA-read again before decoding")

    monkeypatch.setattr(source_module, "sha256_file", unexpected_hash)
    monkeypatch.setattr(catalog_module, "sha256_file", unexpected_hash)
    with FileMetadataCache(cache_path) as cache:
        with BoundedFrameDecoder(
            workers=1,
            digest_resolver=cache.resolve_digest,
        ) as decoder:
            decoded = decoder.decode(catalog.samples)
        assert (cache.hits, cache.misses) == (1, 0)

    assert [item.sample_id for item in decoded] == [catalog.samples[0].sample_id]


def test_cached_pending_decode_rejects_changed_source(tmp_path) -> None:
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 4)
    cache_path = tmp_path / "metadata.json"

    with FileMetadataCache(cache_path) as cache:
        catalog = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    before = frame.stat()
    time.sleep(0.002)
    _save_frame(frame, 8)
    os.utime(frame, ns=(before.st_atime_ns, before.st_mtime_ns))

    with FileMetadataCache(cache_path) as cache:
        with BoundedFrameDecoder(
            workers=1,
            digest_resolver=cache.resolve_digest,
        ) as decoder:
            with pytest.raises(ValueError, match="changed after cataloging"):
                decoder.decode(catalog.samples)
        assert (cache.hits, cache.misses) == (0, 1)


def test_metadata_cache_recovers_from_corruption_and_write_failure(tmp_path) -> None:
    frame = tmp_path / "frame.npy"
    _save_frame(frame)
    cache_path = tmp_path / "corrupt.json"
    cache_path.write_text("not-json", encoding="utf-8")

    with FileMetadataCache(cache_path) as cache:
        expected = cache.resolve(frame, True)
    assert json.loads(cache_path.read_text(encoding="utf-8"))["schema"] == "boxmot.file-metadata/v1"

    unwritable_target = tmp_path / "target-is-a-directory"
    unwritable_target.mkdir()
    with FileMetadataCache(unwritable_target) as cache:
        actual = cache.resolve(frame, True)
    assert actual == expected


def test_default_source_metadata_cache_uses_platform_cache(tmp_path, monkeypatch) -> None:
    platform_cache = tmp_path / "platform-cache"
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)

    path = default_source_metadata_cache_path(source)

    assert path.parent == platform_cache / "materialization" / "source-metadata"
    assert path.suffix == ".json"
    assert source not in path.parents


def test_default_source_cache_migrates_legacy_evaluation_entries_without_probing(tmp_path, monkeypatch) -> None:
    platform_cache = tmp_path / "platform-cache"
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 5)
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)
    legacy_path = platform_cache / "evaluation" / "source-catalogs" / "legacy.json"
    canonical_path = default_source_metadata_cache_path(source)
    inspected: list[Path] = []

    def inspect(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        inspected.append(path)
        return inspect_catalog_file(path, include_image_size)

    with FileMetadataCache(
        legacy_path,
        metadata_resolver=inspect,
        write_schema="boxmot.eval-file-metadata/v1",
        discover_legacy_evaluation=False,
    ) as cache:
        expected = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
    assert inspected == [frame.resolve()]
    assert not canonical_path.exists()

    inspected.clear()
    with FileMetadataCache(canonical_path, metadata_resolver=inspect) as cache:
        actual = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (1, 0)

    assert inspected == []
    assert actual == expected
    assert json.loads(canonical_path.read_text(encoding="utf-8"))["schema"] == "boxmot.file-metadata/v1"
    assert list(canonical_path.parent.glob(f".{canonical_path.name}.*.tmp")) == []


def test_arbitrary_source_cache_path_never_scans_legacy_evaluation_entries(tmp_path, monkeypatch) -> None:
    platform_cache = tmp_path / "platform-cache"
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 6)
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)
    legacy_path = platform_cache / "evaluation" / "source-catalogs" / "legacy.json"

    with FileMetadataCache(
        legacy_path,
        write_schema="boxmot.eval-file-metadata/v1",
        discover_legacy_evaluation=False,
    ) as cache:
        catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    inspected: list[Path] = []

    def inspect(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        inspected.append(path)
        return inspect_catalog_file(path, include_image_size)

    custom_path = tmp_path / "custom" / "metadata.json"
    with FileMetadataCache(custom_path, metadata_resolver=inspect) as cache:
        catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (0, 1)

    assert inspected == [frame.resolve()]


def test_migrated_evaluation_entry_is_rejected_when_full_file_identity_changed(tmp_path, monkeypatch) -> None:
    platform_cache = tmp_path / "platform-cache"
    source = tmp_path / "images"
    frame = source / "seq-a" / "000001.npy"
    _save_frame(frame, 1)
    monkeypatch.setattr(cache_module, "user_cache_path", lambda _name: platform_cache)
    legacy_path = platform_cache / "evaluation" / "source-catalogs" / "legacy.json"
    with FileMetadataCache(
        legacy_path,
        write_schema="boxmot.eval-file-metadata/v1",
        discover_legacy_evaluation=False,
    ) as cache:
        initial = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)

    before = frame.stat()
    time.sleep(0.002)
    _save_frame(frame, 9)
    os.utime(frame, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert frame.stat().st_ctime_ns != before.st_ctime_ns
    inspected: list[Path] = []

    def inspect(path: Path, include_image_size: bool) -> CatalogFileMetadata:
        inspected.append(path)
        return inspect_catalog_file(path, include_image_size)

    canonical_path = default_source_metadata_cache_path(source)
    with FileMetadataCache(canonical_path, metadata_resolver=inspect) as cache:
        changed = catalog_local_source(source, split="test", metadata_resolver=cache.resolve)
        assert (cache.hits, cache.misses) == (0, 1)

    assert inspected == [frame.resolve()]
    assert changed.fingerprint != initial.fingerprint
