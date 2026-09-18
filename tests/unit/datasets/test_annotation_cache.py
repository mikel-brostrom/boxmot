from __future__ import annotations

import errno
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets.annotation_cache import load_cached_annotation
from boxmot.datasets.replay_cache import ReplayCacheError


@pytest.mark.parametrize("dtype,shape", [("float64", (2, 9)), ("float32", (0, 13)), ("uint16", (5, 7))])
def test_annotations_preserve_values_and_are_isolated_without_rereading(tmp_path, dtype, shape) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("annotation source")
    expected = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    calls = []

    def reader(path):
        calls.append(path)
        return expected

    options = dict(reader=reader, format="fixture/v1")
    first = load_cached_annotation(source, **options)
    first.fill(99)
    second = load_cached_annotation(source, **options)
    assert calls == [source]
    np.testing.assert_array_equal(second, expected)
    assert second.dtype == expected.dtype
    assert second.flags.c_contiguous and second.flags.writeable


def test_annotation_cache_invalidates_sources_even_with_restored_mtime(tmp_path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    options = dict(reader=np.loadtxt, format="numeric-text/v1")
    np.testing.assert_array_equal(load_cached_annotation(source, **options), [1, 2, 3])
    stat = source.stat()
    source.write_text("4 5 6")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    np.testing.assert_array_equal(load_cached_annotation(source, **options), [4, 5, 6])


@pytest.mark.parametrize("component", ["values.npy", "index.json", "_SUCCESS"])
def test_modified_or_incomplete_annotation_cache_is_rebuilt(tmp_path, component) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    root = tmp_path / "cache"
    calls = []

    def reader(path):
        calls.append(path)
        return np.loadtxt(path)

    options = dict(reader=reader, format="numeric-text/v1", cache_root=root)
    load_cached_annotation(source, **options)
    path = next(root.glob("*/values.npy")).parent
    (path / component).write_bytes(b"corrupt")
    np.testing.assert_array_equal(load_cached_annotation(source, **options), [1, 2, 3])
    assert calls == [source, source]


def test_annotation_parser_contract_has_separate_cache_identity(tmp_path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    first = load_cached_annotation(source, reader=np.loadtxt, format="numeric-text/v1")
    second = load_cached_annotation(source, reader=lambda path: np.loadtxt(path) + 1, format="numeric-text/v2")
    np.testing.assert_array_equal(second, first + 1)


def test_concurrent_annotation_preparation_reads_once(tmp_path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    calls = []
    barrier = threading.Barrier(2)

    def reader(path):
        calls.append(path)
        return np.loadtxt(path)

    def prepare(_):
        barrier.wait(timeout=10)
        return load_cached_annotation(source, reader=reader, format="numeric-text/v1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(prepare, range(2))
    np.testing.assert_array_equal(first, second)
    assert calls == [source]


def test_annotation_reader_failure_does_not_publish_partial_cache(tmp_path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text("invalid")
    root = tmp_path / "cache"
    with pytest.raises(ValueError):
        load_cached_annotation(source, reader=np.loadtxt, format="numeric-text/v1", cache_root=root)
    assert not list(root.glob("*/_SUCCESS"))
    assert not list(root.glob(".*.tmp-*"))


def _fail_cache_operation(monkeypatch, root: Path, stage: str, error: OSError) -> None:
    """Inject real OS failures only into the chosen cache-storage operation."""
    import boxmot.datasets.annotation_cache as cache

    def fail(*args, **kwargs):
        raise error

    if stage in {"root", "locks"}:
        original_mkdir = Path.mkdir
        target = root if stage == "root" else root / ".locks"

        def mkdir(path, *args, **kwargs):
            if path == target:
                raise error
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir)
    elif stage == "lock":
        monkeypatch.setattr(cache, "FileLock", fail)
    elif stage == "staging":
        monkeypatch.setattr(cache.tempfile, "mkdtemp", fail)
    elif stage == "array":
        original_write = cache._ArrayWriter.write

        def write(writer, data):
            original_write(writer, data[:8])
            raise error

        monkeypatch.setattr(cache._ArrayWriter, "write", write)
    elif stage in {"index.json", "_SUCCESS"}:
        original_write_json = cache._write_json

        def write_json(path, value):
            if path.name == stage:
                raise error
            return original_write_json(path, value)

        monkeypatch.setattr(cache, "_write_json", write_json)
    elif stage == "fsync":
        monkeypatch.setattr(cache.os, "fsync", fail)
    elif stage == "publish":
        monkeypatch.setattr(cache.os, "replace", fail)
    else:
        raise AssertionError(stage)


@pytest.mark.parametrize("code", (errno.ENOSPC, errno.EDQUOT))
@pytest.mark.parametrize(
    "stage", ("root", "locks", "lock", "staging", "array", "fsync", "index.json", "_SUCCESS", "publish")
)
def test_cache_capacity_failure_retains_values_and_cleans_only_its_staging(tmp_path, monkeypatch, code, stage) -> None:
    """Storage exhaustion must neither lose input data nor rerun its encoder."""
    source = tmp_path / "gt.txt"
    source.write_text("annotation source")
    root = tmp_path / "cache"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    options = dict(reader=lambda _: expected, cache_root=root)
    load_cached_annotation(source, format="existing/v1", **options)
    existing = next(root.glob("*/values.npy"))
    original_bytes = existing.read_bytes()
    calls = []

    def reader(path):
        calls.append(path)
        return expected

    _fail_cache_operation(monkeypatch, root, stage, OSError(code, "fixture storage failure"))
    actual = load_cached_annotation(source, reader=reader, format="new/v1", cache_root=root)

    assert calls == [source]
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous and actual.flags.writeable and actual.flags.owndata
    actual.fill(-1)
    assert expected[0, 0] == 0
    assert existing.read_bytes() == original_bytes
    assert len(list(root.glob("*/_SUCCESS"))) == 1
    assert not list(root.glob(".*.tmp-*"))


@pytest.mark.parametrize("stage", ("root", "array", "_SUCCESS", "publish"))
@pytest.mark.parametrize("code", (errno.EACCES, errno.EIO))
def test_cache_unrelated_io_failures_propagate(tmp_path, monkeypatch, stage, code) -> None:
    """Capacity fallback must not disguise permission or device errors."""
    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    root = tmp_path / "cache"
    failure = OSError(code, "fixture non-capacity failure")
    _fail_cache_operation(monkeypatch, root, stage, failure)

    with pytest.raises(OSError) as raised:
        load_cached_annotation(source, reader=np.loadtxt, format="fixture/v1", cache_root=root)

    assert raised.value is failure
    assert not list(root.glob("*/_SUCCESS"))
    assert not list(root.glob(".*.tmp-*"))


@pytest.mark.parametrize("full_setup", (False, True))
def test_annotation_reader_capacity_error_is_not_treated_as_cache_failure(tmp_path, monkeypatch, full_setup) -> None:
    """An encoder or parser failure must propagate even if it uses the same errno."""
    source = tmp_path / "gt.txt"
    source.write_text("source")
    root = tmp_path / "cache"
    calls = []
    failure = OSError(errno.ENOSPC, "reader failure")

    def reader(path):
        calls.append(path)
        raise failure

    if full_setup:
        _fail_cache_operation(monkeypatch, root, "root", OSError(errno.ENOSPC, "cache failure"))
    with pytest.raises(OSError) as raised:
        load_cached_annotation(source, reader=reader, format="fixture/v1", cache_root=root)

    assert raised.value is failure
    assert calls == [source]


def test_capacity_fallback_still_rejects_source_changes(tmp_path, monkeypatch) -> None:
    """Returning uncached values must not relax source snapshot validation."""
    import boxmot.datasets.annotation_cache as cache

    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    root = tmp_path / "cache"

    def failed_publication(*args, **kwargs):
        source.write_text("4 5 6")
        raise OSError(errno.ENOSPC, "fixture storage failure")

    monkeypatch.setattr(cache.os, "replace", failed_publication)
    with pytest.raises(ReplayCacheError, match="source changed"):
        load_cached_annotation(source, reader=np.loadtxt, format="fixture/v1", cache_root=root)

    assert not list(root.glob("*/_SUCCESS"))
    assert not list(root.glob(".*.tmp-*"))


def test_capacity_warning_is_emitted_once_per_root(tmp_path, monkeypatch) -> None:
    """Thousands of failed frame-cache writes should produce one useful warning."""
    import boxmot.datasets.annotation_cache as cache

    source = tmp_path / "gt.txt"
    source.write_text("1 2 3")
    root = tmp_path / "cache"
    messages = []
    monkeypatch.setattr(cache.logger, "warning", messages.append)
    _fail_cache_operation(monkeypatch, root, "array", OSError(errno.ENOSPC, "fixture storage failure"))

    for version in range(3):
        values = load_cached_annotation(source, reader=np.loadtxt, format=f"fixture/v{version}", cache_root=root)
        np.testing.assert_array_equal(values, [1, 2, 3])

    assert len(messages) == 1
    assert str(root) in messages[0]
    assert "continuing with uncached" in messages[0]


def test_array_stream_retries_short_writes_without_truncating_numpy_payload() -> None:
    """A successful NPY publication must contain every byte of a short writer."""
    from boxmot.datasets.annotation_cache import _ArrayWriter

    class ShortWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:7])

    expected = np.arange(120, dtype=np.float32).reshape(10, 12)
    stream = ShortWriter()
    np.save(_ArrayWriter(stream), expected, allow_pickle=False)
    stream.seek(0)
    np.testing.assert_array_equal(np.load(stream, allow_pickle=False), expected)


@pytest.mark.parametrize("written", (None, 0))
def test_array_stream_reports_stalled_writes_as_io_error(written) -> None:
    """No-progress writes must not be mistaken for a completed cache array."""
    from boxmot.datasets.annotation_cache import _ArrayWriter

    class StalledWriter(io.BytesIO):
        def write(self, data):
            return written

    with pytest.raises(OSError) as raised:
        np.save(_ArrayWriter(StalledWriter()), np.ones(3), allow_pickle=False)
    assert raised.value.errno == errno.EIO
