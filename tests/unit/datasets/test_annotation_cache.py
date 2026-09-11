from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from boxmot.datasets.annotation_cache import load_cached_annotation


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
