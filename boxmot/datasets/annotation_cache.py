"""Disposable mapped numeric annotations for repeated calibration and scoring."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
from filelock import FileLock

from boxmot.utils import logger

from .manifest import sha256_file
from .replay_cache import ReplayCacheError, _close_arrays, _digest, _read_json, _signature, _write_json

_SCHEMA = "boxmot.annotation-cache/v1"
_CAPACITY_ERRORS = frozenset({errno.ENOSPC, errno.EDQUOT})


class _ArrayWriter:
    """Keep NumPy writes buffered so storage failures retain their OS errno."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def write(self, data: bytes) -> int:
        """Write NumPy's bounded chunks without its errno-less tofile path."""
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = self._stream.write(view[offset:])
            if written is None or written <= 0:
                raise OSError(errno.EIO, "Annotation cache write made no progress.")
            offset += written
        return offset


@lru_cache(maxsize=128)
def _warn_cache_capacity(root: Path) -> None:
    """Report disabled persistence once per active annotation/embedding root."""
    logger.warning(f"Input cache storage is full at {root}; continuing with uncached annotations or embeddings.")


def _load_array(path: Path, identity: dict) -> np.ndarray:
    """Validate a published array and return privately owned writable values."""
    arrays = {}
    try:
        index = _read_json(path / "index.json")
        success = _read_json(path / "_SUCCESS")
        if index["identity"] != identity or success != {"index_sha256": sha256_file(path / "index.json")}:
            raise ReplayCacheError("Annotation cache identity or publication changed.")
        file = path / "values.npy"
        if _signature(file) != index["signature"]:
            raise ReplayCacheError("Annotation cache values changed after preparation.")
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        arrays["values"] = array
        if (
            list(array.shape) != index["shape"]
            or array.dtype.str != index["dtype"]
            or array.dtype.kind not in "biufc"
            or not array.flags.c_contiguous
        ):
            raise ReplayCacheError("Annotation cache array shape or dtype is invalid.")
        return np.array(array, copy=True, order="C")
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ReplayCacheError):
            raise
        raise ReplayCacheError(f"Invalid annotation cache: {path}") from error
    finally:
        _close_arrays(arrays)


def load_cached_annotation(
    source: str | Path,
    *,
    reader: Callable[[Path], np.ndarray],
    format: str,
    cache_root: str | Path | None = None,
) -> np.ndarray:
    """Cache a numeric reader result, preserving its dtype, shape and values.

    ``format`` names the parser contract, including its version and options.
    Source mutation and altered cache arrays trigger preparation again. Returned
    values are writable copies, so scoring cannot change another trial's input.
    A full disk or storage quota skips persistence and returns the parsed values.
    Reader failures and unrelated I/O errors still propagate.
    """
    if not isinstance(format, str) or not format or format != format.strip():
        raise ValueError("Annotation format must be a non-empty canonical string.")
    source = Path(source).resolve()
    identity = {"schema": _SCHEMA, "source": str(source), "signature": _signature(source), "format": format}
    root = (
        source.parent / ".boxmot" / "replay_cache" / "annotations" if cache_root is None else Path(cache_root).resolve()
    )
    path = root / _digest(identity)
    locks = root / ".locks"

    def read_values() -> np.ndarray:
        """Validate parsed values independently of optional cache writes."""
        values = np.asarray(reader(source))
        if values.dtype.kind not in "biufc":
            raise ValueError("Cached annotations must be numeric arrays without object values.")
        if _signature(source) != identity["signature"]:
            raise ReplayCacheError("Annotation source changed during preparation.")
        return values

    def uncached(values: np.ndarray) -> np.ndarray:
        """Retain parsed data and its source guarantees when persistence fails."""
        if _signature(source) != identity["signature"]:
            raise ReplayCacheError("Annotation source changed during preparation.")
        _warn_cache_capacity(root)
        return np.array(values, copy=True, order="C")

    with ExitStack() as resources:
        try:
            root.mkdir(parents=True, exist_ok=True)
            locks.mkdir(exist_ok=True)
            resources.enter_context(FileLock(str(locks / f"{path.name}.lock")))
        except OSError as error:
            if error.errno not in _CAPACITY_ERRORS:
                raise
            return uncached(read_values())
        if path.is_dir() and not path.is_symlink():
            try:
                values = _load_array(path, identity)
                if _signature(source) != identity["signature"]:
                    raise ReplayCacheError("Annotation source changed while opening cached values.")
                return values
            except ReplayCacheError:
                pass
        values = read_values()
        staging = None
        try:
            staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=root))
            file = staging / "values.npy"
            with file.open("wb") as stream:
                np.save(_ArrayWriter(stream), np.array(values, copy=True, order="C"), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            _write_json(
                staging / "index.json",
                {
                    "identity": identity,
                    "signature": _signature(file),
                    "sha256": sha256_file(file),
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                },
            )
            _write_json(staging / "_SUCCESS", {"index_sha256": sha256_file(staging / "index.json")})
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            os.replace(staging, path)
        except OSError as error:
            if error.errno not in _CAPACITY_ERRORS:
                raise
            return uncached(values)
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
    return np.array(values, copy=True, order="C")
