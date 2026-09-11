"""Disposable mapped numeric annotations for repeated calibration and scoring."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from filelock import FileLock

from .manifest import sha256_file
from .replay_cache import ReplayCacheError, _close_arrays, _digest, _read_json, _signature, _write_json

_SCHEMA = "boxmot.annotation-cache/v1"


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
    """
    if not isinstance(format, str) or not format or format != format.strip():
        raise ValueError("Annotation format must be a non-empty canonical string.")
    source = Path(source).resolve()
    identity = {"schema": _SCHEMA, "source": str(source), "signature": _signature(source), "format": format}
    root = (
        source.parent / ".boxmot" / "replay_cache" / "annotations" if cache_root is None else Path(cache_root).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / _digest(identity)
    locks = root / ".locks"
    locks.mkdir(exist_ok=True)
    with FileLock(str(locks / f"{path.name}.lock")):
        if path.is_dir() and not path.is_symlink():
            try:
                values = _load_array(path, identity)
                if _signature(source) != identity["signature"]:
                    raise ReplayCacheError("Annotation source changed while opening cached values.")
                return values
            except ReplayCacheError:
                pass
        values = np.asarray(reader(source))
        if values.dtype.kind not in "biufc":
            raise ValueError("Cached annotations must be numeric arrays without object values.")
        if _signature(source) != identity["signature"]:
            raise ReplayCacheError("Annotation source changed during preparation.")
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=root))
        try:
            file = staging / "values.npy"
            with file.open("wb") as stream:
                np.save(stream, np.array(values, copy=True, order="C"), allow_pickle=False)
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
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return np.array(values, copy=True, order="C")
