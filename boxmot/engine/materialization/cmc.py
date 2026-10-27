"""Persist validated camera analysis as derived artifacts beside a dataset build."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from filelock import FileLock

CMC_MATERIALIZATION_SCHEMA = "boxmot.materialization/cmc/v1"
_Validated = TypeVar("_Validated")


@dataclass(frozen=True)
class CMCMaterialization(Generic[_Validated]):
    """One validated byte snapshot, independent of later cache or run-directory edits."""

    path: Path
    report: bytes
    warps: bytes
    validated: _Validated
    reused: bool


def cmc_materialization_key(inputs: Mapping[str, Any], *, analysis_schema: str) -> str:
    """Keep estimator, source selection, and format variants under distinct keys."""
    identity = {"schema": CMC_MATERIALIZATION_SCHEMA, "analysis_schema": analysis_schema, "inputs": dict(inputs)}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _publication(key: str, report: bytes, warps: bytes) -> dict[str, str]:
    """Bind both artifact files to a completed materialization of this identity."""
    return {
        "schema": CMC_MATERIALIZATION_SCHEMA,
        "key": key,
        "analysis_sha256": hashlib.sha256(report).hexdigest(),
        "warps_sha256": hashlib.sha256(warps).hexdigest(),
    }


def _read_regular(path: Path) -> bytes:
    """Reject indirection or non-file artifacts before reading their contents."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"CMC materialization requires a regular file: {path}")
    return path.read_bytes()


def _load_entry(
    path: Path,
    key: str,
    validate: Callable[[bytes, bytes], _Validated],
) -> CMCMaterialization[_Validated]:
    """Read without acquiring a writable lock, checking the pair and current sources."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"CMC materialization is not a completed directory: {path}")
    marker = json.loads(_read_regular(path / "_SUCCESS"))
    report = _read_regular(path / "analysis.json")
    warps = _read_regular(path / "warps.npz")
    if marker != _publication(key, report, warps):
        raise ValueError("CMC materialization publication marker does not match its artifacts.")
    validated = validate(report, warps)
    return CMCMaterialization(path, report, warps, validated, reused=True)


def _try_load_entry(
    path: Path,
    key: str,
    validate: Callable[[bytes, bytes], _Validated],
) -> CMCMaterialization[_Validated] | None:
    """Treat absent, incomplete, stale, or corrupted derived evidence as a cache miss."""
    try:
        return _load_entry(path, key, validate)
    except (OSError, ValueError):
        return None


def _remove_owned_path(path: Path) -> None:
    """Remove one cache-owned entry without following an invalid directory symlink."""
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _sync_directory(path: Path) -> None:
    """Flush directory metadata where supported by the filesystem."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def materialize_cmc(
    build: str | Path,
    *,
    inputs: Mapping[str, Any],
    analysis_schema: str,
    generate: Callable[[Path], None],
    validate: Callable[[bytes, bytes], _Validated],
) -> CMCMaterialization[_Validated]:
    """Reuse a valid entry or atomically publish one complete analysis under its lock.

    A successful read does not create directories or touch a lock file. Writers
    recheck after acquiring the per-key lock, so concurrent callers estimate
    camera motion once. The callback must validate source epochs as well as the
    report and archive; this storage layer never changes the dataset manifest.
    """
    root = Path(build).resolve() / "cmc"
    if root.is_symlink():
        raise ValueError(f"CMC materialization root must not be a symlink: {root}")
    key = cmc_materialization_key(inputs, analysis_schema=analysis_schema)
    path = root / key
    existing = _try_load_entry(path, key, validate)
    if existing is not None:
        return existing

    root.mkdir(parents=True, exist_ok=True)
    locks = root / ".locks"
    lock_path = locks / f"{key}.lock"
    if locks.is_symlink() or lock_path.is_symlink():
        raise ValueError("CMC materialization lock paths must not contain symlinks.")
    locks.mkdir(exist_ok=True)
    with FileLock(lock_path):
        existing = _try_load_entry(path, key, validate)
        if existing is not None:
            return existing
        for stale in root.glob(f".{key}.tmp-*"):
            _remove_owned_path(stale)
        with tempfile.TemporaryDirectory(prefix=f".{key}.tmp-", dir=root) as temporary:
            staging = Path(temporary)
            generate(staging)
            report = _read_regular(staging / "analysis.json")
            warps = _read_regular(staging / "warps.npz")
            validated = validate(report, warps)
            # Publish the success marker last, after validating the complete
            # snapshot and flushing both generated artifacts.
            for name in ("analysis.json", "warps.npz"):
                with (staging / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            with (staging / "_SUCCESS").open("w", encoding="utf-8") as stream:
                json.dump(_publication(key, report, warps), stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(staging)
            # Only this exact, invalid key can be replaced. Other estimator or
            # selection variants remain available to subsequent workflows.
            if path.exists() or path.is_symlink():
                _remove_owned_path(path)
            os.replace(staging, path)
            _sync_directory(root)
    return CMCMaterialization(path, report, warps, validated, reused=False)
