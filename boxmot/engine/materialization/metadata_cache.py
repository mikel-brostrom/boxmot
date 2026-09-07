"""Persistent, stat-validated metadata for large local source trees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from boxmot.datasets.manifest import canonical_json_bytes

from .catalog import CatalogFileMetadata, CatalogMetadataResolver, inspect_catalog_file

_CACHE_SCHEMA = "boxmot.file-metadata/v1"
_LEGACY_EVALUATION_CACHE_SCHEMA = "boxmot.eval-file-metadata/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROGRESS_INTERVAL = 128
_MAX_STABILITY_ATTEMPTS = 3

StatusCallback = Callable[[str], None]


def _file_identity(path: Path) -> dict[str, str | int]:
    """Return every stable stat field required before reusing file metadata."""

    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mode": int(stat.st_mode),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _identity_key(identity: Mapping[str, str | int]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _cached_metadata(
    entry: object,
    identity: Mapping[str, str | int],
    *,
    include_image_size: bool,
) -> CatalogFileMetadata | None:
    if not isinstance(entry, Mapping) or entry.get("identity") != identity:
        return None
    digest = entry.get("sha256")
    size_bytes = entry.get("size_bytes")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        return None
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes != identity["size"]:
        return None

    raw_image_size = entry.get("image_size")
    image_size: tuple[int, int] | None = None
    if raw_image_size is not None:
        if (
            not isinstance(raw_image_size, list)
            or len(raw_image_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw_image_size)
        ):
            return None
        image_size = (raw_image_size[0], raw_image_size[1])
    if include_image_size and image_size is None:
        return None
    return CatalogFileMetadata(
        sha256=digest,
        size_bytes=size_bytes,
        image_size=image_size,
    )


class FileMetadataCache:
    """Persist hashes and dimensions behind a complete local file identity.

    Cache hits require path, device, inode, mode, size, modification time, and
    change time to match. Misses are hashed while checking that this identity
    remains stable, so interrupted or concurrent source writes are never
    published as valid metadata.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        status_callback: StatusCallback | None = None,
        metadata_resolver: CatalogMetadataResolver | None = None,
        fallback_paths: Iterable[str | Path] = (),
        progress_label: str = "Cataloging source metadata",
        write_schema: str = _CACHE_SCHEMA,
        discover_legacy_evaluation: bool | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.status_callback = status_callback
        self._metadata_resolver = metadata_resolver or inspect_catalog_file
        self._fallback_paths = tuple(Path(item).expanduser() for item in fallback_paths)
        self._progress_label = progress_label
        self._write_schema = write_schema
        self._discover_legacy_evaluation = discover_legacy_evaluation
        self.hits = 0
        self.misses = 0
        self._entries, self._fallback_entries = self._load_entries()
        self._dirty = False

    @property
    def processed(self) -> int:
        return self.hits + self.misses

    @staticmethod
    def _read_entries(path: Path) -> tuple[dict[str, Any], bool]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}, False
        if not isinstance(payload, Mapping) or payload.get("schema") not in {
            _CACHE_SCHEMA,
            _LEGACY_EVALUATION_CACHE_SCHEMA,
        }:
            return {}, False
        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            return {}, False
        return dict(entries), True

    def _is_default_source_cache_path(self) -> bool:
        default_root = user_cache_path("boxmot") / "materialization" / "source-metadata"
        return self.path.parent.resolve() == default_root.expanduser().resolve()

    def _legacy_evaluation_paths(self) -> tuple[Path, ...]:
        discover = self._discover_legacy_evaluation
        if discover is False or (discover is None and not self._is_default_source_cache_path()):
            return ()
        cache_root = user_cache_path("boxmot")
        return tuple(sorted((cache_root / "evaluation" / "source-catalogs").glob("*.json")))

    def _load_entries(self) -> tuple[dict[str, Any], dict[str, Any]]:
        primary, primary_is_valid = self._read_entries(self.path)
        if primary_is_valid:
            return primary, {}

        migrated: dict[str, Any] = {}
        fallback_paths = (*self._fallback_paths, *self._legacy_evaluation_paths())
        for fallback_path in dict.fromkeys(fallback_paths):
            entries, fallback_is_valid = self._read_entries(fallback_path)
            if not fallback_is_valid:
                continue
            for key, entry in entries.items():
                migrated.setdefault(str(key), entry)
        return {}, migrated

    def _emit_progress(self, *, force: bool = False) -> None:
        if self.status_callback is None or (not force and self.processed % _PROGRESS_INTERVAL):
            return
        message = f"{self._progress_label}… {self.processed:,} files ({self.hits:,} cached, {self.misses:,} refreshed)"
        try:
            self.status_callback(message)
        except Exception:
            # Reporting is observational and must not affect source identity.
            return

    def resolve(self, path: Path, include_image_size: bool) -> CatalogFileMetadata:
        """Return cached metadata or read a stable file and retain its result."""

        identity = _file_identity(path)
        key = _identity_key(identity)
        cached = _cached_metadata(
            self._entries.get(key),
            identity,
            include_image_size=include_image_size,
        )
        if cached is None:
            fallback_entry = self._fallback_entries.get(key)
            cached = _cached_metadata(
                fallback_entry,
                identity,
                include_image_size=include_image_size,
            )
            if cached is not None:
                self._entries[key] = fallback_entry
                self._dirty = True
        if cached is not None:
            self.hits += 1
            self._emit_progress()
            return cached

        for _ in range(_MAX_STABILITY_ATTEMPTS):
            before = identity
            metadata = self._metadata_resolver(Path(str(before["path"])), include_image_size)
            after = _file_identity(path)
            if after == before:
                break
            identity = after
            key = _identity_key(identity)
        else:
            raise OSError(f"Source file changed repeatedly while cataloging: {path}")

        self._entries[key] = {
            "identity": identity,
            "sha256": metadata.sha256,
            "size_bytes": metadata.size_bytes,
            "image_size": list(metadata.image_size) if metadata.image_size is not None else None,
        }
        self._dirty = True
        self.misses += 1
        self._emit_progress()
        return metadata

    def resolve_digest(self, path: Path) -> str:
        """Return a stat-validated digest suitable for source decoding."""

        return self.resolve(path, False).sha256

    def save(self) -> None:
        """Atomically publish updates; cache persistence failures are non-fatal."""

        if not self._dirty:
            return
        temporary: Path | None = None
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary = Path(raw_temporary)
            payload = canonical_json_bytes({"schema": self._write_schema, "entries": self._entries})
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self._dirty = False
        except OSError:
            # Callers still hold freshly read metadata, so persistence is only
            # an optimization and must never make cataloging fail.
            return
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def __enter__(self) -> FileMetadataCache:
        self._emit_progress(force=True)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.save()
        self._emit_progress(force=True)


def default_source_metadata_cache_path(source: str | Path) -> Path:
    """Return a platform-cache path isolated by canonical source location."""

    source_path = Path(source).expanduser().resolve()
    key = hashlib.sha256(canonical_json_bytes({"source": str(source_path)})).hexdigest()
    return user_cache_path("boxmot") / "materialization" / "source-metadata" / f"{key}.json"


__all__ = (
    "FileMetadataCache",
    "StatusCallback",
    "default_source_metadata_cache_path",
)
