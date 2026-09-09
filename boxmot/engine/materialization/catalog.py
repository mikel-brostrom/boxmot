"""Finite source catalogs for reproducible materialization."""

from __future__ import annotations

import configparser
import csv
import hashlib
import io
import math
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Mapping

import cv2

from boxmot.datasets.config import validate_sequence_names
from boxmot.datasets.kitti_mots import kitti_mots_frame_paths
from boxmot.datasets.manifest import canonical_json_bytes, sha256_file
from boxmot.datasets.readers.images import NUMPY_IMAGE_EXTENSIONS, probe_numpy_image_size
from boxmot.engine.dataset_variants.fps import select_sequence_frames, validate_dataset_fps
from boxmot.engine.frame_timing import SourceTimestamps
from boxmot.engine.materialization.source import SourceSample
from boxmot.engine.tracking.sources import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, is_appledouble_file

STILL_FRAME_EXTENSIONS = IMAGE_EXTENSIONS | NUMPY_IMAGE_EXTENSIONS
LOCAL_SOURCE_EXTENSIONS = STILL_FRAME_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    """Resolved frame metadata and the content identity used by a build plan."""

    samples: tuple[SourceSample, ...]
    fingerprint: str
    source_root: Path
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("A materialization source catalog must contain at least one sample.")
        if len(self.fingerprint) != 64:
            raise ValueError("SourceCatalog.fingerprint must be a SHA-256 digest.")
        object.__setattr__(self, "source_root", self.source_root.expanduser().resolve())


@dataclass(frozen=True, slots=True)
class CatalogFileMetadata:
    """Content metadata used while constructing a source catalog."""

    sha256: str
    size_bytes: int
    image_size: tuple[int, int] | None = None


CatalogMetadataResolver = Callable[[Path, bool], CatalogFileMetadata]


def default_data_root(explicit: str | Path | None = None) -> Path:
    """Resolve the tracking-dataset root using CLI or the local workspace."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return (Path("datasets") / "mot").resolve()


def resolve_dataset_root(config: Mapping[str, Any], data_root: str | Path | None = None) -> Path:
    """Resolve one canonical dataset storage path beneath the selected data root."""

    configured = str(config.get("root") or "")
    relative = PurePosixPath(configured)
    windows_path = PureWindowsPath(configured)
    if (
        not configured
        or "\\" in configured
        or relative.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in relative.parts
    ):
        raise ValueError("Dataset storage.root must be a safe path relative to the configured data root.")
    base = default_data_root(data_root)
    resolved = base.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Dataset storage.root must remain beneath the configured data root.") from exc
    return resolved


def _resolve_dataset_child(root: Path, value: Any, *, description: str) -> Path:
    """Resolve one configured POSIX path while containing symlink traversal."""

    configured = str(value or "")
    relative = PurePosixPath(configured)
    windows_path = PureWindowsPath(configured)
    if (
        not configured
        or "\\" in configured
        or relative.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in relative.parts
    ):
        raise ValueError(f"Dataset {description} paths must remain beneath storage.root.")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Dataset {description} paths must remain beneath storage.root.") from exc
    return resolved


def _dataset_split_config(config: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    """Return one validated split mapping from a resolved dataset config."""

    split_configs = config.get("splits")
    if not isinstance(split_configs, Mapping) or split not in split_configs:
        available = ", ".join(sorted(str(name) for name in (split_configs or {}))) or "none"
        raise ValueError(f"Unknown dataset split {split!r}; available splits: {available}.")
    split_config = split_configs[split]
    if not isinstance(split_config, Mapping):
        raise ValueError(f"Dataset split {split!r} must be a mapping.")
    return split_config


def resolve_dataset_split_root(
    config: Mapping[str, Any],
    split: str,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve the configured frame root for one dataset split."""

    dataset_root = resolve_dataset_root(config, data_root)
    split_config = _dataset_split_config(config, split)
    return _resolve_dataset_child(dataset_root, split_config.get("path"), description="split")


def resolve_dataset_annotation_root(
    config: Mapping[str, Any],
    split: str,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve the ground-truth root for one dataset split."""

    dataset_root = resolve_dataset_root(config, data_root)
    split_config = _dataset_split_config(config, split)
    annotations = split_config.get("annotations")
    if annotations is not None:
        return _resolve_dataset_child(dataset_root, annotations, description="annotation")
    if str(config.get("layout") or "").lower() == "visdrone":
        return _resolve_dataset_child(dataset_root, "annotations", description="annotation")
    return _resolve_dataset_child(dataset_root, split_config.get("path"), description="split")


def _sample_catalog_record(sample: SourceSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "sequence_id": sample.sequence_id,
        "frame_index": sample.frame_index,
        "timestamp_s": sample.timestamp_s,
        "height": sample.image_size[0],
        "width": sample.image_size[1],
        "image_ref": sample.image_ref,
        "source_sha256": sample.source_sha256,
        "source_frame_index": sample.source_frame_index,
    }


def _catalog(
    samples: list[SourceSample],
    *,
    source_root: Path,
    sources: list[dict[str, Any]],
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> SourceCatalog:
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Source catalog sample IDs must be unique.")
    payload = {
        "kind": kind,
        "samples": [_sample_catalog_record(sample) for sample in samples],
        "sources": sorted(sources, key=lambda item: str(item["ref"])),
        # Caller-supplied catalog metadata is semantic (dataset/split/layout
        # and taxonomy/ground-truth digests).  Machine-local provenance is
        # added to SourceCatalog.metadata only after this digest is computed.
        "metadata": {} if metadata is None else dict(metadata),
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return SourceCatalog(
        samples=tuple(samples),
        fingerprint=fingerprint,
        source_root=source_root,
        metadata={
            "source_kind": kind,
            "source_root_uri": source_root.as_uri(),
            "source_catalog_digest": fingerprint,
            "source_count": len(samples),
            **({} if metadata is None else dict(metadata)),
        },
    )


def _relative_ref(path: Path, root: Path, *, frame_index: int | None = None) -> str:
    try:
        ref = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        ref = path.resolve().as_uri()
    if frame_index is not None:
        ref = f"{ref}#frame={frame_index}"
    return ref


def _sequence_id(source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"source-{digest}"


def _jpeg_size(stream: BinaryIO) -> tuple[int, int]:
    if stream.read(2) != b"\xff\xd8":
        raise ValueError("invalid JPEG signature")
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while True:
        prefix = stream.read(1)
        while prefix and prefix != b"\xff":
            prefix = stream.read(1)
        if not prefix:
            break
        marker_raw = stream.read(1)
        while marker_raw == b"\xff":
            marker_raw = stream.read(1)
        if not marker_raw:
            break
        marker = marker_raw[0]
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        length_raw = stream.read(2)
        if len(length_raw) != 2:
            break
        length = int.from_bytes(length_raw, "big")
        if length < 2:
            break
        if marker in start_of_frame:
            header = stream.read(5)
            if len(header) != 5:
                break
            height = int.from_bytes(header[1:3], "big")
            width = int.from_bytes(header[3:5], "big")
            return height, width
        stream.seek(length - 2, os.SEEK_CUR)
    raise ValueError("JPEG dimensions were not found")


def _tiff_scalar(stream: BinaryIO, raw: bytes, *, endian: str, value_type: int, count: int) -> int:
    sizes = {3: 2, 4: 4}
    if value_type not in sizes or count != 1:
        raise ValueError("unsupported TIFF dimension value")
    size = sizes[value_type]
    if size <= 4:
        data = raw[:size]
    else:  # pragma: no cover - scalar SHORT/LONG values are always inline
        offset = struct.unpack(f"{endian}I", raw)[0]
        current = stream.tell()
        stream.seek(offset)
        data = stream.read(size)
        stream.seek(current)
    return int.from_bytes(data, "little" if endian == "<" else "big")


def _tiff_size(stream: BinaryIO) -> tuple[int, int]:
    byte_order = stream.read(2)
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        raise ValueError("invalid TIFF byte order")
    header = stream.read(6)
    if len(header) != 6 or struct.unpack(f"{endian}H", header[:2])[0] != 42:
        raise ValueError("invalid or unsupported TIFF header")
    stream.seek(struct.unpack(f"{endian}I", header[2:])[0])
    count_raw = stream.read(2)
    if len(count_raw) != 2:
        raise ValueError("truncated TIFF directory")
    entry_count = struct.unpack(f"{endian}H", count_raw)[0]
    dimensions: dict[int, int] = {}
    for _ in range(entry_count):
        entry = stream.read(12)
        if len(entry) != 12:
            raise ValueError("truncated TIFF directory entry")
        tag, value_type, count = struct.unpack(f"{endian}HHI", entry[:8])
        if tag in {256, 257}:
            dimensions[tag] = _tiff_scalar(
                stream,
                entry[8:],
                endian=endian,
                value_type=value_type,
                count=count,
            )
    if 256 not in dimensions or 257 not in dimensions:
        raise ValueError("TIFF dimensions were not found")
    return dimensions[257], dimensions[256]


def _webp_size(stream: BinaryIO) -> tuple[int, int]:
    header = stream.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WEBP":
        raise ValueError("invalid WebP signature")
    while True:
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            break
        kind = chunk_header[:4]
        size = int.from_bytes(chunk_header[4:], "little")
        payload = stream.read(size)
        if len(payload) != size:
            break
        if size % 2:
            stream.read(1)
        if kind == b"VP8X" and len(payload) >= 10:
            width = int.from_bytes(payload[4:7], "little") + 1
            height = int.from_bytes(payload[7:10], "little") + 1
            return height, width
        if kind == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            return ((bits >> 14) & 0x3FFF) + 1, (bits & 0x3FFF) + 1
        if kind == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            return height, width
    raise ValueError("WebP dimensions were not found")


def _probe_image_size(path: Path) -> tuple[int, int]:
    """Read dimensions from an image header without decoding its pixels."""

    suffix = path.suffix.lower()
    if suffix in NUMPY_IMAGE_EXTENSIONS:
        return probe_numpy_image_size(path)
    try:
        with path.open("rb") as stream:
            if suffix == ".png":
                header = stream.read(24)
                if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                    raise ValueError("invalid PNG header")
                width, height = struct.unpack(">II", header[16:24])
                result = (height, width)
            elif suffix in {".jpg", ".jpeg"}:
                result = _jpeg_size(stream)
            elif suffix == ".bmp":
                header = stream.read(26)
                if len(header) != 26 or header[:2] != b"BM":
                    raise ValueError("invalid BMP header")
                width, height = struct.unpack("<ii", header[18:26])
                result = (abs(height), abs(width))
            elif suffix in {".tif", ".tiff"}:
                result = _tiff_size(stream)
            elif suffix == ".webp":
                result = _webp_size(stream)
            else:  # guarded by STILL_FRAME_EXTENSIONS
                raise ValueError(f"unsupported image extension {suffix!r}")
    except (OSError, ValueError, struct.error) as exc:
        raise ValueError(f"Could not read image metadata: {path}") from exc
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"Image dimensions must be positive: {path}")
    return result


def inspect_catalog_file(path: Path, include_image_size: bool) -> CatalogFileMetadata:
    """Read one file's content identity and optional image dimensions."""

    return CatalogFileMetadata(
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        image_size=_probe_image_size(path) if include_image_size else None,
    )


def _image_sample(
    path: Path,
    *,
    split: str,
    sample_id: str,
    sequence_id: str,
    frame_index: int,
    image_size: tuple[int, int],
    image_ref: str,
    source_sha256: str,
) -> SourceSample:
    return SourceSample(
        sample_id=sample_id,
        split=split,
        sequence_id=sequence_id,
        frame_index=frame_index,
        timestamp_s=None,
        image_size=image_size,
        source_uri=path.as_uri(),
        source_sha256=source_sha256,
        image_ref=image_ref,
    )


def _video_samples(
    path: Path,
    *,
    split: str,
    source_root: Path,
    source_sha256: str,
) -> list[SourceSample]:
    """Scan video packet metadata without retrieving decoded pixel arrays."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video source: {path}")
    uri = path.as_uri()
    sequence_id = _sequence_id(_relative_ref(path, source_root))
    samples: list[SourceSample] = []
    source_index = 0
    try:
        timestamps = SourceTimestamps(float(capture.get(cv2.CAP_PROP_FPS)))
        while capture.grab():
            height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            samples.append(
                SourceSample(
                    sample_id=f"{sequence_id}:{source_index:012d}",
                    split=split,
                    sequence_id=sequence_id,
                    frame_index=source_index,
                    timestamp_s=timestamps.resolve(position_ms, frame_index=source_index),
                    image_size=(height, width),
                    source_uri=uri,
                    source_sha256=source_sha256,
                    image_ref=_relative_ref(path, source_root, frame_index=source_index),
                    source_frame_index=source_index,
                )
            )
            source_index += 1
    finally:
        capture.release()
    if not samples:
        raise ValueError(f"Video source contains no readable frames: {path}")
    return samples


def catalog_local_source(
    source: str | Path,
    *,
    split: str = "default",
    metadata_resolver: CatalogMetadataResolver | None = None,
) -> SourceCatalog:
    """Catalog a finite local source without retaining decoded frames."""

    if not split:
        raise ValueError("split must not be empty.")
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix.lower() not in LOCAL_SOURCE_EXTENSIONS:
        raise ValueError(f"Unsupported finite local source type: {path.suffix or '<none>'}")

    source_root = path if path.is_dir() else path.parent
    source_files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file() and not is_appledouble_file(item))
    )
    resolve_metadata = metadata_resolver or inspect_catalog_file
    source_records: dict[Path, dict[str, Any]] = {}
    file_metadata: dict[Path, CatalogFileMetadata] = {}
    for item in source_files:
        suffix = item.suffix.lower()
        if suffix not in LOCAL_SOURCE_EXTENSIONS:
            continue
        metadata = resolve_metadata(item, suffix in STILL_FRAME_EXTENSIONS)
        if suffix in STILL_FRAME_EXTENSIONS and metadata.image_size is None:
            raise ValueError(f"Image metadata resolver omitted dimensions for {item}.")
        file_metadata[item] = metadata
        source_records[item] = {
            "ref": _relative_ref(item, source_root),
            "sha256": metadata.sha256,
            "size_bytes": metadata.size_bytes,
        }
    samples: list[SourceSample] = []
    if path.is_file() and path.suffix.lower() in STILL_FRAME_EXTENSIONS:
        image_size = file_metadata[path].image_size
        assert image_size is not None
        samples.append(
            _image_sample(
                path,
                split=split,
                sample_id=path.name,
                sequence_id=_sequence_id(_relative_ref(path, source_root)),
                frame_index=0,
                image_size=image_size,
                image_ref=_relative_ref(path, source_root),
                source_sha256=str(source_records[path]["sha256"]),
            )
        )
    elif path.is_file():
        samples.extend(
            _video_samples(
                path,
                split=split,
                source_root=source_root,
                source_sha256=str(source_records[path]["sha256"]),
            )
        )
    else:
        frame_indices: dict[Path, int] = {}
        for child in source_files:
            if child.suffix.lower() in STILL_FRAME_EXTENSIONS:
                image_size = file_metadata[child].image_size
                assert image_size is not None
                parent_ref = _relative_ref(child.parent, source_root)
                image_index = frame_indices.get(child.parent, 0)
                samples.append(
                    _image_sample(
                        child,
                        split=split,
                        sample_id=child.relative_to(source_root).as_posix(),
                        sequence_id=_sequence_id(parent_ref),
                        frame_index=image_index,
                        image_size=image_size,
                        image_ref=_relative_ref(child, source_root),
                        source_sha256=str(source_records[child]["sha256"]),
                    )
                )
                frame_indices[child.parent] = image_index + 1
            elif child.suffix.lower() in VIDEO_EXTENSIONS:
                samples.extend(
                    _video_samples(
                        child,
                        split=split,
                        source_root=source_root,
                        source_sha256=str(source_records[child]["sha256"]),
                    )
                )

    sources = list(source_records.values())
    return _catalog(samples, source_root=source_root, sources=sources, kind="local")


def _sequence_rate(sequence_root: Path) -> float | None:
    path = sequence_root / "seqinfo.ini"
    if not path.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    value = parser.getfloat("Sequence", "frameRate", fallback=0.0)
    return value if math.isfinite(value) and value > 0 else None


def _read_sequence_timestamps(path: Path, frame_count: int) -> tuple[tuple[float, ...], CatalogFileMetadata]:
    """Validate a complete sequence timeline and hash the exact parsed bytes."""

    contents = path.read_bytes()
    timestamps: list[float] = []
    try:
        rows = csv.reader(io.StringIO(contents.decode("utf-8"), newline=""), strict=True)
        if next(rows, None) != ["frame_id", "timestamp_s"]:
            raise ValueError(f"{path} must have exactly the columns frame_id,timestamp_s.")
        for row in rows:
            if len(row) != 2:
                raise ValueError(f"{path} line {rows.line_num} must contain exactly two values.")
            try:
                frame_id = int(row[0])
            except ValueError as exc:
                raise ValueError(f"{path} line {rows.line_num} frame_id must be an integer.") from exc
            expected_frame_id = len(timestamps) + 1
            if frame_id != expected_frame_id or frame_id > frame_count:
                raise ValueError(f"{path} frame_id values must cover frames 1..{frame_count} exactly once in order.")
            try:
                timestamp = float(row[1])
            except ValueError as exc:
                raise ValueError(f"{path} line {rows.line_num} timestamp_s must be a finite number.") from exc
            if not math.isfinite(timestamp):
                raise ValueError(f"{path} line {rows.line_num} timestamp_s must be a finite number.")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError(f"{path} timestamp_s values must increase strictly.")
            timestamps.append(timestamp)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"{path} must contain valid UTF-8 CSV data: {exc}") from exc
    if len(timestamps) != frame_count:
        raise ValueError(
            f"{path} must contain one timestamp for every image; expected {frame_count}, got {len(timestamps)}."
        )
    return tuple(timestamps), CatalogFileMetadata(
        sha256=hashlib.sha256(contents).hexdigest(),
        size_bytes=len(contents),
    )


def catalog_mot_dataset(
    config: Mapping[str, Any],
    *,
    split: str | None = None,
    data_root: str | Path | None = None,
    metadata_resolver: CatalogMetadataResolver | None = None,
    fps: float | None = None,
) -> SourceCatalog:
    """Resolve a tracking dataset split beneath the selected tracking-data root.

    ``metadata_resolver`` is an explicit opt-in hook for consumers that can
    safely reuse file metadata. The default always reads and hashes every
    source, which keeps materialization's build identity cryptographically
    fresh.

    An optional sequence-root ``timestamps.csv`` supplies ``frame_id,timestamp_s``
    rows for every sorted image, numbered from one. The validated sidecar takes
    precedence over ``seqinfo.ini`` frame rate and participates in content identity.
    ``fps`` keeps the first frame in each occupied sampling interval, retains its
    capture timestamp, and numbers selected frames contiguously for cache/GT use.
    KITTI MOTS uses native zero-based PNG frame numbers at 10 Hz and pairs
    each image with a 16-bit instance PNG beneath the annotation root.
    """

    if fps is not None:
        fps = validate_dataset_fps(fps)
    layout = str(config.get("layout") or "")
    if layout not in {"mot", "visdrone", "kitti-mots"}:
        raise ValueError(f"Unsupported dataset layout {layout!r}; expected 'mot', 'visdrone', or 'kitti-mots'.")
    is_kitti_mots = layout == "kitti-mots"
    split_name = str(split or config.get("default_split") or "")
    dataset_root = resolve_dataset_root(config, data_root)
    split_config = _dataset_split_config(config, split_name)
    split_root = resolve_dataset_split_root(config, split_name, data_root)
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Dataset split does not exist: {split_root}. Set --data-root explicitly to use another dataset root."
        )
    annotations_root = None
    if is_kitti_mots and bool(split_config.get("has_ground_truth")) != (split_config.get("annotations") is not None):
        raise ValueError("KITTI MOTS splits must declare annotations exactly when has_ground_truth is true.")
    if split_config.get("annotations") is not None:
        annotations_root = resolve_dataset_annotation_root(config, split_name, data_root)
        if not annotations_root.is_dir():
            raise FileNotFoundError(f"Dataset annotation directory does not exist: {annotations_root}.")

    samples: list[SourceSample] = []
    sources: list[dict[str, Any]] = []
    ground_truth_sources: list[dict[str, Any]] = []
    timestamp_sources: list[dict[str, Any]] = []
    frame_sampling: dict[str, list[int]] = {}
    resolve_metadata = metadata_resolver or inspect_catalog_file
    sequence_roots = sorted(item for item in split_root.iterdir() if item.is_dir() and not is_appledouble_file(item))
    if is_kitti_mots:
        sequence_roots = [path for path in sequence_roots if not path.name.startswith(".")]
    if "sequences" in split_config:
        selected_sequences = set(validate_sequence_names(split_config["sequences"]))
        missing_sequences = selected_sequences - {path.name for path in sequence_roots}
        if missing_sequences:
            missing = ", ".join(sorted(missing_sequences))
            raise FileNotFoundError(f"Dataset sequence directories do not exist: {missing}.")
        sequence_roots = [path for path in sequence_roots if path.name in selected_sequences]
    for sequence_root in sequence_roots:
        image_root = sequence_root / "img1"
        if not image_root.is_dir():
            image_root = sequence_root
        image_paths = (
            list(kitti_mots_frame_paths(sequence_root))
            if is_kitti_mots
            else sorted(
                item
                for item in image_root.iterdir()
                if item.is_file() and not is_appledouble_file(item) and item.suffix.lower() in STILL_FRAME_EXTENSIONS
            )
        )
        native_indices = [int(path.stem) for path in image_paths] if is_kitti_mots else list(range(len(image_paths)))
        timestamp_path = sequence_root / "timestamps.csv"
        timestamps = None
        if timestamp_path.exists() or timestamp_path.is_symlink():
            timestamp_ref = timestamp_path.relative_to(dataset_root).as_posix()
            resolved_timestamp_path = _resolve_dataset_child(dataset_root, timestamp_ref, description="timestamp")
            if not resolved_timestamp_path.is_file():
                raise ValueError(f"Sequence timestamps.csv must be a regular file: {timestamp_path}.")
            timestamps, timestamp_metadata = _read_sequence_timestamps(resolved_timestamp_path, len(image_paths))
            timestamp_record = {
                "ref": timestamp_ref,
                "sha256": timestamp_metadata.sha256,
                "size_bytes": timestamp_metadata.size_bytes,
            }
            sources.append(timestamp_record)
            timestamp_sources.append(timestamp_record)
        frame_rate = _sequence_rate(sequence_root) if timestamps is None else None
        if is_kitti_mots and timestamps is None and frame_rate is None:
            frame_rate = 10.0
        sequence_timestamps = (
            timestamps
            if timestamps is not None
            else tuple(None if frame_rate is None else index / frame_rate for index in native_indices)
        )
        selected_indices = (
            tuple(range(len(image_paths)))
            if fps is None
            else select_sequence_frames(sequence_timestamps, fps=fps, sequence_id=sequence_root.name)
        )
        if fps is not None:
            frame_sampling[sequence_root.name] = [native_indices[index] + 1 for index in selected_indices]
        for frame_index, source_index in enumerate(selected_indices):
            image_path = image_paths[source_index]
            if is_kitti_mots and fps is None:
                frame_index = native_indices[source_index]
            sample_id = f"{split_name}:{sequence_root.name}:{frame_index}"
            image_ref = _relative_ref(image_path, dataset_root)
            file_metadata = resolve_metadata(image_path, True)
            if file_metadata.image_size is None:
                raise ValueError(f"Image metadata resolver omitted dimensions for {image_path}.")
            samples.append(
                SourceSample(
                    sample_id=sample_id,
                    split=split_name,
                    sequence_id=sequence_root.name,
                    frame_index=frame_index,
                    timestamp_s=sequence_timestamps[source_index],
                    image_size=file_metadata.image_size,
                    source_uri=image_path.as_uri(),
                    source_sha256=file_metadata.sha256,
                    image_ref=image_ref,
                )
            )
            sources.append(
                {
                    "ref": image_ref,
                    "sha256": file_metadata.sha256,
                    "size_bytes": file_metadata.size_bytes,
                }
            )
            if is_kitti_mots and annotations_root is not None:
                annotation_path = annotations_root / sequence_root.name / image_path.name
                if not annotation_path.is_file():
                    raise FileNotFoundError(f"KITTI MOTS instance annotation does not exist: {annotation_path}.")
                annotation_metadata = resolve_metadata(annotation_path, True)
                if annotation_metadata.image_size != file_metadata.image_size:
                    raise ValueError(
                        f"KITTI MOTS annotation dimensions do not match image dimensions: {annotation_path}."
                    )
                annotation_record = {
                    "ref": _relative_ref(annotation_path, dataset_root),
                    "sha256": annotation_metadata.sha256,
                    "size_bytes": annotation_metadata.size_bytes,
                }
                sources.append(annotation_record)
                ground_truth_sources.append(annotation_record)
        metadata_files = [sequence_root / "seqinfo.ini"]
        metadata_files.extend(
            sorted(path for path in (sequence_root / "gt").glob("*") if not is_appledouble_file(path))
            if (sequence_root / "gt").is_dir()
            else []
        )
        if annotations_root is not None and not is_kitti_mots:
            annotation_path = annotations_root / f"{sequence_root.name}.txt"
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Dataset annotation for sequence {sequence_root.name!r} does not exist: {annotation_path}."
                )
            annotation_metadata = resolve_metadata(annotation_path, False)
            annotation_record = {
                "ref": _relative_ref(annotation_path, dataset_root),
                "sha256": annotation_metadata.sha256,
                "size_bytes": annotation_metadata.size_bytes,
            }
            sources.append(annotation_record)
            ground_truth_sources.append(annotation_record)
        if layout == "visdrone" and annotations_root is None:
            metadata_files.append(dataset_root / "annotations" / f"{sequence_root.name}.txt")
        for metadata_path in metadata_files:
            if not metadata_path.is_file():
                continue
            file_metadata = resolve_metadata(metadata_path, False)
            record = {
                "ref": _relative_ref(metadata_path, dataset_root),
                "sha256": file_metadata.sha256,
                "size_bytes": file_metadata.size_bytes,
            }
            sources.append(record)
            if metadata_path.suffix.lower() in {".txt", ".csv"} and (
                metadata_path.parent.name in {"gt", "annotations"}
            ):
                ground_truth_sources.append(record)

    class_taxonomy = config.get("classes") or {}
    return _catalog(
        samples,
        source_root=dataset_root,
        sources=sources,
        kind="dataset",
        metadata={
            "dataset_id": str(config.get("id") or ""),
            "split": split_name,
            "layout": layout,
            "class_taxonomy_digest": hashlib.sha256(canonical_json_bytes(class_taxonomy)).hexdigest(),
            "ground_truth_digest": hashlib.sha256(
                canonical_json_bytes(
                    ground_truth_sources
                    if fps is None
                    else {"sources": ground_truth_sources, "frame_sampling": frame_sampling}
                )
            ).hexdigest(),
            **({"fps": fps, "frame_sampling": frame_sampling} if fps is not None else {}),
            **(
                {"timestamps_digest": hashlib.sha256(canonical_json_bytes(timestamp_sources)).hexdigest()}
                if timestamp_sources
                else {}
            ),
        },
    )


__all__ = (
    "CatalogFileMetadata",
    "CatalogMetadataResolver",
    "SourceCatalog",
    "catalog_local_source",
    "catalog_mot_dataset",
    "default_data_root",
    "inspect_catalog_file",
    "resolve_dataset_annotation_root",
    "resolve_dataset_root",
    "resolve_dataset_split_root",
)
