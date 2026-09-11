"""Opt-in derived NumPy replay inputs beside immutable Parquet builds.

``prepare_replay_sequence`` publishes ``replay_cache/<identity>/`` atomically.
Each entry contains ``index.json``, ``_SUCCESS`` and read-only mapped geometry,
score, class and optional embedding ``.npy`` arrays. Parquet remains authoritative;
this format is a disposable runtime optimization, never a dataset publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock

from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes

from ._parquet import iter_selected_batches
from .cached import CachedVisionDataset, DatasetSample, _MaskPayloadStream
from .manifest import DatasetManifest, canonical_json_bytes, sha256_file
from .masks import MASK_CODEC, unpack_mask_batch
from .readers import read_rgb_chw_uint8
from .schema import (
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MANIFEST_FILENAME,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SUCCESS_FILENAME,
    masks_schema,
)
from .storage import artifact_files, resolve_artifact_path
from .validation import DatasetValidationError, validate_published_build

_CACHE_SCHEMA = "boxmot.replay-cache/v1"
_INDEX = "index.json"
_ARRAY_DTYPES = {
    "geometry": np.dtype("float32"),
    "scores": np.dtype("float32"),
    "class_ids": np.dtype("int64"),
    "embeddings": np.dtype("float32"),
}


class ReplayCacheError(ValueError):
    """Raised when a derived replay entry must be prepared again."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayCacheError(f"Invalid replay metadata: {path}") from error
    if not isinstance(result, dict):
        raise ReplayCacheError(f"Replay metadata must be an object: {path}")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("wb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _signature(path: Path) -> list[int]:
    """Bind a verified digest to the same ordinary file and mutation epoch."""
    if path.is_symlink() or not path.is_file():
        raise ReplayCacheError(f"Replay inputs require a regular file: {path}")
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _source_signature(build: Path, manifest: DatasetManifest) -> dict[str, list[int]]:
    result = {name: _signature(build / name) for name in (MANIFEST_FILENAME, SUCCESS_FILENAME)}
    for artifact in manifest.artifacts:
        directory = resolve_artifact_path(build, artifact.path)
        actual = tuple(path.relative_to(build).as_posix() for path in artifact_files(directory))
        if actual != tuple(shard.path for shard in artifact.shards):
            raise DatasetValidationError(f"Dataset artifact {artifact.name!r} shard list differs from its manifest.")
        for shard in artifact.shards:
            result[shard.path] = _signature(resolve_artifact_path(build, shard.path))
    return result


def _source_context(build: Path) -> tuple[DatasetManifest, dict[str, list[int]]]:
    manifest = DatasetManifest.load(build)
    try:
        success = json.loads((build / SUCCESS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError("Dataset publication marker is invalid.") from error
    if success != {"schema": SCHEMA_ID, "build_id": manifest.build_id}:
        raise DatasetValidationError("Dataset publication marker does not match its manifest.")
    expected_artifacts = {SAMPLES_ARTIFACT, INSTANCES_ARTIFACT}
    for name, published in (
        (MASKS_ARTIFACT, manifest.publish.masks),
        (EMBEDDINGS_ARTIFACT, manifest.publish.embeddings),
    ):
        if published:
            expected_artifacts.add(name)
        elif (build / name).exists():
            raise DatasetValidationError(f"Dataset build contains unpublished {name!r} artifact data.")
    if set(manifest.artifacts_by_name) != expected_artifacts:
        raise DatasetValidationError("Dataset artifact set differs from its published content.")
    return manifest, _source_signature(build, manifest)


def _identity(
    manifest: DatasetManifest, *, sequence_id: str, split: str | None, load_embeddings: bool
) -> dict[str, Any]:
    names = [SAMPLES_ARTIFACT, INSTANCES_ARTIFACT]
    if load_embeddings:
        if not manifest.publish.embeddings:
            raise DatasetValidationError(
                "Embeddings were requested, but this dataset build does not publish embeddings."
            )
        names.append(EMBEDDINGS_ARTIFACT)
    return {
        "schema": _CACHE_SCHEMA,
        "dataset_schema": manifest.schema,
        "dataset_schema_version": manifest.schema_version,
        "build_id": manifest.build_id,
        "box_type": manifest.box_type,
        "sequence_id": sequence_id,
        "split": split,
        "load_embeddings": load_embeddings,
        "artifacts": [manifest.artifact(name).to_dict() for name in names],
    }


def _validate_source_once(build: Path, manifest: DatasetManifest, signatures: dict, cache_root: Path) -> None:
    """Hash immutable source artifacts once per mutation epoch across workers."""
    content = {
        "schema": manifest.schema,
        "build_id": manifest.build_id,
        "artifacts": [artifact.to_dict() for artifact in manifest.artifacts],
    }
    directory = cache_root / ".sources"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f"{_digest(content)}.json"
    expected = {"content": content, "build": str(build), "signatures": signatures}
    with FileLock(str(receipt.with_suffix(".lock"))):
        if receipt.exists():
            try:
                if _read_json(receipt) == expected:
                    return
            except ReplayCacheError:
                pass
        validate_published_build(build, manifest=manifest)
        if _source_signature(build, manifest) != signatures:
            raise DatasetValidationError("Dataset artifacts changed while preparing replay inputs.")
        temporary = receipt.with_suffix(".tmp")
        try:
            _write_json(temporary, expected)
            os.replace(temporary, receipt)
        finally:
            temporary.unlink(missing_ok=True)


def _close_arrays(arrays: dict[str, np.ndarray]) -> None:
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    arrays.clear()


def _load_entry(
    path: Path, *, expected_identity: dict[str, Any] | None = None, expected_build: Path | None = None
) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate the publication and open immutable arrays without payload scans."""
    arrays: dict[str, np.ndarray] = {}
    try:
        if path.is_symlink() or not path.is_dir():
            raise ReplayCacheError(f"Replay cache is not a published directory: {path}")
        success = _read_json(path / SUCCESS_FILENAME)
        index = _read_json(path / _INDEX)
        identity = index["identity"]
        if identity["schema"] != _CACHE_SCHEMA or _digest(identity) != path.name:
            raise ReplayCacheError("Replay cache identity or format is invalid.")
        if expected_identity is not None and identity != expected_identity:
            raise ReplayCacheError("Replay cache does not match the requested source selection.")
        if expected_build is not None and index["build"] != str(expected_build):
            raise ReplayCacheError("Replay cache source location changed.")
        if success != {"identity": path.name, "index_sha256": sha256_file(path / _INDEX)}:
            raise ReplayCacheError("Replay cache publication marker does not match its metadata.")
        manifest, signatures = _source_context(Path(index["build"]))
        if (
            _identity(
                manifest,
                sequence_id=identity["sequence_id"],
                split=identity["split"],
                load_embeddings=identity["load_embeddings"],
            )
            != identity
        ):
            raise ReplayCacheError("Replay source identity changed.")
        if index["source_signature"] != signatures:
            raise ReplayCacheError("Replay source files changed after validation.")
        if index["image_root"] != str(manifest.metadata.get("source_root_uri", index["build"])):
            raise ReplayCacheError("Replay source image root changed.")
        expected_names = {"geometry", "scores", "class_ids"} | (
            {"embeddings"} if identity["load_embeddings"] else set()
        )
        if set(index["arrays"]) != expected_names:
            raise ReplayCacheError("Replay cache array set is invalid.")
        count = index["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ReplayCacheError("Replay cache row count is invalid.")
        dimension = int(manifest.artifact(EMBEDDINGS_ARTIFACT).metadata["dim"]) if identity["load_embeddings"] else None
        for name in sorted(expected_names):
            record = index["arrays"][name]
            file = path / f"{name}.npy"
            if _signature(file) != record["signature"]:
                raise ReplayCacheError(f"Replay cache array {name!r} changed after hashing.")
            # The digest was checked when this stat signature was recorded.
            # Header validation stays cheap and mmap avoids reading every row.
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            arrays[name] = array
            shape = (
                (count, 5 if manifest.box_type == "obb" else 4)
                if name == "geometry"
                else ((count, dimension) if name == "embeddings" else (count,))
            )
            if (
                array.shape != shape
                or array.dtype != _ARRAY_DTYPES[name]
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                raise ReplayCacheError(f"Replay cache array {name!r} has an invalid shape, dtype or ownership.")
        cursor = 0
        sample_ids = set()
        previous_order = None
        for sample in index["samples"]:
            start, end = sample["offset"]
            order = (sample["split"], sample["sequence_id"], sample["frame_index"], sample["sample_id"])
            if start != cursor or not isinstance(end, int) or end < start or end > count:
                raise ReplayCacheError("Replay cache offsets are invalid.")
            if sample["sample_id"] in sample_ids or (previous_order is not None and order <= previous_order):
                raise ReplayCacheError("Replay cache samples are not unique and canonically ordered.")
            if sample["sequence_id"] != identity["sequence_id"] or (
                identity["split"] is not None and sample["split"] != identity["split"]
            ):
                raise ReplayCacheError("Replay sample selection is invalid.")
            if len(sample["instance_ids"]) != end - start or len(sample["detection_indices"]) != end - start:
                raise ReplayCacheError("Replay detection keys do not match cached rows.")
            if sample["detection_indices"] != list(range(end - start)):
                raise ReplayCacheError("Replay detection indices must retain canonical row order.")
            sample_ids.add(sample["sample_id"])
            previous_order, cursor = order, end
        if not index["samples"] or cursor != count:
            raise ReplayCacheError("Replay cache samples do not cover the array rows.")
        return index, arrays
    except (ReplayCacheError, OSError, EOFError, KeyError, TypeError, ValueError, OverflowError) as error:
        _close_arrays(arrays)
        if isinstance(error, ReplayCacheError):
            raise
        raise ReplayCacheError(f"Invalid replay cache: {path}") from error


def _write_entry(staging: Path, build: Path, manifest: DatasetManifest, signatures: dict, identity: dict) -> None:
    stream = CachedVisionDataset._stream_sequence(
        build, sequence_id=identity["sequence_id"], split=identity["split"], load_embeddings=identity["load_embeddings"]
    )
    dataset = stream._dataset
    count = sum(len(rows) for rows in dataset._instances_by_sample.values())
    shapes = {"geometry": (count, 5 if manifest.box_type == "obb" else 4), "scores": (count,), "class_ids": (count,)}
    if identity["load_embeddings"]:
        shapes["embeddings"] = (count, int(manifest.artifact(EMBEDDINGS_ARTIFACT).metadata["dim"]))
    arrays = {}
    samples = []
    cursor = 0
    try:
        for name, shape in shapes.items():
            arrays[name] = np.lib.format.open_memmap(
                staging / f"{name}.npy", mode="w+", dtype=_ARRAY_DTYPES[name], shape=shape
            )
        with closing(iter(stream)) as samples_iterator:
            for sample in samples_iterator:
                detections = sample.detections
                end = cursor + len(detections)
                arrays["geometry"][cursor:end] = detections.geometry.values.numpy()
                arrays["scores"][cursor:end] = detections.scores.numpy()
                arrays["class_ids"][cursor:end] = detections.class_ids.numpy()
                if identity["load_embeddings"]:
                    arrays["embeddings"][cursor:end] = detections.embeddings.numpy()
                samples.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "sequence_id": sample.sequence_id,
                        "frame_index": sample.frame_index,
                        "timestamp_s": sample.timestamp_s,
                        "image_size": list(sample.image_size),
                        "image_ref": sample.image_ref,
                        "instance_ids": list(detections.instance_ids or ()),
                        "detection_indices": [
                            row["detection_index"] for row in dataset._instances_by_sample.get(sample.sample_id, ())
                        ],
                        "offset": [cursor, end],
                    }
                )
                cursor = end
        if cursor != count:
            raise DatasetValidationError("Replay stream row count changed during preparation.")
        for array in arrays.values():
            array.flush()
    finally:
        _close_arrays(arrays)
    records = {}
    for name in shapes:
        file = staging / f"{name}.npy"
        with file.open("rb") as handle:
            os.fsync(handle.fileno())
        records[name] = {"sha256": sha256_file(file), "signature": _signature(file)}
    index = {
        "identity": identity,
        "build": str(build),
        "image_root": str(manifest.metadata.get("source_root_uri", build)),
        "source_signature": signatures,
        "count": count,
        "samples": samples,
        "arrays": records,
    }
    _write_json(staging / _INDEX, index)
    _write_json(
        staging / SUCCESS_FILENAME, {"identity": _digest(identity), "index_sha256": sha256_file(staging / _INDEX)}
    )


def prepare_replay_sequence(
    build: str | Path,
    *,
    sequence_id: str,
    split: str | None = None,
    load_embeddings: bool = True,
    cache_root: str | Path | None = None,
) -> Path:
    """Prepare detection arrays and optional embeddings for one exact sequence.

    The default directory is ``build.parent.parent / 'replay_cache'``. Content
    identities contain no execution device. A process lock serializes builders;
    incomplete or modified entries are rebuilt from validated source Parquet.
    """
    if not isinstance(sequence_id, str) or not sequence_id or sequence_id != sequence_id.strip():
        raise ValueError("sequence_id must be a non-empty canonical string.")
    if split is not None and (not isinstance(split, str) or not split or split != split.strip()):
        raise ValueError("split must be a non-empty canonical string or None.")
    if not isinstance(load_embeddings, bool):
        raise TypeError("load_embeddings must be a boolean.")
    build = Path(build).resolve()
    root = (build.parent.parent / "replay_cache") if cache_root is None else Path(cache_root).resolve()
    if root == build or root.is_relative_to(build):
        raise ValueError("Replay cache must be outside the immutable dataset build.")
    manifest, signatures = _source_context(build)
    identity = _identity(manifest, sequence_id=sequence_id, split=split, load_embeddings=load_embeddings)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _digest(identity)
    lock_dir = root / ".locks"
    lock_dir.mkdir(exist_ok=True)
    with FileLock(str(lock_dir / f"{path.name}.lock")):
        if path.exists():
            try:
                _, arrays = _load_entry(path, expected_identity=identity, expected_build=build)
                _close_arrays(arrays)
                return path
            except ReplayCacheError:
                pass
        _validate_source_once(build, manifest, signatures, root)
        for stale in root.glob(f".{path.name}.tmp-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=root))
        try:
            _write_entry(staging, build, manifest, signatures, identity)
            current_manifest, current_signatures = _source_context(build)
            if (
                current_signatures != signatures
                or _identity(current_manifest, sequence_id=sequence_id, split=split, load_embeddings=load_embeddings)
                != identity
            ):
                raise DatasetValidationError("Dataset source changed while preparing replay inputs.")
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            os.replace(staging, path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return path


class ReplaySequence:
    """Reusable read-only mappings that yield independent writable samples."""

    def __init__(self, path: Path, *, load_images: bool, load_masks: bool, load_embeddings: bool) -> None:
        self.cache_path = path
        self._index, self._arrays = _load_entry(path)
        self._closed = False
        try:
            self.build = Path(self._index["build"])
            self.manifest = DatasetManifest.load(self.build)
            self.load_images, self.load_masks, self.load_embeddings = load_images, load_masks, load_embeddings
            if load_embeddings and "embeddings" not in self._arrays:
                raise ReplayCacheError("This replay cache was prepared without embeddings.")
            if load_images and not self.manifest.publish.image_references:
                raise DatasetValidationError(
                    "Images were requested, but this dataset build does not publish image references."
                )
            if load_masks and not self.manifest.publish.masks:
                raise DatasetValidationError("Masks were requested, but this dataset build does not publish masks.")
            self.encoder_fingerprint = (
                str(self.manifest.artifact(EMBEDDINGS_ARTIFACT).metadata["encoder_fingerprint"])
                if load_embeddings
                else None
            )
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._index["samples"])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample["sample_id"] for sample in self._index["samples"])

    def close(self) -> None:
        """Release mappings without invalidating tensors already yielded."""
        _close_arrays(self._arrays)
        self._closed = True

    def validate(self) -> None:
        """Reject stale retained views without rescanning array payloads.

        Workers should evict and prepare the sequence again on
        :class:`ReplayCacheError`. The source and cache file mutation epochs,
        publication metadata and array headers are checked on every reuse.
        """
        if self._closed:
            raise ReplayCacheError("Replay sequence is closed.")
        index, arrays = _load_entry(self.cache_path)
        _close_arrays(arrays)
        if index != self._index:
            raise ReplayCacheError("Replay cache was republished after this view was opened.")

    def _iter_selected_batches(self, name: str, **selection: Any) -> Iterator[Any]:
        """Read keyed masks without reopening detection or sample Parquet."""
        yield from iter_selected_batches(self.build, self.manifest, name, **selection)

    def __iter__(self) -> Iterator[DatasetSample]:
        if self._closed:
            raise ReplayCacheError("Replay sequence is closed.")
        with ExitStack() as resources:
            yield from self._iter_samples(resources)

    def _iter_samples(self, resources: ExitStack) -> Iterator[DatasetSample]:
        """Own optional Parquet handles for exactly one frame iteration."""
        masks = None
        if self.load_masks:
            masks = _MaskPayloadStream(
                self,
                name=MASKS_ARTIFACT,
                sample_ids=set(self.sample_ids),
                expected_keys=frozenset(
                    (sample["sample_id"], instance_id)
                    for sample in self._index["samples"]
                    for instance_id in sample["instance_ids"]
                ),
                expected_schema=masks_schema(),
                artifact_label="Mask",
            )
            resources.callback(masks.close)
        geometry_class = OrientedBoxes if self.manifest.box_type == "obb" else Boxes
        for row in self._index["samples"]:
            if self._closed:
                raise ReplayCacheError("Replay sequence was closed during iteration.")
            start, end = row["offset"]
            values = {
                name: torch.from_numpy(array[start:end].copy())
                for name, array in self._arrays.items()
                if name != "embeddings" or self.load_embeddings
            }
            sample_masks = None
            if masks is not None:
                rows = masks.take(tuple((row["sample_id"], instance_id) for instance_id in row["instance_ids"]))
                height, width = row["image_size"]
                if any(
                    mask["height"] != height or mask["width"] != width or mask["codec"] != MASK_CODEC for mask in rows
                ):
                    raise DatasetValidationError(
                        "Mask rows must use the selected sample dimensions and canonical codec."
                    )
                sample_masks = MaskBatch(unpack_mask_batch((mask["data"] for mask in rows), height, width))
            detections = Detections(
                geometry=geometry_class(values["geometry"]),
                scores=values["scores"],
                class_ids=values["class_ids"],
                sample_id=row["sample_id"],
                instance_ids=tuple(row["instance_ids"]),
                masks=sample_masks,
                embeddings=values.get("embeddings"),
            )
            frame = None
            if self.load_images:
                if not row["image_ref"]:
                    raise DatasetValidationError(f"Sample {row['sample_id']!r} does not publish an image reference.")
                image = read_rgb_chw_uint8(row["image_ref"], self._index["image_root"])
                if tuple(image.shape) != (3, *row["image_size"]):
                    raise DatasetValidationError(
                        f"Decoded image for sample {row['sample_id']!r} does not match its dimensions."
                    )
                frame = Frame(
                    image=image,
                    sample_id=row["sample_id"],
                    sequence_id=row["sequence_id"],
                    frame_index=row["frame_index"],
                    timestamp_s=row["timestamp_s"],
                    source_uri=row["image_ref"],
                )
            yield DatasetSample(
                sample_id=row["sample_id"],
                split=row["split"],
                sequence_id=row["sequence_id"],
                frame_index=row["frame_index"],
                timestamp_s=row["timestamp_s"],
                image_size=tuple(row["image_size"]),
                image_ref=row["image_ref"],
                frame=frame,
                detections=detections,
            )
        if masks is not None:
            masks.finish()


def open_replay_sequence(
    cache_path: str | Path, *, load_images: bool = False, load_masks: bool = False, load_embeddings: bool = True
) -> ReplaySequence:
    """Open a prepared cache; call ``close()`` when evicting a retained view."""
    for name, value in (("load_images", load_images), ("load_masks", load_masks), ("load_embeddings", load_embeddings)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    path = Path(cache_path).absolute()
    lock_dir = path.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / f"{path.name}.lock")):
        return ReplaySequence(path, load_images=load_images, load_masks=load_masks, load_embeddings=load_embeddings)
