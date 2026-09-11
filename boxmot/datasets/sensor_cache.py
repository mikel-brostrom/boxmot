"""Disposable, immutable replay inputs for configured sensor modalities.

Preparation decodes each source frame once and publishes compact array files
atomically. Workers reopen those arrays by path; yielded tensors, RGB frames,
annotation arrays and RLE objects always belong to the caller. Filtering and
fusion remain the consumer's responsibility, so tracker trials share inputs
without sharing mutable tracking state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch
from filelock import FileLock

from boxmot.datasets.inputs import DatasetInputs, SequenceInputs
from boxmot.datasets.readers.boxes3d import (
    KittiObjectLabels,
    TrackingLabels3D,
    read_kitti_object_labels,
    read_kitti_tracking_labels,
)
from boxmot.datasets.readers.images import read_rgb_chw_uint8
from boxmot.datasets.readers.masks import read_instance_png
from boxmot.datasets.sequence import MultimodalSequence, SensorFrame, _class_ids, _instance_options, _single_path
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, Frame, MaskBatch

_SCHEMA = "boxmot.sensor-replay-cache/v3"


class SensorReplayCacheError(ValueError):
    """A derived sensor entry is invalid and must be prepared again."""


def _json_bytes(value: Any) -> bytes:
    """Produce deterministic metadata without executable serialization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _signature(path: Path) -> list[int]:
    """Bind a file's content to its identity and mutation epoch."""
    if not path.is_file():
        raise SensorReplayCacheError(f"Sensor cache input is missing or is not a file: {path}")
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _source_state(sequence: SequenceInputs) -> dict[str, list[int]]:
    """Detect changed, missing or newly added files before reusing a sequence."""
    sources: dict[str, list[int]] = {}
    for modality in sequence.modalities.values():
        for declared in modality.paths:
            path = Path(declared).resolve()
            if path.is_dir():
                files = sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
                )
                # Directory epochs also record empty prediction directories.
                stat = path.stat()
                sources[str(path)] = [stat.st_dev, stat.st_ino, -1, stat.st_mtime_ns, stat.st_ctime_ns]
                sources.update((str(candidate.resolve()), _signature(candidate)) for candidate in files)
            else:
                sources[str(path)] = _signature(path)
    return sources


def _context(dataset: DatasetInputs, sequence: SequenceInputs, load_images: bool) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "dataset_id": dataset.id,
        "root": str(dataset.root.resolve()),
        "split": dataset.split,
        "sequence_id": sequence.sequence_id,
        "classes": dataset.classes,
        "fps": dataset.fps,
        "load_images": load_images,
        "modalities": {
            role: {
                "format": modality.format,
                "paths": [str(path.resolve()) for path in modality.paths],
                "options": modality.options,
            }
            for role, modality in sequence.modalities.items()
        },
    }


def _source_files(sequence: SequenceInputs, digests: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Freeze directory membership, including logical names of linked inputs."""
    result: dict[str, list[tuple[str, str]]] = {}
    for modality in sequence.modalities.values():
        for declared in modality.paths:
            directory = Path(declared).resolve()
            if directory.is_dir():
                result[str(directory)] = [
                    (path.name, digests[str(path.resolve())])
                    for path in sorted(directory.iterdir())
                    if path.is_file() and not path.name.startswith(".")
                ]
    return result


def _loads_detection_masks(context: dict[str, Any]) -> bool:
    """Preserve an explicitly disabled mask channel in the cached contract."""
    selected = context["modalities"].get("detections_2d", {}).get("options", {}).get("load_masks", True)
    if type(selected) is not bool:
        raise SensorReplayCacheError("Sensor replay mask selection must be boolean.")
    return selected


def _write_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _read_index(path: Path) -> dict[str, Any]:
    try:
        index = json.loads((path / "index.json").read_bytes())
        success = json.loads((path / "_SUCCESS").read_bytes())
        if not isinstance(index, dict) or index.get("schema") != _SCHEMA:
            raise SensorReplayCacheError(f"Invalid sensor replay schema: {path}")
        if success != {"schema": _SCHEMA, "index_sha256": _digest(index)}:
            raise SensorReplayCacheError(f"Invalid sensor replay publication marker: {path}")
        identity = {"context": index["context"], "sources": index["source_signatures"]}
        if path.name != _digest(identity):
            raise SensorReplayCacheError(f"Sensor replay directory does not match its input identity: {path}")
        _validate_index(index)
        for name, record in index["arrays"].items():
            if Path(name).name != name or not name.endswith(".bin"):
                raise SensorReplayCacheError(f"Invalid sensor replay array name: {name}")
            payload = path / name
            signature = _signature(payload)
            expected_size = math.prod(record["shape"]) * np.dtype(record["dtype"]).itemsize
            if signature[2] != expected_size or signature != record["signature"]:
                raise SensorReplayCacheError(f"Sensor replay array changed: {payload}")
        return index
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        if isinstance(error, SensorReplayCacheError):
            raise
        raise SensorReplayCacheError(f"Invalid sensor replay entry: {path}") from error


def _validate_index(index: dict[str, Any]) -> None:
    """Require exact array schemas and complete, nonoverlapping frame ranges."""
    context, arrays, frames = index["context"], index["arrays"], index["frames"]
    if not isinstance(index["source_files"], dict):
        raise SensorReplayCacheError("Sensor replay source directory inventory is invalid.")
    for records in index["source_files"].values():
        names = []
        for name, digest in records:
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise SensorReplayCacheError("Sensor replay source directory entry is invalid.")
            names.append(name)
        if names != sorted(set(names)):
            raise SensorReplayCacheError("Sensor replay source directory entries must be unique and sorted.")
    height, width = index["image_size"]
    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise SensorReplayCacheError("Sensor replay image dimensions must be positive integers.")
    if not isinstance(frames, list) or not frames:
        raise SensorReplayCacheError("Sensor replay requires a nonempty image timeline.")
    if type(context["load_images"]) is not bool:
        raise SensorReplayCacheError("Sensor replay RGB selection must be boolean.")
    if type(context["fps"]) not in {int, float} or not math.isfinite(context["fps"]) or context["fps"] <= 0:
        raise SensorReplayCacheError("Sensor replay fps must be a positive finite number.")

    def count(name: str) -> int:
        value = arrays[f"{name}.bin"]["shape"][0]
        if type(value) is not int or value < 0:
            raise SensorReplayCacheError(f"Invalid sensor replay row count: {name}")
        return value

    count2d, count3d = count("boxes2d"), count("boxes3d")
    expected = {
        "boxes2d": (np.float32, (count2d, 4)),
        "scores2d": (np.float32, (count2d,)),
        "classes2d": (np.int64, (count2d,)),
        "boxes3d": (np.float32, (count3d, 7)),
        "scores3d": (np.float32, (count3d,)),
        "classes3d": (np.int64, (count3d,)),
    }
    if _loads_detection_masks(context):
        expected["masks2d"] = (np.uint8, (count2d, (height * width + 7) // 8))
    modalities = context["modalities"]
    for role, shape in (("calibration", (len(frames), 3, 4)), ("poses", (len(frames), 4, 4))):
        if role in modalities:
            expected["projection" if role == "calibration" else role] = (np.float32, shape)
    if context["load_images"]:
        expected["images"] = (np.uint8, (len(frames), 3, height, width))
    if "ground_truth" in modalities:
        expected["ground_truth"] = (np.uint8, (count("ground_truth"),))
    if "ground_truth_3d" in modalities:
        rows = count("gt3d_boxes")
        expected["gt3d_boxes"] = (np.float64, (rows, 7))
        for field in ("frame_indices", "track_ids", "class_ids"):
            expected[f"gt3d_{field}"] = (np.int64, (rows,))
        expected["gt3d_source_rows"] = (np.uint8, (count("gt3d_source_rows"),))
        metadata = index["ground_truth_3d"]
        if type(metadata["row_count"]) is not int or metadata["row_count"] < rows:
            raise SensorReplayCacheError("Sensor replay 3D annotation count is invalid.")
        if len(metadata["source_sha256"]) != 64:
            raise SensorReplayCacheError("Sensor replay 3D annotation provenance is invalid.")
    elif index["ground_truth_3d"] is not None:
        raise SensorReplayCacheError("Sensor replay contains undeclared 3D annotations.")
    if "ground_truth_objects" in modalities:
        expected["gt_object_rows"] = (np.uint8, (count("gt_object_rows"),))
        metadata = index["ground_truth_objects"]
        if metadata["frame_count"] != len(frames) or type(metadata["frame_count"]) is not int:
            raise SensorReplayCacheError("Sensor replay object annotation timeline is invalid.")
        if type(metadata["row_count"]) is not int or metadata["row_count"] < 0:
            raise SensorReplayCacheError("Sensor replay object annotation count is invalid.")
        if not isinstance(metadata["source_sha256"], str) or len(metadata["source_sha256"]) != 64:
            raise SensorReplayCacheError("Sensor replay object annotation provenance is invalid.")
    elif index["ground_truth_objects"] is not None:
        raise SensorReplayCacheError("Sensor replay contains undeclared object annotations.")
    if set(arrays) != {f"{name}.bin" for name in expected}:
        raise SensorReplayCacheError("Sensor replay arrays differ from its declared modalities.")
    for name, (dtype, shape) in expected.items():
        record = arrays[f"{name}.bin"]
        if np.dtype(record["dtype"]) != np.dtype(dtype) or tuple(record["shape"]) != shape:
            raise SensorReplayCacheError(f"Invalid sensor replay array dtype or shape: {name}")
        if any(type(dimension) is not int for dimension in record["shape"]):
            raise SensorReplayCacheError(f"Invalid sensor replay array dimensions: {name}")
        if len(record["sha256"]) != 64:
            raise SensorReplayCacheError(f"Invalid sensor replay payload digest: {name}")
    totals = {"detections_2d": count2d, "detections_3d": count3d}
    if "ground_truth" in modalities:
        totals["ground_truth"] = count("ground_truth")
    offsets = dict.fromkeys(totals, 0)
    for ordinal, frame in enumerate(frames):
        if type(frame["frame_index"]) is not int or frame["frame_index"] != ordinal:
            raise SensorReplayCacheError("Sensor replay frames must follow the zero-based image timeline.")
        if not isinstance(frame["image_path"], str) or not Path(frame["image_path"]).is_absolute():
            raise SensorReplayCacheError("Sensor replay image paths must be absolute.")
        for role in totals:
            start, end = frame[role]
            if (
                type(start) is not int
                or type(end) is not int
                or start != offsets[role]
                or not start <= end <= totals[role]
            ):
                raise SensorReplayCacheError(f"Sensor replay frame offsets are invalid: {role}")
            offsets[role] = end
    if offsets != totals:
        raise SensorReplayCacheError("Sensor replay frame offsets do not cover its payload arrays.")


def _encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Store scoring masks as compressed COCO runs without retaining dense GT."""
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def _write_entry(
    path: Path,
    dataset: DatasetInputs,
    sequence_inputs: SequenceInputs,
    context: dict[str, Any],
    sources: dict[str, list[int]],
    progress: Callable[[str], None] | None,
) -> None:
    """Stream observations and annotations into aligned, bounded-memory arrays."""
    sequence = MultimodalSequence(sequence_inputs, classes=dataset.classes, fps=dataset.fps, split=dataset.split)
    height, width = sequence.image_size
    packed_width = (height * width + 7) // 8
    formats: dict[str, tuple[Any, tuple[int, ...]]] = {
        "boxes2d": (np.float32, (4,)),
        "scores2d": (np.float32, ()),
        "classes2d": (np.int64, ()),
        "boxes3d": (np.float32, (7,)),
        "scores3d": (np.float32, ()),
        "classes3d": (np.int64, ()),
    }
    if _loads_detection_masks(context):
        formats["masks2d"] = (np.uint8, (packed_width,))
    modalities = sequence_inputs.modalities
    if "calibration" in modalities:
        formats["projection"] = (np.float32, (3, 4))
    if "poses" in modalities:
        formats["poses"] = (np.float32, (4, 4))
    if context["load_images"]:
        formats["images"] = (np.uint8, (3, height, width))
    annotations = modalities.get("ground_truth")
    annotation_root = None if annotations is None else _single_path(annotations, "instance-png", "ground_truth")
    annotation_options = {} if annotations is None else _instance_options(annotations.options)
    if annotations is not None:
        formats["ground_truth"] = (np.uint8, ())
    counts = dict.fromkeys(formats, 0)
    frames: list[dict[str, Any]] = []
    with ExitStack() as resources:
        handles = {name: resources.enter_context((path / f"{name}.bin").open("wb")) for name in formats}

        def append(name: str, values: np.ndarray) -> None:
            values = np.asarray(values, dtype=formats[name][0])
            values.tofile(handles[name])
            counts[name] += len(values)

        for frame in sequence:
            detection_start, spatial_start = counts["boxes2d"], counts["boxes3d"]
            for suffix, detections in (("2d", frame.detections), ("3d", frame.detections_3d)):
                append(f"boxes{suffix}", detections.geometry.values.numpy())
                append(f"scores{suffix}", detections.scores.numpy())
                append(f"classes{suffix}", detections.class_ids.numpy())
            if "masks2d" in formats:
                if frame.detections.masks is None:
                    raise SensorReplayCacheError("Sensor replay is missing its declared detection masks.")
                masks = frame.detections.masks.values.numpy()
                append("masks2d", np.packbits(masks.reshape(len(masks), height * width), axis=1, bitorder="little"))
            elif frame.detections.masks is not None:
                raise SensorReplayCacheError("Sensor replay received masks when their reader channel is disabled.")
            if frame.camera is not None:
                append("projection", frame.camera.projection.numpy()[None])
                if frame.camera.camera_to_world is not None:
                    append("poses", frame.camera.camera_to_world.numpy()[None])
            image_path = sequence.frame_paths[frame.frame_index]
            if context["load_images"]:
                image = read_rgb_chw_uint8(image_path.as_uri(), dataset.root)
                if tuple(image.shape) != (3, height, width):
                    raise ValueError(f"Sensor image dimensions changed: {image_path}")
                append("images", image.numpy()[None])
            row = {
                "frame_index": frame.frame_index,
                "image_path": str(image_path),
                "detections_2d": [detection_start, counts["boxes2d"]],
                "detections_3d": [spatial_start, counts["boxes3d"]],
            }
            if annotation_root is not None:
                canonical_frame = Frame(
                    image=torch.empty((3, height, width), dtype=torch.uint8),
                    sample_id=frame.detections.sample_id,
                )
                tracks, ignore = read_instance_png(
                    annotation_root / image_path.with_suffix(".png").name,
                    canonical_frame,
                    class_ids=_class_ids(dataset.classes),
                    ignore_class_ids=_class_ids(dataset.classes, evaluation="ignore"),
                    **annotation_options,
                )
                payload = _json_bytes(
                    {
                        "ids": tracks.track_ids.tolist(),
                        "classes": tracks.class_ids.tolist(),
                        "masks": [_encode_mask(mask) for mask in tracks.masks.values.numpy()],
                        "ignore": _encode_mask(ignore.numpy()) if bool(ignore.any()) else None,
                    }
                )
                start = counts["ground_truth"]
                append("ground_truth", np.frombuffer(payload, dtype=np.uint8))
                row["ground_truth"] = [start, counts["ground_truth"]]
            frames.append(row)
            if progress is not None and (frame.frame_index + 1) % 100 == 0:
                progress(f"Input cache: {sequence.sequence_id} ({frame.frame_index + 1}/{len(sequence)} frames)")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())

    def write_annotation_array(name: str, values: np.ndarray) -> None:
        """Publish numeric or UTF-8 annotation payloads without object serialization."""
        values = np.ascontiguousarray(values)
        formats[name] = (values.dtype, values.shape[1:])
        counts[name] = len(values)
        with (path / f"{name}.bin").open("wb") as handle:
            values.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())

    ground_truth_3d = None
    if "ground_truth_3d" in modalities:
        declaration = modalities["ground_truth_3d"]
        annotations_3d = read_kitti_tracking_labels(
            _single_path(declaration, "kitti-tracking-labels", "ground_truth_3d"),
            frame_count=len(sequence),
            classes=dataset.classes,
            options=declaration.options,
        )
        for name in ("frame_indices", "track_ids", "class_ids", "boxes"):
            write_annotation_array(f"gt3d_{name}", getattr(annotations_3d, name))
        write_annotation_array(
            "gt3d_source_rows", np.frombuffer(_json_bytes(annotations_3d.source_rows), dtype=np.uint8)
        )
        ground_truth_3d = {"row_count": annotations_3d.row_count, "source_sha256": annotations_3d.source_sha256}
    ground_truth_objects = None
    if "ground_truth_objects" in modalities:
        declaration = modalities["ground_truth_objects"]
        annotations_objects = read_kitti_object_labels(
            _single_path(declaration, "kitti-object-labels", "ground_truth_objects"), frame_count=len(sequence)
        )
        write_annotation_array(
            "gt_object_rows", np.frombuffer(_json_bytes(annotations_objects.frame_rows), dtype=np.uint8)
        )
        ground_truth_objects = {
            "frame_count": len(annotations_objects.frame_rows),
            "row_count": sum(map(len, annotations_objects.frame_rows)),
            "source_sha256": annotations_objects.source_sha256,
        }
    if progress is not None:
        progress(f"Input cache: prepared {sequence.sequence_id} ({len(sequence)} frames)")
    source_digests = {name: _file_digest(Path(name)) for name, signature in sources.items() if signature[2] >= 0}
    index = {
        "schema": _SCHEMA,
        "context": context,
        "source_signatures": sources,
        "source_sha256": source_digests,
        "source_files": _source_files(sequence_inputs, source_digests),
        "image_size": [height, width],
        "missing_3d_frames": sequence.missing_3d_frames,
        "frames": frames,
        "ground_truth_3d": ground_truth_3d,
        "ground_truth_objects": ground_truth_objects,
        "arrays": {
            f"{name}.bin": {
                "dtype": np.dtype(dtype).str,
                "shape": [counts[name], *tail],
                "signature": _signature(path / f"{name}.bin"),
                "sha256": _file_digest(path / f"{name}.bin"),
            }
            for name, (dtype, tail) in formats.items()
        },
    }
    if _source_state(sequence_inputs) != sources:
        raise SensorReplayCacheError("Sensor inputs changed while preparing their replay cache. Retry preparation.")
    _write_json(path / "index.json", index)
    _write_json(path / "_SUCCESS", {"schema": _SCHEMA, "index_sha256": _digest(index)})


def prepare_sensor_sequence(
    dataset: DatasetInputs,
    sequence_id: str,
    *,
    cache_root: Path | None = None,
    load_images: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Prepare every declared modality once, reusing unchanged inputs on disk.

    Image headers and their timeline are always indexed. RGB pixels are cached
    only when requested. Input signatures include directory membership and all
    configured source files, so removed, added and edited payloads invalidate the
    entry before a study starts. Workers opening a prepared path do not revisit
    the raw dataset. An open view is an immutable study snapshot; call prepare
    again before a later study to observe source edits.
    """
    if type(load_images) is not bool:
        raise TypeError("load_images must be a boolean.")
    selected = [sequence for sequence in dataset.sequences if sequence.sequence_id == sequence_id]
    if len(selected) != 1:
        raise ValueError(f"Sensor dataset must contain exactly one selected sequence {sequence_id!r}.")
    sequence = selected[0]
    context = _context(dataset, sequence, load_images)
    root = Path(cache_root) if cache_root is not None else dataset.root / ".boxmot" / "replay_cache"
    root = root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock_root = root / ".locks"
    lock_root.mkdir(exist_ok=True)
    # Creating the cache must precede source directory epochs when a modality
    # uses the dataset root itself. Hidden cache contents are excluded.
    sources = _source_state(sequence)
    path = root / _digest({"context": context, "sources": sources})
    with FileLock(str(lock_root / f"{path.name}.lock")):
        if path.is_dir():
            try:
                index = _read_index(path)
                if index["context"] == context and index["source_signatures"] == sources:
                    if _source_state(sequence) != sources:
                        raise SensorReplayCacheError("Sensor inputs changed while opening their replay cache.")
                    return path
            except SensorReplayCacheError:
                pass
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=root))
        try:
            _write_entry(staging, dataset, sequence, context, sources, progress)
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            os.replace(staging, path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return path


class SensorReplaySequence(Sequence[SensorFrame]):
    """Read-only mapped observations yielding independent writable frames."""

    def __init__(self, path: Path) -> None:
        self.cache_path = Path(path).absolute()
        self._index = _read_index(self.cache_path)
        self._arrays: dict[str, np.ndarray] = {}
        self._closed = False
        try:
            for name, record in self._index["arrays"].items():
                shape, dtype = tuple(record["shape"]), np.dtype(record["dtype"])
                self._arrays[Path(name).stem] = (
                    np.memmap(self.cache_path / name, mode="r", dtype=dtype, shape=shape)
                    if all(shape)
                    else np.empty(shape, dtype=dtype)
                )
            context = self._index["context"]
            self.sequence_id, self.split, self.fps = context["sequence_id"], context["split"], context["fps"]
            self.image_size = tuple(self._index["image_size"])
            self.frame_paths = tuple(Path(row["image_path"]) for row in self._index["frames"])
            self.missing_3d_frames = {path: tuple(frames) for path, frames in self._index["missing_3d_frames"].items()}
            self.load_images = context["load_images"]
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._index["frames"])

    def _row(self, index: int) -> tuple[int, dict[str, Any]]:
        if self._closed:
            raise SensorReplayCacheError("Sensor replay sequence is closed.")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Sensor replay indices must be integers or slices.")
        row = self._index["frames"][index]
        return int(row["frame_index"]), row

    @overload
    def __getitem__(self, index: int) -> SensorFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SensorFrame, ...]: ...

    def __getitem__(self, index: int | slice) -> SensorFrame | tuple[SensorFrame, ...]:
        """Return fresh canonical observations without re-reading source files."""
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        frame_index, row = self._row(index)
        sample_id = f"{self.split}:{self.sequence_id}:{frame_index}"
        left, right = row["detections_2d"]

        def tensor(name: str, begin: int, stop: int) -> torch.Tensor:
            return torch.from_numpy(self._arrays[name][begin:stop].copy())

        masks = None
        if "masks2d" in self._arrays:
            values = np.unpackbits(
                self._arrays["masks2d"][left:right], axis=1, count=int(np.prod(self.image_size)), bitorder="little"
            ).reshape(right - left, *self.image_size)
            masks = MaskBatch(torch.from_numpy(values.astype(np.bool_)))
        detections = Detections(
            Boxes(tensor("boxes2d", left, right)),
            tensor("scores2d", left, right),
            tensor("classes2d", left, right),
            sample_id,
            masks=masks,
        )
        return SensorFrame(
            frame_index,
            self.image_size,
            detections,
            self.read_spatial(index),
            self.read_camera(index),
            frame_index / self.fps,
        )

    def read_spatial(self, index: int) -> Detections3D:
        """Read just 3D predictions, without unpacking unrelated 2D masks."""
        frame_index, row = self._row(index)
        start, end = row["detections_3d"]
        return Detections3D(
            Boxes3D(torch.from_numpy(self._arrays["boxes3d"][start:end].copy())),
            torch.from_numpy(self._arrays["scores3d"][start:end].copy()),
            torch.from_numpy(self._arrays["classes3d"][start:end].copy()),
            f"{self.split}:{self.sequence_id}:{frame_index}",
        )

    def read_camera(self, index: int) -> CameraModel | None:
        """Read calibration and pose without loading predictions or masks."""
        frame_index, _ = self._row(index)
        camera = None
        if "projection" in self._arrays:
            camera = CameraModel(
                torch.from_numpy(self._arrays["projection"][frame_index].copy()),
                self.image_size,
                None if "poses" not in self._arrays else torch.from_numpy(self._arrays["poses"][frame_index].copy()),
            )
        return camera

    def read_image(self, index: int) -> torch.Tensor:
        """Read independent RGB pixels cached for image-consuming workflows."""
        frame_index, _ = self._row(index)
        if "images" not in self._arrays:
            raise SensorReplayCacheError("Sensor replay was prepared without RGB images; use load_images=True.")
        return torch.from_numpy(self._arrays["images"][frame_index].copy())

    def ground_truth(self, frame_index: int) -> tuple[np.ndarray, np.ndarray, list[dict], dict | None] | None:
        """Return independent IDs, classes, COCO masks and an optional ignore RLE."""
        _, row = self._row(frame_index)
        if "ground_truth" not in row:
            return None
        start, end = row["ground_truth"]
        record = json.loads(self._arrays["ground_truth"][start:end].tobytes())
        for mask in [*record["masks"], record["ignore"]]:
            if mask is not None:
                mask["counts"] = mask["counts"].encode("ascii")
        return (
            np.asarray(record["ids"], dtype=np.int64),
            np.asarray(record["classes"], dtype=np.int64),
            record["masks"],
            record["ignore"],
        )

    def ground_truth_3d(self) -> TrackingLabels3D | None:
        """Return cached identity-bearing 3D annotations and source provenance."""
        if self._closed:
            raise SensorReplayCacheError("Sensor replay sequence is closed.")
        metadata = self._index["ground_truth_3d"]
        if metadata is None:
            return None
        source_rows = json.loads(self._arrays["gt3d_source_rows"].tobytes())
        if (
            not isinstance(source_rows, list)
            or len(source_rows) != metadata["row_count"]
            or any(not isinstance(row, str) for row in source_rows)
        ):
            raise SensorReplayCacheError("Sensor replay tracking annotation rows are invalid.")
        return TrackingLabels3D(
            **{
                name: self._arrays[f"gt3d_{name}"].copy()
                for name in ("frame_indices", "track_ids", "class_ids", "boxes")
            },
            source_rows=tuple(source_rows),
            **metadata,
        )

    def ground_truth_objects(self) -> KittiObjectLabels | None:
        """Return immutable native object labels aligned to every cached frame."""
        if self._closed:
            raise SensorReplayCacheError("Sensor replay sequence is closed.")
        metadata = self._index["ground_truth_objects"]
        if metadata is None:
            return None
        frames = json.loads(self._arrays["gt_object_rows"].tobytes())
        if (
            not isinstance(frames, list)
            or len(frames) != metadata["frame_count"]
            or any(not isinstance(rows, list) or any(not isinstance(row, str) for row in rows) for rows in frames)
            or sum(map(len, frames)) != metadata["row_count"]
        ):
            raise SensorReplayCacheError("Sensor replay object annotation rows are invalid.")
        return KittiObjectLabels(
            frame_rows=tuple(tuple(rows) for rows in frames), source_sha256=metadata["source_sha256"]
        )

    def source_sha256(self, path: Path) -> str:
        """Return the verified digest of a raw input without reopening its file."""
        return self._index["source_sha256"][str(Path(path).resolve())]

    def source_files(self, directory: Path) -> tuple[tuple[str, str], ...]:
        """Return source filenames and hashes from the immutable input snapshot."""
        return tuple(tuple(record) for record in self._index["source_files"][str(Path(directory).resolve())])

    def validate(self, *, dataset: DatasetInputs | None = None) -> None:
        """Check publication and mapped-file epochs without rescanning raw inputs."""
        if self._closed:
            raise SensorReplayCacheError("Sensor replay sequence is closed.")
        if _read_index(self.cache_path) != self._index:
            raise SensorReplayCacheError("Sensor replay entry was republished after this view was opened.")
        if dataset is not None:
            selected = [sequence for sequence in dataset.sequences if sequence.sequence_id == self.sequence_id]
            if len(selected) != 1 or _context(dataset, selected[0], self.load_images) != self._index["context"]:
                raise SensorReplayCacheError("Sensor replay inputs do not match the selected dataset configuration.")

    def close(self) -> None:
        """Release mappings without invalidating any sample already returned."""
        for values in self._arrays.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays.clear()
        self._closed = True

    def __reduce__(self) -> tuple[Any, tuple[Path]]:
        """Send only the immutable cache path to process workers."""
        if self._closed:
            raise SensorReplayCacheError("Sensor replay sequence is closed.")
        return open_sensor_sequence, (self.cache_path,)


def open_sensor_sequence(path: str | Path) -> SensorReplaySequence:
    """Open a prepared entry; consumers own its lifetime and call ``close()``."""
    path = Path(path).expanduser().absolute()
    locks = path.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / f"{path.name}.lock")):
        return SensorReplaySequence(path)
