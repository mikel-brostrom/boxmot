from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from numpy.lib import format as npy_format

from boxmot.data.dataset import (
    _collect_seq_info,
    _list_sequence_frames,
    _sequence_img_dir,
    _sequence_name_from_img_dir,
)


def _read_image_cv2(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(str(path))
        # 8-channel MMOT multispectral: extract pseudo-RGB as BGR for cv2/YOLO.
        # Per the MMOT paper, RGB proxy = bands 5,3,2 (0-indexed: 4,2,1).
        # BGR order for YOLO: B=band2(idx1), G=band3(idx2), R=band5(idx4).
        if arr.ndim == 3 and arr.shape[2] == 8:
            arr = arr[:, :, [1, 2, 4]]
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _clear_device_cache(device: str) -> None:
    dev_lower = str(device).lower()
    if dev_lower.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _count_embedding_rows(path: Path) -> int:
    """Count rows in an embedding cache (.npy)."""
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape[0]
    except Exception:
        return 0


def _existing_cache_path(path: Path) -> Optional[Path]:
    """Return the path if it exists, otherwise None."""
    if path.exists():
        return path
    return None


def _existing_embedding_cache_path(path: Path) -> Optional[Path]:
    """Return *path* if it exists, else None."""
    if path.exists():
        return path
    return None


def _load_embedding_cache_array(path: Path) -> np.ndarray:
    """Load an embedding cache .npy file and ensure 2-D shape."""
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _load_numeric_cache_array(path: Path) -> np.ndarray:
    """Load a numeric .npy cache file."""
    return np.load(path)


# Tokens accepted by ``BOXMOT_REID_BACKEND`` (matches the C++ runtime selector).
_OPENCV_RUNTIME_TOKENS = {"opencv", "cv", "dnn", "opencv_dnn"}
REID_CROP_SCHEMA_VERSION = 2


def _artifact_signature(path: Path) -> tuple[tuple[str, int, int, int], ...] | None:
    """Return a cacheable signature for a file or directory artifact."""
    try:
        if path.is_file():
            stat = path.stat()
            return ((path.name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),)
        if path.is_dir():
            entries = []
            for artifact_file in sorted(item for item in path.rglob("*") if item.is_file()):
                stat = artifact_file.stat()
                entries.append(
                    (
                        artifact_file.relative_to(path).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    )
                )
            return tuple(entries)
    except OSError:
        return None
    return None


@lru_cache(maxsize=64)
def _artifact_sha256(
    resolved_path: str,
    signature: tuple[tuple[str, int, int, int], ...],
) -> str:
    """Hash artifact bytes, caching only while its complete stat signature matches."""
    del signature  # The signature is intentionally part of the cache key.
    path = Path(resolved_path)
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for artifact_file in files:
        relative = artifact_file.name if path.is_file() else artifact_file.relative_to(path).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        with artifact_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reid_artifact_token(path: Path) -> str:
    """Return a short content token for an existing ReID artifact."""
    signature = _artifact_signature(path)
    if signature is None:
        return ""
    try:
        digest = _artifact_sha256(str(path.resolve()), signature)
    except OSError:
        return ""
    return f"_w{digest[:16]}"


def _onnx_runtime_token() -> str:
    """Resolve the ONNX runtime token (``ort`` / ``opencv``) from the env."""
    raw = os.environ.get("BOXMOT_REID_BACKEND", "").strip().lower()
    return "opencv" if raw in _OPENCV_RUNTIME_TOKENS else "ort"


def _resolve_reid_runtime(suffix: str, *, tracker_backend: str | None) -> str:
    """Map a ReID weights suffix + tracker backend to its runtime token."""
    suffix = (suffix or "").lower()
    is_cpp = bool(tracker_backend) and str(tracker_backend).lower() == "cpp"

    if is_cpp:
        return _onnx_runtime_token()

    if suffix == ".onnx":
        return _onnx_runtime_token()
    if suffix == ".pt":
        return "pytorch"
    if suffix == ".engine":
        return "tensorrt"
    if suffix == ".xml":
        return "openvino"
    if suffix == ".tflite":
        return "tflite"
    if suffix in {"", "."}:
        return "pytorch"
    return suffix.lstrip(".") or "default"


def reid_cache_key(
    reid_model: str | os.PathLike,
    *,
    tracker_backend: str | None = None,
) -> str:
    """Return the producer/model portion of a ReID embedding cache path.

    The producer is the effective implementation that generated the vectors,
    not the tracker algorithm consuming them. Keeping it as the first path
    component makes Python and native cache provenance immediately visible::

        python/lmbn_n_duke-pt-pytorch-w0123456789abcdef
        cpp/lmbn_n_duke-onnx-ort-w0123456789abcdef

    Crop semantics are deliberately kept in :func:`reid_preprocess_cache_key`
    so model identity and preprocessing identity are separate path levels.
    """
    p = Path(reid_model)
    stem = p.stem or p.name or str(reid_model)
    model_format = p.suffix.lower().lstrip(".") or "model"
    runtime = _resolve_reid_runtime(p.suffix, tracker_backend=tracker_backend)
    producer = "cpp" if (tracker_backend and str(tracker_backend).lower() == "cpp") else "python"
    artifact = _reid_artifact_token(p).replace("_w", "-w", 1)
    return f"{producer}/{stem}-{model_format}-{runtime}{artifact}"


def reid_preprocess_cache_key(reid_preprocess: str | None = None) -> str:
    """Return the versioned preprocessing portion of an embedding cache path."""
    if not reid_preprocess:
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS

        reid_preprocess = DEFAULT_PREPROCESS
    name = str(reid_preprocess).strip().replace("/", "-").replace("\\", "-")
    return f"{name}-cropv{REID_CROP_SCHEMA_VERSION}"


def _flat_reid_cache_key(
    reid_model: str | os.PathLike,
    *,
    tracker_backend: str | None,
    include_artifact: bool,
    include_crop_schema: bool,
) -> str:
    """Return a cache key emitted by the previous flattened layout."""
    p = Path(reid_model)
    name = p.name or str(reid_model)
    base = name.replace(".", "_")
    runtime = _resolve_reid_runtime(p.suffix, tracker_backend=tracker_backend)
    stack = "cpp" if (tracker_backend and str(tracker_backend).lower() == "cpp") else "py"
    artifact = _reid_artifact_token(p) if include_artifact else ""
    crop_schema = f"_cropv{REID_CROP_SCHEMA_VERSION}" if include_crop_schema else ""
    return f"{base}_{runtime}_{stack}{artifact}{crop_schema}"


def reid_cache_dir_candidates(
    embeddings_root: str | os.PathLike,
    reid_model: str | os.PathLike,
    *,
    reid_preprocess: str | None = None,
    tracker_backend: str | None = None,
    allow_legacy: bool = False,
) -> tuple[Path, ...]:
    """Return canonical and compatible embedding directories in priority order.

    Fingerprinted, crop-versioned layouts are always considered compatible.
    Unversioned layouts are returned only when the caller has independently
    established trusted artifact provenance via ``allow_legacy``.
    """
    root = Path(embeddings_root)
    preprocess = str(reid_preprocess or "").strip()
    if not preprocess:
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS

        preprocess = DEFAULT_PREPROCESS

    candidates = [
        root / reid_cache_key(reid_model, tracker_backend=tracker_backend) / reid_preprocess_cache_key(preprocess),
        root
        / _flat_reid_cache_key(
            reid_model,
            tracker_backend=tracker_backend,
            include_artifact=True,
            include_crop_schema=True,
        )
        / preprocess,
    ]

    if allow_legacy:
        candidates.append(
            root
            / _flat_reid_cache_key(
                reid_model,
                tracker_backend=tracker_backend,
                include_artifact=False,
                include_crop_schema=True,
            )
            / preprocess
        )
        candidates.append(
            root
            / _flat_reid_cache_key(
                reid_model,
                tracker_backend=tracker_backend,
                include_artifact=False,
                include_crop_schema=False,
            )
            / preprocess
        )
        if not tracker_backend or str(tracker_backend).lower() != "cpp":
            candidates.append(root / Path(reid_model).stem / preprocess)

    return tuple(dict.fromkeys(candidates))


def find_existing_reid_cache_file(
    embeddings_root: str | os.PathLike,
    reid_model: str | os.PathLike,
    sequence_name: str,
    *,
    reid_preprocess: str | None = None,
    tracker_backend: str | None = None,
    expected_rows: int | None = None,
    allow_legacy: bool = False,
) -> Path | None:
    """Find the highest-priority valid embedding file for one sequence.

    When ``expected_rows`` is supplied, incomplete or corrupt higher-priority
    files are skipped. This lets a complete trusted legacy file win over a
    partial canonical file left by an interrupted migration.
    """
    if _artifact_signature(Path(reid_model)) is None and not allow_legacy:
        return None

    filename = f"{Path(sequence_name).stem}.npy"
    for directory in reid_cache_dir_candidates(
        embeddings_root,
        reid_model,
        reid_preprocess=reid_preprocess,
        tracker_backend=tracker_backend,
        allow_legacy=allow_legacy,
    ):
        candidate = directory / filename
        if not candidate.is_file():
            continue
        try:
            array = np.load(candidate, mmap_mode="r")
        except Exception:  # noqa: BLE001 - a broken cache is a miss, not fatal
            continue
        if array.ndim != 2:
            continue
        if array.shape[0] > 0 and array.shape[1] == 0:
            continue
        if expected_rows is not None and int(array.shape[0]) != int(expected_rows):
            continue
        return candidate
    return None


class AppendableNpyWriter:
    """Append row chunks to a standard `.npy` file without buffering the full array."""

    def __init__(
        self,
        path: Path,
        *,
        dtype: np.dtype = np.float32,
        trailing_shape: Optional[tuple[int, ...]] = None,
        empty_trailing_shape: Optional[tuple[int, ...]] = None,
    ):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.trailing_shape = tuple(trailing_shape) if trailing_shape is not None else None
        self.empty_trailing_shape = (
            tuple(empty_trailing_shape) if empty_trailing_shape is not None else self.trailing_shape
        )
        self.rows = 0
        self._fp = None
        self._data_offset = None
        self._version = (2, 0)

        if self.path.exists():
            self._open_existing()
        elif self.trailing_shape is not None:
            self._initialize_file(self.trailing_shape)

    def _header_dict(self) -> dict:
        if self.trailing_shape is None:
            raise ValueError("Cannot build NPY header before trailing shape is known")
        return {
            "descr": npy_format.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": (int(self.rows), *self.trailing_shape),
        }

    def _sync_header(self) -> None:
        if self._fp is None:
            return

        self._fp.seek(0)
        if self._version == (1, 0):
            npy_format.write_array_header_1_0(self._fp, self._header_dict())
        else:
            npy_format.write_array_header_2_0(self._fp, self._header_dict())

        new_offset = self._fp.tell()
        if self._data_offset is not None and new_offset != self._data_offset:
            raise RuntimeError(
                f"NPY header resize changed data offset for {self.path}: "
                f"{self._data_offset} -> {new_offset}"
            )
        self._fp.flush()

    def _initialize_file(self, trailing_shape: tuple[int, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trailing_shape = tuple(trailing_shape)
        self._fp = open(self.path, "wb+")
        npy_format.write_array_header_2_0(self._fp, self._header_dict())
        self._data_offset = self._fp.tell()
        self._fp.seek(self._data_offset)

    def _open_existing(self) -> None:
        self._fp = open(self.path, "rb+")
        major, minor = npy_format.read_magic(self._fp)
        self._version = (major, minor)
        if self._version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(self._fp)
        elif self._version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(self._fp)
        else:
            raise ValueError(f"Unsupported npy version for append: {self._version}")
        if fortran_order:
            raise ValueError(f"Fortran-order npy append is not supported: {self.path}")

        self.dtype = np.dtype(dtype)
        self.rows = int(shape[0]) if len(shape) > 0 else 0
        self.trailing_shape = tuple(shape[1:]) if len(shape) > 1 else ()
        if self.rows == 0 and self.trailing_shape == (0,):
            self._fp.close()
            self._fp = None
            self.trailing_shape = None
            self.path.unlink(missing_ok=True)
            return
        self._data_offset = self._fp.tell()
        self._fp.seek(0, os.SEEK_END)

    def append(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=self.dtype)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim < 2:
            raise ValueError(f"AppendableNpyWriter expects row-major arrays, got shape {arr.shape}")

        if self.trailing_shape is None:
            self._initialize_file(tuple(arr.shape[1:]))
        elif tuple(arr.shape[1:]) != self.trailing_shape:
            raise ValueError(
                f"Appended array shape mismatch for {self.path}: "
                f"expected (*, {self.trailing_shape}), got {arr.shape}"
            )

        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        self._fp.seek(0, os.SEEK_END)
        self._fp.write(arr.tobytes(order="C"))
        self.rows += int(arr.shape[0])
        self._sync_header()

    def close(self) -> None:
        if self._fp is None:
            if self.empty_trailing_shape is None:
                return
            self._initialize_file(self.empty_trailing_shape)

        self._sync_header()
        self._fp.close()
        self._fp = None


def _max_frame_id(path: Path) -> int:
    """Return the maximum frame id (first column) in a dets .npy cache."""
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
            return 0
        return int(np.max(arr[:, 0]))
    except Exception:
        return 0


def _saved_detection_column_count(path: Path) -> int:
    """Return the number of columns in a detection .npy cache."""
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            return 0
        return int(arr.shape[1])
    except Exception:
        return 0


def _serialize_eval_detections(dets: np.ndarray, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Serialize detector output for cache files and return the boxes used for ReID crops."""
    if dets.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    if dets.shape[1] == 7:
        frame_col = np.full((dets.shape[0], 1), float(frame_id), dtype=np.float32)
        exported = np.concatenate([frame_col, dets], axis=1).astype(np.float32)
        reid_boxes = dets[:, :5].astype(np.float32)
        return exported, reid_boxes

    if dets.shape[1] == 6:
        frame_col = np.full((dets.shape[0], 1), float(frame_id), dtype=np.float32)
        boxes = dets[:, :4].astype(np.float32)
        confs = dets[:, 4:5].astype(np.float32)
        clss = dets[:, 5:6].astype(np.float32)
        exported = np.concatenate([frame_col, boxes, confs, clss], axis=1).astype(np.float32)
        return exported, boxes

    raise ValueError(f"Unsupported detection shape for serialization: {dets.shape}")


__all__ = [
    "AppendableNpyWriter",
    "REID_CROP_SCHEMA_VERSION",
    "_clear_device_cache",
    "_collect_seq_info",
    "_count_embedding_rows",
    "_existing_cache_path",
    "_list_sequence_frames",
    "_max_frame_id",
    "_read_image_cv2",
    "_saved_detection_column_count",
    "_sequence_img_dir",
    "_sequence_name_from_img_dir",
    "_serialize_eval_detections",
    "find_existing_reid_cache_file",
    "reid_cache_dir_candidates",
    "reid_cache_key",
    "reid_preprocess_cache_key",
]
