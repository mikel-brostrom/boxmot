from __future__ import annotations

import os
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


def _count_data_lines(path: Path, skip_header: bool = False) -> int:
    """Count non-header lines in a txt file, tolerating missing files."""
    try:
        with open(path, "r") as handle:
            if skip_header:
                return sum(1 for line in handle if not line.startswith("#"))
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def _count_embedding_rows(path: Path) -> int:
    """Count rows in an embedding cache (.npy or .txt)."""
    if path.suffix == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
            return arr.shape[0]
        except Exception:
            return 0
    return _count_data_lines(path, skip_header=True)


def _legacy_cache_txt_path(path: Path) -> Path:
    return path.with_suffix(".txt")


def _existing_cache_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path

    legacy_path = _legacy_cache_txt_path(path)
    if legacy_path.exists():
        return legacy_path

    return None


def _load_numeric_cache_array(path: Path, *, comments: str | None = "#") -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, comments=comments)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    return np.atleast_2d(arr).astype(np.float32, copy=False)


def _migrate_legacy_numeric_cache(
    source_path: Path,
    target_path: Path,
    *,
    comments: str | None = "#",
) -> bool:
    if source_path.suffix != ".txt" or target_path.exists():
        return False

    arr = _load_numeric_cache_array(source_path, comments=comments)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, arr)
    return True


def _existing_embedding_cache_path(path: Path) -> Optional[Path]:
    return _existing_cache_path(path)


# Tokens accepted by ``BOXMOT_REID_BACKEND`` (matches the C++ runtime selector).
_OPENCV_RUNTIME_TOKENS = {"opencv", "cv", "dnn", "opencv_dnn"}


def _onnx_runtime_token() -> str:
    """Resolve the ONNX runtime token (``ort`` / ``opencv``) from the env."""
    raw = os.environ.get("BOXMOT_REID_BACKEND", "").strip().lower()
    return "opencv" if raw in _OPENCV_RUNTIME_TOKENS else "ort"


def _resolve_reid_runtime(suffix: str, *, tracker_backend: str | None) -> str:
    """Map a ReID weights suffix + tracker backend to its runtime token.

    The runtime token differentiates embedding caches that came out of
    distinct execution stacks (ONNX Runtime vs OpenCV-DNN vs PyTorch, etc.)
    so they don't silently overwrite each other on the same model file.
    """
    suffix = (suffix or "").lower()
    is_cpp = bool(tracker_backend) and str(tracker_backend).lower() == "cpp"

    # The native C++ replay path always runs ONNX models (it auto-exports
    # ``.pt`` weights at load time), so the runtime token is determined
    # entirely by ``BOXMOT_REID_BACKEND``.
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
        # Bare model identifier with no suffix: treat as the historical
        # PyTorch default (the only runtime that ever supported it).
        return "pytorch"
    return suffix.lstrip(".") or "default"


def reid_cache_key(
    reid_model: str | os.PathLike,
    *,
    tracker_backend: str | None = None,
) -> str:
    """Return the directory key used to bucket cached ReID embeddings.

    Format: ``<stem>_<ext-no-dot>_<runtime>_<stack>``. Examples:

    * ``lmbn_n_duke.pt`` (Python)               → ``lmbn_n_duke_pt_pytorch_py``
    * ``lmbn_n_duke.onnx`` (Python, ORT)        → ``lmbn_n_duke_onnx_ort_py``
    * ``lmbn_n_duke.onnx`` (Python, OpenCV-DNN) → ``lmbn_n_duke_onnx_opencv_py``
    * ``lmbn_n_duke.onnx`` (C++, ORT)           → ``lmbn_n_duke_onnx_ort_cpp``
    * ``lmbn_n_duke.pt`` (C++, OpenCV-DNN)      → ``lmbn_n_duke_pt_opencv_cpp``

    The runtime token comes from the file suffix and the
    ``BOXMOT_REID_BACKEND`` env var; the stack token (``py`` / ``cpp``) keeps
    Python- and C++-generated embeddings strictly separated even when the
    runtime selector matches, since the two stacks can differ in
    preprocessing, normalization, or pad/letterbox handling.
    """
    p = Path(reid_model)
    name = p.name if p.suffix else str(reid_model)
    base = name.replace(".", "_")
    runtime = _resolve_reid_runtime(p.suffix, tracker_backend=tracker_backend)
    stack = "cpp" if (tracker_backend and str(tracker_backend).lower() == "cpp") else "py"
    return f"{base}_{runtime}_{stack}"


def legacy_reid_cache_keys(
    reid_model: str | os.PathLike,
    *,
    tracker_backend: str | None = None,
) -> list[str]:
    """Return historical cache keys to consult on read for backwards compat.

    These are checked **only** if no cache exists under the canonical
    :func:`reid_cache_key` directory, so legacy on-disk caches generated by
    earlier versions remain loadable instead of silently regenerating.
    Order matters: most-recent-format first.
    """
    p = Path(reid_model)
    name = p.name if p.suffix else str(reid_model)
    suffix = p.suffix.lower()
    is_cpp = bool(tracker_backend) and str(tracker_backend).lower() == "cpp"
    base = name.replace(".", "_")
    runtime = _resolve_reid_runtime(suffix, tracker_backend=tracker_backend)
    keys: list[str] = []
    # Pre-stack-suffix scheme: <base>_<runtime>
    keys.append(f"{base}_{runtime}")
    # Pre-runtime scheme: <name>[__cpp]
    keys.append(f"{name}__cpp" if is_cpp else name)
    # Pre-suffix legacy: stem-only directory, only valid for ``.pt`` requests
    # on the Python backend (the only path that ever wrote it).
    if not is_cpp and suffix == ".pt":
        stem = p.stem
        if stem and stem != name:
            keys.append(stem)
    return keys


def resolve_embedding_dir(
    embs_root: Path,
    reid_model: str | os.PathLike,
    preprocess_name: str,
    *,
    tracker_backend: str | None = None,
) -> Path:
    """Return the embedding directory for *reid_model*, preferring the new
    runtime-aware key but falling back to historical layouts when the new
    directory does not exist on disk. Used on the read side.
    """
    key = reid_cache_key(reid_model, tracker_backend=tracker_backend)
    new_dir = Path(embs_root) / key / preprocess_name
    if new_dir.exists():
        return new_dir
    for legacy_key in legacy_reid_cache_keys(reid_model, tracker_backend=tracker_backend):
        if legacy_key == key:
            continue
        legacy_dir = Path(embs_root) / legacy_key / preprocess_name
        if legacy_dir.exists():
            return legacy_dir
    return new_dir


def _load_embedding_cache_array(path: Path) -> np.ndarray:
    return _load_numeric_cache_array(path, comments="#")


def _migrate_legacy_embedding_cache(source_path: Path, target_path: Path) -> bool:
    return _migrate_legacy_numeric_cache(source_path, target_path, comments="#")


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
    """Return the maximum frame id (first column) in a dets txt, skipping headers."""
    if path.suffix == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
                return 0
            return int(np.max(arr[:, 0]))
        except Exception:
            return 0

    max_frame_id = 0
    try:
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    frame_value = int(float(parts[0]))
                except Exception:
                    continue
                if frame_value > max_frame_id:
                    max_frame_id = frame_value
    except FileNotFoundError:
        return 0
    return max_frame_id


def _saved_detection_column_count(path: Path) -> int:
    """Return the number of columns in the first non-header detections row."""
    if path.suffix == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                return 0
            return int(arr.shape[1])
        except Exception:
            return 0

    try:
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return len(line.replace(",", " ").split())
    except FileNotFoundError:
        return 0
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
    "_clear_device_cache",
    "_collect_seq_info",
    "_count_embedding_rows",
    "_existing_cache_path",
    "_existing_embedding_cache_path",
    "_list_sequence_frames",
    "_load_embedding_cache_array",
    "_load_numeric_cache_array",
    "_max_frame_id",
    "_migrate_legacy_embedding_cache",
    "_migrate_legacy_numeric_cache",
    "reid_cache_key",
    "legacy_reid_cache_keys",
    "resolve_embedding_dir",
    "_read_image_cv2",
    "_saved_detection_column_count",
    "_sequence_img_dir",
    "_sequence_name_from_img_dir",
    "_serialize_eval_detections",
]