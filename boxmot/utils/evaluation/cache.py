from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from numpy.lib import format as npy_format


def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _list_sequence_frames(img_dir: Path) -> list[Path]:
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))


def _sequence_name_from_img_dir(img_dir: Path) -> str:
    return img_dir.parent.name if img_dir.name == "img1" else img_dir.name


def _read_image_cv2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths: list[Path] = []
    seq_info: dict[str, int] = {}
    for seq_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        img_dir = _sequence_img_dir(seq_dir)
        frame_files = _list_sequence_frames(img_dir)
        if not frame_files:
            continue
        seq_paths.append(img_dir)
        seq_info[seq_dir.name] = len(frame_files)
    return seq_paths, seq_info


def _clear_device_cache(device: str) -> None:
    dev_lower = str(device).lower()
    if dev_lower.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        # Empty file with shape (0, 0) has unknown trailing shape — treat as
        # uninitialised so the first append() determines the real shape.
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
        except Exception:  # noqa: BLE001
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
                except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    "_read_image_cv2",
    "_saved_detection_column_count",
    "_sequence_img_dir",
    "_sequence_name_from_img_dir",
    "_serialize_eval_detections",
]
