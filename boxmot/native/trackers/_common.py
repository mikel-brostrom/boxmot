"""Typed ABI v2 plumbing shared by low-level native tracker bindings.

This module is deliberately structure-independent. It accepts only typed,
CPU-resident NumPy buffers and returns copied NumPy buffers; conversion to and
from BoxMOT domain structures belongs to :mod:`boxmot.trackers.common.native`.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.native import _common as native_common

ABI_VERSION = 2

FloatPointer = ctypes.POINTER(ctypes.c_float)
Int64Pointer = ctypes.POINTER(ctypes.c_int64)
UInt8Pointer = ctypes.POINTER(ctypes.c_uint8)


class CDetectionBatchV2(ctypes.Structure):
    """ctypes mirror of ``BoxMOTDetectionBatchV2``."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("geometry", FloatPointer),
        ("scores", FloatPointer),
        ("class_ids", Int64Pointer),
        ("detection_indices", Int64Pointer),
        ("embeddings", FloatPointer),
        ("rows", ctypes.c_int64),
        ("geometry_cols", ctypes.c_int32),
        ("embedding_cols", ctypes.c_int32),
    ]


class CImageV2(ctypes.Structure):
    """ctypes mirror of ``BoxMOTImageV2``."""

    _fields_ = [
        ("data", UInt8Pointer),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("channels", ctypes.c_int32),
    ]


class CTrackBatchV2(ctypes.Structure):
    """ctypes mirror of the library-owned ``BoxMOTTrackBatchV2``."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("geometry", FloatPointer),
        ("scores", FloatPointer),
        ("track_ids", Int64Pointer),
        ("class_ids", Int64Pointer),
        ("detection_indices", Int64Pointer),
        ("rows", ctypes.c_int64),
        ("geometry_cols", ctypes.c_int32),
    ]


CTrackBatchV2Pointer = ctypes.POINTER(CTrackBatchV2)

LIVE_UPDATE_V2_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.POINTER(CDetectionBatchV2),
    ctypes.POINTER(CImageV2),
    ctypes.POINTER(CTrackBatchV2Pointer),
]
LIVE_RESULT_FREE_V2_ARGTYPES = [CTrackBatchV2Pointer]


@dataclass(frozen=True, slots=True)
class NativeTrackBatch:
    """Owned NumPy representation copied from one native update result."""

    geometry: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray
    class_ids: np.ndarray
    detection_indices: np.ndarray

    def __post_init__(self) -> None:
        rows = _require_array(self.geometry, name="geometry", dtype=np.float32, rank=2)
        if self.geometry.shape[1] not in {4, 5}:
            raise ValueError("geometry must have four AABB or five OBB columns.")
        for name, value, dtype in (
            ("scores", self.scores, np.float32),
            ("track_ids", self.track_ids, np.int64),
            ("class_ids", self.class_ids, np.int64),
            ("detection_indices", self.detection_indices, np.int64),
        ):
            if _require_array(value, name=name, dtype=dtype, rank=1) != rows:
                raise ValueError(f"{name} must have {rows} rows.")


def _require_array(
    value: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    rank: int,
) -> int:
    """Validate one ABI input without converting, copying, or coercing it."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}.")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {value.dtype}.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}.")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return int(value.shape[0])


def ensure_tracker_library(
    *,
    tracker_name: str,
    display_name: str,
    build_lock: threading.Lock,
    force_rebuild: bool = False,
) -> Path:
    """Resolve or build one typed native tracker library."""

    target = f"{tracker_name}_capi"
    return native_common.build_native_target(
        tracker_name=tracker_name,
        display_name=display_name,
        target=target,
        candidates=native_common.candidate_libraries(tracker_name),
        force_rebuild=force_rebuild,
        not_found_message=f"Native {display_name} build succeeded but the {target} shared library was not found.",
        build_lock=build_lock,
    )


def configure_update_v2(library: ctypes.CDLL, *, prefix: str) -> tuple[Any, Any]:
    """Configure and return a tracker's ABI v2 update/free functions."""

    update = getattr(library, f"boxmot_{prefix}_update_v2")
    update.argtypes = LIVE_UPDATE_V2_ARGTYPES
    update.restype = ctypes.c_int
    result_free = getattr(library, f"boxmot_{prefix}_result_free_v2")
    result_free.argtypes = LIVE_RESULT_FREE_V2_ARGTYPES
    result_free.restype = None
    return update, result_free


def _as_float_pointer(array: np.ndarray) -> FloatPointer:
    if array.size == 0:
        return FloatPointer()
    return array.ctypes.data_as(FloatPointer)


def _as_int64_pointer(array: np.ndarray) -> Int64Pointer:
    if array.size == 0:
        return Int64Pointer()
    return array.ctypes.data_as(Int64Pointer)


def _image_input(image: np.ndarray | None) -> CImageV2 | None:
    if image is None:
        return None
    _require_array(image, name="image", dtype=np.uint8, rank=3)
    if image.shape[2] != 3:
        raise ValueError(f"image must have shape [H,W,3], got {image.shape}.")
    return CImageV2(
        data=image.ctypes.data_as(UInt8Pointer),
        rows=int(image.shape[0]),
        cols=int(image.shape[1]),
        channels=3,
    )


def call_update_v2(
    update_fn: Any,
    result_free_fn: Any,
    *,
    handle: Any,
    geometry: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    detection_indices: np.ndarray,
    embeddings: np.ndarray | None,
    image: np.ndarray | None,
    display_name: str,
    last_error: Any,
) -> NativeTrackBatch:
    """Invoke a native update once and copy its library-owned typed result."""

    rows = _require_array(geometry, name="geometry", dtype=np.float32, rank=2)
    geometry_cols = int(geometry.shape[1])
    if geometry_cols not in {4, 5}:
        raise ValueError("geometry must have four AABB or five OBB columns.")
    for name, value, dtype in (
        ("scores", scores, np.float32),
        ("class_ids", class_ids, np.int64),
        ("detection_indices", detection_indices, np.int64),
    ):
        if _require_array(value, name=name, dtype=dtype, rank=1) != rows:
            raise ValueError(f"{name} must have {rows} rows.")
    embedding_cols = 0
    if embeddings is not None:
        if _require_array(embeddings, name="embeddings", dtype=np.float32, rank=2) != rows:
            raise ValueError(f"embeddings must have {rows} rows.")
        embedding_cols = int(embeddings.shape[1])

    batch = CDetectionBatchV2(
        abi_version=ABI_VERSION,
        geometry=_as_float_pointer(geometry),
        scores=_as_float_pointer(scores),
        class_ids=_as_int64_pointer(class_ids),
        detection_indices=_as_int64_pointer(detection_indices),
        embeddings=FloatPointer() if embeddings is None else _as_float_pointer(embeddings),
        rows=rows,
        geometry_cols=geometry_cols,
        embedding_cols=embedding_cols,
    )
    image_input = _image_input(image)
    result = CTrackBatchV2Pointer()
    ok = update_fn(
        handle,
        ctypes.byref(batch),
        None if image_input is None else ctypes.byref(image_input),
        ctypes.byref(result),
    )
    if ok == 0:
        if result:
            result_free_fn(result)
        raise RuntimeError(last_error())
    if not result:
        raise RuntimeError(f"Native {display_name} returned success without a result object.")

    try:
        native = result.contents
        if native.abi_version != ABI_VERSION:
            raise RuntimeError(f"Native {display_name} returned unsupported ABI version {native.abi_version}.")
        output_rows = int(native.rows)
        output_cols = int(native.geometry_cols)
        if output_rows < 0 or output_cols != geometry_cols:
            raise RuntimeError(
                f"Native {display_name} returned an invalid output shape ({output_rows}, {output_cols})."
            )
        if output_rows:
            pointers = (
                native.geometry,
                native.scores,
                native.track_ids,
                native.class_ids,
                native.detection_indices,
            )
            if not all(bool(pointer) for pointer in pointers):
                raise RuntimeError(f"Native {display_name} returned null data for a non-empty result.")
            output_geometry = np.ctypeslib.as_array(
                native.geometry, shape=(output_rows * output_cols,)
            ).reshape(output_rows, output_cols).copy()
            output_scores = np.ctypeslib.as_array(native.scores, shape=(output_rows,)).copy()
            output_track_ids = np.ctypeslib.as_array(native.track_ids, shape=(output_rows,)).copy()
            output_class_ids = np.ctypeslib.as_array(native.class_ids, shape=(output_rows,)).copy()
            output_detection_indices = np.ctypeslib.as_array(
                native.detection_indices, shape=(output_rows,)
            ).copy()
        else:
            output_geometry = np.empty((0, output_cols), dtype=np.float32)
            output_scores = np.empty((0,), dtype=np.float32)
            output_track_ids = np.empty((0,), dtype=np.int64)
            output_class_ids = np.empty((0,), dtype=np.int64)
            output_detection_indices = np.empty((0,), dtype=np.int64)
    finally:
        result_free_fn(result)

    return NativeTrackBatch(
        geometry=output_geometry,
        scores=output_scores,
        track_ids=output_track_ids,
        class_ids=output_class_ids,
        detection_indices=output_detection_indices,
    )


__all__ = (
    "ABI_VERSION",
    "CDetectionBatchV2",
    "CImageV2",
    "CTrackBatchV2",
    "CTrackBatchV2Pointer",
    "FloatPointer",
    "Int64Pointer",
    "NativeTrackBatch",
    "call_update_v2",
    "configure_update_v2",
    "ensure_tracker_library",
)
