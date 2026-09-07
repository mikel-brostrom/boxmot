"""Local image references decoded into canonical Frame tensors."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import torch


class ImageDecodeError(ValueError):
    """Raised when an image reference cannot be decoded canonically."""


_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
NUMPY_IMAGE_EXTENSIONS = frozenset({".npy"})
_FRAME_FRAGMENT = re.compile(r"frame=(0|[1-9][0-9]*)")
_MMOT_3CH_BGR_INDICES = (1, 2, 4)


def _local_path(value: str | Path, *, description: str) -> tuple[Path, str]:
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme not in {"", "file"}:
        raise ImageDecodeError(
            f"Materialized {description} must be a local path or file URI; "
            f"nonlocal scheme {parsed.scheme!r} is unsupported."
        )
    if parsed.netloc not in {"", "localhost"}:
        raise ImageDecodeError(f"Materialized {description} refers to nonlocal host {parsed.netloc!r}.")
    if parsed.query:
        raise ImageDecodeError(f"Materialized {description} must not contain a query string.")
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        if not path.is_absolute():
            raise ImageDecodeError(f"Materialized {description} file URI must contain an absolute path.")
        return path, parsed.fragment
    return Path(unquote(parsed.path)), parsed.fragment


def _local_image_path(reference: str, root: str | Path) -> tuple[Path, int | None]:
    root_path, root_fragment = _local_path(root, description="image root")
    if root_fragment:
        raise ImageDecodeError("Materialized image roots must not contain a fragment.")
    path, fragment = _local_path(reference, description="image reference")
    if not path.is_absolute():
        resolved_root = root_path.expanduser().resolve()
        path = (resolved_root / path).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ImageDecodeError(f"Materialized image reference escapes its source root: {reference!r}.") from exc
    else:
        path = path.expanduser().resolve()

    frame_index = None
    if fragment:
        match = _FRAME_FRAGMENT.fullmatch(fragment)
        if match is None:
            raise ImageDecodeError(
                f"Materialized image reference has unsupported fragment {fragment!r}; expected '#frame=N'."
            )
        frame_index = int(match.group(1))
        if path.suffix.lower() not in _VIDEO_EXTENSIONS:
            raise ImageDecodeError("The '#frame=N' fragment is only valid for a local video reference.")
    elif path.suffix.lower() in _VIDEO_EXTENSIONS:
        raise ImageDecodeError("A local video reference must select one frame with '#frame=N'.")
    return path, frame_index


def _read_video_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ImageDecodeError(f"Unable to open video: {path}")
        seeked = frame_index == 0 or capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok = False
        image = None
        if seeked:
            ok, image = capture.read()
        else:
            for _ in range(frame_index + 1):
                ok, image = capture.read()
                if not ok:
                    break
        if not ok or image is None:
            raise ImageDecodeError(f"Unable to decode frame {frame_index} from video: {path}")
        return image
    finally:
        capture.release()


def _open_numpy_image(path: str | Path) -> np.ndarray:
    """Memory-map and validate a NumPy image without reading its pixels."""

    resolved = Path(path)
    try:
        image = np.load(resolved, mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ImageDecodeError(f"Could not decode NumPy source frame: {resolved}") from exc
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0 or image.shape[2] < 3:
        raise ImageDecodeError(
            "NumPy source frames must be non-empty uint8 HWC arrays with at least three channels; "
            f"got {image.shape}, {image.dtype}: {resolved}"
        )
    return image


def probe_numpy_image_size(path: str | Path) -> tuple[int, int]:
    """Validate a NumPy image header and return its ``(height, width)``."""

    image = _open_numpy_image(path)
    return int(image.shape[0]), int(image.shape[1])


def read_numpy_bgr_uint8(path: str | Path) -> np.ndarray:
    """Decode a NumPy image into the OpenCV BGR contract used by BoxMOT.

    The MMOT 3-channel OBB checkpoint was trained with ``npy2rgb=True``. That
    path selected zero-based indices ``[1, 2, 4]`` at its BGR boundary before
    Ultralytics reversed them for the network. Consequently, the model sees
    zero-based RGB bands ``[4, 2, 1]`` (one-based bands 5, 3, and 2). Other
    arrays with extra channels use their first three channels, matching the
    legacy BoxMOT loader.
    """

    image = _open_numpy_image(path)
    if image.shape[2] == 8:
        return np.take(image, _MMOT_3CH_BGR_INDICES, axis=2)
    return np.array(image[..., :3], dtype=np.uint8, order="C", copy=True)


def read_rgb_chw_uint8(uri: str, root: str | Path) -> torch.Tensor:
    """Decode a local reference as contiguous CPU RGB ``uint8 [3,H,W]``."""

    if not isinstance(uri, str) or not uri:
        raise ImageDecodeError("Materialized image references must be non-empty strings.")
    path, frame_index = _local_image_path(uri, root)
    if path.suffix.lower() in NUMPY_IMAGE_EXTENSIONS:
        bgr = read_numpy_bgr_uint8(path)
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR) if frame_index is None else _read_video_frame(path, frame_index)
    if bgr is None:
        raise ImageDecodeError(f"Unable to decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


__all__ = (
    "ImageDecodeError",
    "NUMPY_IMAGE_EXTENSIONS",
    "probe_numpy_image_size",
    "read_numpy_bgr_uint8",
    "read_rgb_chw_uint8",
)
