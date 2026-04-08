from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from boxmot.utils import logger as LOGGER

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MANIFEST_EXTS = {".txt", ".streams"}


def _yield_image(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    image = cv2.imread(str(path))
    if image is not None:
        yield str(path), image


def _yield_video(source: str, vid_stride: int) -> Iterator[tuple[str, np.ndarray]]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        LOGGER.error(f"Could not open source: {source}")
        return

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % vid_stride == 0:
                yield source, frame
    finally:
        capture.release()


def _yield_manifest(path: Path, vid_stride: int) -> Iterator[tuple[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            entry = raw_line.strip()
            if not entry or entry.startswith("#"):
                continue

            resolved = entry
            if "://" not in entry:
                entry_path = Path(entry)
                if not entry_path.is_absolute():
                    entry_path = (path.parent / entry_path).resolve()
                resolved = str(entry_path)

            yield from iter_source(resolved, vid_stride=vid_stride)


def _yield_directory(path: Path, vid_stride: int) -> Iterator[tuple[str, np.ndarray]]:
    for child in sorted(path.iterdir()):
        suffix = child.suffix.lower()
        if child.is_dir():
            continue
        if suffix in IMAGE_EXTS:
            yield from _yield_image(child)
        elif suffix in VIDEO_EXTS:
            yield from _yield_video(str(child), vid_stride=vid_stride)


def iter_source(source: Any, vid_stride: int = 1) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(path, frame)`` pairs from images, videos, manifests, URLs, or webcams."""
    if isinstance(source, np.ndarray):
        yield "", source
        return

    if isinstance(source, (list, tuple)):
        for item in source:
            yield from iter_source(item, vid_stride=vid_stride)
        return

    if isinstance(source, (str, Path)):
        source_str = str(source)

        if "://" in source_str:
            yield from _yield_video(source_str, vid_stride=vid_stride)
            return

        if any(ch in source_str for ch in "*?[]"):
            for match in sorted(glob(source_str)):
                yield from iter_source(match, vid_stride=vid_stride)
            return

        path = Path(source_str)
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTS:
                yield from _yield_image(path)
                return
            if suffix in MANIFEST_EXTS:
                yield from _yield_manifest(path, vid_stride=vid_stride)
                return
            yield from _yield_video(str(path), vid_stride=vid_stride)
            return

        if path.is_dir():
            yield from _yield_directory(path, vid_stride=vid_stride)
            return

        try:
            cam_id = int(source_str)
        except ValueError:
            LOGGER.error(f"Could not open source: {source}")
            return

        yield from _yield_video(str(cam_id), vid_stride=vid_stride)
        return

    try:
        cam_id = int(source)
    except (TypeError, ValueError):
        LOGGER.error(f"Could not open source: {source}")
        return

    yield from _yield_video(str(cam_id), vid_stride=vid_stride)


__all__ = ("IMAGE_EXTS", "MANIFEST_EXTS", "VIDEO_EXTS", "iter_source")