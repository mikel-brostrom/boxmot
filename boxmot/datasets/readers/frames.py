"""Lightweight numeric image discovery without loading pixel or tensor libraries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def numeric_frame_paths(sequence_root: Path, *, contiguous: bool = False) -> tuple[Path, ...]:
    """Discover numeric image names, rejecting duplicate frame indices.

    Sparse indices are retained unless the reader needs a contiguous zero-based
    timeline. AppleDouble metadata is ignored without opening image headers.
    """
    sequence_root = Path(sequence_root)
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Image sequence directory does not exist: {sequence_root}")
    indexed: dict[int, Path] = {}
    for path in sequence_root.iterdir():
        if path.name.startswith("._") or not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if not path.stem.isascii() or not path.stem.isdecimal():
            raise ValueError(f"Image frame filenames must have non-negative numeric stems: {path}")
        frame_index = int(path.stem)
        if frame_index in indexed:
            raise ValueError(
                f"Image sequence has duplicate frame index {frame_index}: {indexed[frame_index]} and {path}"
            )
        indexed[frame_index] = path
    if not indexed:
        raise ValueError(f"Image sequence contains no image frames: {sequence_root}")
    indices = sorted(indexed)
    if contiguous and indices != list(range(len(indices))):
        raise ValueError(f"Sequence images must cover contiguous zero-based frames: {sequence_root}")
    return tuple(indexed[index] for index in indices)


def image_sequence_size(frame_paths: Sequence[Path]) -> tuple[int, int]:
    """Validate matching image headers and return height and width, without pixels."""
    from PIL import Image

    image_size = None
    for path in frame_paths:
        with Image.open(path) as image:
            size = (image.height, image.width)
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"Image dimensions {size} differ from {image_size}: {path}")
    if image_size is None:
        raise ValueError("An image sequence must contain at least one frame.")
    return image_size
