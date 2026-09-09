"""Lazy KITTI MOTS images and instance PNGs in canonical tensor structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import torch

    from boxmot.structures import Frame, Tracks


def kitti_mots_frame_paths(sequence_root: Path) -> tuple[Path, ...]:
    """Discover PNG frames in numeric order without decoding their pixels.

    Native, zero-based frame numbers may be sparse. Duplicate numbers with
    different zero padding are ambiguous and rejected. macOS AppleDouble
    files are metadata, so their ``._`` names are ignored.
    """

    sequence_root = Path(sequence_root)
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"KITTI MOTS sequence directory does not exist: {sequence_root}")
    indexed: dict[int, Path] = {}
    for path in sequence_root.iterdir():
        if path.name.startswith("._") or not path.is_file() or path.suffix.lower() != ".png":
            continue
        if not path.stem.isascii() or not path.stem.isdecimal():
            raise ValueError(f"KITTI MOTS frame filenames must have non-negative numeric stems: {path}")
        frame_index = int(path.stem)
        if frame_index in indexed:
            raise ValueError(
                f"KITTI MOTS sequence has duplicate frame index {frame_index}: {indexed[frame_index]} and {path}"
            )
        indexed[frame_index] = path
    if not indexed:
        raise ValueError(f"KITTI MOTS sequence contains no PNG frames: {sequence_root}")
    return tuple(indexed[index] for index in sorted(indexed))


@dataclass(frozen=True, slots=True)
class KittiMotsSample:
    """One RGB frame, optional annotated tracks, and a separate ignore region.

    Metadata is available on ``frame``. Annotated frames always provide an
    ``ignore_mask`` with shape ``[H,W]`` and boolean dtype, including when no
    pixels are ignored. Unannotated frames set both annotation fields to None.
    """

    frame: Frame
    ground_truth: Tracks | None
    ignore_mask: torch.Tensor | None


def _read_ground_truth(path: Path, frame: Frame) -> tuple[Tracks, torch.Tensor]:
    """Decode native uint16 IDs while preserving classes and ignored pixels."""

    import cv2
    import numpy as np
    import torch

    from boxmot.structures import Boxes, MaskBatch, Tracks

    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"Unable to decode KITTI MOTS instance PNG: {path}")
    if labels.dtype != np.uint16 or labels.ndim != 2:
        raise ValueError(
            f"KITTI MOTS instance PNG must be single-channel uint16, got {labels.shape}, {labels.dtype}: {path}"
        )
    if labels.shape != frame.image_size:
        raise ValueError(
            f"KITTI MOTS instance PNG dimensions {labels.shape} "
            f"do not match image dimensions {frame.image_size}: {path}"
        )

    object_ids = np.unique(labels)
    valid = (object_ids == 0) | (object_ids == 10000) | ((object_ids >= 1000) & (object_ids < 3000))
    if not valid.all():
        raise ValueError(f"KITTI MOTS instance PNG contains unsupported labels {object_ids[~valid].tolist()}: {path}")
    object_ids = object_ids[(object_ids != 0) & (object_ids != 10000)].astype(np.int64)
    masks = labels[None, :, :] == object_ids[:, None, None]
    boxes = np.empty((len(object_ids), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        boxes[index] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    tracks = Tracks(
        geometry=Boxes(torch.from_numpy(boxes)),
        track_ids=torch.from_numpy(object_ids),
        scores=torch.ones(len(object_ids), dtype=torch.float32),
        class_ids=torch.from_numpy(object_ids // 1000),
        detection_indices=torch.full((len(object_ids),), -1, dtype=torch.int64),
        sample_id=frame.sample_id,
        masks=MaskBatch(torch.from_numpy(masks)),
    )
    return tracks, torch.from_numpy(labels == 10000)


class KittiMotsDataset(Sequence[KittiMotsSample]):
    """Load raw KITTI MOTS frames and optional uint16 instance annotations.

    Args:
        image_root: Directory containing sequence folders, such as KITTI's
            ``training/image_02`` or ``testing/image_02``.
        instances_root: Directory containing matching sequence folders and
            identically named PNG annotations. Omit for unannotated data.
        split: Split label used in sample IDs; it does not choose sequences.
        sequence_ids: Optional exact sequence names, preserving zero padding.

    Construction indexes paths and checks annotation pairing. Pixels are
    decoded only on indexing. Native frame numbers are preserved, with
    timestamps in seconds at KITTI's 10 FPS. Each sample ID is
    ``{split}:{sequence_id}:{frame_index}``.

    Ground-truth classes retain KITTI's IDs (1 car, 2 pedestrian), and track
    IDs retain the full PNG value ``class_id * 1000 + instance_id``. Masks
    cover the full frame; boxes have exclusive maximum pixel coordinates.
    Background (0) is excluded and ignored pixels (10000) are kept separately.
    """

    def __init__(
        self,
        image_root: str | Path,
        instances_root: str | Path | None = None,
        *,
        split: str = "train",
        sequence_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(split, str) or not split or split != split.strip() or ":" in split:
            raise ValueError("split must be a non-empty string without surrounding whitespace or ':'.")
        self.image_root = Path(image_root).expanduser().resolve()
        self.instances_root = None if instances_root is None else Path(instances_root).expanduser().resolve()
        self.split = split
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"KITTI MOTS image root does not exist: {self.image_root}")
        if self.instances_root is not None and not self.instances_root.is_dir():
            raise FileNotFoundError(f"KITTI MOTS instances root does not exist: {self.instances_root}")

        available = {
            path.name: path for path in self.image_root.iterdir() if path.is_dir() and not path.name.startswith(".")
        }
        if sequence_ids is None:
            selected = tuple(sorted(available))
        else:
            if isinstance(sequence_ids, (str, bytes)) or not isinstance(sequence_ids, Sequence):
                raise TypeError("sequence_ids must be a sequence of exact directory names, not a string.")
            selected = tuple(sequence_ids)
            if not selected:
                raise ValueError("sequence_ids must contain at least one sequence name.")
            if any(
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or any(character in name for character in ("/", "\\", ":"))
                or name != name.strip()
                for name in selected
            ):
                raise ValueError("sequence_ids must contain non-empty directory names without paths or ':'.")
            if len(set(selected)) != len(selected):
                raise ValueError("sequence_ids must not contain duplicate names.")
            missing = sorted(set(selected) - available.keys())
            if missing:
                raise ValueError(f"KITTI MOTS image root does not contain requested sequences: {missing}")
            selected = tuple(sorted(selected))
        if sequence_ids is None and not selected:
            raise ValueError(f"KITTI MOTS image root contains no sequence directories: {self.image_root}")

        self.sequence_ids = selected
        paths: list[tuple[str, Path, Path | None]] = []
        for sequence_id in selected:
            sequence_root = available[sequence_id]
            if ":" in sequence_id or not sequence_root.resolve().is_relative_to(self.image_root):
                raise ValueError(f"Invalid KITTI MOTS sequence directory: {sequence_root}")
            for image_path in kitti_mots_frame_paths(sequence_root):
                annotation_path = None
                if self.instances_root is not None:
                    annotation_path = self.instances_root / sequence_id / image_path.name
                    if not annotation_path.is_file():
                        raise FileNotFoundError(
                            f"Missing paired KITTI MOTS instance PNG for {image_path}: {annotation_path}"
                        )
                paths.append((sequence_id, image_path, annotation_path))
        self._paths = tuple(paths)

    def __len__(self) -> int:
        """Return the number of selected frames without decoding images."""

        return len(self._paths)

    @overload
    def __getitem__(self, index: int) -> KittiMotsSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KittiMotsSample, ...]: ...

    def __getitem__(self, index: int | slice) -> KittiMotsSample | tuple[KittiMotsSample, ...]:
        """Decode one frame or a slice of frames and matching annotations."""

        from boxmot.datasets.readers.images import read_rgb_chw_uint8
        from boxmot.structures import Frame

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"KITTI MOTS dataset indices must be integers or slices, got {type(index).__name__}.")
        sequence_id, image_path, annotation_path = self._paths[index]
        frame_index = int(image_path.stem)
        frame = Frame(
            image=read_rgb_chw_uint8(image_path.as_uri(), self.image_root.as_uri()),
            sample_id=f"{self.split}:{sequence_id}:{frame_index}",
            sequence_id=sequence_id,
            frame_index=frame_index,
            timestamp_s=frame_index / 10.0,
            source_uri=image_path.as_uri(),
        )
        ground_truth, ignore_mask = (
            (None, None) if annotation_path is None else _read_ground_truth(annotation_path, frame)
        )
        return KittiMotsSample(frame=frame, ground_truth=ground_truth, ignore_mask=ignore_mask)


__all__ = ("KittiMotsDataset", "KittiMotsSample", "kitti_mots_frame_paths")
