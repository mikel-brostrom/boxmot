from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ._validation import (
    normalize_row_indices,
    validate_finite,
    validate_nonempty_string,
    validate_scores_and_classes,
    validate_tensor,
)
from .geometry import Boxes, Boxes3D, Geometry, OrientedBoxes
from .masks import MaskBatch


@dataclass(frozen=True, slots=True, eq=False)
class Tracks:
    """Tracker outputs with optional full-frame masks aligned to track rows."""

    geometry: Geometry
    track_ids: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    detection_indices: torch.Tensor
    sample_id: str
    masks: MaskBatch | None = None

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.track_ids.shape[0])

    @property
    def is_obb(self) -> bool:
        return isinstance(self.geometry, OrientedBoxes)

    def validate(self) -> None:
        if not isinstance(self.geometry, (Boxes, OrientedBoxes)):
            raise TypeError(f"Tracks.geometry must be Boxes or OrientedBoxes, got {type(self.geometry).__name__}.")
        self.geometry.validate()
        validate_tensor(self.track_ids, name="Tracks.track_ids", dtype=torch.int64, ndim=1)
        validate_tensor(self.scores, name="Tracks.scores", dtype=torch.float32, ndim=1)
        validate_tensor(self.class_ids, name="Tracks.class_ids", dtype=torch.int64, ndim=1)
        validate_tensor(self.detection_indices, name="Tracks.detection_indices", dtype=torch.int64, ndim=1)
        validate_finite(self.scores, name="Tracks.scores")
        validate_nonempty_string(self.sample_id, name="Tracks.sample_id")
        if self.track_ids.numel() and bool((self.track_ids < 0).any()):
            raise ValueError("Tracks.track_ids must be non-negative.")
        if self.scores.numel() and (bool((self.scores < 0).any()) or bool((self.scores > 1).any())):
            raise ValueError("Tracks.scores must be in the inclusive range [0, 1].")
        if self.class_ids.numel() and bool((self.class_ids < 0).any()):
            raise ValueError("Tracks.class_ids must be non-negative.")

        count = len(self)
        aligned = (
            ("geometry", len(self.geometry)),
            ("scores", len(self.scores)),
            ("class IDs", len(self.class_ids)),
            ("detection indices", len(self.detection_indices)),
        )
        for name, row_count in aligned:
            if row_count != count:
                raise ValueError(f"Track IDs and {name} must be aligned, got {count} and {row_count} rows.")

        if self.track_ids.numel() and self.track_ids.unique().numel() != self.track_ids.numel():
            raise ValueError("Tracks.track_ids must be unique.")
        if self.detection_indices.numel() and bool((self.detection_indices < -1).any()):
            raise ValueError("Tracks.detection_indices may only use -1 for an unmatched track.")

        if self.masks is not None:
            if not isinstance(self.masks, MaskBatch):
                raise TypeError(f"Tracks.masks must be a MaskBatch or None, got {type(self.masks).__name__}.")
            self.masks.validate()
            if len(self.masks) != count:
                raise ValueError(f"Track IDs and masks must be aligned, got {count} and {len(self.masks)} rows.")

    def select(self, indices: torch.Tensor) -> Tracks:
        """Select or reorder rows while preserving track-aligned masks."""
        selected = normalize_row_indices(indices, count=len(self))
        return Tracks(
            geometry=self.geometry.select(selected),
            track_ids=self.track_ids.index_select(0, selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            detection_indices=self.detection_indices.index_select(0, selected),
            sample_id=self.sample_id,
            masks=None if self.masks is None else self.masks.select(selected),
        )

    def with_masks(self, masks: MaskBatch) -> Tracks:
        """Return track rows carrying aligned full-frame masks."""
        return replace(self, masks=masks)

    def to_aabb_rows(self) -> torch.Tensor:
        """Serialize AABB tracks as ``[x1,y1,x2,y2,id,score,class,det_idx]`` float32 rows."""
        if not isinstance(self.geometry, Boxes):
            raise TypeError("Cannot serialize oriented tracks as AABB rows.")
        return torch.cat(
            (
                self.geometry.values,
                self.track_ids.to(dtype=torch.float32)[:, None],
                self.scores[:, None],
                self.class_ids.to(dtype=torch.float32)[:, None],
                self.detection_indices.to(dtype=torch.float32)[:, None],
            ),
            dim=1,
        ).contiguous()

    def to_obb_rows(self) -> torch.Tensor:
        """Serialize OBB tracks as ``[cx,cy,w,h,angle,id,score,class,det_idx]`` float32 rows."""
        if not isinstance(self.geometry, OrientedBoxes):
            raise TypeError("Cannot serialize axis-aligned tracks as OBB rows.")
        return torch.cat(
            (
                self.geometry.values,
                self.track_ids.to(dtype=torch.float32)[:, None],
                self.scores[:, None],
                self.class_ids.to(dtype=torch.float32)[:, None],
                self.detection_indices.to(dtype=torch.float32)[:, None],
            ),
            dim=1,
        ).contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class Tracks3D:
    """Camera-space 3D track rows, including objects outside the image view.

    ``detection_indices`` refers to the independent current Detections3D batch,
    with -1 for tracks without a current 3D observation.
    """

    geometry: Boxes3D
    track_ids: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    detection_indices: torch.Tensor
    sample_id: str

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.geometry)

    def validate(self) -> None:
        """Validate spatial tracks independently of their visible 2D projections."""
        if not isinstance(self.geometry, Boxes3D):
            raise TypeError("Tracks3D.geometry must be Boxes3D.")
        self.geometry.validate()
        validate_scores_and_classes(self.scores, self.class_ids, owner="Tracks3D")
        validate_tensor(self.track_ids, name="Tracks3D.track_ids", dtype=torch.int64, ndim=1)
        validate_tensor(self.detection_indices, name="Tracks3D.detection_indices", dtype=torch.int64, ndim=1)
        validate_nonempty_string(self.sample_id, name="Tracks3D.sample_id")
        if any(
            len(values) != len(self) for values in (self.track_ids, self.scores, self.class_ids, self.detection_indices)
        ):
            raise ValueError("Tracks3D geometry and all metadata must be aligned.")
        if self.track_ids.numel() and bool((self.track_ids < 0).any()):
            raise ValueError("Tracks3D.track_ids must be non-negative.")
        if self.track_ids.unique().numel() != self.track_ids.numel():
            raise ValueError("Tracks3D.track_ids must be unique.")
        if self.detection_indices.numel() and bool((self.detection_indices < -1).any()):
            raise ValueError("Tracks3D.detection_indices may only use -1 for an unmatched track.")

    def select(self, indices: torch.Tensor) -> Tracks3D:
        """Select or reorder aligned 3D tracking rows."""
        selected = normalize_row_indices(indices, count=len(self))
        return Tracks3D(
            geometry=self.geometry.select(selected),
            track_ids=self.track_ids.index_select(0, selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            detection_indices=self.detection_indices.index_select(0, selected),
            sample_id=self.sample_id,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MultimodalTracks:
    """Independent image and spatial outputs sharing one track-ID namespace.

    The collections can differ in length: an off-camera 3D object needs no
    artificial image box. Matching IDs identify the same object in both views.
    """

    image_tracks: Tracks
    spatial_tracks: Tracks3D

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(set(self.image_tracks.track_ids.tolist()) | set(self.spatial_tracks.track_ids.tolist()))

    @property
    def sample_id(self) -> str:
        """Return the common frame identity."""
        return self.spatial_tracks.sample_id

    def validate(self) -> None:
        """Validate both collections and their shared identity metadata."""
        if not isinstance(self.image_tracks, Tracks) or not isinstance(self.spatial_tracks, Tracks3D):
            raise TypeError("MultimodalTracks requires image Tracks and spatial Tracks3D.")
        self.image_tracks.validate()
        self.spatial_tracks.validate()
        if self.image_tracks.sample_id != self.spatial_tracks.sample_id:
            raise ValueError("MultimodalTracks collections must identify the same sample.")
        classes = dict(zip(self.image_tracks.track_ids.tolist(), self.image_tracks.class_ids.tolist()))
        for track_id, class_id in zip(self.spatial_tracks.track_ids.tolist(), self.spatial_tracks.class_ids.tolist()):
            if track_id in classes and classes[track_id] != class_id:
                raise ValueError("MultimodalTracks must assign the same class to a shared track ID.")


__all__ = ("MultimodalTracks", "Tracks", "Tracks3D")
