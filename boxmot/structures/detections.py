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
class Detections:
    """Detection rows and optional row-aligned perception enrichments."""

    geometry: Geometry
    scores: torch.Tensor
    class_ids: torch.Tensor
    sample_id: str
    instance_ids: tuple[str, ...] | None = None
    masks: MaskBatch | None = None
    embeddings: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @property
    def is_obb(self) -> bool:
        return isinstance(self.geometry, OrientedBoxes)

    def validate(self) -> None:
        if not isinstance(self.geometry, (Boxes, OrientedBoxes)):
            raise TypeError(f"Detections.geometry must be Boxes or OrientedBoxes, got {type(self.geometry).__name__}.")
        self.geometry.validate()
        validate_tensor(self.scores, name="Detections.scores", dtype=torch.float32, ndim=1)
        validate_tensor(self.class_ids, name="Detections.class_ids", dtype=torch.int64, ndim=1)
        validate_finite(self.scores, name="Detections.scores")
        validate_nonempty_string(self.sample_id, name="Detections.sample_id")
        if self.scores.numel() and (bool((self.scores < 0).any()) or bool((self.scores > 1).any())):
            raise ValueError("Detections.scores must be in the inclusive range [0, 1].")
        if self.class_ids.numel() and bool((self.class_ids < 0).any()):
            raise ValueError("Detections.class_ids must be non-negative.")

        count = len(self)
        if len(self.geometry) != count:
            raise ValueError(f"Geometry and scores must be aligned, got {len(self.geometry)} and {count} rows.")
        if len(self.class_ids) != count:
            raise ValueError(f"Class IDs and scores must be aligned, got {len(self.class_ids)} and {count} rows.")

        if self.instance_ids is not None:
            if not isinstance(self.instance_ids, tuple):
                raise TypeError("Detections.instance_ids must be a tuple of strings or None.")
            if len(self.instance_ids) != count:
                raise ValueError(
                    f"Instance IDs and scores must be aligned, got {len(self.instance_ids)} and {count} rows."
                )
            for instance_id in self.instance_ids:
                validate_nonempty_string(instance_id, name="Detections.instance_ids entry")
            if len(set(self.instance_ids)) != len(self.instance_ids):
                raise ValueError("Detections.instance_ids must be unique.")

        if self.masks is not None:
            if not isinstance(self.masks, MaskBatch):
                raise TypeError(f"Detections.masks must be a MaskBatch or None, got {type(self.masks).__name__}.")
            self.masks.validate()
            if len(self.masks) != count:
                raise ValueError(f"Masks and scores must be aligned, got {len(self.masks)} and {count} rows.")

        if self.embeddings is not None:
            validate_tensor(self.embeddings, name="Detections.embeddings", dtype=torch.float32, ndim=2)
            validate_finite(self.embeddings, name="Detections.embeddings")
            if self.embeddings.shape[0] != count:
                raise ValueError(
                    f"Embeddings and scores must be aligned, got {self.embeddings.shape[0]} and {count} rows."
                )
            if self.embeddings.shape[1] <= 0:
                raise ValueError("Detections.embeddings must have a positive embedding dimension.")

    def select(self, indices: torch.Tensor) -> Detections:
        """Select or reorder rows while preserving every aligned field."""
        selected = normalize_row_indices(indices, count=len(self))
        instance_ids = None
        if self.instance_ids is not None:
            instance_ids = tuple(self.instance_ids[index] for index in selected.tolist())
        return Detections(
            geometry=self.geometry.select(selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            sample_id=self.sample_id,
            instance_ids=instance_ids,
            masks=None if self.masks is None else self.masks.select(selected),
            embeddings=None if self.embeddings is None else self.embeddings.index_select(0, selected),
        )

    def with_instance_ids(self, instance_ids: tuple[str, ...]) -> Detections:
        """Return a detection batch carrying stable row-aligned instance IDs."""
        return replace(self, instance_ids=instance_ids)

    def with_masks(self, masks: MaskBatch) -> Detections:
        """Return a detection batch enriched with aligned full-frame masks."""
        return replace(self, masks=masks)

    def with_embeddings(self, embeddings: torch.Tensor) -> Detections:
        """Return a detection batch enriched with aligned appearance vectors."""
        return replace(self, embeddings=embeddings)

    def to_aabb_rows(self) -> torch.Tensor:
        """Serialize AABB detections as ``[x1,y1,x2,y2,score,class]`` float32 rows."""
        if not isinstance(self.geometry, Boxes):
            raise TypeError("Cannot serialize oriented detections as AABB rows.")
        return torch.cat(
            (self.geometry.values, self.scores[:, None], self.class_ids.to(dtype=torch.float32)[:, None]),
            dim=1,
        ).contiguous()

    def to_obb_rows(self) -> torch.Tensor:
        """Serialize OBB detections as ``[cx,cy,w,h,angle,score,class]`` float32 rows."""
        if not isinstance(self.geometry, OrientedBoxes):
            raise TypeError("Cannot serialize axis-aligned detections as OBB rows.")
        return torch.cat(
            (self.geometry.values, self.scores[:, None], self.class_ids.to(dtype=torch.float32)[:, None]),
            dim=1,
        ).contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class Detections3D:
    """Independent 3D detector rows; their count need not match 2D detections."""

    geometry: Boxes3D
    scores: torch.Tensor
    class_ids: torch.Tensor
    sample_id: str

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.geometry)

    def validate(self) -> None:
        """Validate aligned spatial geometry, confidence, classes, and sample ID."""
        if not isinstance(self.geometry, Boxes3D):
            raise TypeError("Detections3D.geometry must be Boxes3D.")
        self.geometry.validate()
        validate_scores_and_classes(self.scores, self.class_ids, owner="Detections3D")
        validate_nonempty_string(self.sample_id, name="Detections3D.sample_id")
        if len(self.scores) != len(self) or len(self.class_ids) != len(self):
            raise ValueError("Detections3D geometry, scores, and class IDs must be aligned.")

    def select(self, indices: torch.Tensor) -> Detections3D:
        """Select or reorder only this independent detector's rows."""
        selected = normalize_row_indices(indices, count=len(self))
        return Detections3D(
            geometry=self.geometry.select(selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            sample_id=self.sample_id,
        )


__all__ = ("Detections", "Detections3D")
