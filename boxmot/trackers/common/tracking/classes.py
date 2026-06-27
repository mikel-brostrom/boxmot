from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


def normalize_class_names(class_names: Mapping[int, str] | Sequence[str] | None) -> dict[int, str]:
    """Normalize model class metadata into ``{class_id: name}`` form."""
    if class_names is None:
        return {}

    if isinstance(class_names, Mapping):
        return {
            int(class_id): str(name)
            for class_id, name in class_names.items()
        }

    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        return {
            int(class_id): str(name)
            for class_id, name in enumerate(class_names)
        }

    return {}


def normalize_class_ids(class_ids: Iterable[int] | None) -> frozenset[int] | None:
    """Normalize configured detector class IDs, preserving ``None`` as open-set."""
    if class_ids is None:
        return None
    return frozenset(int(class_id) for class_id in class_ids)


@dataclass(frozen=True)
class ClassCatalog:
    """Detector class metadata used by tracker validation and per-class state."""

    class_ids: frozenset[int] | None = None
    names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        *,
        class_ids: Iterable[int] | None = None,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
    ) -> "ClassCatalog":
        names = normalize_class_names(class_names)
        normalized_ids = normalize_class_ids(class_ids)
        if normalized_ids is None and names:
            normalized_ids = frozenset(names)
        return cls(class_ids=normalized_ids, names=names)

    @property
    def is_open(self) -> bool:
        """Return whether detections from any non-negative class ID are allowed."""
        return self.class_ids is None

    def validate_ids(self, class_ids: Iterable[int]) -> None:
        """Raise a clear error when detector class IDs do not match this catalog."""
        ids = {int(class_id) for class_id in class_ids}
        negative = sorted(class_id for class_id in ids if class_id < 0)
        if negative:
            raise ValueError(f"Detector returned negative class id(s): {negative}")

        if self.class_ids is None:
            return

        unknown = sorted(ids - self.class_ids)
        if unknown:
            available = ", ".join(str(class_id) for class_id in sorted(self.class_ids))
            raise ValueError(
                "Detector returned class id(s) not present in the tracker class catalog: "
                f"{unknown}. Available class ids: {available}"
            )

    def validate_detections(self, dets: np.ndarray, layout) -> None:
        """Validate class IDs from a detection array using its active layout."""
        if dets is None or dets.size == 0:
            return
        self.validate_ids(layout.classes(dets).astype(int).tolist())

    def detected_ids(self, dets: np.ndarray, layout) -> set[int]:
        """Return validated class IDs present in a detection array."""
        if dets is None or dets.size == 0:
            return set()
        ids = {int(class_id) for class_id in layout.classes(dets)}
        self.validate_ids(ids)
        return ids

    def name(self, class_id: int) -> str | None:
        """Return the configured display name for ``class_id`` when known."""
        return self.names.get(int(class_id))
