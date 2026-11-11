"""Persistent point handles with separate edits for the current video frame."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PromptPoint:
    """An immutable point handle whose identity survives position or label edits."""

    id: int
    position: tuple[float, float]
    label: int


class PointEditor:
    """Retain saved handles without automatically resubmitting their coordinates."""

    def __init__(self) -> None:
        self.saved: dict[int, dict[int, PromptPoint]] = {}
        self.drafts: dict[int, dict[int, PromptPoint | None]] = {}
        self._next_id = 1

    def visible(self, object_id: int) -> tuple[PromptPoint, ...]:
        """Return this object's saved points with current edits and deletions applied."""
        points = dict(self.saved.get(object_id, {}))
        for point_id, point in self.drafts.get(object_id, {}).items():
            if point is None:
                points.pop(point_id, None)
            else:
                points[point_id] = point
        return tuple(points[point_id] for point_id in sorted(points))

    def changed(self, object_id: int) -> tuple[PromptPoint, ...]:
        """Return only new, moved, or relabeled points for current-frame inference."""
        return tuple(point for _, point in sorted(self.drafts.get(object_id, {}).items()) if point is not None)

    def put(
        self,
        object_id: int,
        position: tuple[float, float],
        label: int,
        *,
        point_id: int | None = None,
    ) -> PromptPoint:
        """Add a handle or edit one belonging to this object without changing its ID."""
        x, y = map(float, position)
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Point coordinates must be finite.")
        if label not in (0, 1):
            raise ValueError("Point labels must be 0 or 1.")
        saved = self.saved.get(object_id, {})
        if point_id is not None and point_id not in saved and point_id not in self.drafts.get(object_id, {}):
            raise KeyError(f"Point {point_id} does not belong to object {object_id}.")
        if point_id is None:
            point_id = self._next_id
            self._next_id += 1
        point = PromptPoint(point_id, (x, y), int(label))
        if point == saved.get(point_id):
            self.drafts.get(object_id, {}).pop(point_id, None)
            self._drop_empty_draft(object_id)
        else:
            self.drafts.setdefault(object_id, {})[point_id] = point
        return point

    def delete(self, object_id: int, point_id: int) -> None:
        """Hide a saved handle or cancel a point that has not yet been committed."""
        if point_id in self.saved.get(object_id, {}):
            self.drafts.setdefault(object_id, {})[point_id] = None
        else:
            self.drafts.get(object_id, {}).pop(point_id, None)
            self._drop_empty_draft(object_id)

    def snapshot(self, object_id: int) -> dict[int, PromptPoint | None]:
        """Copy one object's draft state for undo; point values are immutable."""
        return dict(self.drafts.get(object_id, {}))

    def restore(self, object_id: int, snapshot: Mapping[int, PromptPoint | None]) -> None:
        """Restore an owned copy of an earlier draft without reusing allocated IDs."""
        if snapshot:
            self.drafts[object_id] = dict(snapshot)
        else:
            self.drafts.pop(object_id, None)

    def discard(self) -> None:
        """Discard frame edits while retaining all successfully committed handles."""
        self.drafts.clear()

    def commit(self, removed_ids: Iterable[int] = ()) -> None:
        """Save frame edits and remove handles belonging to deleted objects."""
        removed = set(removed_ids)
        for object_id in self.drafts:
            if object_id in removed:
                continue
            points = {point.id: point for point in self.visible(object_id)}
            if points:
                self.saved[object_id] = points
            else:
                self.saved.pop(object_id, None)
        for object_id in removed:
            self.saved.pop(object_id, None)
        self.drafts.clear()

    def _drop_empty_draft(self, object_id: int) -> None:
        """Keep objects without pending changes out of the draft mapping."""
        if not self.drafts.get(object_id):
            self.drafts.pop(object_id, None)
