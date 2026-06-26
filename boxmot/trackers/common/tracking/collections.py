from __future__ import annotations

from collections import deque
from collections.abc import Mapping


TRACK_STATE_GROUPS = {
    "active": ("active_tracks",),
    "pool": ("trackers", "_tracks"),
    "lost": ("lost_stracks", "lost_tracks"),
    "removed": ("removed_stracks", "removed_tracks"),
}
LIVE_STATE_GROUPS = ("active", "pool", "lost")
TRACK_COLLECTION_ATTRS = tuple(
    dict.fromkeys(attr_name for attr_names in TRACK_STATE_GROUPS.values() for attr_name in attr_names)
)


def validate_track_group(group: str) -> None:
    if group not in TRACK_STATE_GROUPS:
        available = ", ".join(sorted(TRACK_STATE_GROUPS))
        raise ValueError(f"Unknown track group: {group!r}. Available groups are: {available}")


def track_collection_attrs(group: str | None = None) -> tuple[str, ...]:
    """Return concrete tracker collection attributes for one canonical group."""
    if group is None:
        return TRACK_COLLECTION_ATTRS
    validate_track_group(group)
    return TRACK_STATE_GROUPS[group]


def tracks_from_mapping(attrs: Mapping[str, object], group: str) -> list:
    """Collect tracks from a state mapping using a canonical lifecycle group."""
    tracks = []
    for attr_name in track_collection_attrs(group):
        value = attrs.get(attr_name)
        if value:
            tracks.extend(list(value))
    return tracks


def tracks_from_owner(owner, group: str) -> list:
    """Collect tracks from a tracker instance using a canonical lifecycle group."""
    tracks = []
    for attr_name in track_collection_attrs(group):
        value = getattr(owner, attr_name, None)
        if value:
            tracks.extend(list(value))
    return tracks


def owner_has_track_collection(owner, group: str) -> bool:
    """Return whether a tracker exposes any concrete collection for ``group``."""
    return any(
        hasattr(owner, attr_name) and getattr(owner, attr_name) is not None
        for attr_name in track_collection_attrs(group)
    )


def empty_track_collection_like(value):
    """Return an empty collection with the same container type when supported."""
    if isinstance(value, deque):
        return deque(maxlen=value.maxlen)
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    if isinstance(value, set):
        return set()
    return None
