from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Literal

import numpy as np


class TrackState(Enum):
    """Canonical lifecycle states shared by bbox tracker infrastructure."""

    TENTATIVE = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


@dataclass
class TrackMeta:
    """Canonical metadata shared by tracker-local track implementations."""

    id: int
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    start_frame: int = 0
    frame_id: int = 0
    conf: float = 0.0
    cls: int = -1
    det_ind: int = -1
    is_activated: bool = False
    lost_region: Literal["central", "marginal"] | None = None


class TrackIdAllocator:
    """Instance-owned monotonically increasing track ID allocator."""

    def __init__(self, start: int = 0) -> None:
        self.next_id = int(start)

    def alloc(self) -> int:
        track_id = self.next_id
        self.next_id += 1
        return track_id

    def reset(self, start: int = 0) -> None:
        self.next_id = int(start)


class TrackLifecycleMixin:
    """Shared lifecycle boilerplate for tracker-local track classes."""

    track_id: ClassVar[int] = 0
    is_activated: ClassVar[bool] = False
    state: ClassVar[object | None] = None

    history: ClassVar[OrderedDict] = OrderedDict()
    features: ClassVar[list] = []
    curr_feature: ClassVar[np.ndarray | None] = None
    conf: ClassVar[float] = 0.0
    score: ClassVar[float] = 0.0
    start_frame: ClassVar[int] = 0
    frame_id: ClassVar[int] = 0
    time_since_update: ClassVar[int] = 0
    location: ClassVar[tuple[float, float]] = (np.inf, np.inf)

    lost_state: ClassVar[object | None] = None
    long_lost_state: ClassVar[object | None] = None
    removed_state: ClassVar[object | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "history" not in cls.__dict__:
            cls.history = OrderedDict()
        if "features" not in cls.__dict__:
            cls.features = []

    @property
    def end_frame(self) -> int:
        return int(self.frame_id)

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def _set_lifecycle_state(
        self,
        local_state_attr: str,
        common_state: TrackState,
    ) -> None:
        local_state = getattr(self, local_state_attr, None)
        if local_state is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not define {local_state_attr}")
        self.state = local_state
        sync_track_meta(self, common_state)

    def mark_lost(self) -> None:
        self._set_lifecycle_state("lost_state", TrackState.LOST)

    def mark_long_lost(self) -> None:
        self._set_lifecycle_state("long_lost_state", TrackState.LOST)

    def mark_removed(self) -> None:
        self._set_lifecycle_state("removed_state", TrackState.REMOVED)


def _track_meta_id(track) -> int:
    if hasattr(track, "id"):
        return int(getattr(track, "id"))
    if hasattr(track, "track_id"):
        return int(getattr(track, "track_id"))
    raise AttributeError(f"{track.__class__.__name__} does not expose an id")


def sync_track_meta(track, state: TrackState | None = None) -> TrackMeta:
    """Create or refresh canonical metadata for a tracker-local track object."""
    meta = getattr(track, "meta", None)
    if meta is None:
        meta = TrackMeta(id=_track_meta_id(track))
        track.meta = meta

    if state is not None:
        meta.state = state

    for attr_name in (
        "id",
        "age",
        "hits",
        "hit_streak",
        "time_since_update",
        "start_frame",
        "frame_id",
        "conf",
        "cls",
        "det_ind",
        "is_activated",
        "lost_region",
    ):
        if hasattr(track, attr_name):
            setattr(meta, attr_name, getattr(track, attr_name))

    return meta
