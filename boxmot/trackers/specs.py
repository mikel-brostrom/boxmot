from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias

from boxmot.structures import GeometryKind

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...]

TRACKER_BACKENDS = frozenset({"python", "cpp"})
_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
_OPTION_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class TrackerFamily(str, Enum):
    """Primary representation maintained by a tracker."""

    BOX = "box"
    MASK = "mask"
    MULTIMODAL = "multimodal"

    def __str__(self) -> str:
        return self.value


GEOMETRY_MODES = frozenset(kind.value for kind in GeometryKind)


@dataclass(frozen=True, slots=True)
class TrackerCapabilities:
    """Static input and representation capabilities of a tracker algorithm.

    ``requires_*`` records requirements shared by every supported
    configuration. Optional configuration-dependent needs are expressed by
    ``accepts_*`` here and by :class:`TrackerRequirements` on the resolved
    tracker instance.
    """

    family: TrackerFamily
    geometry_kinds: frozenset[GeometryKind]
    requires_embeddings: bool = False
    accepts_embeddings: bool = False
    requires_masks: bool = False
    accepts_masks: bool = False
    requires_frame: bool = False
    accepts_frame: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.family, TrackerFamily):
            raise TypeError("TrackerCapabilities.family must be a TrackerFamily.")
        if not isinstance(self.geometry_kinds, frozenset):
            raise TypeError("TrackerCapabilities.geometry_kinds must be a frozenset.")
        if not self.geometry_kinds:
            raise ValueError("TrackerCapabilities.geometry_kinds must not be empty.")
        if any(not isinstance(kind, GeometryKind) for kind in self.geometry_kinds):
            raise TypeError("TrackerCapabilities.geometry_kinds must contain only GeometryKind values.")

        for input_name in ("embeddings", "masks", "frame"):
            requires = getattr(self, f"requires_{input_name}")
            accepts = getattr(self, f"accepts_{input_name}")
            if not isinstance(requires, bool):
                raise TypeError(f"TrackerCapabilities.requires_{input_name} must be bool.")
            if not isinstance(accepts, bool):
                raise TypeError(f"TrackerCapabilities.accepts_{input_name} must be bool.")
            if requires and not accepts:
                raise ValueError(f"TrackerCapabilities requiring {input_name} must also accept {input_name}.")


def _validate_immutable_json_value(value: JSONValue, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers.")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_immutable_json_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} must contain only immutable JSON values, not {type(value).__name__}.")


def _validate_options(options: tuple[tuple[str, JSONValue], ...]) -> None:
    if not isinstance(options, tuple):
        raise TypeError("TrackerSpec.options must be a tuple of key/value pairs")
    keys: list[str] = []
    for index, entry in enumerate(options):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"TrackerSpec.options[{index}] must be a (key, value) tuple.")
        key, value = entry
        if not isinstance(key, str) or _OPTION_PATTERN.fullmatch(key) is None:
            raise ValueError(f"TrackerSpec.options[{index}] key must be a canonical lowercase identifier.")
        keys.append(key)
        _validate_immutable_json_value(value, path=f"Tracker option {key!r}")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("TrackerSpec.options keys must be unique and sorted")


@dataclass(frozen=True, slots=True)
class TrackerSpec:
    """Normalized tracker selection.

    The spec contains only algorithm selection and immutable configuration.
    Model construction belongs to detector, segmentor, or appearance encoder
    factories and is deliberately absent here.
    """

    name: str
    backend: str = "python"
    geometry: str = "aabb"
    per_class: bool = False
    class_ids: tuple[int, ...] | None = None
    class_names: tuple[tuple[int, str], ...] = ()
    options: tuple[tuple[str, JSONValue], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _IDENTIFIER_PATTERN.fullmatch(self.name) is None:
            raise ValueError(f"Tracker name must use its canonical lowercase identifier: {self.name!r}")
        if not isinstance(self.backend, str):
            raise ValueError(f"Unknown tracker backend: {self.backend!r}.")
        normalize_tracker_backend(self.backend)
        if not isinstance(self.geometry, str) or self.geometry not in GEOMETRY_MODES:
            available = ", ".join(sorted(GEOMETRY_MODES))
            raise ValueError(f"Unknown tracker geometry {self.geometry!r}. Available modes are: {available}")
        if not isinstance(self.per_class, bool):
            raise TypeError("TrackerSpec.per_class must be bool")
        if self.class_ids is not None:
            if not isinstance(self.class_ids, tuple) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.class_ids
            ):
                raise TypeError("TrackerSpec.class_ids must be a tuple of non-negative integers or None")
            if tuple(sorted(set(self.class_ids))) != self.class_ids:
                raise ValueError("TrackerSpec.class_ids must be unique and sorted")
        if not isinstance(self.class_names, tuple):
            raise TypeError("TrackerSpec.class_names must be a tuple of (id, name) pairs")
        for index, entry in enumerate(self.class_names):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"TrackerSpec.class_names[{index}] must be an (id, name) tuple")
            class_id, name = entry
            if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
                raise TypeError("TrackerSpec class-name IDs must be non-negative integers")
            if not isinstance(name, str) or not name:
                raise TypeError("TrackerSpec class names must be non-empty strings")
        if tuple(sorted(self.class_names)) != self.class_names or len({key for key, _ in self.class_names}) != len(
            self.class_names
        ):
            raise ValueError("TrackerSpec.class_names must have unique IDs and be sorted")
        _validate_options(self.options)

    @property
    def option_dict(self) -> dict[str, JSONValue]:
        """Return a fresh mutable mapping for constructor dispatch."""

        return dict(self.options)


def normalize_tracker_backend(backend: Any, *, default: str = "python") -> str:
    """Return a canonical tracker backend identifier."""

    raw_backend = default if backend is None else backend
    if not isinstance(raw_backend, str) or raw_backend not in TRACKER_BACKENDS:
        available = ", ".join(sorted(TRACKER_BACKENDS))
        raise ValueError(f"Unknown tracker backend: {backend!r}. Available backends are: {available}")
    return raw_backend


def parse_tracker_spec(
    spec: Any,
    *,
    default_backend: str = "python",
    class_specs: Mapping[str, TrackerSpec] | None = None,
) -> TrackerSpec:
    """Parse tracker spec strings and tracker instances into a normalized form.

    Tracker strings must contain only the tracker name. Select the backend with
    the separate ``tracker_backend`` field.
    """

    normalized_default_backend = normalize_tracker_backend(default_backend)

    if isinstance(spec, TrackerSpec):
        tracker_name = str(spec.name)
        if tracker_name != tracker_name.strip().lower():
            raise ValueError(f"Tracker name must use its canonical lowercase identifier: {tracker_name!r}")
        return TrackerSpec(
            name=tracker_name,
            backend=normalize_tracker_backend(spec.backend, default=normalized_default_backend),
            geometry=spec.geometry,
            per_class=spec.per_class,
            class_ids=spec.class_ids,
            class_names=spec.class_names,
            options=spec.options,
        )

    if isinstance(spec, str):
        if not spec:
            raise ValueError("Tracker spec cannot be empty.")
        if spec != spec.strip().lower():
            raise ValueError(f"Tracker name must use its canonical lowercase identifier: {spec!r}")
        if ":" in spec or "@" in spec:
            raise ValueError(
                "Tracker spec must be a tracker name only. Set tracker_backend to either 'python' or 'cpp'."
            )

        return TrackerSpec(name=spec, backend=normalized_default_backend)

    tracker_class = spec if isinstance(spec, type) else type(spec)
    class_path = f"{tracker_class.__module__}.{tracker_class.__qualname__}"
    registered_spec = class_specs.get(class_path) if class_specs is not None else None
    if registered_spec is None:
        kind = "class" if isinstance(spec, type) else "instance"
        raise ValueError(f"The provided tracker {kind} is not registered: {class_path}")

    return registered_spec


__all__ = (
    "GEOMETRY_MODES",
    "GeometryKind",
    "JSONScalar",
    "JSONValue",
    "TRACKER_BACKENDS",
    "TrackerCapabilities",
    "TrackerFamily",
    "TrackerSpec",
    "normalize_tracker_backend",
    "parse_tracker_spec",
)
