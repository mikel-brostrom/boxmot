"""Immutable algorithm settings shared by the concrete tracker schemas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from functools import lru_cache
from numbers import Integral, Real
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

_ConfigT = TypeVar("_ConfigT", bound="TrackerConfig")


@lru_cache(maxsize=None)
def _field_types(config_type: type[TrackerConfig]) -> dict[str, Any]:
    """Resolve postponed annotations once for each concrete schema."""
    return get_type_hints(config_type)


def _matches_type(value: object, annotation: Any) -> bool:
    """Check scalar configuration types without accepting booleans as numbers."""
    if get_origin(annotation) in (Union, UnionType):
        return any(_matches_type(value, member) for member in get_args(annotation))
    if annotation is float:
        return isinstance(value, Real) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, Integral) and not isinstance(value, bool)
    return isinstance(value, annotation)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackerConfig:
    """Shared algorithm settings, independent of models and input metadata.

    Args:
        max_age: Missing-frame lifetime, or shared display history for trackers
            with their own buffer or regional expiry rules.
        max_obs: Number of observations retained in track history. The runtime
            can increase this to retain a complete ``max_age`` interval.
        min_hits: Confirmation hits, or shared display filtering for trackers
            with separate activation rules.
        iou_threshold: Geometric association threshold. Its useful range depends
            on the selected similarity; values above one can disable matching.
        asso_func: Canonical geometry similarity used during association.

    Each concrete schema owns its defaults. YAML presets and serialized specs
    supply overrides through ``from_mapping`` rather than defining defaults.
    """

    max_age: int = field(default=30, metadata={"ge": 0})
    max_obs: int = field(default=50, metadata={"ge": 1})
    min_hits: int = field(default=3, metadata={"ge": 0})
    iou_threshold: float = 0.3
    asso_func: str = field(
        default="iou",
        metadata={"choices": ("iou", "giou", "diou", "ciou", "hmiou", "centroid")},
    )

    def __post_init__(self) -> None:
        """Validate scalar types, finite numbers, bounds, and enum choices."""
        annotations = _field_types(type(self))
        for setting in dataclass_fields(self):
            value = getattr(self, setting.name)
            if value is None and setting.metadata.get("none_uses_default", False):
                value = setting.default
                object.__setattr__(self, setting.name, value)
            annotation = annotations[setting.name]
            if not _matches_type(value, annotation):
                raise TypeError(f"{setting.name} must have type {annotation}, got {type(value).__name__}.")
            if value is None:
                continue
            if isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"{setting.name} must be finite.")
                if type(value) not in (int, float):
                    value = int(value) if isinstance(value, Integral) else float(value)
                    object.__setattr__(self, setting.name, value)
                for bound, compare, symbol in (
                    ("ge", lambda left, right: left >= right, ">="),
                    ("gt", lambda left, right: left > right, ">"),
                    ("le", lambda left, right: left <= right, "<="),
                ):
                    if bound in setting.metadata and not compare(value, setting.metadata[bound]):
                        raise ValueError(f"{setting.name} must be {symbol} {setting.metadata[bound]}.")
            if "choices" in setting.metadata and value not in setting.metadata["choices"]:
                raise ValueError(f"{setting.name} must be one of {setting.metadata['choices']!r}.")

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the canonical parameter names accepted by this schema."""
        return tuple(setting.name for setting in dataclass_fields(cls))

    @classmethod
    def from_mapping(cls: type[_ConfigT], values: Mapping[str, object]) -> _ConfigT:
        """Resolve serialized overrides using the schema's canonical defaults."""
        if not isinstance(values, Mapping):
            raise TypeError(f"{cls.__name__}.from_mapping requires a mapping.")
        unknown = set(values).difference(cls.fields())
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise TypeError(f"Unknown {cls.__name__} parameter(s): {names}.")
        return cls(**dict(values))

    @classmethod
    def resolve(cls: type[_ConfigT], config: _ConfigT | None) -> _ConfigT:
        """Choose defaults or validate the concrete configuration's identity."""
        if config is None:
            return cls()
        if type(config) is not cls:
            raise TypeError(f"config must be a {cls.__name__} object or None.")
        return config

    def to_dict(self) -> dict[str, object]:
        """Return detached scalar settings suitable for specs and serialization."""
        return {name: getattr(self, name) for name in self.fields()}


__all__ = ("TrackerConfig",)
