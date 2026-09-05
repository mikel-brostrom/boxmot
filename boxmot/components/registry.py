"""Immutable registries for lazily imported component factories."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

FactoryT = TypeVar("FactoryT", bound=Callable[..., Any])

_BACKEND_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")


def _validate_reference(backend: str, reference: object) -> str:
    if not isinstance(reference, str):
        raise TypeError(
            f"Registry target for backend {backend!r} must be a 'module:callable' string, "
            f"not {type(reference).__name__}."
        )
    module_name, separator, attribute = reference.partition(":")
    if (
        separator != ":"
        or ":" in attribute
        or not module_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute.isidentifier()
    ):
        raise ValueError(
            f"Registry target for backend {backend!r} must use 'module:callable' syntax, got {reference!r}."
        )
    return reference


@dataclass(frozen=True, slots=True)
class LazyComponentRegistry(Generic[FactoryT]):
    """Map canonical backend names to factories without importing them eagerly."""

    component: str
    entries: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component or self.component != self.component.strip():
            raise ValueError("component must be a non-empty canonical label.")
        if not isinstance(self.entries, Mapping):
            raise TypeError(f"entries must be a mapping, not {type(self.entries).__name__}.")

        validated: dict[str, str] = {}
        for backend, reference in self.entries.items():
            if not isinstance(backend, str) or _BACKEND_PATTERN.fullmatch(backend) is None:
                raise ValueError(f"Registry backend names must be canonical lowercase identifiers, got {backend!r}.")
            validated[backend] = _validate_reference(backend, reference)
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(validated.items()))))

    def resolve(self, backend: str) -> FactoryT:
        """Import and return the factory registered for ``backend``."""
        if not isinstance(backend, str) or backend not in self.entries:
            available = ", ".join(self.entries) or "(none)"
            raise ValueError(f"Unknown {self.component} backend {backend!r}. Available backends: {available}.")

        reference = self.entries[backend]
        module_name, attribute = reference.split(":", 1)
        target = getattr(import_module(module_name), attribute)
        if not callable(target):
            raise TypeError(f"Registered {self.component} backend {backend!r} target {reference!r} is not callable.")
        return cast(FactoryT, target)


__all__ = ("LazyComponentRegistry",)
