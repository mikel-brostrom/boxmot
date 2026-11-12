"""Optional visual attribute extraction, imported only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import AttributeConfig
    from .models import AttributeResult, AttributeSample
    from .session import AttributeBackend, AttributeEnricher, create_attribute_enricher

_EXPORTS = {
    "AttributeConfig": ".config",
    "AttributeResult": ".models",
    "AttributeSample": ".models",
    "AttributeBackend": ".session",
    "AttributeEnricher": ".session",
    "create_attribute_enricher": ".session",
}

__all__ = (
    "AttributeConfig",
    "AttributeResult",
    "AttributeSample",
    "AttributeBackend",
    "AttributeEnricher",
    "create_attribute_enricher",
)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
