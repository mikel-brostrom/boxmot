# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.trackers.common.motion.cmc.base import BaseCMC


def _normalize(name: str) -> str:
    """Normalize user input to a canonical registry key."""
    return name.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class _LazyLoader:
    """Lazily import and return a CMC class by module and attribute name."""
    module: str
    attr: str

    def __call__(self) -> type[BaseCMC]:
        from boxmot.trackers.common.motion.cmc.base import BaseCMC

        mod = import_module(self.module)
        cls = getattr(mod, self.attr)
        # Optional: basic sanity check to fail fast if registry is misconfigured.
        if not issubclass(cls, BaseCMC):
            raise TypeError(f"{self.module}.{self.attr} is not a BaseCMC subclass.")
        return cls


# Registry of known methods (lazy-loaded).
_CMC_REGISTRY: Mapping[str, Callable[[], type[BaseCMC]]] = {
    "ecc": _LazyLoader("boxmot.trackers.common.motion.cmc.ecc", "ECC"),
    "orb": _LazyLoader("boxmot.trackers.common.motion.cmc.orb", "ORB"),
    "sof": _LazyLoader("boxmot.trackers.common.motion.cmc.sof", "SOF"),
    "sift": _LazyLoader("boxmot.trackers.common.motion.cmc.sift", "SIFT"),
}


def available_cmc_methods() -> tuple[str, ...]:
    """Return the list of supported CMC method keys."""
    return tuple(sorted(_CMC_REGISTRY.keys()))


def get_cmc_method(name: str | None) -> type[BaseCMC] | None:
    """
    Resolve a CMC method name to its class.

    Returns None only when name is None (useful for "disabled" configs).
    Raises ValueError for unknown non-None names to fail fast and clearly.
    """
    if name is None:
        return None

    key = _normalize(name)
    loader = _CMC_REGISTRY.get(key)
    if loader is None:
        raise ValueError(
            f"Unknown cmc_method={name!r}. "
            f"Supported values: {', '.join(available_cmc_methods())}"
        )
    return loader()


def create_cmc(method: str | None, *, enabled: bool = True, **kwargs) -> BaseCMC | None:
    """Create a CMC estimator or return ``None`` for disabled CMC."""
    if not enabled:
        return None
    cls = get_cmc_method(method)
    return None if cls is None else cls(**kwargs)
