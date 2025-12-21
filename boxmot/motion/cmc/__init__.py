# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Iterable, Mapping, Optional, Type

from boxmot.motion.cmc.base_cmc import BaseCMC


def _normalize(name: str) -> str:
    """Normalize user input to a canonical registry key."""
    return name.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class _LazyLoader:
    """Lazily import and return a CMC class by module and attribute name."""
    module: str
    attr: str

    def __call__(self) -> Type[BaseCMC]:
        mod = import_module(self.module)
        cls = getattr(mod, self.attr)
        # Optional: basic sanity check to fail fast if registry is misconfigured.
        if not issubclass(cls, BaseCMC):
            raise TypeError(f"{self.module}.{self.attr} is not a BaseCMC subclass.")
        return cls


# Registry of known methods (lazy-loaded).
_CMC_REGISTRY: Mapping[str, Callable[[], Type[BaseCMC]]] = {
    "ecc": _LazyLoader("boxmot.motion.cmc.ecc", "ECC"),
    "orb": _LazyLoader("boxmot.motion.cmc.orb", "ORB"),
    "sof": _LazyLoader("boxmot.motion.cmc.sof", "SOF"),
    "sift": _LazyLoader("boxmot.motion.cmc.sift", "SIFT"),
}


def available_cmc_methods() -> tuple[str, ...]:
    """Return the list of supported CMC method keys."""
    return tuple(sorted(_CMC_REGISTRY.keys()))


def get_cmc_method(name: Optional[str]) -> Optional[Type[BaseCMC]]:
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


def create_cmc(name: Optional[str], /, **kwargs) -> Optional[BaseCMC]:
    """
    Convenience factory: create and return an instance.
    """
    cls = get_cmc_method(name)
    return None if cls is None else cls(**kwargs)
