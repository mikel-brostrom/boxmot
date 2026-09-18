"""Lightweight validation for ordered evaluation postprocessing stages."""

from __future__ import annotations

from collections.abc import Sequence

POSTPROCESSING_METHODS = ("gsi", "gbrc", "gta")


def normalize_postprocessing(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return requested stages in application order, rejecting ambiguous pipelines.

    GTA consumes original observations and their cached embeddings. GSI and
    GBRC modify geometry and insert synthetic observations, so GTA runs first.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        methods = (value,)
    elif isinstance(value, Sequence):
        methods = tuple(value)
    else:
        raise ValueError("postprocessing must be a method name or an ordered sequence of method names.")
    for method in methods:
        if not isinstance(method, str) or method not in POSTPROCESSING_METHODS:
            raise ValueError(f"Unknown postprocessing method {method!r}; choose from gsi, gbrc, gta.")
    if len(set(methods)) != len(methods):
        raise ValueError("Each postprocessing method may be selected only once.")
    if "gta" in methods and methods[0] != "gta":
        raise ValueError(
            "GTA must run before GSI or GBRC: select --postprocessing gta first, "
            "so association uses original observations before interpolation and smoothing."
        )
    return methods
