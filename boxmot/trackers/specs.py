from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from boxmot.utils.misc import dataclass_slots_kwargs

TRACKER_BACKENDS = frozenset({"python", "cpp"})
_TRACKER_BACKEND_ALIASES = {
    "py": "python",
    "python": "python",
    "c++": "cpp",
    "cpp": "cpp",
    "native": "cpp",
}


@dataclass(frozen=True, **dataclass_slots_kwargs())
class TrackerSpec:
    """Normalized tracker selection.

    ``name`` is the registered tracker id used throughout BoxMOT. ``backend``
    selects whether that tracker should run through the Python implementation or
    a native C++ implementation when one exists.
    """

    name: str
    backend: str = "python"


def normalize_tracker_backend(backend: Any, *, default: str = "python") -> str:
    """Return a normalized tracker backend identifier."""

    raw_backend = default if backend in {None, ""} else backend
    normalized = _TRACKER_BACKEND_ALIASES.get(str(raw_backend).strip().lower())
    if normalized is None:
        available = ", ".join(sorted(TRACKER_BACKENDS))
        raise ValueError(f"Unknown tracker backend: {backend!r}. Available backends are: {available}")
    return normalized


def parse_tracker_spec(
    spec: Any,
    *,
    default_backend: str = "python",
    class_to_name: Mapping[str, str] | None = None,
) -> TrackerSpec:
    """Parse tracker spec strings and tracker instances into a normalized form.

    Supported string forms are:

    - ``bytetrack``
    - ``bytetrack:cpp``
    - ``cpp:bytetrack``
    - ``bytetrack@cpp``
    """

    normalized_default_backend = normalize_tracker_backend(default_backend)

    if isinstance(spec, TrackerSpec):
        return TrackerSpec(
            name=str(spec.name).strip().lower(),
            backend=normalize_tracker_backend(spec.backend, default=normalized_default_backend),
        )

    if isinstance(spec, str):
        raw_spec = spec.strip()
        if not raw_spec:
            raise ValueError("Tracker spec cannot be empty.")

        for separator in ("@", ":"):
            if separator not in raw_spec:
                continue
            left, right = (part.strip() for part in raw_spec.split(separator, 1))
            if not left or not right:
                break

            left_backend = _TRACKER_BACKEND_ALIASES.get(left.lower())
            right_backend = _TRACKER_BACKEND_ALIASES.get(right.lower())
            if left_backend is not None:
                return TrackerSpec(name=right.lower(), backend=left_backend)
            if right_backend is not None:
                return TrackerSpec(name=left.lower(), backend=right_backend)
            break

        return TrackerSpec(name=raw_spec.lower(), backend=normalized_default_backend)

    tracker_backend = normalize_tracker_backend(
        getattr(spec, "tracker_backend", None),
        default=normalized_default_backend,
    )

    if class_to_name is not None and spec is not None:
        class_name = spec.__class__.__name__.lower()
        tracker_name = class_to_name.get(class_name)
        if tracker_name is not None:
            return TrackerSpec(name=tracker_name, backend=tracker_backend)

    tracker_name = getattr(spec, "tracker_name", None)
    if tracker_name is not None:
        return TrackerSpec(name=str(tracker_name).strip().lower(), backend=tracker_backend)

    raise ValueError("Could not infer a registered tracker name from the provided tracker spec.")


__all__ = (
    "TRACKER_BACKENDS",
    "TrackerSpec",
    "normalize_tracker_backend",
    "parse_tracker_spec",
)
