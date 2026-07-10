from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

TRACKER_BACKENDS = frozenset({"python", "cpp"})


@dataclass(frozen=True, slots=True)
class TrackerSpec:
    """Normalized tracker selection.

    ``name`` is the registered tracker id used throughout BoxMOT. ``backend``
    is either ``"python"`` or ``"cpp"``.
    """

    name: str
    backend: str = "python"


def normalize_tracker_backend(backend: Any, *, default: str = "python") -> str:
    """Return a normalized tracker backend identifier."""

    raw_backend = default if backend in {None, ""} else backend
    normalized = str(raw_backend).strip().lower()
    if normalized not in TRACKER_BACKENDS:
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

    Tracker strings must contain only the tracker name. Select the backend with
    the separate ``tracker_backend`` field.
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
        if ":" in raw_spec or "@" in raw_spec:
            raise ValueError(
                "Tracker spec must be a tracker name only. "
                "Set tracker_backend to either 'python' or 'cpp'."
            )

        return TrackerSpec(name=raw_spec.lower(), backend=normalized_default_backend)

    if isinstance(spec, type):
        tracker_backend = normalize_tracker_backend(
            getattr(spec, "tracker_backend", None),
            default=normalized_default_backend,
        )
        if class_to_name is not None:
            tracker_name = class_to_name.get(spec.__name__.lower())
            if tracker_name is not None:
                return TrackerSpec(name=tracker_name, backend=tracker_backend)

        tracker_name = getattr(spec, "tracker_name", None)
        if tracker_name is not None:
            return TrackerSpec(name=str(tracker_name).strip().lower(), backend=tracker_backend)

        raise ValueError("Could not infer a registered tracker name from the provided tracker class.")

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
