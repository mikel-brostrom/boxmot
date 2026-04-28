from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from boxmot.utils.misc import dataclass_slots_kwargs


@dataclass(frozen=True, **dataclass_slots_kwargs())
class NativeReplayBackend:
    """Registered native replay backend for a tracker."""

    tracker_name: str
    process_sequence: Callable[..., tuple[str, list[int], dict[str, Any]]]
    ensure_built: Callable[..., Path]
    source_dir: Path


@dataclass(frozen=True, **dataclass_slots_kwargs())
class NativeLiveBackend:
    """Registered native live-tracking backend for a tracker."""

    tracker_name: str
    create_tracker: Callable[..., Any]
    ensure_built: Callable[..., Path]
    source_dir: Path


def _load_botsort_backend() -> NativeReplayBackend:
    from boxmot.native.botsort_cpp import ensure_botsort_cpp_executable, process_sequence_cpp

    return NativeReplayBackend(
        tracker_name="botsort",
        process_sequence=process_sequence_cpp,
        ensure_built=ensure_botsort_cpp_executable,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "botsort",
    )


def _load_botsort_live_backend() -> NativeLiveBackend:
    from boxmot.native.botsort_cpp import create_botsort_live_tracker, ensure_botsort_cpp_library

    return NativeLiveBackend(
        tracker_name="botsort",
        create_tracker=create_botsort_live_tracker,
        ensure_built=ensure_botsort_cpp_library,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "botsort",
    )


def _load_bytetrack_backend() -> NativeReplayBackend:
    from boxmot.native.bytetrack_cpp import ensure_bytetrack_cpp_executable, process_sequence_cpp

    return NativeReplayBackend(
        tracker_name="bytetrack",
        process_sequence=process_sequence_cpp,
        ensure_built=ensure_bytetrack_cpp_executable,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "bytetrack",
    )


def _load_bytetrack_live_backend() -> NativeLiveBackend:
    from boxmot.native.bytetrack_cpp import create_bytetrack_live_tracker, ensure_bytetrack_cpp_library

    return NativeLiveBackend(
        tracker_name="bytetrack",
        create_tracker=create_bytetrack_live_tracker,
        ensure_built=ensure_bytetrack_cpp_library,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "bytetrack",
    )


def _load_sfsort_backend() -> NativeReplayBackend:
    from boxmot.native.sfsort_cpp import ensure_sfsort_cpp_executable, process_sequence_cpp

    return NativeReplayBackend(
        tracker_name="sfsort",
        process_sequence=process_sequence_cpp,
        ensure_built=ensure_sfsort_cpp_executable,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "sfsort",
    )


def _load_sfsort_live_backend() -> NativeLiveBackend:
    from boxmot.native.sfsort_cpp import create_sfsort_live_tracker, ensure_sfsort_cpp_library

    return NativeLiveBackend(
        tracker_name="sfsort",
        create_tracker=create_sfsort_live_tracker,
        ensure_built=ensure_sfsort_cpp_library,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "sfsort",
    )


def _load_occluboost_backend() -> NativeReplayBackend:
    from boxmot.native.occluboost_cpp import ensure_occluboost_cpp_executable, process_sequence_cpp

    return NativeReplayBackend(
        tracker_name="occluboost",
        process_sequence=process_sequence_cpp,
        ensure_built=ensure_occluboost_cpp_executable,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "occluboost",
    )


def _load_occluboost_live_backend() -> NativeLiveBackend:
    from boxmot.native.occluboost_cpp import create_occluboost_live_tracker, ensure_occluboost_cpp_library

    return NativeLiveBackend(
        tracker_name="occluboost",
        create_tracker=create_occluboost_live_tracker,
        ensure_built=ensure_occluboost_cpp_library,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "occluboost",
    )


def _load_ocsort_backend() -> NativeReplayBackend:
    from boxmot.native.ocsort_cpp import ensure_ocsort_cpp_executable, process_sequence_cpp

    return NativeReplayBackend(
        tracker_name="ocsort",
        process_sequence=process_sequence_cpp,
        ensure_built=ensure_ocsort_cpp_executable,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "ocsort",
    )


def _load_ocsort_live_backend() -> NativeLiveBackend:
    from boxmot.native.ocsort_cpp import create_ocsort_live_tracker, ensure_ocsort_cpp_library

    return NativeLiveBackend(
        tracker_name="ocsort",
        create_tracker=create_ocsort_live_tracker,
        ensure_built=ensure_ocsort_cpp_library,
        source_dir=Path(__file__).resolve().parents[2] / "native" / "trackers" / "ocsort",
    )


_NATIVE_REPLAY_BACKENDS = {
    "botsort": _load_botsort_backend,
    "bytetrack": _load_bytetrack_backend,
    "occluboost": _load_occluboost_backend,
    "ocsort": _load_ocsort_backend,
    "sfsort": _load_sfsort_backend,
}

_NATIVE_LIVE_BACKENDS = {
    "botsort": _load_botsort_live_backend,
    "bytetrack": _load_bytetrack_live_backend,
    "occluboost": _load_occluboost_live_backend,
    "ocsort": _load_ocsort_live_backend,
    "sfsort": _load_sfsort_live_backend,
}


def supported_native_replay_trackers() -> tuple[str, ...]:
    return tuple(sorted(_NATIVE_REPLAY_BACKENDS))


def supported_native_live_trackers() -> tuple[str, ...]:
    return tuple(sorted(_NATIVE_LIVE_BACKENDS))


def has_native_replay_backend(tracker_name: str) -> bool:
    return str(tracker_name).strip().lower() in _NATIVE_REPLAY_BACKENDS


def has_native_live_backend(tracker_name: str) -> bool:
    return str(tracker_name).strip().lower() in _NATIVE_LIVE_BACKENDS


def get_native_replay_backend(tracker_name: str) -> NativeReplayBackend:
    normalized_name = str(tracker_name).strip().lower()
    factory = _NATIVE_REPLAY_BACKENDS.get(normalized_name)
    if factory is None:
        available = ", ".join(supported_native_replay_trackers())
        raise ValueError(
            f"tracker_backend='cpp' is not available for tracker='{normalized_name}'. "
            f"Available native replay trackers: {available}"
        )
    return factory()


def get_native_live_backend(tracker_name: str) -> NativeLiveBackend:
    normalized_name = str(tracker_name).strip().lower()
    factory = _NATIVE_LIVE_BACKENDS.get(normalized_name)
    if factory is None:
        available = ", ".join(supported_native_live_trackers())
        raise ValueError(
            f"tracker_backend='cpp' is not available for live tracker='{normalized_name}'. "
            f"Available native live trackers: {available}"
        )
    return factory()


__all__ = (
    "NativeLiveBackend",
    "NativeReplayBackend",
    "get_native_live_backend",
    "get_native_replay_backend",
    "has_native_live_backend",
    "has_native_replay_backend",
    "supported_native_live_trackers",
    "supported_native_replay_trackers",
)
