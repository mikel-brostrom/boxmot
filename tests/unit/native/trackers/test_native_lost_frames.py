"""Validate the unscaled lost-track lifetime through the compiled tracker ABIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig, ByteTrack, ByteTrackConfig
from boxmot.native.trackers import botsort as botsort_binding
from boxmot.native.trackers import bytetrack as bytetrack_binding
from boxmot.trackers.botsort.native import NativeBotSortTracker
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker

from ._helpers import update_rows


@pytest.fixture(params=["botsort", "bytetrack"])
def tracker_implementations(request: pytest.FixtureRequest) -> tuple:
    """Pair each typed Python tracker with its native configuration ABI."""
    if request.param == "botsort":
        return (
            botsort_binding,
            botsort_binding.BotSortLibrary,
            botsort_binding.ensure_botsort_cpp_library,
            NativeBotSortTracker,
            BotSort,
            BotSortConfig,
            {"use_embeddings": False, "use_cmc": False},
        )
    return (
        bytetrack_binding,
        bytetrack_binding.ByteTrackLibrary,
        bytetrack_binding.ensure_bytetrack_cpp_library,
        NativeByteTrackTracker,
        ByteTrack,
        ByteTrackConfig,
        {},
    )


def test_native_lifetime_rejects_stale_configuration_abi(
    tracker_implementations: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never pass the shortened configuration struct to an old constructor."""
    binding, library_type, *_ = tracker_implementations
    calls = []
    stale_library = SimpleNamespace(
        **{f"boxmot_{binding._TRACKER_NAME}_create": lambda _config: calls.append("unsafe create")}
    )
    monkeypatch.setattr(binding.ctypes, "CDLL", lambda _path: stale_library)

    with pytest.raises(RuntimeError, match="incompatible configuration ABI; rebuild the native library"):
        library_type(Path("stale-native-library"))

    assert calls == []


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("max_lost_frames", [0, 1, 4])
@pytest.mark.parametrize("expire", [False, True], ids=["retained", "expired"])
def test_native_lifetime_and_reset_match_python(
    tracker_implementations: tuple, geometry: str, max_lost_frames: int, expire: bool
) -> None:
    """Short lifetimes govern identity retention equally before and after reset."""
    _, library_type, ensure_library, native_type, python_type, config_type, options = tracker_implementations
    options = {**options, "max_lost_frames": max_lost_frames}
    python_tracker = python_type(config=config_type(**options), is_obb=geometry == "obb")
    native_tracker = native_type(options, geometry=geometry, library=library_type(ensure_library()))
    box = [40, 40, 80, 60] if geometry == "aabb" else [60, 50, 40, 20, 0.1]
    rows = np.asarray([[*box, 0.95, 0]], dtype=np.float32)
    empty = np.empty((0, rows.shape[1]), dtype=np.float32)

    # Keep the existing lifecycle order: a newly lost track enters the pool at
    # the end of the update, and a removal leaves the pool on the next update.
    missed_updates = max(3, max_lost_frames + 2) if expire else max(1, max_lost_frames)
    try:
        for _ in range(2):
            first_python = update_rows(python_tracker, rows)
            first_native = update_rows(native_tracker, rows)
            assert len(first_python) == len(first_native) == 1
            for _ in range(missed_updates):
                assert len(update_rows(python_tracker, empty)) == 0
                assert len(update_rows(native_tracker, empty)) == 0

            # New tracks born after the first update need one confirmation.
            returned_python = update_rows(python_tracker, rows)
            returned_native = update_rows(native_tracker, rows)
            assert len(returned_python) == len(returned_native) == (0 if expire else 1)
            confirmed_python = update_rows(python_tracker, rows)
            confirmed_native = update_rows(native_tracker, rows)
            assert len(confirmed_python) == len(confirmed_native) == 1
            assert bool(confirmed_python[0, len(box)] != first_python[0, len(box)]) is expire
            assert bool(confirmed_native[0, len(box)] != first_native[0, len(box)]) is expire
            np.testing.assert_allclose(confirmed_native[:, : len(box)], confirmed_python[:, : len(box)], atol=1e-5)
            python_tracker.reset()
            native_tracker.reset()
    finally:
        native_tracker.close()


def test_native_lifetime_rejects_negative_values(tracker_implementations: tuple) -> None:
    """Direct native callers also receive the nonnegative lifetime contract."""
    _, library_type, ensure_library, _, _, config_type, options = tracker_implementations
    config = config_type(**options).to_dict()
    config["max_lost_frames"] = -1
    library = library_type(ensure_library())

    with pytest.raises(RuntimeError, match="max_lost_frames must be nonnegative"):
        library.create(config)
