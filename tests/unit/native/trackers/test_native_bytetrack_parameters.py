"""ByteTrack's native confidence and duplicate controls match Python behavior."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from boxmot import ByteTrack, ByteTrackConfig
from boxmot.native.trackers import bytetrack as native_binding
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker
from boxmot.trackers.common.config import load_tracker_defaults


@pytest.fixture(scope="module")
def library() -> native_binding.ByteTrackLibrary:
    """Compile and reuse the native library without model inference."""
    return native_binding.ByteTrackLibrary(native_binding.ensure_bytetrack_cpp_library())


def _rows(is_obb: bool, *, confidence: float = 0.99, offset: float = 0.0) -> np.ndarray:
    """Use the same stationary box in either supported representation."""
    geometry = [20 + offset, 20, 20, 20, 0.0] if is_obb else [10 + offset, 10, 30 + offset, 30]
    return np.asarray([[*geometry, confidence, 0]], dtype=np.float32)


def _compare_sequence(
    library: native_binding.ByteTrackLibrary,
    is_obb: bool,
    sequence: Sequence[np.ndarray],
    **options: object,
) -> list[np.ndarray]:
    """Compare actual tracks while exercising config propagation and reset."""
    python = ByteTrack(ByteTrackConfig(**options), is_obb=is_obb)
    native = NativeByteTrackTracker(options, geometry="obb" if is_obb else "aabb", library=library)
    outputs: list[np.ndarray] = []
    try:
        for replay in range(2):
            if replay:
                python.reset()
                native.reset()
            outputs = []
            native_ids: dict[float, float] = {}
            python_ids: dict[float, float] = {}
            id_column = 5 if is_obb else 4
            for detections in sequence:
                expected = python.update(detections)
                actual = native.update(detections)
                assert actual.shape == expected.shape
                np.testing.assert_allclose(actual[:, :id_column], expected[:, :id_column], atol=1e-5)
                np.testing.assert_array_equal(actual[:, id_column + 1 :], expected[:, id_column + 1 :])
                # The native allocator starts at one; Python starts at zero.
                for native_id, python_id in zip(actual[:, id_column], expected[:, id_column], strict=True):
                    assert native_ids.setdefault(native_id, python_id) == python_id
                    assert python_ids.setdefault(python_id, native_id) == native_id
                outputs.append(actual)
    finally:
        native.close()
    return outputs


@pytest.mark.parametrize("is_obb", [False, True], ids=["aabb", "obb"])
@pytest.mark.parametrize("fuse", [False, True])
def test_first_pass_confidence_fusion_changes_native_assignments(
    library: native_binding.ByteTrackLibrary, is_obb: bool, fuse: bool,
) -> None:
    """Low-confidence identical geometry matches only when score fusion is disabled."""
    outputs = _compare_sequence(
        library, is_obb, [_rows(is_obb), _rows(is_obb, confidence=0.65)],
        track_thresh=0.5, match_thresh=0.3, fuse_first_associate=fuse, duplicate_distance_thresh=0.0,
    )
    assert len(outputs[-1]) == (0 if fuse else 1)
    if not fuse:
        id_column = 5 if is_obb else 4
        assert outputs[-1][0, id_column] == outputs[0][0, id_column]


@pytest.mark.parametrize("is_obb", [False, True], ids=["aabb", "obb"])
@pytest.mark.parametrize("fuse", [False, True])
def test_tentative_confidence_fusion_changes_native_confirmation(
    library: native_binding.ByteTrackLibrary, is_obb: bool, fuse: bool,
) -> None:
    """Later-born tracks use the independently configured tentative fusion policy."""
    outputs = _compare_sequence(
        library, is_obb,
        [np.empty((0, 7 if is_obb else 6), dtype=np.float32), _rows(is_obb), _rows(is_obb, confidence=0.65)],
        track_thresh=0.5, unconfirmed_match_thresh=0.3, fuse_unconfirmed_associate=fuse,
    )
    assert len(outputs[1]) == 0
    assert len(outputs[-1]) == (0 if fuse else 1)


@pytest.mark.parametrize("is_obb", [False, True], ids=["aabb", "obb"])
@pytest.mark.parametrize("threshold, survivors", [(0.0, 1), (0.1, 1), (0.3, 0)])
def test_duplicate_distance_changes_native_active_lost_suppression(
    library: native_binding.ByteTrackLibrary, is_obb: bool, threshold: float, survivors: int,
) -> None:
    """An unmatched nearby detection confirms only when duplicate suppression permits it."""
    outputs = _compare_sequence(
        library, is_obb, [_rows(is_obb), _rows(is_obb, offset=2.0), _rows(is_obb, offset=2.0)],
        match_thresh=0.0, duplicate_distance_thresh=threshold,
    )
    assert len(outputs[-1]) == survivors


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_c_api_rejects_invalid_duplicate_distance(
    library: native_binding.ByteTrackLibrary, threshold: float,
) -> None:
    """Direct ABI callers receive validation even when they bypass typed configs."""
    cfg = {**load_tracker_defaults("bytetrack"), "duplicate_distance_thresh": threshold}
    with pytest.raises(RuntimeError, match="duplicate_distance_thresh"):
        library.create(cfg)
