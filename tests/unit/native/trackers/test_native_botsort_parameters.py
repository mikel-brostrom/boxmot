"""Compiled BotSort preserves configurable appearance and lifecycle behavior."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig
from boxmot.native.trackers.botsort import BotSortLibrary, ensure_botsort_cpp_library
from boxmot.trackers.botsort.native import NativeBotSortTracker

from ._helpers import update_rows


@pytest.fixture(scope="module")
def library() -> BotSortLibrary:
    """Build and load the current native configuration ABI once."""
    return BotSortLibrary(ensure_botsort_cpp_library())


def _rows(is_obb: bool, *, offset: float = 0.0, confidence: float = 0.99) -> np.ndarray:
    """Represent the same square through each supported detection geometry."""
    box = [20 + offset, 20, 20, 20, 0.0] if is_obb else [10 + offset, 10, 30 + offset, 30]
    return np.asarray([[*box, confidence, 0]], dtype=np.float32)


def _pair(library: BotSortLibrary, is_obb: bool, **options: object) -> tuple[BotSort, NativeBotSortTracker]:
    """Use identical resolved algorithm settings without image-model dependencies."""
    config = BotSortConfig.from_mapping({"use_cmc": False, "use_embeddings": False, **options})
    return (
        BotSort(config=config, is_obb=is_obb),
        NativeBotSortTracker(config.to_dict(), geometry="obb" if is_obb else "aabb", library=library),
    )


def _assert_parity(actual: np.ndarray, expected: np.ndarray, is_obb: bool) -> None:
    """Compare identity continuity while accounting for the native one-based allocator."""
    normalized = actual.copy()
    normalized[:, 5 if is_obb else 4] -= 1
    np.testing.assert_allclose(normalized, expected, atol=1e-5)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_native_tentative_fusion_matches_python(library: BotSortLibrary, is_obb: bool, fuse: bool) -> None:
    """Low confidence blocks tentative confirmation only when score fusion is enabled."""
    python, native = _pair(
        library, is_obb, track_high_thresh=0.5, new_track_thresh=0.5,
        unconfirmed_match_thresh=0.3, fuse_unconfirmed_associate=fuse,
    )
    frames = (
        np.empty((0, 7 if is_obb else 6), dtype=np.float32),
        _rows(is_obb),
        _rows(is_obb, confidence=0.65),
    )
    try:
        for rows in frames:
            expected, actual = update_rows(python, rows), update_rows(native, rows)
            _assert_parity(actual, expected, is_obb)
        assert len(actual) == (0 if fuse else 1)
    finally:
        native.close()


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("threshold, remaining", [(0.1, 1), (0.15, 1), (0.3, 0)])
def test_native_duplicate_cutoff_matches_python(
    library: BotSortLibrary, is_obb: bool, threshold: float, remaining: int,
) -> None:
    """A nearby tentative identity can confirm only if duplicate suppression retained it."""
    python, native = _pair(library, is_obb, match_thresh=0.0, duplicate_distance_thresh=threshold)
    try:
        for rows in (_rows(is_obb), _rows(is_obb, offset=2.0), _rows(is_obb, offset=2.0)):
            expected, actual = update_rows(python, rows), update_rows(native, rows)
            _assert_parity(actual, expected, is_obb)
        assert len(actual) == remaining
        if remaining:
            assert actual[0, 5 if is_obb else 4] == 2
    finally:
        native.close()


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.9, 1.0])
def test_native_appearance_memory_changes_matching_like_python(
    library: BotSortLibrary, is_obb: bool, alpha: float,
) -> None:
    """A later ambiguous frame exposes whether EMA retained old or recent appearance."""
    python, native = _pair(
        library, is_obb, use_embeddings=True, alpha_fixed_emb=alpha,
        proximity_thresh=1.0, appearance_thresh=0.25, duplicate_distance_thresh=0.0,
        fuse_first_associate=False,
    )
    original = np.vstack((_rows(is_obb), _rows(is_obb, offset=40)))
    ambiguous = np.repeat(_rows(is_obb, offset=20), 2, axis=0)
    features = np.eye(2, dtype=np.float32)
    recent = np.asarray([[0.1, 1.0], [1.0, 0.1]], dtype=np.float32)
    recent /= np.linalg.norm(recent, axis=1, keepdims=True)
    try:
        for rows, embeddings in ((original, features), (original, recent), (ambiguous, features)):
            expected = update_rows(python, rows, embeddings=embeddings)
            actual = update_rows(native, rows, embeddings=embeddings)
            expected = expected[np.argsort(expected[:, -1])]
            actual = actual[np.argsort(actual[:, -1])]
            _assert_parity(actual, expected, is_obb)
        expected_ids = [2, 1] if alpha < 0.5 else [1, 2]
        assert actual[:, 5 if is_obb else 4].tolist() == expected_ids
    finally:
        native.close()


@pytest.mark.parametrize("parameter", ["alpha_fixed_emb", "duplicate_distance_thresh"])
@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_native_rejects_invalid_parameter_bounds(library: BotSortLibrary, parameter: str, value: float) -> None:
    """Direct ABI callers receive the same finite unit-interval contract."""
    options = BotSortConfig(use_cmc=False, use_embeddings=False).to_dict()
    options[parameter] = value
    with pytest.raises(RuntimeError, match=parameter):
        library.create(options)
