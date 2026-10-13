"""Native OccluBoost exposes the same appearance and confidence policies as Python."""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import OccluBoostConfig
from boxmot.native.trackers import occluboost as binding
from boxmot.trackers.occluboost.native import NativeOccluBoostTracker, _resolve_tracker_config
from boxmot.trackers.occluboost.tracker import OccluBoost

from ._helpers import update_rows


@pytest.fixture(scope="module")
def library() -> binding.OccluBoostLibrary:
    """Use the actual compiled ABI, including its independent configuration validation."""
    return binding.OccluBoostLibrary(binding.ensure_occluboost_cpp_library())


def _rows(geometry: str, confidence: float = 1.0, shift: float = 0.0) -> np.ndarray:
    """Represent one 20-by-40 box in either geometry."""
    box = [20 + shift, 30, 20, 40, 0] if geometry == "obb" else [10 + shift, 10, 30 + shift, 50]
    return np.asarray([[*box, confidence, 0]], dtype=np.float32)


def _run(
    library: binding.OccluBoostLibrary,
    geometry: str,
    overrides: dict[str, object],
    frames: list[np.ndarray],
    embeddings: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Compare every public output while running independent Python/native states."""
    config = OccluBoostConfig.from_mapping(
        {
            "use_cmc": False,
            "use_embeddings": embeddings is not None,
            "use_dlo_boost": False,
            "use_duo_boost": False,
            "use_second_pass": False,
            "recovery_max_age": 0,
            "min_box_area": 0,
            "aspect_ratio_thresh": 10.0,
            "min_hits": 0,
            "confirm_hits": 1,
            "new_track_thresh": 0.5,
            "instant_confirm_thresh": 0.5,
            "det_thresh": 0.5,
            "iou_threshold": 0.1,
            "duplicate_iou_thresh": 1.0,
            **overrides,
        }
    )
    python = OccluBoost(config=config, is_obb=geometry == "obb")
    native = NativeOccluBoostTracker(config.to_dict(), geometry=geometry, library=library)
    outputs = []
    native_ids: dict[int, int] = {}
    python_ids: dict[int, int] = {}
    try:
        for index, rows in enumerate(frames):
            appearance = None if embeddings is None else embeddings[index]
            expected = update_rows(python, rows, embeddings=appearance)
            actual = update_rows(native, rows, embeddings=appearance)
            assert actual.shape == expected.shape
            # Allocator origins and unmatched-detection order differ. Match
            # detections while enforcing a persistent bijection of track IDs.
            expected = expected[np.argsort(expected[:, -1])]
            actual = actual[np.argsort(actual[:, -1])]
            for expected_row, actual_row in zip(expected, actual, strict=True):
                python_id, native_id = int(expected_row[-4]), int(actual_row[-4])
                assert native_ids.setdefault(python_id, native_id) == native_id
                assert python_ids.setdefault(native_id, python_id) == python_id
                expected_row[-4] = native_id
            np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-5)
            outputs.append(actual)
    finally:
        native.close()
    return outputs


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("alpha,power,expected", [(0.2, 1.0, 0.52), (0.2, 2.0, 0.328), (0.8, 1.0, 0.28)])
def test_native_soft_dlo_blend_and_power_match_python(library, geometry, alpha, power, expected) -> None:
    """Nondefault soft boost controls change the emitted detection score."""
    outputs = _run(
        library, geometry,
        {
            "det_thresh": 0.1, "new_track_thresh": 0.1, "use_dlo_boost": True,
            "use_rich_s": False, "use_sb": True, "use_vt": False,
            "dlo_soft_alpha": alpha, "dlo_soft_power": power,
        },
        [_rows(geometry), _rows(geometry, 0.2, 5.0)],
    )
    assert outputs[1][0, -3] == pytest.approx(expected)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_native_basic_dlo_preserves_confidence_bounds(library, geometry) -> None:
    """Basic boosting still saturates valid scores when the coefficient exceeds one."""
    outputs = _run(
        library, geometry,
        {"use_dlo_boost": True, "use_rich_s": False, "use_sb": False, "use_vt": False, "dlo_boost_coef": 1.5},
        [_rows(geometry), _rows(geometry, 0.6)],
    )
    assert outputs[-1][0, -3] == 1.0


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("parameter,missing", [("dlo_threshold_start", False), ("dlo_threshold_end", True)])
@pytest.mark.parametrize("threshold,boosted", [(0.5, True), (0.7, False)])
def test_native_dlo_visibility_thresholds_match_python(
    library, geometry, parameter, missing, threshold, boosted
) -> None:
    """Initial visibility and the missing-track floor independently govern promotion."""
    frames = [_rows(geometry)]
    if missing:
        frames.append(np.empty((0, 7 if geometry == "obb" else 6), dtype=np.float32))
    frames.append(_rows(geometry, 0.2, 5.0))
    outputs = _run(
        library, geometry,
        {
            "use_dlo_boost": True, "use_rich_s": False, "use_sb": False, "use_vt": True,
            "dlo_threshold_start": 0.95, "dlo_threshold_end": 0.4, parameter: threshold,
        },
        frames,
    )
    assert bool(len(outputs[-1])) is boosted


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("motion_gate,overlap_gate,count", [(1e8, 0.3, 0), (0.1, 0.3, 1), (0.1, 0.8, 2)])
def test_native_duo_motion_and_overlap_gates_match_python(library, geometry, motion_gate, overlap_gate, count) -> None:
    """A distant candidate pair is promoted only through the configured DUO gates."""
    candidates = np.concatenate((_rows(geometry, 0.2, 100), _rows(geometry, 0.3, 105)))
    outputs = _run(
        library, geometry,
        {"use_duo_boost": True, "duo_mahalanobis_thresh": motion_gate, "duo_iou_thresh": overlap_gate},
        [_rows(geometry), candidates],
    )
    assert len(outputs[-1]) == count


def test_native_duo_obb_ties_follow_python_neighbor_order(library) -> None:
    """Equal-confidence overlap groups use the same deterministic representative order."""
    candidates = np.concatenate([_rows("obb", 0.2, shift) for shift in (100, 101, 102)])
    outputs = _run(
        library, "obb", {"use_duo_boost": True, "duo_mahalanobis_thresh": 0.1, "duo_iou_thresh": 0.5},
        [_rows("obb"), candidates],
    )
    assert len(outputs[-1]) == 2


@pytest.mark.parametrize("threshold,fraction,recovered", [(0.7, 0.5, True), (0.9, 0.5, False), (0.7, 0.9, False)])
def test_native_appearance_rescue_gates_match_python(library, threshold, fraction, recovered) -> None:
    """First-pass appearance rescue retains identity only when both gates permit it."""
    outputs = _run(
        library, "aabb",
        {"iou_threshold": 0.5, "appearance_thresh": threshold, "appearance_iou_fraction": fraction},
        [_rows("aabb"), _rows("aabb", shift=20 * 0.7 / 1.3)],
        [np.asarray([[1.0, 0.0]], np.float32), np.asarray([[0.8, 0.6]], np.float32)],
    )
    assert bool(outputs[0][0, -4] == outputs[1][0, -4]) is recovered


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("alpha,recovered", [(0.2, True), (0.8, False)])
def test_native_embedding_ema_controls_later_recovery(library, geometry, alpha, recovered) -> None:
    """The first-pass EMA baseline changes whether later appearance recovery succeeds."""
    outputs = _run(
        library, geometry,
        {
            "alpha_fixed_emb": alpha, "iou_threshold": 0.5, "appearance_iou_fraction": 1.0,
            "recovery_max_age": 2, "recovery_appearance_thresh": 0.7, "recovery_iou_thresh": 0.1,
        },
        [_rows(geometry), _rows(geometry), _rows(geometry, shift=20 * 0.7 / 1.3)],
        [np.asarray([embedding], np.float32) for embedding in ([1, 0], [0, 1], [0, 1])],
    )
    assert bool(outputs[0][0, -4] == outputs[-1][0, -4]) is recovered


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize(
    "parameter,values", [("track_confidence_decay", (0.5, 0.95)), ("track_confidence_warmup", (2, 10))]
)
def test_native_track_confidence_changes_rich_dlo_scores(library, geometry, parameter, values) -> None:
    """Confidence decay and warmup reach both native geometry paths and affect scores."""
    scores = []
    for value in values:
        outputs = _run(
            library, geometry,
            {
                "det_thresh": 0.1, "new_track_thresh": 0.1, "use_dlo_boost": True,
                "use_rich_s": True, "use_sb": True, "use_vt": False,
                "dlo_soft_alpha": 0.0, "dlo_soft_power": 1.0, parameter: value,
            },
            [_rows(geometry), _rows(geometry, 0.2, 5.0)],
        )
        scores.append(outputs[-1][0, -3])
    assert abs(scores[0] - scores[1]) > 0.001


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha_fixed_emb": 1.1}, {"appearance_thresh": -1.1}, {"appearance_iou_fraction": 1.1},
        {"dlo_soft_alpha": -0.1}, {"dlo_soft_power": 0.0},
        {"dlo_threshold_start": 0.5, "dlo_threshold_end": 0.8},
        {"duo_mahalanobis_thresh": 0.0}, {"duo_iou_thresh": 1.1},
        {"track_confidence_decay": 0.0}, {"track_confidence_warmup": 0},
    ],
)
def test_native_c_api_validates_confidence_policies(library, overrides) -> None:
    """Direct C callers cannot bypass bounds or reversed DLO threshold validation."""
    config = binding._build_c_config(_resolve_tracker_config({"use_cmc": False}))
    for name, value in overrides.items():
        setattr(config, name, value)
    handle = library._create(ctypes.byref(config))
    try:
        assert not handle
        assert "appearance or confidence policy" in library._last_error()
    finally:
        if handle:
            library.destroy(handle)


def test_native_rejects_previous_v2_config_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    """The longer configuration must never be passed to an older shared library."""
    stale = SimpleNamespace(boxmot_occluboost_create_v2=lambda _config: pytest.fail("called incompatible create"))
    monkeypatch.setattr(binding.ctypes, "CDLL", lambda _path: stale)
    with pytest.raises(RuntimeError, match="incompatible configuration ABI"):
        binding.OccluBoostLibrary(Path("stale-occluboost"))
