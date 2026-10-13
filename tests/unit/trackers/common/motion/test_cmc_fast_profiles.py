"""Bounded CMC work must retain useful motion estimates and sequence isolation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from boxmot.trackers.common.motion.cmc.base import BaseCMC
from boxmot.trackers.common.motion.cmc.ecc import ECC
from boxmot.trackers.common.motion.cmc.sof import SOF

_METHODS = (ECC, SOF)


def _textured_frame(height: int = 360, width: int = 640) -> np.ndarray:
    """Build repeatable multiscale detail that survives the default downsampling."""
    random = np.random.default_rng(2491)
    texture = random.integers(0, 256, (height // 4, width // 4), dtype=np.uint8)
    image = cv2.resize(texture, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(120):
        center = tuple(random.integers((12, 12), (width - 12, height - 12)).tolist())
        radius = int(random.integers(3, 12))
        cv2.circle(image, center, radius, int(random.integers(0, 256)), -1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _warp_error(actual: np.ndarray, expected: np.ndarray, shape: tuple[int, int]) -> float:
    """Measure displacement error in original pixels over the scene interior."""
    height, width = shape
    points = np.array(
        [[x * width, y * height, 1.0] for y in (0.2, 0.5, 0.8) for x in (0.2, 0.5, 0.8)],
        dtype=np.float64,
    )
    return float(np.max(np.linalg.norm(points @ (actual - expected).T, axis=1)))


@pytest.mark.parametrize("cmc_class", _METHODS)
def test_default_profile_recovers_nonzero_translation(cmc_class: type[BaseCMC]) -> None:
    """A cheap identity fallback must not pass as successful camera compensation."""
    image = _textured_frame()
    expected = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    shifted = cv2.warpAffine(image, expected, (image.shape[1], image.shape[0]))
    estimator = cmc_class()
    np.testing.assert_array_equal(estimator.apply(image), np.eye(2, 3, dtype=np.float32))

    cv2.setRNGSeed(0)
    actual = estimator.apply(shifted)

    assert actual.dtype == np.float32
    assert BaseCMC.is_valid_transform(actual)
    assert _warp_error(actual, expected, image.shape[:2]) < 2.0


def test_default_sof_profile_recovers_rotation_and_scale() -> None:
    """Feature budgets must preserve rotation/scale, beyond a translation-only fit."""
    image = _textured_frame()
    expected = cv2.getRotationMatrix2D((320, 180), 1.5, 1.01).astype(np.float32)
    expected[:, 2] += np.array([3.0, -2.0], dtype=np.float32)
    shifted = cv2.warpAffine(image, expected, (640, 360))
    estimator = SOF()
    estimator.apply(image)

    cv2.setRNGSeed(0)
    actual = estimator.apply(shifted)

    assert BaseCMC.is_valid_transform(actual)
    assert _warp_error(actual, expected, image.shape[:2]) < 3.0


def test_sof_tracks_through_feature_reuse_and_refresh_without_drift() -> None:
    """Sparse feature refreshes must retain accurate inter-frame camera motion."""
    image = _textured_frame()
    estimator = SOF()
    expected_step = np.float32([[1, 0, 2.5], [0, 1, -1.5]])
    accumulated = np.eye(3, dtype=np.float64)
    estimator.apply(image)
    for index in range(1, 13):
        expected_total = expected_step.copy()
        expected_total[:, 2] *= index
        frame = cv2.warpAffine(image, expected_total, (640, 360))
        cv2.setRNGSeed(0)
        actual = estimator.apply(frame)
        assert _warp_error(actual, expected_step, image.shape[:2]) < 2.0
        homogeneous = np.eye(3, dtype=np.float64)
        homogeneous[:2] = actual
        accumulated = homogeneous @ accumulated

    assert _warp_error(accumulated[:2], expected_total, image.shape[:2]) < 4.0


@pytest.mark.parametrize("cmc_class", _METHODS)
@pytest.mark.parametrize("channels", (1, 3))
def test_default_preprocessing_does_not_enlarge_small_inputs(cmc_class: type[BaseCMC], channels: int) -> None:
    """Small grayscale and BGR inputs do not need a full-size work image."""
    shape = (24, 32) if channels == 1 else (24, 32, channels)
    image = np.zeros(shape, dtype=np.uint8)
    processed = cmc_class().preprocess(image)

    assert processed.ndim == 2
    assert processed.shape[0] <= shape[0]
    assert processed.shape[1] <= shape[1]
    assert processed.size > 0


@pytest.mark.parametrize("cmc_class", _METHODS)
def test_grayscale_sequences_match_equivalent_bgr_sequences(cmc_class: type[BaseCMC]) -> None:
    """Accepting grayscale frames must preserve the estimated original-image warp."""
    bgr = _textured_frame()
    shifted = cv2.warpAffine(bgr, np.float32([[1, 0, 5], [0, 1, -3]]), (640, 360))
    color_estimator, gray_estimator = cmc_class(), cmc_class()
    for frame in (bgr, shifted):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.setRNGSeed(0)
        expected = color_estimator.apply(frame)
        cv2.setRNGSeed(0)
        actual = gray_estimator.apply(gray)
        np.testing.assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("cmc_class", _METHODS)
def test_original_resolution_change_reinitializes_even_with_same_work_size(cmc_class: type[BaseCMC]) -> None:
    """Cached features cannot connect distinct image coordinates after resizing."""
    estimator = cmc_class(scale=None, target_width=128)
    first = _textured_frame()
    # Same aspect ratio and twice the original resolution produce equal work sizes.
    larger = cv2.resize(first, (1280, 720), interpolation=cv2.INTER_LINEAR)
    estimator.apply(first)
    changed = estimator.apply(larger)

    np.testing.assert_array_equal(changed, np.eye(2, 3, dtype=np.float32))
    assert BaseCMC.is_valid_transform(estimator.apply(larger))


@pytest.mark.parametrize("cmc_class", _METHODS)
def test_grayscale_input_buffer_can_be_reused(cmc_class: type[BaseCMC]) -> None:
    """History must retain the previous image when the caller reuses its storage."""
    frame = cv2.cvtColor(_textured_frame(72, 96), cv2.COLOR_BGR2GRAY)
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 1], [0, 1, -1]]), (96, 72))
    shared_estimator, separate_estimator = cmc_class(), cmc_class()
    shared_estimator.apply(frame)
    separate_estimator.apply(frame.copy())

    np.copyto(frame, shifted)
    cv2.setRNGSeed(0)
    actual = shared_estimator.apply(frame)
    cv2.setRNGSeed(0)
    expected = separate_estimator.apply(shifted)

    assert np.linalg.norm(expected[:, 2]) > 0.5
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_sof_reused_keypoints_becoming_foreground_cannot_estimate_camera_motion() -> None:
    """A new detection mask also excludes points tracked from the preceding frame."""
    image = _textured_frame()
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, -3]]), (640, 360))
    estimator = SOF()
    estimator.apply(image)
    assert estimator.initialized

    actual = estimator.apply(shifted, np.array([[0, 0, 640, 360]], dtype=np.float32))

    np.testing.assert_array_equal(actual, np.eye(2, 3, dtype=np.float32))
    assert estimator.prev_keypoints is None or len(estimator.prev_keypoints) == 0
    assert not estimator.initialized


@pytest.mark.parametrize("cmc_class", _METHODS)
def test_fully_masked_frames_remain_identity_and_recover(cmc_class: type[BaseCMC]) -> None:
    """Foreground masking must not supply spurious camera motion or poison state."""
    image = _textured_frame()
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, -3]]), (640, 360))
    estimator = cmc_class()
    detections = np.array([[0, 0, 640, 360]], dtype=np.float32)
    original_detections = detections.copy()
    for frame in (image, shifted):
        np.testing.assert_array_equal(estimator.apply(frame, detections), np.eye(2, 3, dtype=np.float32))

    np.testing.assert_array_equal(detections, original_detections)
    estimator.reset()
    estimator.apply(image)
    cv2.setRNGSeed(0)
    actual = estimator.apply(shifted)
    assert _warp_error(actual, np.float32([[1, 0, 5], [0, 1, -3]]), image.shape[:2]) < 2.0
