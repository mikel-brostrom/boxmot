"""Association-local KCF reuse preserves exact affinities and custom scoring."""

from __future__ import annotations

import numpy as np
import pytest

import boxmot.trackers.maf_hda.appearance as appearance_module
from boxmot import MafHda, MafHdaConfig
from boxmot.trackers.maf_hda.appearance import MaskedKCF, _AppearanceScorer
from boxmot.trackers.maf_hda.tracker import _Observation


def _scene() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Create separated, equally sized textured candidates with mask holes."""
    image = np.random.default_rng(7).integers(0, 255, (64, 128, 3), dtype=np.uint8)
    boxes = [np.array([8., 12., 32., 52.]), np.array([72., 12., 96., 52.])]
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = True
        mask[y1 + 7:y1 + 13, x1 + 5:x1 + 10] = False
        masks.append(mask)
    return image, boxes, masks


@pytest.mark.parametrize("different_grid", [False, True])
def test_scoring_reuses_observation_features_and_template_spectra_exactly(
    different_grid: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only identical crop/FHOG grids share features, while every track keeps its model."""
    image, boxes, masks = _scene()
    models = [
        MaskedKCF(image, boxes[0], masks[0], template_size=32),
        MaskedKCF(image, boxes[1], masks[1], template_size=32, sigma=0.6),
        MaskedKCF(image, boxes[0], masks[0], template_size=48 if different_grid else 32),
    ]
    expected = np.array([[model.score(image, box, mask) for box, mask in zip(boxes, masks)] for model in models])
    descriptor_calls: list[tuple[int, ...]] = []
    spectrum_calls: list[tuple[int, ...]] = []
    fhog, fft2 = appearance_module.fhog, np.fft.fft2

    def record_fhog(patch: np.ndarray) -> np.ndarray:
        descriptor_calls.append(patch.shape)
        return fhog(patch)

    def record_fft(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        if values.ndim == 3:
            spectrum_calls.append(values.shape)
        return fft2(values, *args, **kwargs)

    monkeypatch.setattr(appearance_module, "fhog", record_fhog)
    monkeypatch.setattr(np.fft, "fft2", record_fft)
    scorer = _AppearanceScorer(image)
    actual = np.array([
        [scorer.score(model, column, box, mask) for column, (box, mask) in enumerate(zip(boxes, masks))]
        for model in models
    ])
    np.testing.assert_array_equal(actual, expected)
    feature_count = len(boxes) * (2 if different_grid else 1)
    assert len(descriptor_calls) == feature_count
    assert len(spectrum_calls) == feature_count + len(models)
    assert len(scorer._observations) == len(boxes)


def test_new_stage_uses_updated_images_masks_and_templates() -> None:
    """No cached validation, descriptors, or spectra survive a frame/model update."""
    image, boxes, masks = _scene()
    model = MaskedKCF(image, boxes[0], masks[0], template_size=32, interp_factor=0.5)
    original = _AppearanceScorer(image).score(model, 0, boxes[0], masks[0])
    changed = 255 - image
    model.update(changed, boxes[0], masks[0])
    scorer = _AppearanceScorer(changed)
    assert scorer.score(model, 0, boxes[0], masks[0]) == model.score(changed, boxes[0], masks[0])
    assert _AppearanceScorer(changed).score(model, 0, boxes[0], np.zeros_like(masks[0])) == 0
    assert _AppearanceScorer(image).score(model, 0, boxes[0], masks[0]) == model.score(image, boxes[0], masks[0])
    assert original > 0


@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_custom_scores_are_called_for_each_allowed_pair(override: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching must not bypass a user-supplied affinity or a subclass's behavior."""
    image, boxes, masks = _scene()
    calls: list[int] = []

    class CustomKCF(MaskedKCF):
        def score(self, image: np.ndarray, box: np.ndarray, mask: np.ndarray) -> float:
            """Supply a deliberately stateful score to expose bypassed calls."""
            calls.append(len(calls))
            return 0.25 + len(calls) / 10

    model = (CustomKCF if override == "subclass" else MaskedKCF)(image, boxes[0], masks[0], template_size=32)
    if override == "instance":
        monkeypatch.setattr(model, "score", CustomKCF.score.__get__(model))
    elif override == "class":
        monkeypatch.setattr(MaskedKCF, "score", CustomKCF.score)
    scorer = _AppearanceScorer(image)
    actual = [scorer.score(model, 0, boxes[0], masks[0]) for _ in range(3)]
    assert actual == [0.35, 0.45, 0.55]
    assert len(calls) == 3
    assert not scorer._spectra


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_spectrum_eviction_bounds_memory_without_changing_affinities(capacity: int) -> None:
    """A crowded association may recompute evicted FFTs but never retain every pair."""
    image, boxes, masks = _scene()
    models = [MaskedKCF(image, box, mask, template_size=32) for box, mask in zip(boxes, masks)]
    spectrum_size = np.fft.fft2(models[0]._template).nbytes
    budget = capacity * spectrum_size
    scorer = _AppearanceScorer(image, max_cache_bytes=budget)
    for model in models:
        for column in (0, 1, 0):
            assert scorer.score(model, column, boxes[column], masks[column]) == model.score(
                image, boxes[column], masks[column]
            )
            assert scorer._cache_bytes <= budget
            assert len(scorer._spectra) <= capacity


def test_tracker_association_reuses_features_without_changing_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live association invokes local reuse for multiple eligible identities."""
    image, boxes, masks = _scene()
    tracker = MafHda(MafHdaConfig(
        s2ta_mode="appearance", t2ta_mode="appearance", appearance_gate=False,
    ))
    observations = [_Observation(box, mask, 0.95, 0, column) for column, (box, mask) in enumerate(zip(boxes, masks))]
    tracks = [tracker._new_track(observation, image) for observation in observations]
    expected, expected_covariances, expected_weights = tracker._associate(tracks, observations, image)
    fhog = appearance_module.fhog
    calls: list[int] = []

    def record_fhog(patch: np.ndarray) -> np.ndarray:
        calls.append(1)
        return fhog(patch)

    monkeypatch.setattr(appearance_module, "fhog", record_fhog)
    actual, actual_covariances, actual_weights = tracker._associate(tracks, observations, image)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_covariances, expected_covariances)
    np.testing.assert_array_equal(actual_weights, expected_weights)
    assert len(calls) == len(observations)
