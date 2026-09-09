"""Masked-KCF appearance and FHOG numerical contracts."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.multimodal.maf_hda.appearance import MaskedKCF, _lab_histograms
from boxmot.trackers.multimodal.maf_hda.fhog import fhog


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic textured foreground with holes inside its box."""
    rng = np.random.default_rng(14)
    image = rng.integers(0, 255, (80, 96, 3), dtype=np.uint8)
    bbox = np.array([24.0, 16.0, 64.0, 64.0])
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[18:62, 26:62] = True
    mask[32:48, 38:50] = False
    return image, bbox, mask


def test_fhog_horizontal_gradient_has_the_source_channel_layout() -> None:
    """A horizontal ramp has one signed/unsigned orientation and four textures."""
    patch = np.zeros((32, 32, 3), dtype=np.uint8)
    patch[..., 0] = np.arange(32, dtype=np.uint8)
    features = fhog(patch)
    assert features.shape == (31, 6, 6)
    expected = np.zeros(31, dtype=np.float32)
    expected[[0, 18]] = 0.4  # Four clipped 0.2 block contributions, scaled by 1/2.
    expected[27:] = 0.2 / np.sqrt(18)
    np.testing.assert_allclose(features[:, 3, 3], expected, atol=1e-7)
    assert not fhog(np.zeros_like(patch)).any()


def test_lab_uses_source_centroids_and_cell_probabilities() -> None:
    """Black BGR pixels select the source's darkest Lab centroid (index seven)."""
    features = _lab_histograms(np.zeros((24, 32, 3), dtype=np.uint8))
    assert features.shape == (15, 4, 6)
    np.testing.assert_array_equal(features[7], 1)
    np.testing.assert_array_equal(features.sum(axis=0), 1)


def test_score_preserves_model_and_ignores_masked_background() -> None:
    """Candidate order and excluded object-box pixels cannot change appearance memory."""
    image, bbox, mask = _scene()
    model = MaskedKCF(image, bbox, mask)
    template, alphaf = model._template.copy(), model._alphaf.copy()
    original_image, original_box, original_mask = image.copy(), bbox.copy(), mask.copy()
    score = model.score(image, bbox, mask)
    changed = image.copy()
    background = ~mask[16:64, 24:64]
    changed[16:64, 24:64][background] = 255 - changed[16:64, 24:64][background]
    assert model.score(changed, bbox, mask) == pytest.approx(score)
    assert 0 < score < 1
    np.testing.assert_array_equal(model._template, template)
    np.testing.assert_array_equal(model._alphaf, alphaf)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(bbox, original_box)
    np.testing.assert_array_equal(mask, original_mask)


def test_empty_foreground_neither_matches_nor_poison_trains() -> None:
    """An empty mask is missing appearance evidence, including at initialization."""
    image, bbox, mask = _scene()
    empty = np.zeros_like(mask)
    model = MaskedKCF(image, bbox, empty)
    assert model.score(image, bbox, mask) == 0
    model.update(image, bbox, mask)
    before = model.score(image, bbox, mask)
    template, alphaf = model._template.copy(), model._alphaf.copy()
    assert model.score(image, bbox, empty) == 0
    model.update(image, bbox, empty)
    np.testing.assert_array_equal(model._template, template)
    np.testing.assert_array_equal(model._alphaf, alphaf)
    assert model.score(image, bbox, mask) == before


def test_matching_texture_scores_above_a_distractor_at_another_position() -> None:
    """The learned response distinguishes texture while allowing object displacement."""
    rng = np.random.default_rng(14)
    image = rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)
    first_box = np.array([16.0, 16.0, 56.0, 80.0])
    candidate_box = np.array([104.0, 16.0, 144.0, 80.0])
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[16:80, 16:56] = True
    mask[16:80, 104:144] = True
    model = MaskedKCF(image, first_box, mask)
    distractor = model.score(image, candidate_box, mask)
    assert model.score(image, first_box, mask) > distractor + 0.1
    moved = image.copy()
    moved[16:80, 104:144] = image[16:80, 16:56]
    assert model.score(moved, candidate_box, mask) > distractor + 0.1


@pytest.mark.parametrize("bbox", [[-8, -6, 24, 35], [0, 0, 1, 70], [95, 0, 96, 80]])
def test_border_and_thin_targets_keep_bounded_finite_features(bbox: list[int]) -> None:
    """Clipped and one-pixel-wide objects must not trigger invalid FHOG normalization."""
    image, _, _ = _scene()
    mask = np.ones(image.shape[:2], dtype=bool)
    box = np.asarray(bbox, dtype=np.float64)
    original_box = box.copy()
    model = MaskedKCF(image, box, mask)
    assert model._template.shape[0] == 46
    assert min(model._template.shape[1:]) >= 2
    assert max(model._template_wh) <= 104
    assert np.isfinite(model._template).all()
    assert np.isfinite(model.score(image, box, mask))
    np.testing.assert_array_equal(box, original_box)
    assert model.score(image, np.array([120, 10, 140, 40]), mask) == 0


def test_explicit_update_trains_the_accepted_appearance() -> None:
    """Scoring leaves memory intact; an accepted new texture changes both model terms."""
    image, bbox, mask = _scene()
    model = MaskedKCF(image, bbox, mask)
    template, alphaf = model._template.copy(), model._alphaf.copy()
    changed = 255 - image
    model.score(changed, bbox, mask)
    np.testing.assert_array_equal(model._template, template)
    model.update(changed, bbox, mask)
    assert not np.array_equal(model._template, template)
    assert not np.array_equal(model._alphaf, alphaf)
