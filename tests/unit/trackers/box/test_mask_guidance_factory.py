"""Each Python box tracker accepts guidance without eagerly loading its model."""

from __future__ import annotations

import pytest

from boxmot.trackers import MaskGuidance, MaskGuidanceConfig, create_tracker


@pytest.mark.parametrize(
    "tracker_name",
    ["boosttrack", "botsort", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "sfsort", "strongsort"],
)
@pytest.mark.parametrize("prebuilt", [False, True])
def test_factory_accepts_optional_guidance_without_loading_models(monkeypatch, tracker_name, prebuilt) -> None:
    from boxmot.segmentors.propagation import edgetam

    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", lambda *a, **kw: pytest.fail("Eager temporal model loading"))
    config = MaskGuidanceConfig("missing-checkpoint.pt", device="cpu", max_objects=4)
    guidance = MaskGuidance(config) if prebuilt else config

    tracker = create_tracker(tracker_name, mask_guidance=guidance, asso_func="iou")

    assert tracker.requirements.frame_pixels
    assert not tracker.requirements.masks
    assert tracker._mask_guidance.config == config
    assert tracker._mask_guidance._propagator is None
    if prebuilt:
        assert tracker._mask_guidance is guidance


def test_factory_rejects_effective_non_iou_default_without_model_loading(monkeypatch) -> None:
    from boxmot.segmentors.propagation import edgetam

    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", lambda *a, **kw: pytest.fail("Loaded invalid-run model"))
    with pytest.raises(ValueError, match="asso_func='iou'"):
        create_tracker("hybridsort", mask_guidance=MaskGuidanceConfig("missing-checkpoint.pt", device="cpu"))
