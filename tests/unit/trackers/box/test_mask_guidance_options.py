"""Tunable guidance scalars preserve ownership and configuration precedence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxmot import ByteTrack
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig, TrackerSpec, create_tracker
from boxmot.trackers.common.config import (
    flatten_tracker_options,
    load_tracker_config,
    load_tracker_defaults,
    load_tracker_schema,
    nest_tracker_options,
)
from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS, mask_guidance_config_from_options

TRACKERS = (
    "boosttrack",
    "botsort",
    "bytetrack",
    "deepocsort",
    "hybridsort",
    "occluboost",
    "ocsort",
    "sfsort",
    "strongsort",
)
OPTIONS = {
    "edgetam.min_coverage": 0.8,
    "edgetam.min_fill": 0.2,
    "edgetam.prompt_overlap": 0.3,
    "edgetam.max_objects": 4,
}


def test_options_helper_uses_missing_defaults_and_ignores_unrelated_options() -> None:
    config = mask_guidance_config_from_options("model.pt", "cpu", {"match_thresh": 0.7, "edgetam.min_fill": 0.2})

    assert config == MaskGuidanceConfig("model.pt", "cpu", min_fill=0.2)


@pytest.mark.parametrize("tracker_name", TRACKERS)
def test_nested_edgetam_schema_and_runtime_defaults_roundtrip(tracker_name: str) -> None:
    schema = load_tracker_schema(tracker_name)
    defaults = load_tracker_defaults(tracker_name)

    assert set(schema["edgetam"]) == set(MASK_GUIDANCE_OPTIONS.values())
    assert {key: defaults[key] for key in MASK_GUIDANCE_OPTIONS} == {
        "edgetam.min_coverage": 0.90,
        "edgetam.min_fill": 0.05,
        "edgetam.prompt_overlap": 0.10,
        "edgetam.max_objects": 96,
    }
    assert flatten_tracker_options(nest_tracker_options(defaults)) == defaults


def test_partial_authored_edgetam_yaml_overlays_only_selected_leaves(tmp_path) -> None:
    profile = tmp_path / "tracker.yaml"
    profile.write_text("edgetam:\n  max_objects: 4\n  min_fill: 0.2\n")
    options = load_tracker_config("bytetrack", profile, {"edgetam": {"min_coverage": 0.8}})

    assert {key: options[key] for key in MASK_GUIDANCE_OPTIONS} == {**OPTIONS, "edgetam.prompt_overlap": 0.1}
    assert load_tracker_config("bytetrack", profile, include_defaults=False) == {
        "edgetam.max_objects": 4,
        "edgetam.min_fill": 0.2,
    }


@pytest.mark.parametrize("tracker_name", TRACKERS)
def test_public_constructor_accepts_nested_edgetam_parameters(tracker_name: str) -> None:
    from boxmot.trackers.common.config import get_tracker_config_class
    from boxmot.trackers.common.registry import _load_tracker_class, get_tracker_definition

    tracker_class = _load_tracker_class(get_tracker_definition(tracker_name))
    config = MaskGuidanceConfig("model.pt", "cpu")
    parameters = {field: OPTIONS[key] for key, field in MASK_GUIDANCE_OPTIONS.items()}

    tracker = tracker_class(
        config=get_tracker_config_class(tracker_name)(asso_func="iou"),
        mask_guidance=config,
        edgetam=parameters,
    )

    assert tracker._mask_guidance.config == mask_guidance_config_from_options("model.pt", "cpu", OPTIONS)
    assert config.max_objects == 96
    assert parameters == {field: OPTIONS[key] for key, field in MASK_GUIDANCE_OPTIONS.items()}


@pytest.mark.parametrize("value", [True, 1, "yes", [], (1,)])
def test_constructor_rejects_nonmapping_edgetam_parameters(value: object) -> None:
    with pytest.raises(TypeError, match="edgetam must be a mapping"):
        ByteTrack(edgetam=value)


@pytest.mark.parametrize("field", ["enabled", "unknown"])
def test_edgetam_has_parameters_only_and_rejects_unknown_fields(field: str) -> None:
    with pytest.raises(TypeError, match="Unknown edgetam"):
        ByteTrack(edgetam={field: True})
    with pytest.raises(TypeError, match="Unknown edgetam"):
        flatten_tracker_options({"edgetam": {field: True}})


@pytest.mark.parametrize("value", [True, None, "yes"])
def test_authored_edgetam_group_requires_mapping(value: object) -> None:
    with pytest.raises(TypeError, match="edgetam must be a mapping"):
        flatten_tracker_options({"edgetam": value})


def test_nested_and_dotted_duplicate_edgetam_leaves_reject() -> None:
    with pytest.raises(ValueError, match="more than once"):
        flatten_tracker_options({"edgetam": {"max_objects": 4}, "edgetam.max_objects": 8})


@pytest.mark.parametrize("field", MASK_GUIDANCE_OPTIONS.values())
def test_removed_flat_edgetam_parameters_are_rejected(field: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ByteTrack(**{f"mask_guidance_{field}": 1})
    with pytest.raises(TypeError, match="configure guidance parameters under 'edgetam'"):
        create_tracker("bytetrack", **{f"mask_guidance_{field}": 1})


@pytest.mark.parametrize("tracker_name", TRACKERS)
def test_factory_spec_options_override_injected_config_without_mutating_it(tracker_name: str) -> None:
    config = MaskGuidanceConfig("model.pt", "cpu", max_objects=9, min_coverage=0.7)
    tracker = create_tracker(tracker_name, asso_func="iou", mask_guidance=config, **OPTIONS)

    assert tracker._mask_guidance.config == mask_guidance_config_from_options("model.pt", "cpu", OPTIONS)
    assert config.max_objects == 9
    assert config.min_coverage == 0.7


@pytest.mark.parametrize("tracker_name", TRACKERS)
def test_prebuilt_matching_options_preserve_runtime_identity(tracker_name: str) -> None:
    guidance = MaskGuidance(mask_guidance_config_from_options("model.pt", "cpu", OPTIONS))
    tracker = create_tracker(tracker_name, asso_func="iou", mask_guidance=guidance, **OPTIONS)

    assert tracker._mask_guidance is guidance
    assert guidance._propagator is None


@pytest.mark.parametrize("key", OPTIONS)
@pytest.mark.parametrize("factory", [False, True])
def test_prebuilt_conflicting_options_reject_without_mutation(key: str, factory: bool) -> None:
    guidance = MaskGuidance(MaskGuidanceConfig("model.pt", "cpu"))
    original_config = guidance.config
    mask = np.ones((3, 3), dtype=bool)
    guidance._masks[17] = mask
    construct = lambda **kwargs: create_tracker("bytetrack", **kwargs) if factory else ByteTrack(**kwargs)

    with pytest.raises(ValueError, match="conflict with the prebuilt component"):
        construct(mask_guidance=guidance, edgetam={MASK_GUIDANCE_OPTIONS[key]: OPTIONS[key]})

    assert guidance.config is original_config
    assert guidance._masks[17] is mask
    assert guidance._propagator is None


@pytest.mark.parametrize("field", ["min_coverage", "min_fill", "prompt_overlap"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True), None, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_config_rejects_invalid_probability(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        MaskGuidanceConfig("model.pt", "cpu", **{field: value})


@pytest.mark.parametrize("field", ["min_coverage", "min_fill"])
@pytest.mark.parametrize("value", [0, 1, np.float32(0.5)])
def test_config_accepts_inclusive_probability_bounds(field: str, value: float) -> None:
    config = MaskGuidanceConfig("model.pt", "cpu", **{field: value})
    assert getattr(config, field) == value
    assert isinstance(getattr(config, field), float)


def test_prompt_overlap_requires_positive_threshold() -> None:
    with pytest.raises(ValueError, match="prompt_overlap"):
        MaskGuidanceConfig("model.pt", "cpu", prompt_overlap=0)
    assert MaskGuidanceConfig("model.pt", "cpu", prompt_overlap=1).prompt_overlap == 1


@pytest.mark.parametrize("key", OPTIONS)
@pytest.mark.parametrize("value", [True, "1", float("nan"), -1])
def test_disabled_guidance_still_validates_explicit_knobs(key: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=MASK_GUIDANCE_OPTIONS[key]):
        ByteTrack(edgetam={MASK_GUIDANCE_OPTIONS[key]: value})


def test_factory_keyword_overrides_spec_knob_and_config() -> None:
    spec = TrackerSpec("bytetrack", options=(("edgetam.max_objects", 7),))
    tracker = create_tracker(
        spec, mask_guidance=MaskGuidanceConfig("model.pt", "cpu", max_objects=11), edgetam={"max_objects": 3}
    )

    assert tracker._mask_guidance.config.max_objects == 3


@pytest.mark.parametrize("key", OPTIONS)
def test_factory_rejects_native_explicit_guidance_knob_before_loading(monkeypatch, key: str) -> None:
    from boxmot.trackers.common import factory

    monkeypatch.setattr(
        factory, "_load_native_tracker_class", lambda *args: pytest.fail("Loaded invalid native tracker")
    )
    with pytest.raises(ValueError, match="Native trackers do not support masks"):
        create_tracker("bytetrack", backend="cpp", **{key: OPTIONS[key]})
    with pytest.raises(ValueError, match="Native trackers do not support masks"):
        create_tracker("bytetrack", backend="cpp", edgetam={MASK_GUIDANCE_OPTIONS[key]: OPTIONS[key]})


def test_native_config_strips_inherited_guidance_defaults(monkeypatch) -> None:
    from boxmot.trackers.common import native

    monkeypatch.setattr(native, "load_tracker_defaults", lambda _: {"track_thresh": 0.5, **OPTIONS})
    resolved = native.load_native_tracker_config("bytetrack", None)

    assert resolved["track_thresh"] == 0.5
    assert not any(key.startswith("edgetam") for key in resolved)


@pytest.mark.parametrize("key", OPTIONS)
def test_direct_native_config_rejects_explicit_guidance_knob(key: str) -> None:
    from boxmot.trackers.common.native import load_native_tracker_config

    with pytest.raises(ValueError, match="Native trackers do not support mask guidance options"):
        load_native_tracker_config("bytetrack", {key: OPTIONS[key]})
    with pytest.raises(ValueError, match="Native trackers do not support mask guidance options"):
        load_native_tracker_config("bytetrack", {"edgetam": {MASK_GUIDANCE_OPTIONS[key]: OPTIONS[key]}})


@pytest.mark.parametrize("min_coverage,min_fill,adjusted", [(0.5, 0.5, True), (0.51, 0.5, False), (0.5, 0.51, False)])
def test_runtime_applies_configured_coverage_and_fill_boundaries(
    min_coverage: float, min_fill: float, adjusted: bool
) -> None:
    mask = np.zeros((2, 20), dtype=bool)
    mask[:, :10] = True
    config = replace(MaskGuidanceConfig("model.pt", "cpu"), min_coverage=min_coverage, min_fill=min_fill)
    guidance = MaskGuidance(config)
    guidance._masks[17] = mask
    costs = np.array([[0.75, 0.75]])

    result = guidance.condition(costs, [17], np.array([[0, 0, 20, 1], [0, 1, 20, 2]]), threshold=0.8)

    np.testing.assert_allclose(result, [[0.25, 0.25]] if adjusted else [[0.75, 0.75]])
    np.testing.assert_array_equal(costs, [[0.75, 0.75]])
