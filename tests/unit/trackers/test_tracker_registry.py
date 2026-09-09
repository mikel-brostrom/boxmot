"""Tracker registry tests for immutable v24 construction specifications."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import yaml

import boxmot.trackers.factory as tracker_factory
import boxmot.trackers.registry as tracker_registry
from boxmot._tracker_exports import _TRACKER_MANIFEST
from boxmot.structures import GeometryKind
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.config import TRACKER_CONFIGS_DIR, load_tracker_config, load_tracker_schema
from boxmot.trackers.protocols import TrackerRequirements
from boxmot.trackers.registry import supported_native_trackers
from boxmot.trackers.specs import TrackerCapabilities, TrackerFamily, TrackerSpec


def test_tracker_public_mappings_are_derived_from_the_lazy_manifest() -> None:
    assert tracker_registry.TRACKER_MAPPING == {
        name: definition.class_path for name, definition in tracker_registry.TRACKER_DEFINITIONS.items()
    }
    for name, entry in _TRACKER_MANIFEST.items():
        assert tracker_registry.TRACKER_CLASS_SPECS[entry.class_path] == TrackerSpec(name)
        if entry.native_class_path is not None:
            assert tracker_registry.TRACKER_CLASS_SPECS[entry.native_class_path] == TrackerSpec(name, backend="cpp")


def test_native_registry_matches_the_lazy_tracker_manifest() -> None:
    expected = {name for name, entry in _TRACKER_MANIFEST.items() if entry.native_class_path is not None}

    assert set(supported_native_trackers()) == expected


def test_native_factory_validates_registered_geometry_modes() -> None:
    definition = tracker_registry.TrackerDefinition(
        name="bytetrack",
        class_path="boxmot.trackers.box.bytetrack.tracker.ByteTrack",
        capabilities=TrackerCapabilities(
            family=TrackerFamily.BOX,
            geometry_kinds=frozenset({GeometryKind.AABB, GeometryKind.OBB}),
            accepts_frame=True,
        ),
        native_class_path="boxmot.trackers.box.bytetrack.native.NativeByteTrackTracker",
        native_geometry_kinds=frozenset({GeometryKind.AABB}),
    )

    with pytest.raises(ValueError, match="does not support OBB geometry"):
        tracker_factory._create_native_tracker(
            TrackerSpec("bytetrack", backend="cpp", geometry="obb"),
            definition,
            GeometryKind.OBB,
        )


def test_tracker_definition_captures_component_requirements() -> None:
    strongsort = tracker_registry.get_tracker_definition("strongsort").capabilities
    bytetrack = tracker_registry.get_tracker_definition("bytetrack").capabilities

    assert strongsort.requires_embeddings is True
    assert strongsort.accepts_embeddings is True
    assert bytetrack.requires_embeddings is False
    assert bytetrack.accepts_embeddings is False
    assert tracker_registry.get_tracker_definition("bytetrack").accepts_per_class is True


def test_registered_capabilities_describe_every_tracker_family_and_input() -> None:
    definitions = tracker_registry.TRACKER_DEFINITIONS
    expected_embeddings = {"boosttrack", "botsort", "deepocsort", "hybridsort", "occluboost", "strongsort"}
    expected_multimodal = {"maf_hda", "sam2mot"}

    assert set(definitions) == set(_TRACKER_MANIFEST)
    assert {name for name, item in definitions.items() if item.capabilities.family is TrackerFamily.BOX} == (
        set(definitions) - expected_multimodal
    )
    assert {
        name for name, item in definitions.items() if item.capabilities.family is TrackerFamily.MULTIMODAL
    } == expected_multimodal
    for name, item in definitions.items():
        expected_geometry = {GeometryKind.AABB} if name == "maf_hda" else {GeometryKind.AABB, GeometryKind.OBB}
        assert item.capabilities.geometry_kinds == expected_geometry
    assert {name for name, item in definitions.items() if item.capabilities.accepts_embeddings} == expected_embeddings
    assert {name for name, item in definitions.items() if item.capabilities.requires_embeddings} == {"strongsort"}
    assert {name for name, item in definitions.items() if item.capabilities.accepts_masks} == expected_multimodal
    assert {name for name, item in definitions.items() if item.capabilities.requires_masks} == expected_multimodal
    assert all(item.capabilities.accepts_frame for item in definitions.values())
    assert {name for name, item in definitions.items() if item.capabilities.requires_frame} == {
        "maf_hda",
        "sam2mot",
        "strongsort",
    }


@pytest.mark.parametrize("tracker_name", tuple(tracker_registry.TRACKER_DEFINITIONS))
def test_direct_tracker_instances_match_registered_capabilities(tracker_name: str) -> None:
    definition = tracker_registry.get_tracker_definition(tracker_name)
    tracker = tracker_registry.get_tracker_class(tracker_name)()

    assert tracker.capabilities == definition.capabilities


def test_factory_rejects_an_unsupported_geometry_before_construction() -> None:
    definition = tracker_registry.TrackerDefinition(
        name="aabb-only",
        class_path="example.AabbOnly",
        capabilities=TrackerCapabilities(
            family=TrackerFamily.BOX,
            geometry_kinds=frozenset({GeometryKind.AABB}),
        ),
    )

    with pytest.raises(ValueError, match=r"does not support geometry kind 'obb'.*\['aabb'\]"):
        tracker_factory._validate_geometry(TrackerSpec("aabb-only", geometry="obb"), definition)


def test_factory_checks_resolved_requirements_against_static_capabilities() -> None:
    configurable = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        accepts_embeddings=True,
    )
    tracker = SimpleNamespace(requirements=TrackerRequirements(embeddings=True))

    assert tracker_factory._bind_and_validate_capabilities(tracker, configurable) is tracker
    assert tracker.capabilities is configurable

    unsupported = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=frozenset({GeometryKind.AABB}),
    )
    with pytest.raises(ValueError, match="static capabilities do not accept embeddings"):
        tracker_factory._bind_and_validate_capabilities(
            SimpleNamespace(requirements=TrackerRequirements(embeddings=True)),
            unsupported,
        )

    always_required = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        requires_frame=True,
        accepts_frame=True,
    )
    with pytest.raises(ValueError, match="omitted frame"):
        tracker_factory._bind_and_validate_capabilities(
            SimpleNamespace(requirements=TrackerRequirements()),
            always_required,
        )


def test_tracker_lookup_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown tracker type"):
        tracker_registry.get_tracker_definition("unknown_tracker")
    with pytest.raises(ValueError, match="Unknown tracker type"):
        tracker_registry.get_tracker_config("unknown_tracker")


def test_tracker_config_precedence_and_partial_custom_overlay(tmp_path) -> None:
    custom_path = tmp_path / "bytetrack.yaml"
    custom_path.write_text(yaml.safe_dump({"track_thresh": 0.7}), encoding="utf-8")

    resolved = load_tracker_config(
        "bytetrack",
        custom_path,
        {"track_buffer": 45, "match_thresh": 0.8},
        {"match_thresh": 0.75},
    )

    assert resolved["min_conf"] == 0.1
    assert resolved["track_thresh"] == 0.7
    assert resolved["track_buffer"] == 45
    assert resolved["match_thresh"] == 0.75


def test_sfsort_config_includes_default_obb_angle_damping() -> None:
    assert load_tracker_config("sfsort")["obb_theta_damping"] == 0.8


@pytest.mark.parametrize("tracker_name", tuple(tracker_registry.TRACKER_DEFINITIONS))
def test_all_python_configs_expose_canonical_association_choices(tracker_name: str) -> None:
    association = load_tracker_schema(tracker_name)["asso_func"]

    assert association["type"] == "choice"
    assert association["options"] == ["iou", "giou", "diou", "ciou", "hmiou", "centroid"]
    assert load_tracker_config(tracker_name)["asso_func"] == association["default"]


def test_create_tracker_merges_spec_options_then_applies_fixed_spec_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Tracker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.requirements = TrackerRequirements()

    monkeypatch.setattr(tracker_factory, "_load_tracker_class", lambda _definition: _Tracker)

    tracker_factory.create_tracker(
        TrackerSpec(
            "bytetrack",
            geometry="obb",
            per_class=True,
            class_ids=(0, 3),
            class_names=((0, "person"), (3, "car")),
            options=(
                ("asso_func", "giou"),
                ("is_obb", False),
                ("match_thresh", 0.75),
                ("per_class", False),
                ("track_buffer", 45),
            ),
        )
    )

    assert captured["asso_func"] == "giou"
    assert captured["match_thresh"] == 0.75
    assert captured["track_buffer"] == 45
    assert captured["is_obb"] is True
    assert captured["per_class"] is True
    assert captured["class_ids"] == (0, 3)
    assert captured["class_names"] == {0: "person", 3: "car"}


def test_create_tracker_rejects_old_keyword_factory_surface() -> None:
    with pytest.raises(TypeError, match="spec must be TrackerSpec"):
        tracker_factory.create_tracker("bytetrack")  # type: ignore[arg-type]


def test_create_tracker_rejects_unknown_algorithm_options() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'legacy_option'"):
        tracker_factory.create_tracker(TrackerSpec("bytetrack", options=(("legacy_option", True),)))


def test_create_tracker_dispatches_native_spec_without_model_options(monkeypatch) -> None:
    expected = object()
    monkeypatch.setattr(tracker_factory, "_create_native_tracker", lambda _spec, _definition, _kind: expected)

    monkeypatch.setattr(tracker_factory, "_bind_and_validate_capabilities", lambda tracker, _capabilities: tracker)
    assert tracker_factory.create_tracker(TrackerSpec("bytetrack", backend="cpp")) is expected


@pytest.mark.parametrize(
    ("tracker_name", "preset_name"),
    (
        ("botsort", "botsort-mot17-ablation"),
        ("occluboost", "occluboost-mot17-ablation"),
        ("occluboost", "occluboost-mot17-test"),
    ),
)
def test_builtin_preset_declares_and_strips_tracker_identity(tracker_name: str, preset_name: str) -> None:
    assert "tracker" not in load_tracker_config(tracker_name, preset_name)


def test_builtin_preset_rejects_wrong_tracker_identity() -> None:
    with pytest.raises(ValueError, match='is for "botsort", not "bytetrack"'):
        load_tracker_config("bytetrack", "botsort-mot17-ablation")


@pytest.mark.parametrize("tracker_name", tuple(tracker_registry.TRACKER_DEFINITIONS))
def test_tracker_defaults_are_scalar_constructor_parameters(tracker_name: str) -> None:
    tracker_class = tracker_registry.get_tracker_class(tracker_name)
    accepted = set(inspect.signature(BaseTracker.__init__).parameters)
    for owner in tracker_class.mro():
        if "__init__" in owner.__dict__:
            accepted.update(inspect.signature(owner.__init__).parameters)
    accepted -= {"self", "args", "kwargs", "per_class", "class_ids", "class_names", "is_obb"}

    defaults = load_tracker_config(tracker_name)

    assert set(defaults) <= accepted
    assert all(isinstance(value, (str, int, float, bool, type(None))) for value in defaults.values())


def test_tracker_config_rejects_collection_values(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump({"track_thresh": [0.5, 0.7]}), encoding="utf-8")

    with pytest.raises(ValueError, match="not nested or collection values"):
        load_tracker_config("bytetrack", config_path)


def test_builtin_config_path_is_registered() -> None:
    assert tracker_registry.get_tracker_config("botsort") == TRACKER_CONFIGS_DIR / "botsort.yaml"
