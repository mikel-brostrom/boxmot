import inspect

import pytest
import yaml

import boxmot.trackers.registry as tracker_registry
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.config import TRACKER_CONFIGS_DIR, load_tracker_config, load_tracker_schema
from boxmot.trackers.registry import (
    REID_TRACKERS,
    TRACKER_CLASS_TO_NAME,
    TRACKER_DEFINITIONS,
    TRACKER_MAPPING,
    get_tracker_config,
    get_tracker_definition,
)


def test_tracker_public_mappings_are_derived_from_definitions():
    assert TRACKER_MAPPING == {name: definition.class_path for name, definition in TRACKER_DEFINITIONS.items()}
    assert REID_TRACKERS == [name for name, definition in TRACKER_DEFINITIONS.items() if definition.needs_reid]
    assert TRACKER_CLASS_TO_NAME == {
        definition.class_name.lower(): name for name, definition in TRACKER_DEFINITIONS.items()
    }


def test_tracker_definition_captures_constructor_metadata():
    strongsort = get_tracker_definition("strongsort")
    bytetrack = get_tracker_definition("bytetrack")

    assert strongsort.needs_reid is True
    assert strongsort.accepts_per_class is True
    assert bytetrack.needs_reid is False
    assert bytetrack.accepts_per_class is True


def test_get_tracker_config_preserves_unknown_tracker_path_behavior():
    assert get_tracker_config("botsort") == TRACKER_CONFIGS_DIR / "botsort.yaml"
    assert get_tracker_config("unknown_tracker") == TRACKER_CONFIGS_DIR / "unknown_tracker.yaml"


def test_get_tracker_definition_rejects_unknown_tracker():
    with pytest.raises(ValueError, match="Unknown tracker type: 'unknown_tracker'"):
        get_tracker_definition("unknown_tracker")


def test_tracker_config_precedence_and_partial_custom_overlay(tmp_path):
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


def test_sfsort_canonical_config_includes_obb_theta_damping():
    defaults = load_tracker_config("sfsort")

    assert defaults["obb_theta_damping"] == 0.8


@pytest.mark.parametrize("tracker_name", TRACKER_DEFINITIONS)
def test_all_python_tracker_configs_expose_canonical_association_choices(tracker_name):
    association = load_tracker_schema(tracker_name)["asso_func"]

    assert association["type"] == "choice"
    assert association["options"] == ["iou", "giou", "diou", "ciou", "hmiou", "centroid"]
    assert association["default"] in association["options"]
    assert load_tracker_config(tracker_name)["asso_func"] == association["default"]


def test_create_tracker_applies_tuned_values_before_tracker_kwargs(monkeypatch, tmp_path):
    captured = {}
    custom_path = tmp_path / "bytetrack.yaml"
    custom_path.write_text(yaml.safe_dump({"track_thresh": 0.7}), encoding="utf-8")

    class _Tracker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tracker_registry, "_load_tracker_class", lambda definition: _Tracker)

    tracker_registry.create_tracker(
        "bytetrack",
        tracker_config=custom_path,
        evolve_param_dict={"track_buffer": 45, "match_thresh": 0.8},
        tracker_kwargs={"match_thresh": 0.75, "asso_func": "giou"},
        per_class=False,
    )

    assert captured["min_conf"] == 0.1
    assert captured["track_thresh"] == 0.7
    assert captured["track_buffer"] == 45
    assert captured["match_thresh"] == 0.75
    assert captured["asso_func"] == "giou"


def test_create_tracker_can_skip_warmup_for_a_shared_reid_model(monkeypatch):
    class _Model:
        def __init__(self):
            self.warmup_calls = 0

        def warmup(self):
            self.warmup_calls += 1

    class _Tracker:
        def __init__(self, **kwargs):
            self.model = kwargs["reid_model"]

    model = _Model()
    monkeypatch.setattr(tracker_registry, "_load_tracker_class", lambda definition: _Tracker)

    tracker = tracker_registry.create_tracker(
        "botsort",
        reid_model=model,
        warmup_model=False,
    )

    assert tracker.model is model
    assert model.warmup_calls == 0


@pytest.mark.parametrize(
    ("tracker_name", "preset_name"),
    [
        ("botsort", "botsort-mot17-ablation"),
        ("occluboost", "occluboost-mot17-ablation"),
        ("occluboost", "occluboost-mot17-test"),
        ("occluboost", "occluboost-sportsmot-val"),
    ],
)
def test_builtin_preset_declares_and_strips_tracker_identity(tracker_name, preset_name):
    resolved = load_tracker_config(tracker_name, preset_name)

    assert "tracker" not in resolved


def test_builtin_preset_rejects_wrong_tracker_identity():
    with pytest.raises(ValueError, match='is for "botsort", not "bytetrack"'):
        load_tracker_config("bytetrack", "botsort-mot17-ablation")


@pytest.mark.parametrize("tracker_name", TRACKER_DEFINITIONS)
def test_tracker_defaults_are_scalar_constructor_parameters(tracker_name):
    tracker_class = tracker_registry.get_tracker_class(tracker_name)
    accepted = set(inspect.signature(BaseTracker.__init__).parameters)
    for owner in tracker_class.mro():
        if "__init__" in owner.__dict__:
            accepted.update(inspect.signature(owner.__init__).parameters)
    accepted -= {"self", "args", "kwargs", "reid_model", "per_class", "class_ids", "class_names", "is_obb"}

    defaults = load_tracker_config(tracker_name)

    assert set(defaults) <= accepted
    assert all(isinstance(value, (str, int, float, bool, type(None))) for value in defaults.values())


def test_tracker_config_rejects_collection_values(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump({"track_thresh": [0.5, 0.7]}), encoding="utf-8")

    with pytest.raises(ValueError, match="not nested or collection values"):
        load_tracker_config("bytetrack", config_path)
