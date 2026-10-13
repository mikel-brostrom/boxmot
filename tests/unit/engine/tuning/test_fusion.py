"""Native tuning excludes fusion policies that its ABI cannot implement."""

from copy import deepcopy

import pytest

from boxmot.engine.tuning.fusion import condition_fusion_schema, validate_fusion_tuning
from boxmot.engine.tuning.search_space import condition_tracker_schema, flatten_yaml_config, load_yaml_config
from boxmot.engine.tuning.tuner import Tuner
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.native import NATIVE_FUSION_DEFAULTS, load_native_tracker_config
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning
from tests.unit.engine.tuning.test_fixed_search_space import _sample

NATIVE_TRACKERS = ("bytetrack", "botsort", "ocsort", "occluboost", "sfsort")


@pytest.mark.parametrize("tracker", NATIVE_TRACKERS)
@pytest.mark.parametrize("search_backend", ["optuna", "hyperopt", "random"])
def test_native_search_freezes_unsupported_fusion_and_subordinate_controls(tracker, search_backend):
    defaults = load_tracker_defaults(tracker)
    original = load_yaml_config(tracker)
    snapshot = deepcopy(original)
    schema = condition_fusion_schema(original, backend="cpp")
    schema = condition_tracker_schema(schema, defaults, geometry="aabb")
    flat = flatten_yaml_config(schema)
    sampled = _sample(schema, search_backend)
    fusion_keys = {
        *NATIVE_FUSION_DEFAULTS,
        "motion_fusion_lambda",
        "motion_fusion_only_position",
    }.intersection(defaults)
    assert fusion_keys
    assert not fusion_keys.intersection(sampled)
    assert all(flat[key] == {"default": defaults[key]} for key in fusion_keys)
    assert original == snapshot
    for supported in ("fuse_first_associate", "fuse_unconfirmed_associate"):
        if supported in flat:
            assert supported in sampled


@pytest.mark.parametrize("tracker", NATIVE_TRACKERS)
def test_python_fusion_search_is_preserved(tracker):
    original = load_yaml_config(tracker)
    assert condition_fusion_schema(original, backend="python") == original


@pytest.mark.parametrize("tracker", NATIVE_TRACKERS)
def test_native_runtime_accepts_disabled_fusion_defaults(tracker):
    config = load_native_tracker_config(tracker, {})
    for name, disabled in NATIVE_FUSION_DEFAULTS.items():
        if name in config:
            assert config[name] == disabled


@pytest.mark.parametrize(
    ("tracker", "option", "value"),
    [
        ("bytetrack", "motion_fusion", "gate"),
        ("bytetrack", "motion_fusion", "blend"),
        ("botsort", "motion_fusion", "gate"),
        ("botsort", "fuse_iou", True),
        ("ocsort", "motion_fusion", "blend"),
        ("ocsort", "fuse_detection_score", True),
        ("occluboost", "motion_fusion", "gate"),
        ("occluboost", "fuse_iou", True),
        ("occluboost", "fuse_detection_score", True),
        ("sfsort", "fuse_detection_score", True),
    ],
)
def test_native_runtime_rejects_enabled_unsupported_fusion(tracker, option, value):
    with pytest.raises(ValueError, match=rf"Native {tracker}.*{option}.*Python"):
        load_native_tracker_config(tracker, {option: value})
    with pytest.raises(ValueError, match=rf"Native {tracker}.*{option}.*Python"):
        validate_fusion_tuning(tracker, {option: value}, backend="cpp")
    validate_fusion_tuning(tracker, {option: value}, backend="python")


def test_authored_native_fusion_fails_before_evaluation_and_ray(fake_tuning):
    args = fake_tuning.args(tracker="bytetrack", tracker_backend="cpp", calibrate_kf=False, variable_dt=False)
    driver = Tuner(args, baseline_config={"motion_fusion": "blend"})
    with pytest.raises(ValueError, match="Native bytetrack.*motion_fusion"):
        driver.fit()
    assert fake_tuning.captured["events"] == []


def test_native_driver_freezes_fusion_in_every_trial(fake_tuning):
    args = fake_tuning.args(tracker="bytetrack", tracker_backend="cpp", calibrate_kf=False, variable_dt=False)
    driver = Tuner(args)
    driver.fit()
    assert driver._yaml_cfg["motion_fusion"] == {"default": "off"}
    assert fake_tuning.captured["trial_configs"]
    for trial in fake_tuning.captured["trial_configs"]:
        assert trial["motion_fusion"] == "off"
        assert trial["motion_fusion_lambda"] == 0.98
        assert trial["motion_fusion_only_position"] is False
