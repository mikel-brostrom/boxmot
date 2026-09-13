"""Grouped public configuration survives factory, YAML, and search boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from boxmot import KalmanNoiseConfig, OcSort, create_tracker
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import (
    flatten_tracker_options,
    load_tracker_config,
    nest_tracker_options,
)
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature


def test_grouped_runtime_yaml_merges_fields_without_resetting_prior_overrides(tmp_path: Path) -> None:
    profile = tmp_path / "ocsort.yaml"
    profile.write_text(
        "tracker: ocsort\nvariable_dt: true\nkalman_noise:\n  measurement_noise_scale: 2.5\n  reference_dt_s: 0.05\n"
    )
    config = load_tracker_config("ocsort", profile, {"kalman_noise": {"process_velocity_scale": 3.0}})
    assert config["kalman_noise.measurement_noise_scale"] == 2.5
    assert config["kalman_noise.process_velocity_scale"] == 3.0
    assert config["kalman_noise.initial_position_scale"] == 1.0
    tracker = create_tracker(TrackerSpec("ocsort", options=tuple(sorted(config.items()))))
    assert tracker.kalman_noise_config.time_unit == "seconds"
    assert tracker.kalman_noise_config.reference_dt_s == 0.05
    assert tracker.kalman_noise_config.measurement_noise_scale == 2.5


def test_direct_factory_and_yaml_configs_create_the_same_noise(tmp_path: Path) -> None:
    noise = KalmanNoiseConfig(process_position_scale=1.3, measurement_noise_scale=2.5)
    direct = OcSort(kalman_noise=noise, variable_dt=True)
    factory = create_tracker("ocsort", kalman_noise=noise, variable_dt=True)
    profile = tmp_path / "calibrated.yaml"
    profile.write_text(yaml.safe_dump(nest_tracker_options({"kalman_noise": noise, "variable_dt": True})))
    loaded = create_tracker("ocsort", options=load_tracker_config("ocsort", profile))
    assert direct.kalman_noise_config == factory.kalman_noise_config == loaded.kalman_noise_config
    assert noise.time_unit is None  # Resolution must not mutate reusable user settings.
    with pytest.raises(FrozenInstanceError):
        noise.measurement_noise_scale = 9.0


def test_class_config_round_trip_retains_global_fallback_and_class_identity(tmp_path: Path) -> None:
    noise = KalmanNoiseConfig(
        measurement_noise_scale=2.0,
        by_class={
            0: KalmanNoiseConfig(measurement_noise_scale=0.2),
            20000: KalmanNoiseConfig(process_velocity_scale=4.0),
        },
    )
    original = {"per_class": True, "kalman_noise": noise}
    flattened = flatten_tracker_options(original)
    assert flattened["kalman_noise.by_class.20000.process_velocity_scale"] == 4.0
    payload = nest_tracker_options(flattened)
    assert flatten_tracker_options(payload) == flattened
    path = tmp_path / "class-noise.yaml"
    path.write_text(yaml.safe_dump(payload))
    config = load_tracker_config("ocsort", path)
    per_class = config.pop("per_class")
    tracker = create_tracker("ocsort", per_class=per_class, options=config)
    assert tracker.kalman_noise_config.for_class(0).measurement_noise_scale == 0.2
    assert tracker.kalman_noise_config.for_class(1).measurement_noise_scale == 2.0
    assert tracker.kalman_noise_config.for_class(20000).process_velocity_scale == 4.0


def test_empty_class_profile_uses_defaults_instead_of_pooled_overrides(tmp_path: Path) -> None:
    profile = tmp_path / "class-noise.yaml"
    profile.write_text(
        "per_class: true\nkalman_noise:\n  measurement_noise_scale: 4.0\n  by_class:\n    1: {}\n"
    )
    options = load_tracker_config("ocsort", profile)
    tracker = create_tracker("ocsort", per_class=options.pop("per_class"), options=options)
    direct = OcSort(
        per_class=True,
        kalman_noise=KalmanNoiseConfig(measurement_noise_scale=4.0, by_class={1: KalmanNoiseConfig()}),
    )
    assert tracker.kalman_noise_config == direct.kalman_noise_config
    assert tracker.kalman_noise_config.for_class(1).measurement_noise_scale == 1.0
    assert tracker.kalman_noise_config.for_class(2).measurement_noise_scale == 4.0


@pytest.mark.parametrize(
    "options",
    [
        {"kalman_noise": {"measurement_noise_scale": 2.0}, "kalman_noise.measurement_noise_scale": 3.0},
        {"kalman_noise": {"by_class": {"01": {"measurement_noise_scale": 2.0}}}},
        {"kf_measurement_noise_scale": 2.0},
    ],
)
def test_ambiguous_or_removed_config_paths_fail(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        flatten_tracker_options(options)


def test_saved_units_cannot_be_reinterpreted_by_factory() -> None:
    noise = KalmanNoiseConfig(time_unit="frames", measurement_noise_scale=2.0)
    with pytest.raises(ValueError, match="time_unit"):
        create_tracker("ocsort", kalman_noise=noise, variable_dt=True)


@pytest.mark.parametrize("name", ["sfsort", "maf_hda"])
def test_non_kalman_factory_rejects_explicit_noise(name: str) -> None:
    with pytest.raises((TypeError, ValueError), match="[Kk]alman"):
        create_tracker(name, kalman_noise=KalmanNoiseConfig())


def test_native_factory_rejects_nondefault_noise_before_loading_bindings() -> None:
    with pytest.raises(ValueError, match="Python Kalman tracker"):
        create_tracker("ocsort", backend="cpp", kalman_noise=KalmanNoiseConfig(measurement_noise_scale=2.0))


def test_factory_binds_single_class_calibration_to_its_declared_class() -> None:
    signature = {
        **calibration_profile_signature("eagermot", "aabb", {}),
        "class_id": 2,
        "class_name": "pedestrian",
    }
    tracker = create_tracker("eagermot", calibration=signature)
    assert tracker.class_ids == frozenset({2})
    create_tracker("eagermot", class_ids=(2,), calibration=signature)
    for selected in ((1,), (1, 2)):
        with pytest.raises(ValueError, match="profile is for class 2"):
            create_tracker("eagermot", class_ids=selected, calibration=signature)
