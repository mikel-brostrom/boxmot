"""Native workflow options distinguish backend defaults from authored settings."""

from types import SimpleNamespace

import pytest
import yaml

from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.trackers.common.config import load_tracker_config, load_tracker_defaults
from boxmot.trackers.common.factory import create_tracker
from boxmot.trackers.common.protocols import TrackerRequirements
from boxmot.trackers.common.registry import supported_native_trackers
from boxmot.trackers.common.specs import TrackerSpec


def _args(tracker: str, **values) -> SimpleNamespace:
    return SimpleNamespace(tracker=tracker, tracker_backend="cpp", **values)


@pytest.mark.parametrize("tracker", ["botsort", "occluboost"])
@pytest.mark.parametrize("include_defaults", [False, True])
def test_native_defaults_reach_factory_without_python_only_options(monkeypatch, tracker, include_defaults) -> None:
    class NativeTracker:
        requirements = TrackerRequirements()

        def __init__(self, options, *, geometry):
            self.options = options

    monkeypatch.setattr("boxmot.trackers.common.factory._load_native_tracker_class", lambda _: NativeTracker)
    options = resolve_tracker_options(_args(tracker), include_defaults=include_defaults, factory_options=True)
    result = create_tracker(TrackerSpec(tracker, backend="cpp", options=tuple(sorted(options.items()))))
    assert result.options == {}


@pytest.mark.parametrize("tracker, unsupported", [("botsort", "removed_stracks_buffer"), ("occluboost", "adaptive_kf")])
@pytest.mark.parametrize("source", ["override", "config"])
def test_native_explicit_unsupported_options_still_reject_even_at_default_values(
    tmp_path, tracker, unsupported, source
) -> None:
    value = load_tracker_defaults(tracker)[unsupported]
    args = _args(tracker)
    overrides = {unsupported: value}
    if source == "config":
        profile = tmp_path / "tracker.yaml"
        profile.write_text(yaml.safe_dump({"tracker": tracker, **overrides}))
        args.tracker_config = profile
        overrides = None
    options = resolve_tracker_options(args, overrides, include_defaults=True, factory_options=True)
    assert options[unsupported] == value
    with pytest.raises(ValueError, match=f"does not implement.*{unsupported}"):
        create_tracker(TrackerSpec(tracker, backend="cpp", options=tuple(sorted(options.items()))))


def test_native_scalar_config_preserves_override_precedence_without_inherited_defaults(tmp_path) -> None:
    profile = tmp_path / "tracker.yaml"
    profile.write_text("tracker: botsort\nuse_cmc: false\nuse_embeddings: false\nasso_func: giou\n")
    options = resolve_tracker_options(
        _args("botsort", tracker_config=profile, asso_func="centroid"),
        {"use_embeddings": True},
        include_defaults=True,
        factory_options=True,
    )
    assert options == {"use_cmc": False, "use_embeddings": True, "asso_func": "centroid"}


def test_sparse_config_selection_keeps_preset_metadata_validation_and_python_defaults(tmp_path) -> None:
    assert load_tracker_config("botsort", "botsort", include_defaults=False) == {}
    preset = load_tracker_config("botsort", "botsort-mot17-ablation", include_defaults=False)
    assert preset["track_buffer"] == 40
    assert "tracker" not in preset
    # This preset explicitly authors a Python-only option; keep its rejection.
    assert "removed_stracks_buffer" in preset
    options = resolve_tracker_options(_args("botsort", tracker_config="botsort-mot17-ablation"), factory_options=True)
    with pytest.raises(ValueError, match="does not implement.*removed_stracks_buffer"):
        create_tracker(TrackerSpec("botsort", backend="cpp", options=tuple(sorted(options.items()))))
    assert "removed_stracks_buffer" in resolve_tracker_options(
        SimpleNamespace(tracker="botsort", tracker_backend="python"), include_defaults=True
    )
    profile = tmp_path / "mismatch.yaml"
    profile.write_text("tracker: botsort\ntrack_buffer: 40\n")
    with pytest.raises(ValueError, match="botsort.*bytetrack"):
        resolve_tracker_options(_args("bytetrack", tracker_config=profile), factory_options=True)


def test_native_sparse_options_still_validate_timing_and_calibrated_noise(tmp_path) -> None:
    with pytest.raises(ValueError, match="variable_dt"):
        resolve_tracker_options(_args("bytetrack", variable_dt=True), include_defaults=True, factory_options=True)
    profile = tmp_path / "calibrated.yaml"
    profile.write_text("tracker: bytetrack\nkf_measurement_noise_scale: 0.5\n")
    with pytest.raises(ValueError, match="requires a Python Kalman tracker"):
        resolve_tracker_options(_args("bytetrack", tracker_config=profile), include_defaults=True, factory_options=True)


@pytest.mark.parametrize("tracker", supported_native_trackers())
def test_ordinary_native_configuration_keeps_full_defaults(tracker: str) -> None:
    options = resolve_tracker_options(_args(tracker), include_defaults=True)
    assert options == load_tracker_defaults(tracker)


def test_ordinary_native_tracker_config_keeps_inherited_defaults(tmp_path) -> None:
    profile = tmp_path / "tracker.yaml"
    profile.write_text("tracker: botsort\nuse_embeddings: false\n")
    options = resolve_tracker_options(_args("botsort", tracker_config=profile))
    assert options == {**load_tracker_defaults("botsort"), "use_embeddings": False}
