"""Typed algorithm configurations are shared by direct and factory construction."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import boxmot
from boxmot import BotSortConfig, ByteTrackConfig, OcSortConfig, create_tracker
from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_defaults, load_tracker_schema
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.trackers.common.registry import get_tracker_class


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_each_tracker_has_one_public_frozen_algorithm_config(name: str) -> None:
    config_type = get_tracker_config_class(name)
    public_name = _TRACKER_MANIFEST[name].class_path.rsplit(".", 1)[1]
    assert getattr(boxmot, f"{public_name}Config") is config_type
    config = config_type()
    with pytest.raises(FrozenInstanceError):
        config.min_hits = 7
    changed = replace(config, min_hits=7)
    assert changed.min_hits == 7
    assert config.min_hits != changed.min_hits


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_direct_and_factory_construction_resolve_identical_algorithm_defaults(name: str) -> None:
    config_type = get_tracker_config_class(name)
    direct = get_tracker_class(name)()
    factory = create_tracker(name)
    assert direct.config == factory.config == config_type()
    defaults = load_tracker_defaults(name)
    schema = flatten_yaml_config(load_tracker_schema(name))
    for field in fields(config_type):
        assert defaults[field.name] == getattr(direct.config, field.name)
        assert schema[field.name]["default"] == defaults[field.name]


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_config_mappings_roundtrip_and_do_not_share_mutable_state(name: str) -> None:
    config_type = get_tracker_config_class(name)
    config = config_type(min_hits=7)
    mapping = config.to_dict()
    assert config_type.from_mapping(mapping) == config
    mapping["min_hits"] = 9
    assert config.min_hits == 7
    assert config_type.from_mapping(mapping).min_hits == 9
    assert create_tracker(TrackerSpec(name, options=tuple(sorted(config.to_dict().items())))).config == config


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
@pytest.mark.parametrize("value", [True, "3", 1.5, None])
def test_algorithm_configs_validate_integer_types_before_tracker_initialization(name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="min_hits"):
        get_tracker_config_class(name)(min_hits=value)


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_algorithm_configs_reject_unknown_fields_and_mismatched_trackers(name: str) -> None:
    config_type = get_tracker_config_class(name)
    with pytest.raises(TypeError, match="unknown_option"):
        config_type.from_mapping({"unknown_option": 1})
    wrong = ByteTrackConfig() if name != "bytetrack" else BotSortConfig()
    with pytest.raises(TypeError):
        get_tracker_class(name)(config=wrong)
    with pytest.raises((TypeError, ValueError)):
        create_tracker(name, config=wrong)
    with pytest.raises(TypeError, match="min_hits"):
        get_tracker_class(name)(min_hits=7)


def test_factory_overrides_config_without_mutating_it() -> None:
    original = BotSortConfig(match_thresh=0.8, track_buffer=45)
    spec = TrackerSpec("botsort", options=(("match_thresh", 0.7),))
    tracker = create_tracker(spec, config=original, options={"track_buffer": 50}, match_thresh=0.6)
    assert tracker.config.match_thresh == 0.6
    assert tracker.config.track_buffer == 50
    assert original.match_thresh == 0.8
    assert original.track_buffer == 45
    assert dict(spec.options) == {"match_thresh": 0.7}


class _RecordingNativeLibrary:
    """Capture the resolved ABI configuration without building a native library."""

    def __init__(self) -> None:
        self.config: dict[str, object] | None = None

    def create(self, config: Mapping[str, object]) -> object:
        self.config = dict(config)
        return object()

    def destroy(self, handle: object) -> None:
        pass


@pytest.mark.parametrize("name", [name for name, entry in _TRACKER_MANIFEST.items() if entry.native_class_path])
def test_native_factory_resolves_typed_config_and_algorithm_overrides(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Factory config injection reaches each native adapter and its runtime requirements."""
    module = import_module(f"boxmot.trackers.{name}.native")
    library = _RecordingNativeLibrary()
    monkeypatch.setattr(module, f"get_{name}_library", lambda: library)
    config_type = get_tracker_config_class(name)
    appearance_options = {key: False for key in ("use_cmc", "use_embeddings") if key in config_type.fields()}
    original = config_type(max_obs=73, **appearance_options)

    tracker = create_tracker(name, backend="cpp", config=original, asso_func="centroid")
    try:
        assert type(tracker.config) is config_type
        assert tracker.config == replace(original, asso_func="centroid")
        assert original.asso_func == "iou"
        assert library.config["max_obs"] == 73
        assert library.config["asso_func"] == "centroid"
        assert tracker.requirements.frame
        assert tracker.requirements.frame_dimensions_only
        if name == "botsort":
            assert tracker.config.removed_stracks_buffer == BotSortConfig().removed_stracks_buffer
    finally:
        tracker.close()


def test_native_factory_rejects_nondefault_botsort_python_history_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python-only history option is tolerated only at its canonical default."""
    module = import_module("boxmot.trackers.botsort.native")
    library = _RecordingNativeLibrary()
    monkeypatch.setattr(module, "get_botsort_library", lambda: library)
    config = BotSortConfig(removed_stracks_buffer=BotSortConfig().removed_stracks_buffer + 1)

    with pytest.raises(ValueError, match="removed_stracks_buffer"):
        create_tracker("botsort", backend="cpp", config=config)

    assert library.config is None


def test_public_algorithm_configs_load_without_tracker_or_model_dependencies() -> None:
    """Authoring configuration should not initialize any runtime inference library."""
    script = """
import sys
import boxmot
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
for entry in _TRACKER_MANIFEST.values():
    config_name = entry.config_class_path.rsplit(".", 1)[1]
    config = getattr(boxmot, config_name)()
    assert config.to_dict()
assert not any(name.endswith(".tracker") for name in sys.modules if name.startswith("boxmot.trackers."))
assert not any(name in sys.modules for name in ("cv2", "numpy", "torch", "yaml"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_numeric_scalars_are_normalized_for_json_and_factory_specs() -> None:
    """NumPy-derived tuning values retain the immutable JSON serialization contract."""
    config = OcSortConfig(min_hits=np.int64(2), det_thresh=np.float32(0.6))
    assert type(config.min_hits) is int
    assert type(config.det_thresh) is float
    assert OcSortConfig.from_mapping(json.loads(json.dumps(config.to_dict()))) == config
    assert create_tracker("ocsort", config=config).config == config
