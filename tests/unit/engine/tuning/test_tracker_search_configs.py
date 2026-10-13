"""Keep authored tracker searches complete and within algorithm constraints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from boxmot.engine.tuning.search_space import condition_tracker_schema, flatten_yaml_config, load_yaml_config
from boxmot.trackers.common.config import get_tracker_config_class, get_tracker_config_path, load_tracker_defaults
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_tracker_search_profiles_declare_defaults_and_preserve_omitted_fields(name: str) -> None:
    """Search entries expose defaults while omitted settings keep typed values."""
    authored = yaml.safe_load(get_tracker_config_path(name).read_text(encoding="utf-8"))
    entries = flatten_yaml_config(authored)
    algorithm_entries = {
        key: entry
        for key, entry in entries.items()
        if not key.startswith(("kalman.", "edgetam."))
    }

    defaults = get_tracker_config_class(name)().to_dict()
    assert algorithm_entries.keys() <= defaults.keys()
    for parameter, entry in entries.items():
        searchable = "type" in entry or any("type" in case for case in entry.get("cases", {}).values())
        if searchable:
            assert "default" in entry, f"{name}.{parameter} must declare its default"
        if parameter in defaults and "default" in entry:
            assert type(entry["default"]) is type(defaults[parameter])
            assert entry["default"] == defaults[parameter]

    resolved = flatten_yaml_config(load_yaml_config(name))
    for parameter in defaults.keys() - algorithm_entries.keys():
        assert resolved[parameter] == {"default": defaults[parameter]}


@pytest.mark.parametrize(
    "parameter,value",
    [("track_thresh", 0.5), ("fuse_first_associate", 1), ("max_lost_frames", 30.0), ("min_conf", "0.1")],
)
def test_tracker_search_defaults_cannot_drift_from_typed_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parameter: str, value: object,
) -> None:
    """Reject changed values and equal-looking defaults with incompatible types."""
    path = tmp_path / "bytetrack.yaml"
    path.write_text(yaml.safe_dump({parameter: {"default": value}}), encoding="utf-8")
    monkeypatch.setattr("boxmot.trackers.common.config.get_tracker_config_path", lambda name: path)

    with pytest.raises(ValueError, match=f"{parameter!r} must match its typed default"):
        load_yaml_config("bytetrack")


def _search_boundaries(entry: dict[str, Any]) -> Iterator[object]:
    """Exercise numeric extremes and every categorical choice a search can emit."""
    kind = entry.get("type")
    if kind in {"choice", "grid_search"}:
        yield from entry.get("options", entry.get("values", ()))
    elif kind in {"uniform", "loguniform"}:
        low, high = entry["range"]
        assert low < high
        yield low
        yield high
    elif kind in {"randint", "qrandint"}:
        low, high, *quantization = entry["range"]
        step = quantization[0] if quantization else 1
        assert all(type(value) is int for value in (low, high, step))
        assert low < high and step > 0
        yield low
        yield low + ((high - 1 - low) // step) * step


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_tracker_search_boundaries_satisfy_algorithm_config_constraints(name: str) -> None:
    """Invalid search endpoints would otherwise waste trials on construction failures."""
    config_type = get_tracker_config_class(name)
    schema = flatten_yaml_config(load_yaml_config(name))

    for parameter in config_type.fields():
        for candidate in _search_boundaries(schema[parameter]):
            try:
                config_type.from_mapping({parameter: candidate})
            except (TypeError, ValueError) as exc:
                pytest.fail(f"{name}.{parameter} cannot use search value {candidate!r}: {exc}")


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_lost_track_lifetime_is_one_directly_searchable_frame_count(name: str, geometry: str) -> None:
    """Both geometries search the effective lifetime without a second FPS dimension."""
    schema = load_yaml_config(name)
    defaults = load_tracker_defaults(name)
    flat = flatten_yaml_config(condition_tracker_schema(schema, defaults, geometry=geometry))

    assert flat["max_lost_frames"]["type"] == "randint"
    assert flat["max_lost_frames"]["range"] == [1, 200]
    assert not {"track_buffer", "buffer_size", "max_time_lost", "frame_rate"}.intersection(flat)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_occluboost_shared_parameters_remain_searchable_in_both_geometries(geometry: str) -> None:
    """Geometry filtering preserves the complete shared search, including optional stages."""
    schema = load_yaml_config("occluboost")
    defaults = load_tracker_defaults("occluboost")
    for parameter, value in (("asso_func", "iou"), ("use_embeddings", True), ("use_second_pass", True)):
        schema[parameter] = {**schema[parameter], "type": "choice", "options": [value]}
        defaults[parameter] = value
    flat = flatten_yaml_config(condition_tracker_schema(schema, defaults, geometry=geometry))

    ranges = {
        "max_age": [1, 200],
        "det_thresh": [0.05, 0.9],
        "iou_threshold": [0.01, 0.99],
        "new_track_thresh": [0.05, 0.9],
        "instant_confirm_thresh": [0.1, 0.95],
        "recovery_max_age": [1, 120],
        "second_iou_thresh": [0.05, 0.9],
    }
    assert not any(parameter.startswith("obb_") for parameter in flat)
    for parameter, expected in ranges.items():
        assert flat[parameter]["type"] in {"uniform", "randint"}
        assert flat[parameter]["range"] == expected
