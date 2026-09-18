"""Keep authored tracker searches complete and within algorithm constraints."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from boxmot.engine.tuning.search_space import flatten_yaml_config, load_yaml_config
from boxmot.trackers.common.config import get_tracker_config_class, get_tracker_config_path
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST


@pytest.mark.parametrize("name", tuple(_TRACKER_MANIFEST))
def test_tracker_search_profiles_explicitly_cover_every_algorithm_parameter(name: str) -> None:
    """Loader-injected defaults must not conceal missing tuning decisions."""
    authored = yaml.safe_load(get_tracker_config_path(name).read_text(encoding="utf-8"))
    algorithm_entries = {
        key: entry
        for key, entry in flatten_yaml_config(authored).items()
        if not key.startswith(("kalman.", "edgetam."))
    }

    assert algorithm_entries.keys() == set(get_tracker_config_class(name).fields())
    assert all("default" not in entry for entry in algorithm_entries.values())


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
