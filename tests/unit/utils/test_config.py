"""Safe YAML parsing preserves values, isolation, and on-disk edits."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

from boxmot.trackers.common.config import _load_mapping
from boxmot.utils.config import load_yaml_mapping


@pytest.fixture(params=["available", "python"])
def safe_parser(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise both the installed safe parser and the pure-Python fallback."""
    if request.param == "python":
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)


@pytest.fixture(params=[load_yaml_mapping, partial(_load_mapping, label="tracker")], ids=["shared", "tracker"])
def read_config(request: pytest.FixtureRequest, safe_parser: None) -> Callable[[Path], dict[str, Any]]:
    """Apply the same file contract to both configuration entrypoints."""
    return request.param


def test_yaml_preserves_scalar_types_and_aliases(tmp_path: Path, read_config: Callable) -> None:
    """Nested numeric settings and aliases retain their authored types."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "settings: &settings\n  enabled: true\n  count: 3\n  scale: 0.25\n"
        "  type: null\n  label: '003'\n  names: [car, bicyclist]\nalias: *settings\n",
        encoding="utf-8",
    )

    result = read_config(path)

    assert result["settings"] == {
        "enabled": True, "count": 3, "scale": 0.25, "type": None,
        "label": "003", "names": ["car", "bicyclist"],
    }
    assert type(result["settings"]["enabled"]) is bool
    assert type(result["settings"]["count"]) is int
    assert type(result["settings"]["scale"]) is float
    assert result["alias"] is result["settings"]


def test_yaml_reads_are_independent_and_observe_file_edits(tmp_path: Path, read_config: Callable) -> None:
    """Mutating one result cannot leak into later reads, including edited files."""
    path = tmp_path / "config.yaml"
    path.write_text("settings:\n  values: [1, 2]\n", encoding="utf-8")
    first = read_config(path)
    first["settings"]["values"].append(3)

    assert read_config(path) == {"settings": {"values": [1, 2]}}

    path.write_text("settings:\n  values: [4, 5]\n", encoding="utf-8")
    assert read_config(path) == {"settings": {"values": [4, 5]}}


@pytest.mark.parametrize("source", ["value: [1,\n", "value: !!python/tuple [1, 2]\n"])
def test_yaml_rejects_malformed_content_and_python_tags(tmp_path: Path, read_config: Callable, source: str) -> None:
    """Both safe loaders reject invalid YAML and Python object construction."""
    path = tmp_path / "config.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse"):
        read_config(path)


def test_yaml_requires_a_mapping(tmp_path: Path, read_config: Callable) -> None:
    """Valid YAML lists are not configuration mappings."""
    path = tmp_path / "config.yaml"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        read_config(path)
