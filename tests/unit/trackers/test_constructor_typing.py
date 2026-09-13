"""Public tracker constructor hints match meaningful inherited runtime options."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from typing_extensions import Unpack, is_typeddict

import boxmot
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST

_REID_OPTIONS = {"reid_model", "reid_weights", "device", "half", "reid_preprocess"}
_TIMING_OPTIONS = {"variable_dt", "kf_reference_dt_s", "kf_time_unit"}
_NOISE_OPTIONS = {
    "kf_process_position_scale",
    "kf_process_velocity_scale",
    "kf_measurement_noise_scale",
    "kf_initial_position_scale",
    "kf_initial_velocity_scale",
}


def _keyword_parameters(constructor: Any) -> dict[str, inspect.Parameter]:
    """Identify explicitly supported keyword arguments without expanding kwargs."""
    return {
        name: parameter
        for name, parameter in inspect.signature(constructor).parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _inherited_parameters(tracker: type, tracker_name: str) -> dict[str, Any]:
    """Derive relevant options from runtime ancestors and capability restrictions."""
    inherited = {}
    for parent in tracker.__mro__[1:]:
        constructor = parent.__dict__.get("__init__")
        if constructor is None or not inspect.isfunction(constructor):
            continue
        annotations = get_type_hints(constructor)
        for name in _keyword_parameters(constructor):
            inherited.setdefault(name, annotations[name])
    omitted = set(_keyword_parameters(tracker.__init__))
    if not tracker.accepts_embeddings:
        omitted.update(_REID_OPTIONS)
    if not tracker.supports_obb:
        omitted.add("is_obb")
    if not tracker.supports_variable_dt:
        omitted.update(_TIMING_OPTIONS)
    if not (tracker.supports_variable_dt or tracker.supports_kalman_noise):
        omitted.update(_NOISE_OPTIONS)
    if tracker_name in {"bytetrack", "sfsort"}:
        omitted.add("det_thresh")
    return {name: annotation for name, annotation in inherited.items() if name not in omitted}


def test_public_static_tracker_exports_match_manifest() -> None:
    """Editors must see the same canonical tracker classes as lazy runtime imports."""
    source = Path(boxmot.__file__).read_text(encoding="utf-8")
    expected = {entry.class_path.rsplit(".", 1)[1]: entry.class_path for entry in _TRACKER_MANIFEST.values()}
    exports = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) or node.test.id != "TYPE_CHECKING":
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if alias.name in expected:
                        assert alias.asname == alias.name, f"{alias.name} must be an explicit public re-export"
                        exports[alias.name] = f"{statement.module}.{alias.name}"
    assert exports == expected


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
def test_constructor_kwargs_match_meaningful_inherited_options(tracker_name: str) -> None:
    public_name = _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1]
    tracker = getattr(boxmot, public_name)
    parameters = inspect.signature(tracker.__init__).parameters
    variadic = [parameter for parameter in parameters.values() if parameter.kind is inspect.Parameter.VAR_KEYWORD]
    assert len(variadic) == 1
    annotation = get_type_hints(tracker.__init__)[variadic[0].name]
    assert get_origin(annotation) is Unpack, f"{public_name} must expose inherited constructor options with Unpack"
    (options,) = get_args(annotation)
    assert is_typeddict(options)
    assert not options.__required_keys__, f"{public_name} inherited constructor options must remain optional"
    assert not set(options.__annotations__).intersection(_keyword_parameters(tracker.__init__))
    assert get_type_hints(options) == _inherited_parameters(tracker, tracker_name)


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
def test_authored_defaults_are_discoverable_in_public_constructor(tracker_name: str) -> None:
    public_name = _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1]
    constructor = getattr(boxmot, public_name).__init__
    (options,) = get_args(get_type_hints(constructor)["kwargs"])
    advertised = set(_keyword_parameters(constructor)) | set(get_type_hints(options))
    missing = set(load_tracker_defaults(tracker_name)) - advertised
    assert not missing, f"{public_name} config defaults lack constructor autocomplete: {sorted(missing)}"
