"""Public tracker constructor hints match meaningful inherited runtime options."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from typing_extensions import Unpack, is_typeddict

import boxmot
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_TRACKER_NAMES

_REID_OPTIONS = {"reid"}
_TIMING_OPTIONS = {"variable_dt"}
_NOISE_OPTIONS = {"kalman"}


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
def test_reid_configuration_is_explicit_only_for_appearance_trackers(tracker_name: str) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    signature = inspect.signature(tracker.__init__)
    if tracker.accepts_embeddings:
        parameter = signature.parameters["reid"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
        assert get_type_hints(tracker.__init__)["reid"] == ReIDConfig | AppearanceEncoder | None
    else:
        assert "reid" not in signature.parameters


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
@pytest.mark.parametrize("option", ["reid_model", "reid_weights", "device", "half", "reid_preprocess"])
def test_removed_reid_constructor_options_are_rejected(tracker_name: str, option: str) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    with pytest.raises(TypeError, match=option):
        tracker(**{option: None})


@pytest.mark.parametrize("tracker_name", ["bytetrack", "ocsort", "sfsort", "eagermot", "maf_hda"])
@pytest.mark.parametrize("reid", [None, ReIDConfig()])
def test_non_appearance_trackers_reject_explicit_reid_configuration(tracker_name: str, reid) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    with pytest.raises(TypeError, match="does not accept.*[Rr]e[Ii][Dd]"):
        tracker(reid=reid)


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


def _documented_arguments(docstring: str) -> list[tuple[str, str]]:
    """Read names and descriptions from one conventional Google Args section."""
    section = re.search(r"(?ms)^Args:\n(.*?)(?=^\S|\Z)", inspect.cleandoc(docstring))
    assert section is not None, "Constructor tooltip must contain a Google Args section"
    content = section.group(1)
    rows = list(re.finditer(r"(?m)^    (\*{0,2}[a-zA-Z_]\w*)(?: \([^\n]*\))?:", content))
    return [
        (row.group(1), content[row.end() : rows[index + 1].start() if index + 1 < len(rows) else len(content)].strip())
        for index, row in enumerate(rows)
    ]


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
    missing = {name.split(".", 1)[0] for name in load_tracker_defaults(tracker_name)} - advertised
    assert not missing, f"{public_name} config defaults lack constructor autocomplete: {sorted(missing)}"


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
def test_constructor_tooltip_documents_its_own_current_parameters(tracker_name: str) -> None:
    public_name = _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1]
    tracker = getattr(boxmot, public_name)
    constructor = tracker.__dict__["__init__"]
    assert tracker.__dict__["__doc__"], f"{public_name} must own a class overview docstring"
    docstring = constructor.__doc__
    assert docstring and docstring.strip(), f"{public_name}.__init__ must own its tooltip docstring"
    assert inspect.getdoc(constructor) == inspect.cleandoc(docstring)
    assert len(re.findall(r"(?m)^Args:$", inspect.cleandoc(docstring))) == 1
    assert not re.search(r"(?m)^Args:$", inspect.cleandoc(tracker.__doc__ or "")), (
        f"{public_name} constructor arguments must have one canonical docstring on __init__"
    )

    arguments = _documented_arguments(docstring)
    names = [name for name, _ in arguments]
    assert len(names) == len(set(names)), f"{public_name} repeats constructor argument documentation"
    assert set(names) == set(_keyword_parameters(constructor)) | {"**kwargs"}
    assert all(description for _, description in arguments), f"{public_name} has an empty argument description"


@pytest.mark.parametrize("tracker_name", tuple(_TRACKER_MANIFEST))
def test_constructor_kwargs_docs_only_advertise_supported_options(tracker_name: str) -> None:
    public_name = _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1]
    tracker = getattr(boxmot, public_name)
    constructor = tracker.__dict__["__init__"]
    assert constructor.__doc__, f"{public_name}.__init__ must own its tooltip docstring"
    arguments = dict(_documented_arguments(constructor.__doc__))
    (options,) = get_args(get_type_hints(constructor)["kwargs"])
    supported = set(get_type_hints(options))
    kwargs_docs = arguments["**kwargs"]
    advertised = set(re.findall(r"`{1,2}([a-z]\w*)`{1,2}", kwargs_docs))
    assert advertised <= supported, f"{public_name} documents unsupported inherited options: {advertised - supported}"
    assert advertised, f"{public_name} must describe meaningful forwarded constructor options"

    forbidden = set()
    if not tracker.accepts_embeddings:
        forbidden.update(_REID_OPTIONS)
    if not tracker.supports_obb:
        forbidden.add("is_obb")
    if not tracker.supports_variable_dt:
        forbidden.update(_TIMING_OPTIONS)
    if not (tracker.supports_variable_dt or tracker.supports_kalman_noise):
        forbidden.update(_NOISE_OPTIONS)
    if tracker_name in {"bytetrack", "sfsort"}:
        forbidden.add("det_thresh")
    mentioned = set(re.findall(r"\b[a-z]\w*\b", kwargs_docs))
    assert not forbidden.intersection(mentioned), f"{public_name} kwargs docs advertise unavailable settings"


@pytest.mark.parametrize("tracker_name", sorted(KALMAN_NOISE_TRACKER_NAMES))
def test_kalman_configuration_is_an_explicit_keyword_only_parameter(tracker_name: str) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    signature = inspect.signature(tracker.__init__)
    parameter = signature.parameters["kalman"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert get_type_hints(tracker.__init__)["kalman"] == boxmot.KalmanConfig | None
    assert not any(name.startswith("kf_") for name in signature.parameters)


@pytest.mark.parametrize("tracker_name", sorted(KALMAN_NOISE_TRACKER_NAMES))
@pytest.mark.parametrize(
    "option",
    [
        "kf_process_position_scale",
        "kf_process_velocity_scale",
        "kf_measurement_noise_scale",
        "kf_initial_position_scale",
        "kf_initial_velocity_scale",
        "kf_reference_dt_s",
        "kf_time_unit",
        "kalman_noise",
        "variable_dt",
        "adaptive_kf",
        "is_angular",
        "ams_enabled",
        "ams_alpha0",
        "ams_threshold",
        "ams_buffer_size",
        "ams_shrink_ratio",
    ],
)
def test_removed_scalar_kalman_constructor_keywords_are_rejected(tracker_name: str, option: str) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    with pytest.raises(TypeError, match=option):
        tracker(**{option: 1.0})


@pytest.mark.parametrize("tracker_name", ["sfsort", "maf_hda"])
@pytest.mark.parametrize("config", [None, boxmot.KalmanConfig()])
def test_trackers_without_kalman_motion_reject_configuration_objects(tracker_name: str, config) -> None:
    tracker = getattr(boxmot, _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1])
    with pytest.raises(TypeError, match="does not accept kalman"):
        tracker(kalman=config)
