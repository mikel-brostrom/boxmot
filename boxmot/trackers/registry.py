# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.trackers.specs import normalize_tracker_backend
from boxmot.utils import TRACKER_CONFIGS


@dataclass(frozen=True, slots=True)
class TrackerDefinition:
    """Registered tracker metadata used by factories and workflow adapters."""

    name: str
    class_path: str
    config_name: str | None = None
    needs_reid: bool = False
    accepts_per_class: bool = True
    warmup_model: bool = True

    @property
    def config_path(self) -> Path:
        return TRACKER_CONFIGS / f"{self.config_name or self.name}.yaml"

    @property
    def class_name(self) -> str:
        return self.class_path.rsplit(".", 1)[-1]


TRACKER_DEFINITIONS = {
    "strongsort": TrackerDefinition(
        name="strongsort",
        class_path="boxmot.trackers.bbox.strongsort.StrongSort",
        needs_reid=True,
        accepts_per_class=False,
    ),
    "ocsort": TrackerDefinition(
        name="ocsort",
        class_path="boxmot.trackers.bbox.ocsort.OcSort",
    ),
    "bytetrack": TrackerDefinition(
        name="bytetrack",
        class_path="boxmot.trackers.bbox.bytetrack.ByteTrack",
    ),
    "sfsort": TrackerDefinition(
        name="sfsort",
        class_path="boxmot.trackers.bbox.sfsort.SFSORT",
    ),
    "botsort": TrackerDefinition(
        name="botsort",
        class_path="boxmot.trackers.bbox.botsort.BotSort",
        needs_reid=True,
    ),
    "deepocsort": TrackerDefinition(
        name="deepocsort",
        class_path="boxmot.trackers.bbox.deepocsort.DeepOcSort",
        needs_reid=True,
    ),
    "hybridsort": TrackerDefinition(
        name="hybridsort",
        class_path="boxmot.trackers.bbox.hybridsort.HybridSort",
        needs_reid=True,
    ),
    "boosttrack": TrackerDefinition(
        name="boosttrack",
        class_path="boxmot.trackers.bbox.boosttrack.BoostTrack",
        needs_reid=True,
    ),
    "occluboost": TrackerDefinition(
        name="occluboost",
        class_path="boxmot.trackers.bbox.occluboost.OccluBoost",
        needs_reid=True,
    ),
    "sam2mot": TrackerDefinition(
        name="sam2mot",
        class_path="boxmot.trackers.hybrid.sam2mot.sam2mot.Sam2Mot",
    ),
}

TRACKER_MAPPING = {name: definition.class_path for name, definition in TRACKER_DEFINITIONS.items()}
REID_TRACKERS = [name for name, definition in TRACKER_DEFINITIONS.items() if definition.needs_reid]
TRACKER_CLASS_TO_NAME = {definition.class_name.lower(): name for name, definition in TRACKER_DEFINITIONS.items()}


def get_tracker_definition(tracker_type: str) -> TrackerDefinition:
    """Return registered metadata for a tracker type."""

    try:
        return TRACKER_DEFINITIONS[tracker_type]
    except KeyError as exc:
        available = ", ".join(TRACKER_MAPPING.keys())
        raise ValueError(f"Unknown tracker type: '{tracker_type}'. Available trackers are: {available}") from exc


def get_tracker_config(tracker_type):
    """Returns the path to the tracker configuration file."""

    definition = TRACKER_DEFINITIONS.get(tracker_type)
    if definition is not None:
        return definition.config_path
    return TRACKER_CONFIGS / f"{tracker_type}.yaml"


def _load_config_defaults(tracker_config: str | Path) -> dict[str, Any]:
    with open(tracker_config, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f) or {}
    flat_config = flatten_yaml_config(yaml_config)
    return {param: details["default"] for param, details in flat_config.items()}


def _resolve_tracker_args(
    definition: TrackerDefinition,
    tracker_config: str | Path | None,
    evolve_param_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    if evolve_param_dict is not None:
        return evolve_param_dict.copy()

    return _load_config_defaults(tracker_config or definition.config_path)


def _resolve_native_tracker_args(
    tracker_config: str | Path | None,
    evolve_param_dict: dict[str, Any] | None,
    tracker_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if evolve_param_dict is not None:
        cfg_dict = evolve_param_dict.copy()
    elif tracker_config is not None:
        cfg_dict = _load_config_defaults(tracker_config)
    else:
        cfg_dict = None

    if tracker_kwargs:
        cfg_dict = {} if cfg_dict is None else cfg_dict
        cfg_dict.update(tracker_kwargs)
    return cfg_dict


def _load_tracker_class(definition: TrackerDefinition):
    module_path, class_name = definition.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_tracker_class(tracker_type: str):
    """Return the registered tracker class for a tracker type."""

    return _load_tracker_class(get_tracker_definition(tracker_type))


def _build_reid_model(
    *,
    reid_weights=None,
    device=None,
    half=None,
    reid_preprocess=None,
    reid_model=None,
):
    if reid_model is not None or reid_weights is None:
        return reid_model

    from boxmot.reid.core import ReID

    return ReID(
        weights=reid_weights,
        device=device,
        half=half,
        preprocess_name=reid_preprocess,
    ).model


def _create_native_tracker(
    tracker_type: str,
    *,
    definition: TrackerDefinition | None,
    tracker_config=None,
    reid_weights=None,
    evolve_param_dict=None,
    tracker_kwargs=None,
    reid_preprocess=None,
):
    # Lazy import keeps ``boxmot.native`` optional for pure-Python users.
    from boxmot.native.registry import get_native_live_backend

    native = get_native_live_backend(tracker_type)
    cfg_dict = _resolve_native_tracker_args(tracker_config, evolve_param_dict, tracker_kwargs)
    kwargs = {}
    if definition is not None and definition.needs_reid:
        kwargs["reid_weights"] = reid_weights
        if reid_preprocess is not None:
            kwargs["reid_preprocess"] = reid_preprocess
    return native.create_tracker(cfg_dict, **kwargs)


def create_tracker(
    tracker_type,
    tracker_config=None,
    reid_weights=None,
    device=None,
    half=None,
    per_class=None,
    class_ids=None,
    class_names=None,
    evolve_param_dict=None,
    tracker_kwargs=None,
    reid_preprocess=None,
    reid_model=None,
    tracker_backend="python",
    precomputed_reid: bool = False,
):
    """
    Creates and returns an instance of the specified tracker type.

    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification). Used to build a ReID backend
        when ``reid_model`` is not supplied.
    - device: Device to run the ReID backend on (only used when building from ``reid_weights``).
    - half: Whether to use half-precision for the ReID backend (only used when building from ``reid_weights``).
    - per_class: Boolean for class-specific tracking (optional).
    - class_ids: Optional detector class IDs allowed by this tracker.
    - class_names: Optional detector class names keyed by detector class ID.
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    - tracker_kwargs: Constructor overrides applied after default YAML config resolution.
    - reid_preprocess: Preprocessing method for the ReID backend (only used when building from ``reid_weights``).
    - reid_model: Pre-built ReID backend (e.g., ``ReID(...).model``). Takes
        precedence over ``reid_weights`` and lets callers share a single backend across trackers.
    - tracker_backend: Backend to use for the tracker. ``"python"`` (default)
        uses the pure-Python implementation under ``boxmot.trackers``. ``"cpp"``
        delegates to the registered native (C++) live backend via
        :func:`boxmot.native.registry.get_native_live_backend`. The native
        backend is built on demand if it isn't already compiled.
    - precomputed_reid: Whether appearance embeddings will be supplied by the
        caller. When enabled, ReID-capable trackers keep appearance matching
        active without constructing a live ReID backend from ``reid_weights``.

    Returns:
    - An instance of the selected tracker.

    Raises:
    - ValueError: If `tracker_type` is not recognized or the requested
      ``tracker_backend`` is not available for that tracker.
    """

    backend = normalize_tracker_backend(tracker_backend, default="python")
    definition = TRACKER_DEFINITIONS.get(tracker_type)

    if backend == "cpp":
        return _create_native_tracker(
            tracker_type,
            definition=definition,
            tracker_config=tracker_config,
            reid_weights=reid_weights,
            evolve_param_dict=evolve_param_dict,
            tracker_kwargs=tracker_kwargs,
            reid_preprocess=reid_preprocess,
        )

    definition = get_tracker_definition(tracker_type)
    tracker_args = _resolve_tracker_args(definition, tracker_config, evolve_param_dict)
    if tracker_kwargs:
        tracker_args.update(tracker_kwargs)
    tracker_args["per_class"] = per_class
    if class_ids is not None:
        tracker_args["class_ids"] = class_ids
    if class_names is not None:
        tracker_args["class_names"] = class_names

    if definition.needs_reid:
        if precomputed_reid:
            tracker_args["reid_model"] = reid_model
        else:
            tracker_args["reid_model"] = _build_reid_model(
                reid_weights=reid_weights,
                device=device,
                half=half,
                reid_preprocess=reid_preprocess,
                reid_model=reid_model,
            )
            if tracker_args["reid_model"] is None and "with_reid" in tracker_args:
                tracker_args["with_reid"] = False

    if not definition.accepts_per_class:
        tracker_args.pop("per_class", None)

    tracker_class = _load_tracker_class(definition)
    tracker = tracker_class(**tracker_args)
    if definition.warmup_model and hasattr(tracker, "model") and tracker.model is not None:
        tracker.model.warmup()
    return tracker


__all__ = (
    "REID_TRACKERS",
    "TRACKER_CLASS_TO_NAME",
    "TRACKER_DEFINITIONS",
    "TRACKER_MAPPING",
    "TrackerDefinition",
    "create_tracker",
    "get_tracker_config",
    "get_tracker_class",
    "get_tracker_definition",
)
