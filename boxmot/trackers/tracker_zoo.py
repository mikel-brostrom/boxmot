# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import importlib

import yaml

from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.reid.core import ReID
from boxmot.trackers.specs import normalize_tracker_backend
from boxmot.utils import TRACKER_CONFIGS

REID_TRACKERS = ["strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack", "occluboost"]

TRACKER_MAPPING = {
    "strongsort": "boxmot.trackers.bbox.strongsort.strongsort.StrongSort",
    "ocsort"    : "boxmot.trackers.bbox.ocsort.ocsort.OcSort",
    "bytetrack" : "boxmot.trackers.bbox.bytetrack.bytetrack.ByteTrack",
    "sfsort"    : "boxmot.trackers.bbox.sfsort.sfsort.SFSORT",
    "botsort"   : "boxmot.trackers.bbox.botsort.botsort.BotSort",
    "deepocsort": "boxmot.trackers.bbox.deepocsort.deepocsort.DeepOcSort",
    "hybridsort": "boxmot.trackers.bbox.hybridsort.hybridsort.HybridSort",
    "boosttrack": "boxmot.trackers.bbox.boosttrack.boosttrack.BoostTrack",
    "occluboost": "boxmot.trackers.bbox.occluboost.occluboost.OccluBoost",
    "sam2mot"   : "boxmot.trackers.hybrid.sam2mot.sam2mot.Sam2Mot",
}


def get_tracker_config(tracker_type):
    """Returns the path to the tracker configuration file."""
    return TRACKER_CONFIGS / f"{tracker_type}.yaml"


def create_tracker(
    tracker_type,
    tracker_config=None,
    reid_weights=None,
    device=None,
    half=None,
    per_class=None,
    evolve_param_dict=None,
    reid_preprocess=None,
    reid_model=None,
    tracker_backend="python",
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
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    - reid_preprocess: Preprocessing method for the ReID backend (only used when building from ``reid_weights``).
    - reid_model: Pre-built ReID backend (e.g., ``ReID(...).model``). Takes
        precedence over ``reid_weights`` and lets callers share a single backend across trackers.
    - tracker_backend: Backend to use for the tracker. ``"python"`` (default)
        uses the pure-Python implementation under ``boxmot.trackers``. ``"cpp"``
        delegates to the registered native (C++) live backend via
        :func:`boxmot.native.registry.get_native_live_backend`. The native
        backend is built on demand if it isn't already compiled.

    Returns:
    - An instance of the selected tracker.

    Raises:
    - ValueError: If `tracker_type` is not recognized or the requested
      ``tracker_backend`` is not available for that tracker.
    """

    backend = normalize_tracker_backend(tracker_backend, default="python")

    if backend == "cpp":
        # Lazy import to keep ``boxmot.native`` (which pulls in ctypes / CMake
        # plumbing) optional for pure-Python users.
        from boxmot.native.registry import get_native_live_backend

        native = get_native_live_backend(tracker_type)
        cfg_dict = None
        if evolve_param_dict is not None:
            cfg_dict = dict(evolve_param_dict)
        elif tracker_config is not None:
            with open(tracker_config, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
                flat_config = flatten_yaml_config(yaml_config)
                cfg_dict = {
                    param: details["default"] for param, details in flat_config.items()
                }
        # Native live constructors take ``(cfg_dict, reid_weights=...)``; only
        # ReID-aware trackers consume ``reid_weights``. Passing ``reid_weights``
        # to non-ReID trackers is harmless because the unused kwarg is ignored
        # by the wrapper signatures.
        kwargs = {}
        if tracker_type in REID_TRACKERS:
            kwargs["reid_weights"] = reid_weights
            if reid_preprocess is not None:
                kwargs["reid_preprocess"] = reid_preprocess
        return native.create_tracker(cfg_dict, **kwargs)

    if tracker_type not in TRACKER_MAPPING:
        available = ", ".join(TRACKER_MAPPING.keys())
        raise ValueError(f"Unknown tracker type: '{tracker_type}'. Available trackers are: {available}")

    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        if tracker_config is None:
            # Load default tracker config
            tracker_config = get_tracker_config(tracker_type)
        with open(tracker_config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            flat_config = flatten_yaml_config(yaml_config)
            tracker_args = {
                param: details["default"] for param, details in flat_config.items()
            }
    else:
        tracker_args = evolve_param_dict.copy()

    # Prepare arguments
    tracker_args["per_class"] = per_class

    if tracker_type in REID_TRACKERS:
        if reid_model is None and reid_weights is not None:
            reid_model = ReID(
                weights=reid_weights,
                device=device,
                half=half,
                preprocess_name=reid_preprocess,
            ).model
        tracker_args["reid_model"] = reid_model

    # Tracker-specific adjustments
    if tracker_type == "strongsort":
        tracker_args.pop("per_class", None)

    # Dynamically import and instantiate the correct tracker class
    module_path, class_name = TRACKER_MAPPING[tracker_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    tracker_class = getattr(module, class_name)

    # Return the instantiated tracker class with arguments and warmed-up models
    tracker = tracker_class(**tracker_args)
    if hasattr(tracker, "model") and tracker.model is not None:
        tracker.model.warmup()
    return tracker
