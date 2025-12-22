# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import importlib
import yaml

from boxmot.utils import TRACKER_CONFIGS

REID_TRACKERS = ["strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"]

TRACKER_MAPPING = {
    "strongsort": "boxmot.trackers.strongsort.strongsort.StrongSort",
    "ocsort"    : "boxmot.trackers.ocsort.ocsort.OcSort",
    "bytetrack" : "boxmot.trackers.bytetrack.bytetrack.ByteTrack",
    "botsort"   : "boxmot.trackers.botsort.botsort.BotSort",
    "deepocsort": "boxmot.trackers.deepocsort.deepocsort.DeepOcSort",
    "hybridsort": "boxmot.trackers.hybridsort.hybridsort.HybridSort",
    "boosttrack": "boxmot.trackers.boosttrack.boosttrack.BoostTrack",
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
):
    """
    Creates and returns an instance of the specified tracker type.

    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision.
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.

    Returns:
    - An instance of the selected tracker.

    Raises:
    - ValueError: If `tracker_type` is not recognized.
    """

    if tracker_type not in TRACKER_MAPPING:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        if tracker_config is None: 
            # Load default tracker config 
            tracker_config = get_tracker_config(tracker_type)
        with open(tracker_config, "r") as f:
            yaml_config = yaml.safe_load(f)
            tracker_args = {
                param: details["default"] for param, details in yaml_config.items()
            }
    else:
        tracker_args = evolve_param_dict.copy()

    # Prepare arguments
    tracker_args["per_class"] = per_class

    if tracker_type in REID_TRACKERS:
        tracker_args.update({
            "reid_weights": reid_weights,
            "device": device,
            "half": half,
        })

    # Tracker-specific adjustments
    if tracker_type == "strongsort":
        tracker_args.pop("per_class", None)

    # Dynamically import and instantiate the correct tracker class
    module_path, class_name = TRACKER_MAPPING[tracker_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    tracker_class = getattr(module, class_name)

    # Return the instantiated tracker class with arguments and warmed-up models
    tracker = tracker_class(**tracker_args)
    if hasattr(tracker, "model"):
        tracker.model.warmup()
    return tracker
