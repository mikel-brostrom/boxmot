from pathlib import Path

from boxmot.configs import BOXMOT_DEFAULTS, DEFAULT_DETECTOR, DEFAULT_REID, build_mode_namespace, ensure_model_extension, get_mode_default, get_mode_defaults
from boxmot.utils import WEIGHTS


def test_ensure_model_extension_preserves_explicit_export_paths():
    model_path = "models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite"

    resolved = ensure_model_extension(model_path)

    assert resolved == Path(model_path)


def test_ensure_model_extension_keeps_bare_reid_names_in_weights_dir():
    resolved = ensure_model_extension("osnet_x0_25_msmt17")

    assert resolved == WEIGHTS / "osnet_x0_25_msmt17.pt"


def test_build_mode_namespace_uses_shared_runtime_defaults():
    args = build_mode_namespace("eval", {"data": "mot17-mini"}, explicit_keys=set())

    assert args.detector == [DEFAULT_DETECTOR]
    assert args.reid == [DEFAULT_REID]
    assert args.tracker == get_mode_default("eval", "tracker")
    assert args.detector_explicit is False
    assert args.reid_explicit is False
    assert args.project == Path(get_mode_default("eval", "project"))
    assert args.show_timing is False


def test_get_mode_defaults_returns_normalized_merged_defaults():
    defaults = get_mode_defaults("eval")

    assert defaults["detector"] == DEFAULT_DETECTOR
    assert defaults["reid"] == DEFAULT_REID
    assert defaults["tracker"] == get_mode_default("eval", "tracker")
    assert defaults["project"] == Path(get_mode_default("eval", "project"))
    assert defaults["show_timing"] is False
    assert isinstance(defaults["n_threads"], int)
    assert defaults["n_threads"] >= 1


def test_boxmot_defaults_bundle_exposes_typed_mode_defaults():
    assert BOXMOT_DEFAULTS.shared.detector == DEFAULT_DETECTOR
    assert BOXMOT_DEFAULTS.shared.reid == DEFAULT_REID
    assert BOXMOT_DEFAULTS.track.tracker == get_mode_default("track", "tracker")
    assert BOXMOT_DEFAULTS.eval.project == Path(get_mode_default("eval", "project"))
    assert BOXMOT_DEFAULTS.export.include == tuple(get_mode_default("export", "include"))


def test_build_mode_namespace_normalizes_track_and_export_models():
    track_args = build_mode_namespace("track", {"source": "0"}, explicit_keys=set())
    export_args = build_mode_namespace("export", {"weights": "osnet_x0_25_msmt17", "include": ["onnx"]}, explicit_keys={"weights", "include"})

    assert track_args.detector == DEFAULT_DETECTOR
    assert track_args.reid == DEFAULT_REID
    assert export_args.weights == WEIGHTS / "osnet_x0_25_msmt17.pt"
    assert export_args.include == ("onnx",)
