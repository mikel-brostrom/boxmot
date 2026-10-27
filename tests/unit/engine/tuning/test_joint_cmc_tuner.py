"""Joint CMC selection stays frozen through Ray trials, export and resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.components.resolution import component_options
from boxmot.engine.calibration import cmc, kalman, kalman_selection
from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, resolve_cmc_policy
from boxmot.engine.tuning import analysis, postprocessing, tuner
from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_cmc_signature
from boxmot.trackers.common.motion.models import resolve_motion_model_kind, supported_motion_model_kinds
from boxmot.trackers.common.specs import TrackerSpec
from tests.unit.engine.calibration.test_cmc_candidates import _source_build
from tests.unit.engine.calibration.test_kalman import _data, _load_fixture
from tests.unit.engine.tuning.test_calibrated_tuner import _SuggestionTrial
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning


@pytest.mark.parametrize("selected_cmc", ["off", "auto"])
@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_joint_choice_reaches_every_trial_and_saved_best_without_reanalysis_on_resume(
    fake_tuning: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    selected_cmc: str,
    search_alg: str,
    tracker: str,
) -> None:
    """Keep real fitting, bank persistence, actor dispatch and config serialization."""
    captured = fake_tuning.captured
    loads = _load_fixture(monkeypatch, _data())
    warps = {"seq1": np.tile(np.eye(2, 3), (6, 1, 1))}
    warps["seq1"][1:, 0, 2] = 0.25
    mask_threshold = 0.17
    defaults = load_tracker_config(tracker)
    selected_kind = resolve_motion_model_kind(tracker).value
    threshold_name = next(name for name in ("min_conf", "det_thresh", "cmc_min_conf") if name in defaults)
    search_threshold = "track_high_thresh" if tracker == "botsort" else threshold_name
    fit_calls = []
    fitted_kinds = []

    def fit(*args, **kwargs):
        captured["events"].append("calibrate")
        fit_calls.append(kwargs.get("camera_warps"))
        fitted_kinds.append(kwargs["profile_name"])
        return calibrate_kalman(*args, **kwargs)

    def prepare_camera(args, baseline, *, output_dir, **kwargs):
        captured["events"].append("camera_analysis")
        assert baseline["cmc_min_conf"] == mask_threshold
        output_dir.mkdir(parents=True)
        report_path, warps_path = output_dir / "analysis.json", output_dir / "warps.npz"
        _source_build(args.build_path)
        manifest = args.build_path / "manifest.json"
        np.savez(
            warps_path,
            seq1_warps=warps["seq1"].astype(np.float32),
            seq1_frame_indices=np.arange(6, dtype=np.int64),
        )
        policy = SequenceCMCPolicy(sequences=(("seq1", True),))
        image_root = str(args.build_path.parent / "cmc-images")
        references = tuple(f"seq1/{index}.png" for index in range(6))
        for reference in references:
            image = Path(image_root) / reference
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture pixels")
        report_path.write_text(
            json.dumps(
                {
                    "schema": cmc.CMC_ANALYSIS_SCHEMA,
                    "status": "complete",
                    "ground_truth_used": False,
                    "build_path": str(args.build_path.resolve()),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "source_signature": cmc.cmc_build_signature(args.build_path),
                    "split": args.split,
                    "tracker": args.tracker,
                    "backend": args.tracker_backend,
                    "geometry": args.geometry,
                    "per_class": args.per_class,
                    "rule": CMC_ANALYSIS_RULE,
                    "method": "sof",
                    "opencv_version": cv2.__version__,
                    "settings": cmc.cmc_analysis_settings(baseline),
                    "mask_min_conf": mask_threshold,
                    "runtime_mask": cmc.cmc_runtime_mask_settings({**baseline, "cmc_min_conf": mask_threshold}),
                    "class_ids": list(args.tracker_class_ids),
                    "class_names": list(args.tracker_class_names),
                    "policy": policy.to_dict(),
                    "sequences": {"seq1": {
                        "frames": 6, "enabled": True,
                        "sample_ids": [f"seq1/{index}" for index in range(6)],
                        "image_refs": references, "source_image_root": image_root,
                        "image_signature": cmc.cmc_image_signature(references, image_root),
                        "archive_keys": {
                            "warps": "seq1_warps", "runtime_warps": "seq1_warps",
                            "frame_indices": "seq1_frame_indices",
                        },
                    }},
                    "total_frames": 6,
                    "warps_sha256": hashlib.sha256(warps_path.read_bytes()).hexdigest(),
                }
            )
        )
        return SimpleNamespace(
            warps=warps,
            policy=policy,
            report_path=report_path,
            warps_path=warps_path,
        )

    def select(data, *, configs, camera_candidates, **kwargs):
        captured["events"].append("select")
        assert camera_candidates["off"] is None
        assert camera_candidates["auto"] is warps
        candidate = {
            "selected_type": selected_kind,
            "selected_profile": "motion",
            "candidates": {
                kind: {"status": "valid", "selected_profile": "motion", "mean_iou": 0.8} for kind in configs
            },
        }
        return {
            **candidate,
            "method": "grouped_cross_validated_cmc_box_profiles",
            "selected_cmc": selected_cmc,
            "cmc_candidates": {"off": candidate, "auto": candidate},
        }

    def save_results(directory, results, *args, **kwargs):
        # Real Ray creates its trial directories before postprocessing.
        for result in results:
            Path(result.path).mkdir(parents=True, exist_ok=True)
        return postprocessing.save_all_results(directory, results, *args, **kwargs)

    original_suggest = _SuggestionTrial.suggest_float
    sampled_values = (0.31, 0.37) if tracker == "botsort" else (0.11, 0.21)
    sampled_confidences = iter(sampled_values)

    def suggest_float(self, name, low, high, **kwargs):
        if name == search_threshold:
            self.params[name] = next(sampled_confidences)
            return self.params[name]
        return original_suggest(self, name, low, high, **kwargs)

    monkeypatch.setattr(kalman, "calibrate_kalman", fit)
    monkeypatch.setattr(cmc, "prepare_cmc_candidates", prepare_camera)
    monkeypatch.setattr(kalman_selection, "select_kalman_model", select)
    monkeypatch.setattr(tuner, "save_all_results", save_results)
    monkeypatch.setattr(analysis, "generate_tune_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(_SuggestionTrial, "suggest_float", suggest_float)
    original_setup_ray = tuner.Tuner._setup_ray

    def setup_ray(self, *args, **kwargs):
        # This assertion runs at dispatch time, not merely after fit completes.
        assert self.args._cmc_setup.strategy == selected_cmc
        assert (Path(self.args._tune_sequence_progress_dir).parent / "cmc-policy.yaml").is_file()
        return original_setup_ray(self, *args, **kwargs)

    monkeypatch.setattr(tuner.Tuner, "_setup_ray", setup_ray)
    options = {
        "tracker": tracker,
        "calibrate_kf": False,
        "_build_validated": True,
        "variable_dt": False,
        "tracker_class_ids": (0,),
        "tracker_class_names": ((0, "person"),),
        "sequence_names": ("seq1",),
        "seq_info": {"seq1": 6},
        "search_alg": search_alg,
    }
    driver = tuner.Tuner(
        fake_tuning.args(**options, select_kf=True, select_cmc=True), baseline_config={threshold_name: mask_threshold}
    )
    _, directory, *_ = driver.fit()

    assert captured["events"].index("camera_analysis") < captured["events"].index("select")
    assert captured["events"].index("select") < captured["events"].index("ray_init")
    assert driver.args._cmc_setup.policy.apply(TrackerSpec(tracker), "seq1").option_dict["use_cmc"] is (
        selected_cmc == "auto"
    )
    assert driver._make_safe_namespace()._cmc_setup == driver.args._cmc_setup
    assert captured["trial_args"]._cmc_setup == driver.args._cmc_setup
    assert (directory / "cmc-policy.yaml").is_file()
    assert len(loads) == 1
    expected_kinds = (
        {kind.value for kind in supported_motion_model_kinds(is_obb=driver.args.geometry == "obb")}
    )
    assert set(fitted_kinds) == expected_kinds
    assert len(fit_calls) == len(fitted_kinds) == len(expected_kinds)
    assert all(value is (warps if selected_cmc == "auto" else None) for value in fit_calls)
    assert len(captured["trial_configs"]) == 2
    if search_alg == "optuna":
        assert [config[search_threshold] for config in captured["trial_configs"]] == list(sampled_values)

    fixed = {
        "kalman.type": selected_kind,
        "calibration.filter": selected_kind,
        "calibration.profile": "motion",
        "cmc_method": "sof",
        "cmc_min_conf": mask_threshold,
        "calibration.cmc_min_conf": mask_threshold,
        "calibration.cmc_policy": selected_cmc,
        "calibration.cmc_rule": CMC_ANALYSIS_RULE,
        **calibration_cmc_signature(defaults),
    }
    schema = flatten_yaml_config(driver._yaml_cfg)
    for name in ("kalman.type", "use_cmc", "cmc_method", "cmc_min_conf"):
        assert driver._calibrated_fixed_options[name] == driver._runtime_config[name]
        if name in schema:
            assert schema[name] == {"default": driver._runtime_config[name]}
    assert "type" in schema[search_threshold]
    exported_paths = [directory / "best.yaml"]
    exported_paths.extend(
        Path(result.path) / f"{tracker}_{result.metrics['trial_id']}.yaml"
        for result in captured["saved_results"]
    )
    saved = [load_tracker_config(tracker, path) for path in exported_paths]
    for path in exported_paths:
        header = "\n".join(line for line in path.read_text().splitlines() if line.startswith("#"))
        assert "effective cmc" in header.lower()
        assert selected_cmc in header
        assert "seq1" in header
    summary = (directory / "summary.md").read_text()
    cmc_section = summary.split("## Effective CMC Setup", 1)[1].split("\n## ", 1)[0]
    assert selected_cmc in cmc_section
    assert "seq1" in cmc_section
    for config in [driver._runtime_config, *captured["trial_configs"], *saved]:
        assert {key: config[key] for key in fixed} == fixed
        spec = TrackerSpec(tracker, options=component_options(config))
        policy = resolve_cmc_policy(driver.args, spec)
        assert policy == ("auto" if selected_cmc == "auto" else SequenceCMCPolicy())

    snapshots = {path: path.read_bytes() for path in (directory / "kf-tuning").rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Resuming a completed joint choice must not repeat analysis, selection, or calibration.")

    monkeypatch.setattr(cmc, "prepare_cmc_candidates", forbidden)
    monkeypatch.setattr(kalman_selection, "select_kalman_model", forbidden)
    monkeypatch.setattr(kalman, "calibrate_kalman", forbidden)
    captured["restore_enabled"] = True
    resumed = tuner.Tuner(fake_tuning.args(**options, resume_tune=directory))
    resumed.fit()
    assert len(loads) == 1
    assert captured["events"].count("camera_analysis") == 1
    assert captured["events"].count("select") == 1
    assert captured["events"].count("restore") == 1
    assert len(captured["trial_configs"]) == 4
    assert resumed._runtime_config == driver._runtime_config
    assert all({key: config[key] for key in fixed} == fixed for config in captured["trial_configs"])
    assert all(path.read_bytes() == content for path, content in snapshots.items())
    assert load_tracker_config(tracker, directory / "best.yaml") == saved[0]
