"""Guided searches optimize active tracker knobs and preserve their execution profile."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.trackers import edgetam_checkpoint, resolve_tracker_options
from boxmot.engine.tuning import tuner
from boxmot.engine.tuning.backends.hyperopt_backend import yaml_to_hyperopt_space
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.mask_guidance import (
    condition_mask_guidance_schema,
    mask_guidance_trial_resources,
    prepare_mask_guidance_tuning,
    record_mask_guidance_tuning,
)
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import load_yaml_config, normalize_trial_config, yaml_to_tune_space
from boxmot.engine.tuning.trainable import build_tracker_trainable
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS
from tests.unit.engine.tuning.test_calibrated_tuner import _SuggestionTrial


def _args(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "tracker": "bytetrack",
            "tracker_backend": "python",
            "device": "cpu",
            "mask_guidance_weights": Path("edgetam.pt"),
            "mask_guidance_max_objects": None,
            "edgetam": True,
            "sequence_workers": 1,
            "resume_tune": None,
            **overrides,
        }
    )


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
@pytest.mark.parametrize("guided", [False, True])
def test_all_search_backends_only_sample_active_guidance_knobs(backend: str, guided: bool) -> None:
    args = _args(edgetam=guided)
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    original = load_yaml_config(args.tracker)
    snapshot = deepcopy(original)
    schema = condition_mask_guidance_schema(original, args, runtime)
    if backend == "optuna":
        trial = _SuggestionTrial()
        yaml_to_optuna_define_space(schema)(trial)
        sampled = trial.params
    elif backend == "hyperopt":
        sampled = yaml_to_hyperopt_space(schema)
    else:
        from ray import tune

        sampled = yaml_to_tune_space(schema, tune)
    assert original == snapshot
    assert set(MASK_GUIDANCE_OPTIONS).intersection(sampled) == (set(MASK_GUIDANCE_OPTIONS) if guided else set())
    assert ("asso_func" in sampled) is not guided
    if guided:
        assert schema["asso_func"] == {"default": "iou"}
    assert "edgetam" not in sampled
    assert not {"min_coverage", "min_fill", "prompt_overlap", "max_objects"}.intersection(sampled)


def test_nested_edgetam_profile_roundtrips_through_saved_best_yaml(tmp_path: Path) -> None:
    args = _args()
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(args.tracker), args, runtime)
    nested = {"edgetam": {"min_coverage": 0.87, "min_fill": 0.07, "prompt_overlap": 0.12, "max_objects": 16}}
    trial = normalize_trial_config(nested)
    assert set(trial) == set(MASK_GUIDANCE_OPTIONS)
    destination = tmp_path / "best.yaml"
    write_trial_yaml(schema, trial, destination, base_config=runtime, tracker_name="bytetrack")
    saved = yaml.safe_load(destination.read_text())
    assert saved["edgetam"] == nested["edgetam"]
    assert not any(key.startswith("edgetam.") for key in saved)
    restored = load_tracker_config("bytetrack", destination)
    assert {key: restored[key] for key in MASK_GUIDANCE_OPTIONS} == trial


def test_explicit_flag_selects_default_checkpoint_and_ignores_disabled_weights() -> None:
    args = _args(mask_guidance_weights=None)
    assert edgetam_checkpoint(args) == Path("edgetam.pt")
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    assert set(MASK_GUIDANCE_OPTIONS).issubset(runtime)
    args.edgetam = False
    args.mask_guidance_weights = Path("custom-model.pt")
    runtime = prepare_mask_guidance_tuning(args, runtime)
    assert not set(MASK_GUIDANCE_OPTIONS).intersection(runtime)
    assert args.mask_guidance_weights == Path("custom-model.pt")
    assert edgetam_checkpoint(args) is None


def test_cpp_and_manual_cap_do_not_create_inactive_search_dimensions() -> None:
    args = _args(edgetam=False, tracker_backend="cpp")
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(args.tracker), args, runtime)
    assert not set(MASK_GUIDANCE_OPTIONS).intersection(schema)
    assert not set(MASK_GUIDANCE_OPTIONS).intersection(runtime)

    args = _args(mask_guidance_max_objects=7)
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(args.tracker), args, runtime)
    assert schema["edgetam.max_objects"] == {"default": 7}
    trial = _SuggestionTrial()
    yaml_to_optuna_define_space(schema)(trial)
    assert "edgetam.max_objects" not in trial.params
    assert "edgetam.min_coverage" in trial.params


def test_explicit_ordinary_association_override_is_fixed_in_the_search() -> None:
    args = _args(edgetam=False, asso_func="giou")
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(args.tracker), args, runtime)
    assert schema["asso_func"] == {"default": "giou"}
    assert runtime["asso_func"] == "giou"


@pytest.mark.parametrize(
    "tracker",
    [
        "boosttrack",
        "botsort",
        "bytetrack",
        "deepocsort",
        "hybridsort",
        "occluboost",
        "ocsort",
        "sfsort",
        "strongsort",
    ],
)
def test_every_python_box_tracker_has_an_admissible_guided_search(tracker: str) -> None:
    args = _args(tracker=tracker)
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(tracker), args, runtime)
    trial = _SuggestionTrial()
    yaml_to_optuna_define_space(schema)(trial)
    assert schema["asso_func"] == {"default": "iou"}
    assert "asso_func" not in trial.params
    assert set(MASK_GUIDANCE_OPTIONS).issubset(trial.params)


@pytest.mark.parametrize(
    "override",
    [
        {"tracker_backend": "cpp"},
        {"geometry": "obb"},
        {"per_class": True},
        {"asso_func": "diou"},
        {"eval_masks": True},
        {"device": "cuda:2"},
    ],
)
def test_invalid_guided_searches_fail_before_model_loading(override: dict) -> None:
    args = _args(**override)
    with pytest.raises(ValueError):
        prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))


def test_hybrid_uses_iou_without_silently_overriding_authored_non_iou(tmp_path: Path) -> None:
    args = _args(tracker="hybridsort")
    runtime = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    assert runtime["asso_func"] == args.asso_func == "iou"
    config = tmp_path / "tracker.yaml"
    config.write_text("asso_func: diou\n", encoding="utf-8")
    args = _args(tracker="hybridsort", tracker_config=config)
    with pytest.raises(ValueError, match="incompatible override"):
        prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))


@pytest.mark.parametrize(
    "device,guided,gpus", [("cpu", True, 0), ("mps", True, 0), ("cuda", True, 1), ("0", True, 1), ("cuda", False, 0)]
)
def test_temporal_cuda_model_reserves_gpu_even_for_bytetrack(device: str, guided: bool, gpus: int) -> None:
    args = _args(device=device, edgetam=guided, sequence_workers=2)
    assert mask_guidance_trial_resources(args) == {"cpu": 2, "gpu": gpus}


@pytest.mark.parametrize("workers,concurrent,cap", [(None, None, None), (2, 3, 5)])
def test_tune_cli_forwards_guidance_for_existing_build(monkeypatch, workers, concurrent, cap) -> None:
    received = []
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.tuner", SimpleNamespace(main=received.append))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    flags = [
        "tune",
        "--edgetam",
        "--experiment",
        "mot17/ablation-yolox-lmbn.yaml",
        "--tracker",
        "hybridsort",
        "--build",
        "existing-build",
        "--device",
        "mps",
        "--mask-guidance-weights",
        "edgetam.pt",
    ]
    for name, value in (
        ("sequence-workers", workers),
        ("max-concurrent-trials", concurrent),
        ("mask-guidance-max-objects", cap),
    ):
        if value is not None:
            flags += [f"--{name}", str(value)]
    result = CliRunner().invoke(boxmot, flags)
    assert result.exit_code == 0, (result.output, result.exception)
    args = received[0]
    assert args.mask_guidance_weights == Path("edgetam.pt")
    assert args.mask_guidance_max_objects == cap
    assert args.sequence_workers == (workers or 1)
    assert args.max_concurrent_trials == (concurrent or 1)
    assert args.device == "mps" and args.asso_func == "iou"


@pytest.mark.parametrize("flags", [["--tracker-backend", "cpp"], ["--asso-func", "giou"], ["--per-class"]])
def test_invalid_tune_cli_guidance_is_rejected_before_materialization(monkeypatch, flags) -> None:
    monkeypatch.setattr(
        "boxmot.engine.commands.tune._prepare_replay_build", lambda *args, **kwargs: pytest.fail("materialized")
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--edgetam",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--mask-guidance-weights",
            "edgetam.pt",
            *flags,
        ],
    )
    assert result.exit_code == 2
    assert "Mask guidance" in result.output


def test_reused_trial_actor_forwards_model_and_new_guidance_parameters_to_replay(monkeypatch) -> None:
    calls = []

    def evaluate(args, *, evolve_config, replay_session, **kwargs):
        calls.append((args, evolve_config.copy(), replay_session))
        return SimpleNamespace(
            raw={"HOTA": 50.0}, benchmark="fixture", summary_label="all", summary={}, timings={}, exp_dir=None
        )

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    monkeypatch.setattr(tuner, "aggregate_results", dict)
    args = _args(cache_inputs=True)
    actor = build_tracker_trainable(SimpleNamespace(Trainable=object), args)()
    actor.setup({})
    try:
        for coverage, cap in [(0.91, 8), (0.95, 16)]:
            actor.reset_config({"edgetam.min_coverage": coverage, "edgetam.max_objects": cap})
            assert actor.step()["HOTA"] == 50
    finally:
        actor.cleanup()
    assert [call[1]["edgetam.max_objects"] for call in calls] == [8, 16]
    assert [call[1]["edgetam.min_coverage"] for call in calls] == [0.91, 0.95]
    assert all(call[0].mask_guidance_weights == args.mask_guidance_weights for call in calls)
    assert calls[0][2] is calls[1][2] and calls[0][2]._closed


@pytest.mark.parametrize("change", ["checkpoint", "options", "device", "schema", "disabled"])
def test_resume_rejects_changed_guidance_profile(tmp_path: Path, change: str) -> None:
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"checkpoint one")
    args = _args(mask_guidance_weights=checkpoint)
    options = prepare_mask_guidance_tuning(args, resolve_tracker_options(args, include_defaults=True))
    schema = condition_mask_guidance_schema(load_yaml_config(args.tracker), args, options)
    record_mask_guidance_tuning(tmp_path / "run", args, options, schema)
    args.resume_tune = tmp_path / "run"
    record_mask_guidance_tuning(tmp_path / "run", args, options, schema)
    if change == "checkpoint":
        checkpoint.write_bytes(b"checkpoint two")
    elif change == "options":
        options["edgetam.min_coverage"] = 0.97
    elif change == "device":
        args.device = "mps"
    elif change == "schema":
        schema["edgetam.max_objects"] = {"default": 8}
    else:
        args.edgetam = False
    with pytest.raises(ValueError, match="same mask guidance"):
        record_mask_guidance_tuning(tmp_path / "run", args, options, schema)
    assert json.loads((tmp_path / "run" / "mask-guidance.json").read_text())["enabled"]
