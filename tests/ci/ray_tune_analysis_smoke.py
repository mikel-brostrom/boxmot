"""Exercise active-branch diagnostics with real Ray workers and synthetic scores.

Run explicitly with ``python -m tests.ci.ray_tune_analysis_smoke --output PATH``.
This is a deterministic, CPU-only plumbing check, not a tracking benchmark.
It downloads no models or datasets and is not collected by the unit suite.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from boxmot.engine.tuning.analysis import generate_tune_analysis
from boxmot.engine.tuning.analysis_data import load_tune_analysis
from boxmot.engine.tuning.backends.optuna_backend import OptunaBackend
from boxmot.engine.tuning.conditions import SCOPED_PARAMETER_PREFIX
from boxmot.engine.tuning.postprocessing import collect_trial_data, generate_summary, save_results_csv
from boxmot.engine.tuning.search_profile import record_search_profile
from boxmot.engine.tuning.search_space import normalize_trial_config

TRIAL_COUNT = 48
SEED = 17


def _schema() -> dict[str, Any]:
    """Include categorical cases, association applicability, and nested activation."""
    return {
        "motion_fusion": {"type": "choice", "options": ["off", "gate", "blend"], "default": "blend"},
        "motion_fusion_lambda": {
            "type": "uniform", "range": [0.1, 1.0], "default": 0.98,
            "depends_on": "motion_fusion", "cases": {"off": {"default": 0.98}, "gate": {"default": 0.98}},
        },
        "motion_fusion_only_position": {
            "type": "choice", "options": [False, True], "default": True,
            "depends_on": "motion_fusion", "cases": {"off": {"default": False}},
            "activates": {"position_weight": {"type": "uniform", "range": [0.1, 0.9], "default": 0.5}},
        },
        "asso_func": {"type": "choice", "options": ["iou", "centroid"], "default": "iou"},
        "match_thresh": {
            "type": "uniform", "range": [0.1, 0.9], "default": 0.5, "association": ["iou"],
        },
        "centroid_radius": {
            "type": "uniform", "range": [10.0, 100.0], "default": 40.0, "association": ["centroid"],
        },
    }


def _objective(config: dict[str, Any]) -> None:
    """Report deliberately confounded scores from the isolated Ray worker."""
    from ray import tune

    config = normalize_trial_config(config)
    fusion = config["motion_fusion"]
    score = 50.0 + 10.0 * config["motion_fusion_lambda"] if fusion == "blend" else {
        "off": 85.0, "gate": 75.0,
    }[fusion]
    view = {
        "HOTA": score, "MOTA": score - 2.0, "IDF1": 100.0 - score / 2.0,
        "DetA": score - 1.0, "AssA": score + 1.0, "AssRe": score + 2.0,
        "IDSW": 100 - int(score), "IDs": 100, "IDSW_rate": 1.0 - score / 100.0,
        "per_sequence": {
            "synthetic-sequence-a": {"HOTA": score + 1.0},
            "synthetic-sequence-b": {"HOTA": score - 1.0},
        },
    }
    tune.report({
        **{key: value for key, value in view.items() if key != "per_sequence"},
        "_validation": {"summary_label": "single_class", "raw": view},
        "synthetic_objective": True,
        "done": True,
    })


def _check_optimizer(searcher: Any, configurations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check sampled Optuna dimensions independently of runtime defaults and analysis masks."""
    from optuna.trial import TrialState

    # Ray exposes no public accessor for its Optuna study or Ray-to-Optuna trial IDs.
    completed = {trial.number: trial for trial in searcher._ot_study.get_trials()}
    assert len(completed) == len(configurations) == TRIAL_COUNT
    counts: dict[str, int] = {}
    for trial_id, raw_config in configurations.items():
        config = normalize_trial_config(raw_config)
        trial = completed[searcher._ot_trials[trial_id].number]
        assert trial.state == TrialState.COMPLETE
        assert set(trial.params) == set(trial.distributions)
        sampled = {
            label.removeprefix(SCOPED_PARAMETER_PREFIX).rsplit("__", 1)[0]
            if label.startswith(SCOPED_PARAMETER_PREFIX) else label: value
            for label, value in trial.params.items()
        }
        expected = {"motion_fusion", "asso_func"}
        if config["motion_fusion"] == "blend":
            expected.add("motion_fusion_lambda")
        else:
            assert config["motion_fusion_lambda"] == 0.98
        if config["motion_fusion"] != "off":
            expected.add("motion_fusion_only_position")
            if config["motion_fusion_only_position"]:
                expected.add("position_weight")
        expected.add("match_thresh" if config["asso_func"] == "iou" else "centroid_radius")
        assert set(sampled) == expected, (trial_id, sampled, expected)
        assert len(sampled) == len(trial.params), "A canonical parameter must have one optimizer label."
        for name, value in sampled.items():
            assert value == config[name], (trial_id, name)
            counts[name] = counts.get(name, 0) + 1
    return {"completed_trials": len(completed), "sampled_parameters": counts, "inactive_defaults_unsampled": True}


def _check_analysis(directory: Path) -> dict[str, Any]:
    """Verify exact active membership, retained defaults, and real result metadata."""
    analysis = load_tune_analysis(directory)
    assert analysis is not None
    trials = analysis.trials
    assert len(trials) == TRIAL_COUNT
    assert analysis.chronological
    assert analysis.baseline_label == "Verified default baseline"
    assert set(analysis.sequence_hota.columns) == {"synthetic-sequence-a", "synthetic-sequence-b"}
    assert set(trials["motion_fusion"]) == {"off", "gate", "blend"}
    assert set(trials["asso_func"]) == {"iou", "centroid"}
    fusion_active = trials["motion_fusion"] != "off"
    expected = {
        "motion_fusion_lambda": trials["motion_fusion"] == "blend",
        "motion_fusion_only_position": fusion_active,
        "position_weight": fusion_active & trials["motion_fusion_only_position"].eq(True),
        "match_thresh": trials["asso_func"] == "iou",
        "centroid_radius": trials["asso_func"] == "centroid",
    }
    branch_labels = {
        "motion_fusion_lambda": "motion_fusion=" + trials["motion_fusion"],
        "motion_fusion_only_position": "motion_fusion=" + trials["motion_fusion"],
        "position_weight": "motion_fusion=" + trials["motion_fusion"] + ", motion_fusion_only_position=True",
        "match_thresh": "asso_func=" + trials["asso_func"],
        "centroid_radius": "asso_func=" + trials["asso_func"],
    }
    counts: dict[str, Any] = {}
    for name, active in expected.items():
        parameter = analysis.parameters[name]
        assert set(parameter.samples.index) == set(trials.index[active]), name
        assert 0 < len(parameter.samples) < TRIAL_COUNT, name
        for branch, samples in parameter.samples.groupby("branch"):
            assert set(samples.index) == set(trials.index[active & branch_labels[name].eq(branch)]), (name, branch)
        counts[name] = {
            "active": len(parameter.samples), "inactive": int((~active).sum()),
            "branches": parameter.samples.groupby("branch").size().to_dict(),
        }
    active = analysis.parameters["motion_fusion_lambda"].samples
    assert active["value"].eq(0.98).any(), "A genuinely active default must be retained."
    raw = pd.read_csv(directory / "results.csv")
    inactive = raw["motion_fusion"] != "blend"
    assert raw.loc[inactive, "motion_fusion_lambda"].eq(0.98).all()
    active_rho = active["value"].corr(active["HOTA"], method="spearman")
    pooled_rho = trials["motion_fusion_lambda"].corr(trials["HOTA"], method="spearman")
    assert np.isclose(active_rho, 1.0)
    assert not np.isclose(pooled_rho, active_rho)
    for trial_id in trials["trial_id"]:
        result = directory / f"trial_{trial_id}" / "result.json"
        assert result.is_file(), result
        assert any(json.loads(line).get("synthetic_objective") for line in result.read_text().splitlines())
    return {
        "synthetic_objective": True, "completed_trials": len(trials), "seed": SEED,
        "parameters": counts, "active_default_lambda_retained": True,
        "lambda_spearman_active": active_rho, "lambda_spearman_unfiltered": pooled_rho,
        "notes": analysis.notes,
    }


def run(output: Path) -> dict[str, Any]:
    """Run an owned, local two-CPU Ray cluster and generate production diagnostics."""
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("RAY_CHDIR_TO_TRIAL_DIR", "0")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("RAY_process_group_cleanup_enabled", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import ray
    from ray import tune

    if ray.is_initialized():
        raise RuntimeError("Run this smoke test in its own Python process; it owns the Ray runtime.")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {output}")
    schema = _schema()
    args = SimpleNamespace(tracker="synthetic-branch-smoke", search_alg="optuna", geometry="aabb")
    record_search_profile(output, args, schema, {})
    backend = OptunaBackend(
        yaml_cfg=schema, opt_metrics=["HOTA"], opt_modes=["max"], seed=SEED, max_concurrent=1,
    )
    search_alg, param_space = backend.build(tune)
    started = perf_counter()
    constants = ray._private.ray_constants
    previous_uv = constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    try:
        with tempfile.TemporaryDirectory(prefix="boxmot-ray-", dir="/tmp") as temporary:
            ray.init(
                num_cpus=2, include_dashboard=False, log_to_driver=False,
                _node_ip_address="127.0.0.1", _temp_dir=temporary,
                object_store_memory=80 * 1024 * 1024,
            )
            try:
                tuner = tune.Tuner(
                    tune.with_resources(_objective, {"cpu": 1}),
                    param_space=param_space,
                    tune_config=tune.TuneConfig(
                        metric="HOTA", mode="max", num_samples=TRIAL_COUNT, search_alg=search_alg,
                        reuse_actors=True, trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
                    ),
                    run_config=tune.RunConfig(
                        name=output.name, storage_path=str(output.parent), verbose=0,
                    ),
                )
                results = tuner.fit()
                assert not results.errors, results.errors
                optimizer_validation = _check_optimizer(
                    search_alg.searcher, {result.metrics["trial_id"]: result.config for result in results},
                )
                trial_data = collect_trial_data(results)
            finally:
                ray.shutdown()
    finally:
        constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous_uv
        # Also close an owned runtime if initialization only partially completed.
        if ray.is_initialized():
            ray.shutdown()
    save_results_csv(output / "results.csv", trial_data)
    validation = _check_analysis(output)
    for name, counts in validation["parameters"].items():
        assert optimizer_validation["sampled_parameters"][name] == counts["active"], name
    validation["optimizer"] = optimizer_validation
    generate_summary(output, trial_data, schema, args.tracker, ["HOTA"], [], args, emit_logs=False)
    assert generate_tune_analysis(output, tracker_name=args.tracker) == output / "analysis.png"
    artifacts = [
        "analysis.png", "analysis_scatter.png", "analysis_categories.png", "analysis_candidates.png",
        "results.csv", "search-space.json", "summary.md",
    ]
    for name in artifacts:
        assert (output / name).is_file(), name
    summary = (output / "summary.md").read_text()
    assert "motion_fusion=blend" in summary
    assert "motion_fusion=gate" in summary
    assert "motion_fusion_lambda" in summary
    validation.update(artifacts=artifacts, elapsed_seconds=round(perf_counter() - started, 3))
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def main() -> None:
    """Parse an explicit smoke invocation and print its validation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/ray/active-branch-smoke"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
