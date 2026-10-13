"""Tests for evidence selection shared by tuning plots and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boxmot.engine.tuning.analysis_data import analyze_parameters, load_tune_analysis


def _write_run(tmp_path: Path, frame: pd.DataFrame, schema: dict | None = None, fixed: dict | None = None) -> None:
    frame.to_csv(tmp_path / "results.csv", index=False)
    if schema is not None:
        (tmp_path / "search-space.json").write_text(json.dumps({"schema": schema, "fixed_options": fixed or {}}))


def _write_result(tmp_path: Path, trial_id: str, result: dict) -> Path:
    directory = tmp_path / f"trial_{trial_id}"
    directory.mkdir(exist_ok=True)
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    return path


def test_active_cases_exclude_runtime_defaults_and_keep_branch_bounds() -> None:
    schema = {
        "mode": {"type": "choice", "options": ["off", "narrow", "wide"], "default": "off"},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.9, "depends_on": "mode",
            "cases": {"off": {"default": 0.9}, "narrow": {"range": [0.2, 0.4]}},
        },
        "fixed": {"default": 10},
        "constant": {"type": "randint", "range": [1, 10], "default": 3},
    }
    frame = pd.DataFrame({
        "trial_id": ["a", "b", "c", "d", "e"], "HOTA": [60, 61, 62, 63, 64],
        "mode": ["off", "narrow", "wide", "wide", "wide"],
        "weight": [0.9, 0.3, 0.6, np.inf, np.nan], "fixed": 10, "constant": 3,
        "arbitrary_metadata": [1, 2, 3, 4, 5],
    }, index=[10, 20, 30, 40, 50])
    parameters = analyze_parameters(frame, schema)
    assert set(parameters) == {"mode", "weight", "constant"}
    weight = parameters["weight"]
    assert weight.samples.index.tolist() == [20, 30]
    assert weight.samples["value"].tolist() == [0.3, 0.6]
    assert weight.bounds == (0, 1)
    assert weight.branch_bounds == {"mode=narrow": (0.2, 0.4), "mode=wide": (0, 1)}
    assert parameters["constant"].samples["value"].nunique() == 1


@pytest.mark.parametrize("flags", [(False, True), ("False", "True"), (0, 1)])
def test_nested_activation_preserves_missing_booleans(flags: tuple[object, object]) -> None:
    false, true = flags
    schema = {
        "enabled": {
            "type": "choice", "options": [False, True], "default": False,
            "activates": {
                "child": {
                    "type": "choice", "options": [False, True], "default": True,
                    "activates": {"value": {"type": "loguniform", "range": [0.01, 1], "default": 0.1}},
                }
            },
        },
        "missing": {"type": "choice", "options": [False, True], "default": False},
        "numeric_choice": {"type": "choice", "options": [0, 1], "default": 0},
    }
    frame = pd.DataFrame({
        "HOTA": [0, 20, 30, 40, np.inf], "enabled": [false, true, true, None, true],
        "child": [true, false, true, true, true], "value": [0.1, 0.1, 0.2, 0.4, 0.5],
        "missing": None, "numeric_choice": [0, 1, 0, 1, 0],
    })
    parameters = analyze_parameters(frame, schema)
    assert "missing" not in parameters
    assert parameters["enabled"].samples["value"].tolist() == [False, True, True]
    assert parameters["child"].samples.index.tolist() == [1, 2]
    assert parameters["value"].samples.index.tolist() == [2]
    assert parameters["value"].scale == "log"
    assert type(parameters["numeric_choice"].samples.iloc[0]["value"]) is not bool


def test_association_inactive_children_and_json_boolean_cases() -> None:
    schema = {
        "asso_func": {"type": "choice", "options": ["iou", "centroid"], "default": "iou"},
        "feature": {
            "type": "choice", "options": [False, True], "default": True, "association": ["iou"],
            "activates": {"weight": {"type": "uniform", "range": [0, 1], "default": 0.5}},
        },
        "flag": {"type": "choice", "options": [False, True], "default": False},
        "conditional": {
            "type": "uniform", "range": [0, 1], "default": 0.1, "depends_on": "flag",
            "cases": {"false": {"default": 0.1}},
        },
    }
    frame = pd.DataFrame({
        "HOTA": [10, 20, 30], "asso_func": ["centroid", "iou", "iou"],
        "feature": [True, False, True], "weight": [0.5, 0.5, 0.9],
        "flag": [False, True, True], "conditional": [0.1, 0.3, 0.6],
    })
    parameters = analyze_parameters(frame, schema)
    assert parameters["feature"].samples.index.tolist() == [1, 2]
    assert parameters["weight"].samples.index.tolist() == [2]
    assert parameters["conditional"].samples.index.tolist() == [1, 2]


def test_loader_uses_completion_order_recorded_defaults_and_selected_class(tmp_path: Path) -> None:
    schema = {
        "mode": {"type": "choice", "options": ["a", "b"], "default": "b"},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.5, "depends_on": "mode",
            "cases": {"b": {"default": 0.7, "range": [0.5, 1]}},
        },
    }
    _write_run(tmp_path, pd.DataFrame({
        "trial_id": ["001", "002", "003", "bad"], "HOTA": [70, 69.85, 0, np.inf],
        "mode": ["a", "b", "a", "a"], "weight": [0.2, 0.7, 0.1, 0.1],
    }), schema, {"fixed": 5})
    for trial_id, score, timestamp in [("001", 70, 30), ("002", 69.85, 10), ("003", 0, 20)]:
        _write_result(tmp_path, trial_id, {
            "HOTA": score, "timestamp": timestamp, "config": {"fixed": 5},
            "_validation": {"summary_label": "second", "raw": {
                "first": {"DetA": 1, "per_sequence": {"sequence": {"HOTA": 1}}},
                "second": {"DetA": 60, "per_sequence": {"sequence": {"HOTA": score}}},
            }},
        })
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.trials["trial_id"].tolist() == ["002", "003", "001"]
    assert analysis.chronological
    assert analysis.baseline_index == 0
    assert analysis.baseline_label == "Verified default baseline"
    assert analysis.best_index == 2
    assert analysis.near_best.tolist() == [True, False, True]
    assert analysis.trials["DetA"].tolist() == [60, 60, 60]
    assert analysis.sequence_hota["sequence"].tolist() == [69.85, 0, 70]


def test_loader_incomplete_metadata_uses_csv_order_and_first_reference(tmp_path: Path) -> None:
    _write_run(tmp_path, pd.DataFrame({"trial_id": ["b", "a"], "HOTA": [0, 0]}))
    _write_result(tmp_path, "a", {"HOTA": 0, "timestamp": 1})
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.trials["trial_id"].tolist() == ["b", "a"]
    assert not analysis.chronological
    assert analysis.baseline_label == "First recorded trial"
    assert analysis.parameters == {}
    assert analysis.near_best.all()
    assert any("Search schema unavailable" in note for note in analysis.notes)
    assert any("1/2" in note for note in analysis.notes)
    assert any("CSV order" in note for note in analysis.notes)


def test_loader_reads_last_valid_jsonl_and_does_not_average_class_hota(tmp_path: Path) -> None:
    _write_run(tmp_path, pd.DataFrame({"trial_id": ["a"], "HOTA": [70]}), {})
    path = _write_result(tmp_path, "a", {"HOTA": 60, "timestamp": 1})
    result = {
        "HOTA": 70, "timestamp": 2,
        "_validation": {"summary_label": "cls_comb_det_av", "raw": {
            "person": {"DetA": 30, "per_sequence": {"sequence": {"HOTA": 10}}},
            "car": {"DetA": 80, "per_sequence": {"sequence": {"HOTA": 90}}},
            "cls_comb_det_av": {"DetA": 65, "per_sequence": {}},
        }},
    }
    with path.open("a") as stream:
        stream.write("\n" + json.dumps(result) + '\n{"incomplete":')
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.trials.at[0, "completion_timestamp"] == 2
    assert analysis.trials.at[0, "DetA"] == 65
    assert analysis.sequence_hota.empty
    assert any("summary class/aggregate" in note for note in analysis.notes)


def test_baseline_requires_fixed_options_match(tmp_path: Path) -> None:
    _write_run(tmp_path, pd.DataFrame({
        "HOTA": [60, 61], "value": [0.5, 0.5], "fixed": [False, True], "timestamp": [1, 2],
    }), {"value": {"type": "uniform", "range": [0, 1], "default": 0.5}}, {"fixed": True})
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.baseline_index == 1
    assert analysis.baseline_label == "Verified default baseline"


def test_integer_case_keys_round_trip_from_json() -> None:
    schema = {
        "mode": {"type": "choice", "options": [0, 1], "default": 0},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.2,
            "depends_on": "mode", "cases": {"0": {"default": 0.2}},
        },
    }
    parameters = analyze_parameters(pd.DataFrame({
        "mode": [0, 1], "weight": [0.2, 0.7], "HOTA": [60, 61],
    }), schema)
    assert parameters["weight"].samples.index.tolist() == [1]


@pytest.mark.parametrize("default", [False, 0, 1.5])
def test_fixed_selector_case_keys_round_trip_from_json(default: object) -> None:
    schema = json.loads(json.dumps({
        "mode": {"default": default},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.2,
            "depends_on": "mode", "cases": {default: {"default": 0.2}},
        },
    }))
    parameters = analyze_parameters(pd.DataFrame({
        "mode": [default], "weight": [0.2], "HOTA": [60],
    }), schema)
    assert "weight" not in parameters


def test_serialized_case_keys_use_resolved_selector_options() -> None:
    schema = {
        "variant": {"type": "choice", "options": ["text", "flag"], "default": "text"},
        "mode": {
            "type": "choice", "options": ["off", "on"], "default": "off",
            "depends_on": "variant",
            "cases": {"flag": {"options": [False, True], "default": False}},
        },
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.2,
            "depends_on": "mode", "cases": {"false": {"default": 0.2}},
        },
    }
    parameters = analyze_parameters(pd.DataFrame({
        "variant": ["flag", "flag"], "mode": [False, True], "weight": [0.2, 0.7], "HOTA": [60, 61],
    }), schema)
    assert parameters["weight"].samples.index.tolist() == [1]


def test_string_case_keys_preserve_case_sensitive_selector_values() -> None:
    schema = {
        "mode": {"type": "choice", "options": ["LOW", "low"], "default": "LOW"},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.2,
            "depends_on": "mode", "cases": {"low": {"default": 0.2}},
        },
    }
    parameters = analyze_parameters(pd.DataFrame({
        "mode": ["LOW", "low"], "weight": [0.7, 0.2], "HOTA": [60, 61],
    }), schema)
    assert parameters["weight"].samples.index.tolist() == [0]


def test_categorical_options_are_recorded_per_resolved_branch() -> None:
    schema = {
        "mode": {"type": "choice", "options": ["a", "b"], "default": "a"},
        "option": {
            "type": "choice", "options": ["first", "second"], "default": "first",
            "depends_on": "mode",
            "cases": {"b": {"options": ["third", "fourth"], "default": "third"}},
        },
    }
    parameter = analyze_parameters(pd.DataFrame({
        "mode": ["a", "b"], "option": ["first", "third"], "HOTA": [60, 61],
    }), schema)["option"]
    assert parameter.options == ("first", "second", "third", "fourth")
    assert parameter.branch_options == {"mode=a": ("first", "second"), "mode=b": ("third", "fourth")}


def test_saved_200_trial_fusion_activity_uses_branches_not_default_value(tmp_path: Path) -> None:
    """Retain active values equal to defaults while excluding 187 inactive rows."""
    schema = {
        "motion_fusion": {"type": "choice", "options": ["off", "gate", "blend"], "default": "off"},
        "motion_fusion_lambda": {
            "type": "uniform", "range": [0, 1], "default": 0.98, "depends_on": "motion_fusion",
            "cases": {"off": {"default": 0.98}, "gate": {"default": 0.98}},
        },
        "motion_fusion_only_position": {
            "type": "choice", "options": [False, True], "default": False, "depends_on": "motion_fusion",
            "cases": {"off": {"default": False}},
        },
    }
    frame = pd.DataFrame({
        "trial_id": [f"trial-{index}" for index in range(200)],
        "HOTA": np.linspace(60, 70, 200),
        "motion_fusion": ["off"] * 107 + ["gate"] * 80 + ["blend"] * 13,
        "motion_fusion_lambda": [0.98] * 187 + np.linspace(0, 0.9, 12).tolist() + [0.98],
        "motion_fusion_only_position": [False] * 200,
    })
    _write_run(tmp_path, frame, schema)
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    weight = analysis.parameters["motion_fusion_lambda"].samples
    assert len(weight) == 13
    assert weight["branch"].unique().tolist() == ["motion_fusion=blend"]
    assert weight.index.tolist() == list(range(187, 200))
    assert weight.loc[199, "value"] == 0.98
    position = analysis.parameters["motion_fusion_only_position"].samples
    assert position.groupby("branch").size().to_dict() == {"motion_fusion=blend": 13, "motion_fusion=gate": 80}
    assert len(position) == 93
    assert position["value"].eq(False).all()


@pytest.mark.parametrize("alternate", [
    {"type": "loguniform", "range": [1, 100]},
    {"type": "choice", "options": [1, 100]},
])
def test_incompatible_branch_representations_are_omitted_with_reason(alternate: dict) -> None:
    schema = {
        "mode": {"type": "choice", "options": ["a", "b"], "default": "a"},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0,
            "depends_on": "mode", "cases": {"b": alternate},
        },
    }
    frame = pd.DataFrame({"mode": ["a", "b"], "weight": [0, 100], "HOTA": [60, 61]})
    notes: list[str] = []
    parameters = analyze_parameters(frame, schema, notes=notes)
    assert set(parameters) == {"mode"}
    assert len(notes) == 1
    assert "weight" in notes[0] and "incompatible" in notes[0]
    # A distribution on an unobserved branch does not invalidate available evidence.
    single_branch = analyze_parameters(frame.iloc[:1], schema)
    assert single_branch["weight"].samples["value"].tolist() == [0]
    assert single_branch["weight"].scale == "linear"


def test_malformed_optional_metadata_and_incomplete_results_are_ignored(tmp_path: Path) -> None:
    _write_run(tmp_path, pd.DataFrame({"trial_id": ["a", "b"], "HOTA": [50, 60], "MOTA": [30, 40]}))
    _write_result(tmp_path, "a", {
        "HOTA": 50, "done": True, "timestamp": {"bad": 1}, "config": {"HOTA": 1, "MOTA": 2},
        "_validation": {"summary_label": "all", "raw": {"all": {
            "DetA": [20], "per_sequence": {"sequence": {"HOTA": {"bad": 20}}},
        }}},
    })
    _write_result(tmp_path, "b", {"HOTA": 60, "done": False, "timestamp": 1})
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.trials["HOTA"].tolist() == [50, 60]
    assert analysis.trials["MOTA"].tolist() == [30, 40]
    assert analysis.sequence_hota.empty
    assert not analysis.chronological
    assert any("1/2" in note for note in analysis.notes)


@pytest.mark.parametrize("contents", ['{"schema":', '{"schema": {"value": {"depends_on": "absent", "cases": {}}}}'])
def test_corrupt_schema_does_not_lose_metric_analysis(tmp_path: Path, contents: str) -> None:
    _write_run(tmp_path, pd.DataFrame({"HOTA": [40, 50]}))
    (tmp_path / "search-space.json").write_text(contents)
    analysis = load_tune_analysis(tmp_path)
    assert analysis is not None
    assert analysis.best_index == 1
    assert analysis.parameters == {}
    assert any("schema" in note.lower() for note in analysis.notes)


@pytest.mark.parametrize("frame", [pd.DataFrame({"MOTA": [3]}), pd.DataFrame({"HOTA": [np.nan, np.inf]})])
def test_unusable_scores_skip_analysis(tmp_path: Path, frame: pd.DataFrame) -> None:
    _write_run(tmp_path, frame)
    assert load_tune_analysis(tmp_path) is None
