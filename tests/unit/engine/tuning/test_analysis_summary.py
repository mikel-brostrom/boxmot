"""Parameter summaries must describe observed active search values."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from boxmot.engine.tuning.postprocessing import (
    generate_summary,
    parameter_concentration_markdown,
    update_analysis_summary,
)


def test_summary_excludes_inactive_defaults_from_best_trial_ranges() -> None:
    """Successful inactive defaults must not look like a concentrated search."""
    schema = {
        "motion_fusion": {"type": "choice", "default": "off", "options": ["off", "blend"]},
        "motion_fusion_lambda": {
            "type": "uniform", "default": 0.98, "range": [0, 1],
            "depends_on": "motion_fusion", "cases": {"off": {"default": 0.98}},
        },
        "fixed": {"default": 7},
    }
    frame = pd.DataFrame([
        {"trial_id": str(i), "HOTA": 70 - i / 100, "motion_fusion": "off",
         "motion_fusion_lambda": 0.98, "fixed": 7}
        for i in range(12)
    ] + [
        {"trial_id": "blend_a", "HOTA": 65, "motion_fusion": "blend", "motion_fusion_lambda": 0.2},
        {"trial_id": "blend_b", "HOTA": 64, "motion_fusion": "blend", "motion_fusion_lambda": 0.8},
    ])

    rendered = parameter_concentration_markdown(frame, schema)

    assert "**12 near-best trials**" in rendered
    row = next(line for line in rendered.splitlines() if line.startswith("| motion_fusion_lambda |"))
    assert "| 2/14 | 0/10; inactive | 0/12; inactive | inactive |" in row
    assert "0.9800" not in row
    assert "| fixed |" not in rendered
    assert "Converged?" not in rendered
    assert "insensitive" not in rendered


def test_summary_reports_separate_conditional_search_domains() -> None:
    """One parameter can have different recorded bounds in each branch."""
    schema = {
        "asso_func": {"type": "choice", "default": "iou", "options": ["iou", "diou", "centroid"]},
        "match_thresh": {
            "type": "uniform", "default": 0.9, "range": [0.7, 0.9],
            "depends_on": "asso_func",
            "cases": {"diou": {"range": [0.4, 0.85]}, "centroid": {"default": 0.9}},
        },
    }
    frame = pd.DataFrame([
        {"trial_id": "a", "HOTA": 70, "asso_func": "iou", "match_thresh": 0.8},
        {"trial_id": "b", "HOTA": 69.9, "asso_func": "diou", "match_thresh": 0.45},
        {"trial_id": "c", "HOTA": 69.8, "asso_func": "centroid", "match_thresh": 0.9},
    ])

    rendered = parameter_concentration_markdown(frame, schema)

    rows = [line for line in rendered.splitlines() if line.startswith("| match_thresh |")]
    assert len(rows) == 2
    assert any("[0.7, 0.9]" in row and "[0.8000, 0.8000]" in row for row in rows)
    assert any("[0.4, 0.85]" in row and "[0.4500, 0.4500]" in row for row in rows)
    assert not any("[0.9000, 0.9000]" in row for row in rows)
    assert "**3 near-best trials**" in rendered


def test_summary_reports_only_options_from_each_active_branch() -> None:
    """Case-specific categories must not be labeled with other branches' domains."""
    schema = {
        "mode": {"type": "choice", "default": "off", "options": ["off", "a", "b"]},
        "policy": {
            "type": "choice", "default": "fast", "options": ["fast", "slow"],
            "depends_on": "mode", "cases": {"off": {"default": "fast"}, "b": {"options": ["near", "far"]}},
        },
    }
    frame = pd.DataFrame({
        "HOTA": [70, 69.9, 69.8], "mode": ["off", "a", "b"], "policy": ["fast", "slow", "near"],
    })
    rendered = parameter_concentration_markdown(frame, schema)
    rows = [line for line in rendered.splitlines() if line.startswith("| policy |")]
    assert len(rows) == 2
    assert "| mode=a | ['fast', 'slow'] | 1/3 |" in rows[0]
    assert "| mode=b | ['near', 'far'] | 1/3 |" in rows[1]
    assert not any("mode=off" in row for row in rows)


def test_summary_handles_missing_values_observed_constants_and_sparse_categories() -> None:
    """Missing observations do not contribute to active counts or ranges."""
    schema = {
        "threshold": {"type": "uniform", "default": 0.5, "range": [0, 1]},
        "enabled": {"type": "choice", "default": False, "options": [False, True]},
    }
    frame = pd.DataFrame([
        {"trial_id": "a", "HOTA": 70, "threshold": 0.05, "enabled": True},
        {"trial_id": "b", "HOTA": 70, "threshold": 0.05, "enabled": False},
        {"trial_id": "c", "HOTA": 70, "threshold": np.nan, "enabled": None},
        {"trial_id": "d", "HOTA": np.nan, "threshold": 0.9, "enabled": False},
    ])

    rendered = parameter_concentration_markdown(frame, schema)

    assert "**3 near-best trials**" in rendered
    assert "| 2/3 | 2/3; [0.0500, 0.0500] | 2/3; [0.0500, 0.0500] | one observed value; near lower bound |" in rendered
    assert "False: 1, True: 1" in rendered
    assert "None:" not in rendered


def test_fresh_summary_prefers_recorded_schema(tmp_path) -> None:
    """Changing a tracker's current YAML must not rewrite past search bounds."""
    saved_schema = {"threshold": {"type": "uniform", "default": 0.8, "range": [0.7, 0.9]}}
    current_schema = {"threshold": {"type": "uniform", "default": 0.3, "range": [0.1, 0.5]}}
    (tmp_path / "search-space.json").write_text(json.dumps({"schema": saved_schema}))

    path = generate_summary(
        tmp_path,
        [{"trial_id": "a", "metrics": {"HOTA": 70}, "config": {"threshold": 0.8}}],
        current_schema, "bytetrack", ["HOTA"], [], SimpleNamespace(), emit_logs=False,
    )

    rendered = path.read_text()
    assert "[0.7, 0.9]" in rendered
    assert "[0.1, 0.5]" not in rendered


def test_invalid_recorded_schema_does_not_fall_back_to_current_yaml(tmp_path) -> None:
    """A corrupt recording cannot justify current-schema claims about a run."""
    (tmp_path / "search-space.json").write_text("{invalid")
    path = generate_summary(
        tmp_path,
        [{"trial_id": "a", "metrics": {"HOTA": 70}, "config": {"threshold": 0.8}}],
        {"threshold": {"type": "uniform", "default": 0.8, "range": [0, 1]}},
        "bytetrack", ["HOTA"], [], SimpleNamespace(), emit_logs=False,
    )

    assert "Omitted: the recorded search schema is unavailable." in path.read_text()


def test_regenerated_summary_preserves_provenance_and_other_sections(tmp_path) -> None:
    """Regeneration replaces analysis without fabricating a new run date."""
    schema = {"threshold": {"type": "uniform", "default": 0.8, "range": [0.7, 0.9]}}
    (tmp_path / "search-space.json").write_text(json.dumps({"schema": schema}))
    pd.DataFrame([{"trial_id": "a", "HOTA": 70, "threshold": 0.8}]).to_csv(
        tmp_path / "results.csv", index=False,
    )
    prefix = "# Original run\n\n- **Date:** 2024-01-01\n- **Build:** immutable\n\n---\n\n"
    suffix = "---\n\n## HOTA Distribution\n\nOriginal statistics.\n"
    path = tmp_path / "summary.md"
    path.write_text(prefix + "## Parameter Convergence (Top-10 trials)\n\nOld claims.\n\n" + suffix)

    assert update_analysis_summary(tmp_path) == path
    rendered = path.read_text()

    assert rendered.startswith(prefix)
    assert rendered.endswith(suffix)
    assert "## Parameter Concentration" in rendered
    assert "Old claims." not in rendered
    assert "Parameter Convergence" not in rendered
    update_analysis_summary(tmp_path)
    assert path.read_text() == rendered


def test_regenerated_summary_explains_missing_recorded_schema(tmp_path) -> None:
    """CSV-only runs must explicitly omit conditional parameter diagnostics."""
    path = tmp_path / "summary.md"
    path.write_text("# Existing run\n")
    pd.DataFrame([{"trial_id": "a", "HOTA": 70}]).to_csv(tmp_path / "results.csv", index=False)

    update_analysis_summary(tmp_path)

    assert "Omitted: the recorded search schema is unavailable." in path.read_text()
    assert path.read_text().startswith("# Existing run\n")


def test_summary_handles_absent_hota_scores() -> None:
    """Unavailable scores should produce an omission instead of fake ranges."""
    assert "Omitted: HOTA scores are unavailable." in parameter_concentration_markdown(pd.DataFrame(), {})
    assert "Omitted: no finite HOTA scores" in parameter_concentration_markdown(
        pd.DataFrame({"HOTA": [np.nan, np.inf]}), {},
    )


def test_regeneration_without_usable_results_preserves_existing_summary(tmp_path) -> None:
    """Missing results cannot replace an existing report with incomplete data."""
    path = tmp_path / "summary.md"
    path.write_text("# Existing report\n")

    assert update_analysis_summary(tmp_path) is None
    assert path.read_text() == "# Existing report\n"


def test_summary_explains_unresolvable_conditional_schema() -> None:
    """Broken branch metadata should omit diagnostics without crashing export."""
    schema = {
        "threshold": {
            "type": "uniform", "default": 0.5, "range": [0, 1],
            "depends_on": "missing_parent", "cases": {"off": {"default": 0.5}},
        },
    }
    rendered = parameter_concentration_markdown(pd.DataFrame({"HOTA": [70], "threshold": [0.5]}), schema)

    assert "Omitted: the recorded search schema could not be resolved" in rendered


@pytest.mark.parametrize("searchable_parent", [True, False])
def test_summary_explains_omitted_incompatible_branch_distributions(searchable_parent: bool) -> None:
    """Mixed linear/log branches cannot be described using one domain scale."""
    selector = {"default": "linear"}
    if searchable_parent:
        selector.update(type="choice", options=["linear", "log"])
    schema = {
        "mode": selector,
        "threshold": {
            "type": "uniform", "default": 0.5, "range": [0, 1],
            "depends_on": "mode", "cases": {"log": {"type": "loguniform", "range": [1, 100]}},
        },
    }
    frame = pd.DataFrame([
        {"trial_id": "a", "HOTA": 70, "mode": "linear", "threshold": 0.1},
        {"trial_id": "b", "HOTA": 69.9, "mode": "log", "threshold": 10},
    ])

    rendered = parameter_concentration_markdown(frame, schema)

    assert "| threshold |" not in rendered
    assert "threshold" in rendered
    assert "incompatible" in rendered
