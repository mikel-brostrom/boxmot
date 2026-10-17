from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy.stats import ConstantInputWarning

from boxmot.engine.tuning.analysis import _categorical_panels, _numeric_panels, generate_tune_analysis
from boxmot.engine.tuning.analysis_data import load_tune_analysis


def _write_run(directory: Path, *, constant_hota: bool = False) -> pd.DataFrame:
    """Create a small run with active/inactive branches and sequence regressions."""
    trials = 16
    frame = pd.DataFrame(
        {
            "trial_id": [f"{index:08d}" for index in range(trials)],
            "HOTA": np.full(trials, 60.0) if constant_hota else np.linspace(50, 65, trials),
            "MOTA": np.linspace(60, 75, trials),
            "IDF1": np.linspace(70, 80, trials)[::-1],
            "IDSW": np.arange(trials) + 20,
            "threshold": np.linspace(0.1, 0.8, trials),
            "fixed_threshold": 0.3,
            "asso_func": ["iou"] * 12 + ["giou"] * 4,
            "motion_fusion": ["off"] * 5 + ["blend"] * 11,
            "motion_fusion_lambda": [0.98] * 5 + list(np.linspace(0.1, 0.9, 11)),
            "use_cmc": False,
            "new_boolean": [False] * 8 + [True] * 8,
        }
    )
    frame.to_csv(directory / "results.csv", index=False)
    schema = {
        "threshold": {"type": "uniform", "range": [0.1, 0.8], "default": 0.1},
        "fixed_threshold": {"default": 0.3},
        "asso_func": {"type": "choice", "options": ["iou", "giou", "centroid"], "default": "iou"},
        "motion_fusion": {"type": "choice", "options": ["off", "blend"], "default": "off"},
        "motion_fusion_lambda": {
            "type": "uniform",
            "range": [0, 1],
            "default": 0.98,
            "depends_on": "motion_fusion",
            "cases": {"off": {"default": 0.98}},
        },
        "use_cmc": {"default": False},
        "new_boolean": {"type": "choice", "options": [False, True], "default": False},
    }
    (directory / "search-space.json").write_text(json.dumps({"schema": schema, "fixed_options": {}}))
    for index, row in frame.iterrows():
        trial = directory / f"trial_{row['trial_id']}"
        trial.mkdir()
        view = {
            "DetA": 55 + index / 4,
            "AssA": 65 + index / 2,
            "per_sequence": {"sequence-a": {"HOTA": 50 + index}, "sequence-b": {"HOTA": 65 - index}},
        }
        result = {
            "HOTA": row["HOTA"],
            "timestamp": 1000 + index,
            "done": True,
            "_validation": {"raw": {"pedestrian": view}, "summary_label": "pedestrian"},
        }
        (trial / "result.json").write_text(json.dumps(result))
    return frame


@pytest.mark.parametrize("constant_hota", [False, True])
def test_analysis_generates_conditional_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant_hota: bool
) -> None:
    frame = _write_run(tmp_path, constant_hota=constant_hota)
    figures: dict[str, Figure] = {}
    original_save = Figure.savefig

    def capture(figure: Figure, filename: Path, *args, **kwargs) -> None:
        original_save(figure, filename, *args, **kwargs)
        figures[Path(filename).name] = figure

    monkeypatch.setattr(Figure, "savefig", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        warnings.simplefilter("error", FutureWarning)
        assert generate_tune_analysis(tmp_path, tracker_name="fixture") == tmp_path / "analysis.png"

    expected = {"analysis.png", "analysis_scatter.png", "analysis_categories.png", "analysis_candidates.png"}
    assert set(figures) == expected
    for name in expected:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / name).stat().st_size > 1000
    scatter = {ax.get_xlabel(): ax for ax in figures["analysis_scatter.png"].axes}
    assert set(scatter) == {"threshold", "motion_fusion_lambda"}
    np.testing.assert_allclose(scatter["threshold"].collections[0].get_offsets(), frame[["threshold", "HOTA"]])
    np.testing.assert_allclose(
        scatter["motion_fusion_lambda"].collections[0].get_offsets(), frame.loc[5:, ["motion_fusion_lambda", "HOTA"]]
    )
    assert scatter["motion_fusion_lambda"].get_xlim() == pytest.approx((0, 1))
    assert "active n=11" in scatter["motion_fusion_lambda"].get_title()
    titles = " ".join(ax.get_title() for ax in figures["analysis_categories.png"].axes)
    assert "new_boolean" in titles
    assert "use_cmc" not in titles
    assert "Joint sampling" in titles
    assert "centroid\nn=0" in [tick.get_text() for tick in figures["analysis_categories.png"].axes[0].get_xticklabels()]
    correlation = " ".join(text.get_text() for text in scatter["threshold"].get_legend().get_texts())
    assert ("undefined" if constant_hota else "ρ=+1.00") in correlation
    heatmap = figures["analysis_candidates.png"].axes[0]
    if not constant_hota:
        np.testing.assert_allclose(heatmap.collections[0].get_array().reshape(10, 2)[0], [15, -15])
    for figure in figures.values():
        assert all("Importance" not in axis.get_title() and "Impact" not in axis.get_title() for axis in figure.axes)


@pytest.mark.parametrize("values", [(False, True), ("False", "True"), (0, 1)])
def test_analysis_missing_boolean_values_are_not_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: tuple[object, object]
) -> None:
    frame = _write_run(tmp_path)
    false, true = values
    frame["new_boolean"] = [None] * 4 + [false] * 6 + [true] * 6
    frame.to_csv(tmp_path / "results.csv", index=False)
    data = load_tune_analysis(tmp_path)
    assert data is not None
    samples = data.parameters["new_boolean"].samples
    assert samples.groupby("value").size().to_dict() == {False: 6, True: 6}
    np.testing.assert_allclose(samples["HOTA"], frame.loc[4:, "HOTA"])


def test_analysis_without_metadata_explains_omissions_and_removes_stale_plots(tmp_path: Path) -> None:
    pd.DataFrame({"HOTA": [0.0, 1.0, 2.0], "unclassified_number": [1, 2, 3]}).to_csv(
        tmp_path / "results.csv", index=False
    )
    for name in ("analysis_scatter.png", "analysis_categories.png", "analysis_candidates.png"):
        (tmp_path / name).write_bytes(b"stale")
    assert generate_tune_analysis(tmp_path) == tmp_path / "analysis.png"
    assert {path.name for path in tmp_path.glob("*.png")} == {"analysis.png"}
    data = load_tune_analysis(tmp_path)
    assert data is not None and len(data.trials) == 3
    assert data.parameters == {}
    assert data.notes
    assert not data.chronological
    assert "first" in data.baseline_label.lower()


def test_analysis_rejects_missing_or_invalid_scores(tmp_path: Path) -> None:
    assert generate_tune_analysis(tmp_path) is None
    pd.DataFrame({"HOTA": [np.nan, np.inf, "bad"]}).to_csv(tmp_path / "results.csv", index=False)
    assert generate_tune_analysis(tmp_path) is None


def test_categories_without_joint_observations_omit_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _write_run(tmp_path)
    frame.loc[:7, "asso_func"] = None
    frame.loc[8:, "motion_fusion"] = None
    frame.to_csv(tmp_path / "results.csv", index=False)
    data = load_tune_analysis(tmp_path)
    assert data is not None
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _categorical_panels(data, tmp_path)
    assert len(figures) == 1
    titles = [axis.get_title() for axis in figures[0].axes]
    assert any("asso_func" in title for title in titles)
    assert any("motion_fusion" in title for title in titles)
    assert not any("Joint" in title for title in titles)


def test_categorical_comparisons_are_separate_for_each_active_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off defaults must not join gate/blend samples or combine their outcomes."""
    frame = pd.DataFrame({
        "trial_id": list("abcdefghi"),
        "HOTA": [99, 98, 97, 61, 62, 63, 71, 72, 73],
        "motion_fusion": ["off"] * 3 + ["blend"] * 3 + ["gate"] * 3,
        "policy": ["fast"] * 3 + ["fast", "slow", "fast"] + ["position", "full", "position"],
    })
    schema = {
        "motion_fusion": {"type": "choice", "options": ["off", "blend", "gate"], "default": "off"},
        "policy": {
            "type": "choice", "options": ["fast", "slow"], "default": "fast",
            "depends_on": "motion_fusion",
            "cases": {"off": {"default": "fast"}, "gate": {"options": ["position", "full"]}},
        },
    }
    frame.to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "search-space.json").write_text(json.dumps({"schema": schema}))
    data = load_tune_analysis(tmp_path)
    assert data is not None
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    _categorical_panels(data, tmp_path)
    policy_axes = [axis for axis in figures[0].axes if axis.get_title().startswith("policy")]
    assert len(policy_axes) == 2
    for axis, branch, expected_labels in zip(policy_axes, ("blend", "gate"), (("fast", "slow"), ("position", "full"))):
        assert f"motion_fusion={branch} · active n=3" in axis.get_title()
        assert [label.get_text().split("\n")[0] for label in axis.get_xticklabels()] == list(expected_labels)
        np.testing.assert_allclose(
            axis.collections[0].get_offsets()[:, 1], frame.loc[frame["motion_fusion"] == branch, "HOTA"]
        )


def test_numeric_correlations_use_each_branch_not_pooled_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opposite within-branch trends must remain visible after excluding off defaults."""
    frame = pd.DataFrame({
        "trial_id": [str(index) for index in range(36)],
        "HOTA": [90] * 12 + list(range(50, 62)) + list(range(80, 68, -1)),
        "mode": ["off"] * 12 + ["a"] * 12 + ["b"] * 12,
        "weight": [0.98] * 12 + list(np.linspace(0, 1, 12)) * 2,
    })
    schema = {
        "mode": {"type": "choice", "default": "off", "options": ["off", "a", "b"]},
        "weight": {
            "type": "uniform", "range": [0, 1], "default": 0.98,
            "depends_on": "mode", "cases": {"off": {"default": 0.98}},
        },
    }
    frame.to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "search-space.json").write_text(json.dumps({"schema": schema}))
    data = load_tune_analysis(tmp_path)
    assert data is not None
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    _numeric_panels(data, tmp_path)
    axis = figures[0].axes[0]
    assert "active n=24" in axis.get_title()
    assert [label.get_text() for label in axis.get_legend().get_texts()] == [
        "mode=a: n=12; Spearman ρ=+1.00", "mode=b: n=12; Spearman ρ=-1.00",
    ]
    np.testing.assert_allclose(axis.collections[0].get_offsets(), frame.loc[12:23, ["weight", "HOTA"]])
    np.testing.assert_allclose(axis.collections[1].get_offsets(), frame.loc[24:35, ["weight", "HOTA"]])
