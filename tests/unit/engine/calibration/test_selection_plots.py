"""Saved selection reports render accurate KF/CMC comparisons without reruns."""

from __future__ import annotations

import builtins
import copy
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.calibration import selection_plots as plots

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def capture_selection_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Capture module warnings even when BoxMOT disables ancestor propagation."""
    plots.LOGGER.addHandler(caplog.handler)
    try:
        yield
    finally:
        plots.LOGGER.removeHandler(caplog.handler)


def _profile(
    score: float | None,
    *,
    folds: tuple[float, ...] = (0.7, 0.9),
    center_error: float = 0.05,
    center_folds: tuple[float, ...] = (0.04, 0.08),
) -> dict:
    """Build small valid or invalid profile reports using persisted fields."""
    return {
        "status": "invalid" if score is None else "valid",
        "mean_iou": score,
        "normalized_center_error": center_error,
        "folds": [
            {"index": index, "mean_iou": value, "normalized_center_error": center_folds[index]}
            for index, value in enumerate(folds)
        ],
    }


def _kf_report(*, one_frame: bool = False) -> dict:
    """Keep a deliberate score tie whose saved winner is not the first cell."""
    return {
        "version": 3,
        "tracker": "bytetrack",
        "objective": "maximize_one_frame_mean_iou" if one_frame else "maximize_mean_iou",
        "horizons": [1] if one_frame else [1, 3, 5],
        "candidate_order": ["xyah", "xywh"],
        "profile_order": ["default", "measurement"],
        "selected_type": "xywh",
        "selected_profile": "measurement",
        "candidates": {
            "xyah": {"profiles": {"default": _profile(0.8, center_error=0.03), "measurement": _profile(None)}},
            "xywh": {
                "profiles": {
                    "default": _profile(0.6, center_error=0.01),
                    "measurement": _profile(0.8, folds=(0.65, 0.75), center_error=0.06, center_folds=(0.03, 0.05)),
                }
            },
        },
    }


def _joint_report() -> dict:
    """Two policies include different winner configurations and fold ordering."""
    off = _kf_report(one_frame=True)
    auto = copy.deepcopy(off)
    auto.update(selected_type="xyah", selected_profile="default")
    auto["candidates"]["xyah"]["profiles"]["default"] = _profile(
        0.9, folds=(0.85, 0.95), center_error=0.02, center_folds=(0.015, 0.03)
    )
    auto["candidates"]["xyah"]["profiles"]["default"]["folds"].reverse()
    return {
        **auto,
        "version": 4,
        "selected_cmc": "auto",
        "cmc_candidate_order": ["off", "auto"],
        "cmc_candidates": {"off": off, "auto": auto},
    }


@pytest.mark.parametrize("joint", [False, True])
def test_generate_png_without_mutating_report(tmp_path: Path, joint: bool) -> None:
    """Both saved report formats produce actual PNGs and leave source intact."""
    report_path = tmp_path / "selection.json"
    source = json.dumps(_joint_report() if joint else _kf_report(), indent=2)
    report_path.write_text(source)
    previous = plt.get_fignums()

    output = plots.generate_selection_plot(report_path)

    assert output == tmp_path / "selection.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
    assert report_path.read_text() == source
    assert plt.get_fignums() == previous


def test_joint_heatmaps_share_scale_mask_invalid_and_honor_saved_winners() -> None:
    """Persisted tie breaks, invalid cells and scales survive plotting."""
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, _joint_report())
        off, auto = fig.axes[:2]
        assert off.images[0].norm is auto.images[0].norm
        assert off.images[0].get_clim() == (0.6, 0.9)
        assert np.ma.getmaskarray(off.images[0].get_array())[0, 1]
        assert any(text.get_text() == "Invalid" for text in off.texts)
        # A tie at xyah/default must not override the saved xywh/measurement winner.
        assert off.patches[0].get_xy() == (0.52, 0.52)
        assert auto.patches[0].get_xy() == (-0.48, -0.48)
        assert off.patches[0].get_linestyle() == "--"
        assert auto.patches[0].get_linestyle() == "-"
        assert "policy winner" in off.get_title()
        assert "selected" in auto.get_title()
        assert any("One-frame held-out motion IoU" in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_folds_pair_by_saved_index_and_aggregates_are_not_fold_averages() -> None:
    """Compare the same folds and show exact saved aggregate rather than mean."""
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, _joint_report())
        folds = fig.axes[2]
        np.testing.assert_allclose(folds.lines[0].get_ydata(), [0.65, 0.85])
        np.testing.assert_allclose(folds.lines[1].get_ydata(), [0.75, 0.95])
        # Off's fold average is 0.7; the saved trajectory/sequence aggregate is 0.8.
        np.testing.assert_allclose(folds.collections[0].get_offsets(), [[0, 0.8]])
        np.testing.assert_allclose(folds.collections[1].get_offsets(), [[1, 0.9]])
    finally:
        plt.close(fig)


def test_center_heatmaps_share_percent_scale_and_keep_iou_winners() -> None:
    """Center distance is a lower-is-better diagnostic using saved IoU winners."""
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, _joint_report())
        iou_off, iou_auto = fig.axes[:2]
        center_off, center_auto = fig.axes[3:5]
        assert center_off.images[0].norm is center_auto.images[0].norm
        assert center_off.images[0].get_clim() == (1.0, 6.0)
        assert center_off.images[0].norm is not iou_off.images[0].norm
        np.testing.assert_allclose(center_off.images[0].get_array()[1], [1.0, 6.0])
        np.testing.assert_allclose(center_auto.images[0].get_array()[0, 0], 2.0)
        np.testing.assert_allclose(center_off.images[0].cmap(0.0), iou_off.images[0].cmap(1.0))
        np.testing.assert_allclose(center_off.images[0].cmap(1.0), iou_off.images[0].cmap(0.0))
        assert np.ma.getmaskarray(center_off.images[0].get_array())[0, 1]
        assert any(text.get_text() == "Invalid" for text in center_off.texts)
        # The lowest error is xywh/default; selection still comes from saved IoU ranking.
        assert center_off.patches[0].get_xy() == iou_off.patches[0].get_xy() == (0.52, 0.52)
        assert center_auto.patches[0].get_xy() == iou_auto.patches[0].get_xy() == (-0.48, -0.48)
        labels = [text.get_text() for text in fig.texts]
        labels.extend(ax.get_title() for ax in fig.axes)
        labels.extend(ax.get_ylabel() for ax in fig.axes)
        labels.extend(ax.get_xlabel() for ax in fig.axes)
        assert any("lower is better" in label.lower() for label in labels)
        assert any("gt box diagonal" in label.lower() for label in labels)
    finally:
        plt.close(fig)


def test_center_folds_and_aggregates_use_saved_percent_values() -> None:
    """The second fold panel uses paired center errors and exact report aggregates."""
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, _joint_report())
        folds = fig.axes[5]
        np.testing.assert_allclose(folds.lines[0].get_ydata(), [3.0, 1.5])
        np.testing.assert_allclose(folds.lines[1].get_ydata(), [5.0, 3.0])
        # Off's fold average is 4%; its saved trajectory/sequence aggregate is 6%.
        np.testing.assert_allclose(folds.collections[0].get_offsets(), [[0, 6.0]])
        np.testing.assert_allclose(folds.collections[1].get_offsets(), [[1, 2.0]])
    finally:
        plt.close(fig)


def test_center_distance_above_one_is_not_clamped() -> None:
    """Center errors exceeding a full box diagonal remain visible above 100%."""
    report = _joint_report()
    winner = report["cmc_candidates"]["off"]["candidates"]["xywh"]["profiles"]["measurement"]
    winner["normalized_center_error"] = 2.5
    winner["folds"][0]["normalized_center_error"] = 4.0
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, report)
        assert fig.axes[3].images[0].get_array()[1, 1] == 250.0
        assert fig.axes[3].images[0].get_clim()[1] == 250.0
        folds = fig.axes[5]
        assert folds.lines[0].get_ydata()[0] == 400.0
        assert folds.get_ylim()[1] > 400.0
        np.testing.assert_allclose(folds.collections[0].get_offsets(), [[0, 250.0]])
    finally:
        plt.close(fig)


def test_missing_center_value_and_fold_remain_masked() -> None:
    """Missing diagnostics are distinct from failures and do not fabricate fold pairs."""
    report = _joint_report()
    policy = report["cmc_candidates"]["off"]
    del policy["candidates"]["xyah"]["profiles"]["default"]["normalized_center_error"]
    del policy["candidates"]["xywh"]["profiles"]["measurement"]["folds"][0]["normalized_center_error"]
    fig = plt.figure(figsize=(15, 11))
    try:
        plots._draw_selection(fig, report)
        center_off = fig.axes[3]
        assert np.ma.getmaskarray(center_off.images[0].get_array())[0, 0]
        assert np.ma.getmaskarray(center_off.images[0].get_array())[0, 1]
        assert any(text.get_text() == "N/A" for text in center_off.texts)
        assert any(text.get_text() == "Invalid" for text in center_off.texts)
        center_folds = fig.axes[5]
        np.testing.assert_allclose(center_folds.lines[0].get_ydata(), [np.nan, 1.5], equal_nan=True)
        np.testing.assert_allclose(center_folds.lines[1].get_ydata(), [5.0, 3.0])
        # An absent error diagnostic never masks an available IoU score.
        assert not np.ma.getmaskarray(fig.axes[0].images[0].get_array())[0, 0]
    finally:
        plt.close(fig)


def test_report_without_center_errors_still_renders(tmp_path: Path) -> None:
    """Previously persisted IoU reports remain renderable when diagnostics are absent."""
    report = _kf_report()
    for candidate in report["candidates"].values():
        for profile in candidate["profiles"].values():
            del profile["normalized_center_error"]
            for fold in profile["folds"]:
                del fold["normalized_center_error"]
    _, _, panels = plots._selection_panels(report)
    assert np.isnan(panels[0].center_errors).all()
    assert panels[0].center_labels[0, 0] == "N/A"
    assert panels[0].center_labels[0, 1] == "Invalid"
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(report))
    output = plots.generate_selection_plot(report_path)
    assert output is not None
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_kf_only_objective_is_horizon_average() -> None:
    """KF-only ranking describes the actual horizons from the persisted report."""
    report = _kf_report()
    report["horizons"] = [1, 5]
    assert plots._objective_label(report) == "Horizon-average held-out motion IoU (1/5 frames)"


def test_invalid_policy_has_masked_heatmap_and_no_winner() -> None:
    """An ineligible CMC policy remains visible without receiving a zero score."""
    report = _joint_report()
    report["cmc_candidates"]["off"] = {"status": "invalid", "error": "divergence"}
    _, _, panels = plots._selection_panels(report)
    assert panels[0].winner is None
    assert np.isnan(panels[0].scores).all()
    assert (panels[0].labels == "Invalid").all()
    assert np.isnan(panels[0].center_errors).all()
    assert (panels[0].center_labels == "Invalid").all()
    assert panels[1].selected


def test_invalid_family_without_profiles_is_explicitly_labeled() -> None:
    """A failed KF family is invalid, including when no profile ran successfully."""
    report = _kf_report()
    report["candidates"]["xyah"] = {"status": "invalid", "error": "fit failed"}
    _, _, panels = plots._selection_panels(report)
    assert np.isnan(panels[0].scores[0]).all()
    assert (panels[0].labels[0] == "Invalid").all()
    assert np.isnan(panels[0].center_errors[0]).all()
    assert (panels[0].center_labels[0] == "Invalid").all()
    assert panels[0].winner == (1, 1)


def test_missing_report_is_quiet(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Ordinary runs without selection do not emit plotting warnings."""
    assert plots.generate_selection_plot(tmp_path / "missing.json") is None
    assert not caplog.records


@pytest.mark.parametrize("payload", ["not json", "[]", '{"version": 99}', '{"version": 3}'])
def test_malformed_or_unsupported_report_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: str) -> None:
    """Bad saved reports cannot interrupt calibration or create misleading plots."""
    report_path = tmp_path / "selection.json"
    report_path.write_text(payload)
    assert plots.generate_selection_plot(report_path) is None
    assert not report_path.with_suffix(".png").exists()
    assert "Could not generate KF selection plot" in caplog.text


def test_missing_matplotlib_is_nonfatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plotting remains optional for installations without matplotlib."""
    original_import = builtins.__import__

    def missing_plotting(name: str, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("matplotlib unavailable")
        return original_import(name, *args, **kwargs)

    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(_kf_report()))
    monkeypatch.setattr(builtins, "__import__", missing_plotting)
    assert plots.generate_selection_plot(report_path) is None


def test_render_failure_closes_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a failure after figure allocation must not leak pyplot figures."""
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(_kf_report()))
    previous = plt.get_fignums()

    def fail_render(*args) -> None:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(plots, "_draw_selection", fail_render)
    assert plots.generate_selection_plot(report_path) is None
    assert plt.get_fignums() == previous
