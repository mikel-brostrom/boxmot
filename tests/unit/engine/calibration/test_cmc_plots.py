"""CMC plots must distinguish image evidence from the final calibration policy."""

from __future__ import annotations

import builtins
import json
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from boxmot.engine.calibration.cmc_plots import LOGGER, generate_cmc_plot


@pytest.fixture(autouse=True)
def capture_plot_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Capture locally even when other modules disable ancestor propagation."""
    LOGGER.addHandler(caplog.handler)
    try:
        yield
    finally:
        LOGGER.removeHandler(caplog.handler)


@pytest.fixture
def analysis_path(tmp_path: Path) -> Path:
    """Include moving, static, unavailable, and measured-zero camera evidence."""
    report = {
        "schema": "boxmot.cmc-analysis/v2",
        "tracker": "bytetrack",
        "method": "sof",
        "rule": "sof-motion-v1",
        "mask_min_conf": 0.15,
        "split": "validation",
        "settings": {
            "minimum_accepted_fraction": 0.72,
            "minimum_median_motion_fraction_image_diagonal": 0.0025,
        },
        "sequences": {
            "moving": {
                "enabled": True,
                "reason": "reliable_camera_motion",
                "accepted_fraction": 0.8,
                "accepted_transitions": 8,
                "transitions": 10,
                "grid_motion_fraction_image_diagonal": {"median": 0.012},
            },
            "static": {
                "enabled": False,
                "reason": "camera_motion_below_threshold",
                "accepted_fraction": 1,
                "accepted_transitions": 10,
                "transitions": 10,
                "grid_motion_fraction_image_diagonal": {"median": 0.000002},
            },
            "unavailable": {
                "enabled": False,
                "reason": "no_camera_transitions",
                "accepted_fraction": None,
                "accepted_transitions": 0,
                "transitions": 0,
                "grid_motion_fraction_image_diagonal": {"median": None},
            },
            "zero": {
                "enabled": False,
                "reason": "camera_motion_below_threshold",
                "accepted_fraction": 0,
                "accepted_transitions": 0,
                "transitions": 10,
                "grid_motion_fraction_image_diagonal": {"median": 0},
            },
        },
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_cmc_plot_preserves_units_missing_values_and_eligibility(
    analysis_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write a real PNG and inspect the rendered evidence rather than just existence."""
    original = analysis_path.read_bytes()
    figures = []
    savefig = Figure.savefig

    def capture_savefig(figure: Figure, filename: Path, *args, **kwargs) -> None:
        savefig(figure, filename, *args, **kwargs)
        figures.append(figure)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    previous_figures = plt.get_fignums()
    output = generate_cmc_plot(analysis_path)

    assert output == analysis_path.with_suffix(".png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000
    assert analysis_path.read_bytes() == original
    assert plt.get_fignums() == previous_figures
    (figure,) = figures
    assert len(figure.axes) == 2
    accepted, motion = figure.axes
    assert [patch.get_width() for patch in accepted.patches] == pytest.approx([80, 100, 0])
    assert [patch.get_width() for patch in motion.patches] == pytest.approx([1.2, 0.0002, 0])
    assert any(list(line.get_xdata()) == [72, 72] for line in accepted.lines)
    assert any(list(line.get_xdata()) == [0.25, 0.25] for line in motion.lines)
    assert accepted.patches[0].get_facecolor() != accepted.patches[1].get_facecolor()
    assert accepted.patches[1].get_facecolor() == accepted.patches[2].get_facecolor()
    assert accepted.patches[0].get_facecolor() == motion.patches[0].get_facecolor()
    assert motion.get_xscale() == "symlog"
    assert motion.get_xlim()[0] == 0
    assert [label.get_text().splitlines()[0] for label in accepted.get_yticklabels()] == [
        "moving",
        "static",
        "unavailable",
        "zero",
    ]
    accepted_text = [text.get_text() for text in accepted.texts]
    assert "Missing" in accepted_text
    assert "8/10" in accepted_text
    assert "0/0" in accepted_text
    assert "Threshold ≥ 72%" in accepted_text
    motion_text = [text.get_text() for text in motion.texts]
    assert "Missing" in motion_text
    assert "0%" in motion_text
    assert "Threshold ≥ 0.25%" in motion_text
    captions = "\n".join(text.get_text() for text in figure.texts)
    assert "Automatic CMC eligibility" in captions
    assert "1/4 sequences eligible" in captions
    assert "final off/auto policy is selected separately" in captions


def test_cmc_plot_does_not_invent_missing_thresholds_or_decisions(
    analysis_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incomplete summaries remain visibly unknown instead of inheriting defaults."""
    report = json.loads(analysis_path.read_text())
    report.pop("settings")
    report["sequences"] = {"incomplete": {"accepted_fraction": float("nan")}}
    analysis_path.write_text(json.dumps(report))
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    assert generate_cmc_plot(analysis_path) is not None
    (figure,) = figures
    assert "0/1 sequences eligible · 1 unknown" in figure.texts[1].get_text()
    for axis in figure.axes:
        assert not axis.patches
        assert not axis.lines
        assert "Missing" in [text.get_text() for text in axis.texts]
        assert "Threshold unavailable" in [text.get_text() for text in axis.texts]


@pytest.mark.parametrize("content", ["{", "[]", "{}", '{"sequences": {"s": null}}'])
def test_cmc_plot_invalid_report_warns_and_returns_none(
    tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A visualization failure must not abort a completed calibration."""
    analysis = tmp_path / "analysis.json"
    analysis.write_text(content)
    assert generate_cmc_plot(analysis) is None
    assert "Could not generate CMC analysis plot" in caplog.text
    assert not analysis.with_suffix(".png").exists()


def test_cmc_plot_absent_report_is_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """KF-only runs are expected to have no CMC report."""
    assert generate_cmc_plot(tmp_path / "missing.json") is None
    assert caplog.text == ""


def test_cmc_plot_missing_optional_dependency_warns(
    analysis_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Runtime-only installations can still complete camera calibration."""
    original_import = builtins.__import__

    def import_without_matplotlib(name: str, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("Matplotlib not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    assert generate_cmc_plot(analysis_path) is None
    assert "Matplotlib not installed" in caplog.text


def test_cmc_plot_closes_figure_when_save_fails(
    analysis_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Optional artifact writes must not leak open figures when the disk is full."""

    def fail_save(*args, **kwargs) -> None:
        raise OSError("disk full")

    previous_figures = plt.get_fignums()
    monkeypatch.setattr(Figure, "savefig", fail_save)
    assert generate_cmc_plot(analysis_path) is None
    assert "disk full" in caplog.text
    assert plt.get_fignums() == previous_figures
