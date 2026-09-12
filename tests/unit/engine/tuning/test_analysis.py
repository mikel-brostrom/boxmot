from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy.stats import ConstantInputWarning

from boxmot.engine.tuning.analysis import generate_tune_analysis
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS


@pytest.mark.parametrize("constant_hota", [False, True])
def test_analysis_omits_constant_pairs_and_preserves_valid_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant_hota: bool
) -> None:
    trials = 16
    results = pd.DataFrame(
        {
            "trial_id": np.arange(trials),
            "HOTA": np.full(trials, 60.0) if constant_hota else np.linspace(50.0, 65.0, trials),
            "MOTA": np.full(trials, 62.0) if constant_hota else np.linspace(55.0, 80.0, trials),
            "IDF1": np.full(trials, 66.0) if constant_hota else np.linspace(60.0, 85.0, trials),
            "det_thresh": np.linspace(0.1, 0.8, trials),
            # Conditional values have sufficient observations, but become
            # constant once unusable pairs are removed.
            "conditional_threshold": [0.2] * 12 + [np.nan, np.inf, -np.inf, np.nan],
            "variable_dt": True,
            "kf_time_unit": "seconds",
            "kf_reference_dt_s": 0.05,
            **{name: float(index + 2) for index, name in enumerate(KALMAN_NOISE_OPTIONS)},
        }
    )
    results.to_csv(tmp_path / "results.csv", index=False)
    saved_figures: dict[str, Figure] = {}
    savefig = Figure.savefig

    def capture_savefig(figure: Figure, filename: Path, *args, **kwargs) -> None:
        savefig(figure, filename, *args, **kwargs)
        saved_figures[Path(filename).name] = figure

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        output = generate_tune_analysis(tmp_path, tracker_name="botsort")

    assert output == tmp_path / "analysis.png"
    for filename in ("analysis.png", "analysis_scatter.png"):
        artifact = tmp_path / filename
        assert artifact.stat().st_size > 1000
        assert artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    importance_axes = [axis for axis in saved_figures["analysis.png"].axes if "Spearman" in axis.get_xlabel()]
    if constant_hota:
        assert importance_axes == []
    else:
        assert len(importance_axes) == 1
        importance = importance_axes[0]
        assert [tick.get_text() for tick in importance.get_yticklabels()] == ["det_thresh"]
        assert [bar.get_width() for bar in importance.patches] == pytest.approx([1.0])

    scatter = saved_figures["analysis_scatter.png"].axes[0]
    assert scatter.get_xlabel() == "det_thresh"
    np.testing.assert_allclose(scatter.collections[0].get_offsets(), results[["det_thresh", "HOTA"]])
