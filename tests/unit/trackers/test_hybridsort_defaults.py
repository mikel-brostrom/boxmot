"""HybridSORT's plain evaluation command uses its historical baseline settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot import HybridSort, HybridSortConfig, create_tracker
from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.eval.evaluator import _tracker_options
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.specs import TrackerSpec


def _preset_algorithm_config(reference: str) -> HybridSortConfig:
    """Read only algorithm fields from a complete runtime preset."""
    options = load_tracker_config("hybridsort", reference)
    return HybridSortConfig.from_mapping({name: options[name] for name in HybridSortConfig.fields()})


def test_hybridsort_defaults_restore_the_historical_mot17_baseline() -> None:
    """A plain API call must reproduce the documented explicit baseline preset."""
    config = HybridSortConfig()

    assert config == _preset_algorithm_config("hybridsort-mot17-ablation")
    assert HybridSort().config == create_tracker("hybridsort").config == config
    assert (config.max_age, config.max_obs, config.min_hits) == (30, 50, 3)
    assert (config.det_thresh, config.iou_threshold, config.asso_func) == (0.3, 0.3, "iou")
    assert config.use_byte is True
    assert config.adapfs is False
    assert config.longterm_reid_weight == 0.0


def test_sportsmot_preset_preserves_the_previous_tuned_algorithm_settings() -> None:
    """Moving tuning out of global defaults must preserve every authored value."""
    config = _preset_algorithm_config("hybridsort-sportsmot-val")

    assert config.to_dict() == {
        "max_age": 230,
        "max_obs": 90,
        "min_hits": 1,
        "iou_threshold": 0.24623615333496582,
        "asso_func": "diou",
        "cmc_method": "ecc",
        "use_embeddings": True,
        "low_thresh": 0.1,
        "delta_t": 4,
        "inertia": 0.07385224556640951,
        "use_byte": False,
        "longterm_bank_length": 270,
        "alpha": 0.9189916764734039,
        "adapfs": True,
        "track_thresh": 0.3190991353484191,
        "eg_weight_high_score": 3.8961609177336562,
        "eg_weight_low_score": 0.5096125821683565,
        "tcm_first_step": True,
        "tcm_byte_step": True,
        "tcm_byte_step_weight": 1.0,
        "with_longterm_reid": True,
        "longterm_reid_weight": 1.9752492019041523,
        "with_longterm_reid_correction": True,
        "longterm_reid_correction_thresh": 0.11310432273756706,
        "longterm_reid_correction_thresh_low": 0.3479740001301599,
        "det_thresh": 0.38633684113126876,
    }
    assert config != HybridSortConfig()


@pytest.mark.parametrize("reference", [None, "hybridsort-mot17-ablation", "hybridsort-sportsmot-val"])
def test_eval_cli_resolves_the_same_hybridsort_config_as_the_python_api(
    monkeypatch: pytest.MonkeyPatch, reference: str | None
) -> None:
    """Exercise CLI dispatch and replay option resolution without acquiring data."""
    captured: dict[str, SimpleNamespace] = {}

    def capture_workflow(module: str, args: SimpleNamespace) -> None:
        assert module == "boxmot.engine.eval.evaluator"
        captured["args"] = args

    monkeypatch.setattr(_support, "_run_engine_workflow", capture_workflow)
    command = [
        "eval",
        "--experiment",
        "mot17/ablation-yolox-lmbn.yaml",
        "--build",
        "fixture-build",
        "--tracker",
        "hybridsort",
        "--tracker-backend",
        "python",
        "--no-cache-inputs",
    ]
    if reference is not None:
        command.extend(("--tracker-config", reference))

    result = CliRunner().invoke(boxmot, command)

    assert result.exit_code == 0, result.output
    args = captured["args"]
    if reference is None:
        assert args.tracker_config is None
    tracker = create_tracker(TrackerSpec("hybridsort", options=_tracker_options(args, None)))
    expected = HybridSortConfig() if reference is None else _preset_algorithm_config(reference)
    assert tracker.config == expected
