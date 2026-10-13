"""Saved tuning configs explain effective camera policies without changing replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from boxmot.components.resolution import component_options
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, resolve_cmc_policy
from boxmot.engine.tuning import analysis, postprocessing
from boxmot.trackers.common.config import load_tracker_config, nest_tracker_options
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_cmc_signature
from boxmot.trackers.common.specs import TrackerSpec


def _base_config() -> dict:
    """Return a portable bound SOF profile with a deliberately disabled base switch."""
    config = load_tracker_config("bytetrack")
    config.update(
        {
            "use_cmc": False,
            "cmc_method": "sof",
            "cmc_min_conf": 0.1,
            "calibration.cmc_policy": "auto",
            "calibration.cmc_rule": CMC_ANALYSIS_RULE,
            "calibration.cmc_min_conf": 0.1,
            **calibration_cmc_signature(config),
        }
    )
    return config


def _record_policy(directory: Path) -> SequenceCMCPolicy:
    """Record a mixed policy whose decisions cannot be expressed by one Boolean."""
    policy = SequenceCMCPolicy(sequences=(("moving-a", True), ("static", False), ("moving-b", True)))
    (directory / "cmc-policy.yaml").write_text(yaml.safe_dump(policy.to_dict()), encoding="utf-8")
    return policy


def _assert_mixed_policy_report(text: str) -> None:
    """Require both aggregate counts and every recorded sequence decision."""
    assert "auto" in text
    assert "sof" in text.lower()
    assert "enabled (2)" in text.lower()
    assert "disabled (1)" in text.lower()
    for name in ("moving-a", "moving-b", "static"):
        assert name in text


def test_result_export_reports_mixed_cmc_policy_and_preserves_replay_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best and per-trial configs must show the same effective setup as the summary."""
    base = _base_config()
    _record_policy(tmp_path)
    trial_dir = tmp_path / "trial_winner"
    trial_dir.mkdir()
    trial_config = {"match_thresh": 0.83}
    result = SimpleNamespace(
        error=None, path=str(trial_dir), config=trial_config,
        metrics={"trial_id": "winner", "HOTA": 69.45},
    )
    monkeypatch.setattr(analysis, "generate_tune_analysis", lambda *args, **kwargs: None)

    postprocessing.save_all_results(
        tmp_path, [result], {}, "bytetrack", ["HOTA"], [],
        SimpleNamespace(sequence_names=("moving-a", "static", "moving-b")),
        base_config=base, emit_logs=False,
    )

    for path in (tmp_path / "best.yaml", trial_dir / "bytetrack_winner.yaml"):
        text = path.read_text()
        header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
        _assert_mixed_policy_report(header)
        assert yaml.safe_load(text) == nest_tracker_options({"tracker": "bytetrack", **base, **trial_config})
        assert "#" in next(line for line in text.splitlines() if line.startswith("use_cmc:"))
        loaded = load_tracker_config("bytetrack", path)
        spec = TrackerSpec("bytetrack", options=component_options(loaded))
        assert resolve_cmc_policy(SimpleNamespace(), spec) == "auto"
    summary = (tmp_path / "summary.md").read_text()
    cmc_section = summary.split("## Effective CMC Setup", 1)[1].split("\n## ", 1)[0]
    _assert_mixed_policy_report(cmc_section)


def test_refresh_adds_effective_cmc_report_without_changing_saved_provenance(tmp_path: Path) -> None:
    """Refreshing an existing run changes presentation, never selected tracker values."""
    base = _base_config()
    _record_policy(tmp_path)
    original_config = nest_tracker_options({"tracker": "bytetrack", **base, "match_thresh": 0.83})
    best_path = tmp_path / "best.yaml"
    best_path.write_text(yaml.safe_dump(original_config, sort_keys=False), encoding="utf-8")
    schema = {"match_thresh": {"type": "uniform", "default": 0.8, "range": [0.7, 0.9]}}
    (tmp_path / "search-space.json").write_text(json.dumps({"schema": schema}), encoding="utf-8")
    pd.DataFrame([{"trial_id": "winner", "HOTA": 69.45, "match_thresh": 0.83}]).to_csv(
        tmp_path / "results.csv", index=False,
    )
    summary_path = tmp_path / "summary.md"
    provenance = "# Tuning Summary: bytetrack\n\n- **Date:** 2024-01-01 12:34\n- **Build:** immutable\n"
    summary_path.write_text(
        provenance + "\n## Best Trial: `winner`\n\nOriginal metrics.\n"
        "\n## Parameter Convergence\n\nOld claim.\n"
        "\n## HOTA Distribution\n\nOriginal distribution.\n", encoding="utf-8",
    )

    assert postprocessing.update_analysis_summary(tmp_path) == summary_path

    best_text, summary_text = best_path.read_text(), summary_path.read_text()
    assert yaml.safe_load(best_text) == original_config
    _assert_mixed_policy_report(best_text)
    _assert_mixed_policy_report(summary_text)
    assert summary_text.startswith(provenance)
    assert "Original metrics." in summary_text
    assert "Original distribution." in summary_text
    assert "Old claim." not in summary_text
    assert summary_text.count("## Effective CMC Setup") == 1
    assert best_text.lower().count("# effective cmc setup") == 1

    postprocessing.update_analysis_summary(tmp_path)

    assert best_path.read_text() == best_text
    assert summary_path.read_text() == summary_text
