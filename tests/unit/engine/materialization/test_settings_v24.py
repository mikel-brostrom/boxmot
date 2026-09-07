from __future__ import annotations

import pytest

from boxmot.engine.materialization.settings import load_executor_settings


def test_executor_defaults_match_v24_contract() -> None:
    settings = load_executor_settings()

    assert settings["detect"] == {
        "batch_size": 16,
        "workers": 1,
        "executor": "inline",
        "retries": 2,
        "retry_backoff_s": 0.0,
    }
    assert settings["segment"]["batch_size"] == 8
    assert settings["embed"]["batch_size"] == 64
    assert settings["finalize"]["retries"] == 0
    assert settings["writer"] == {"instance_rows_per_shard": 50_000, "workers": 1}
    assert 1 <= settings["decode"]["workers"] <= 4


def test_executor_yaml_and_yaml_typed_overrides(tmp_path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("detect:\n  batch_size: 4\n", encoding="utf-8")

    settings = load_executor_settings(
        plan,
        ("detect.retry_backoff_s=0.25", "embed.executor=process", "embed.workers=2"),
    )

    assert settings["detect"]["batch_size"] == 4
    assert settings["detect"]["retry_backoff_s"] == 0.25
    assert settings["embed"]["workers"] == 2
    assert settings["embed"]["executor"] == "process"


@pytest.mark.parametrize(
    "override",
    ("unknown.workers=1", "detect.unknown=1", "detect.workers=0", "detect.workers=true", "bad"),
)
def test_executor_overrides_reject_unknown_or_invalid_values(override) -> None:
    with pytest.raises((TypeError, ValueError)):
        load_executor_settings(overrides=(override,))
