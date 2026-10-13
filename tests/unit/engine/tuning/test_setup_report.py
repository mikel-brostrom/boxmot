"""Saved setup descriptions distinguish base defaults from effective camera policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from boxmot.engine.config.cmc import SequenceCMCPolicy
from boxmot.engine.tuning.setup_report import describe_cmc_setup


@pytest.mark.parametrize("decisions", [(True, False), (False, False), (True, True)])
def test_prepared_policy_reports_each_sequence_and_preserves_config(tmp_path, decisions) -> None:
    """Even all-on or all-off observations preserve automatic selection semantics."""
    config = {"use_cmc": False, "cmc_method": "sof", "calibration.cmc_policy": "auto"}
    policy = SequenceCMCPolicy(sequences=tuple(zip(("moving", "static"), decisions)))
    args = SimpleNamespace(_cmc_setup=SimpleNamespace(policy=policy, strategy="auto"))
    (tmp_path / "cmc-policy.yaml").write_text("invalid: evidence")

    report = describe_cmc_setup(config, args=args, tune_dir=tmp_path)

    assert report is not None
    assert report.enabled == tuple(name for name, active in policy.sequences if active)
    assert report.disabled == tuple(name for name, active in policy.sequences if not active)
    header = report.yaml_header()
    assert f"CMC enabled ({sum(decisions)}):" in header
    assert f"CMC disabled ({2 - sum(decisions)}):" in header
    assert "Base use_cmc: false; the sequence policy overrides" in header
    assert "Saved calibration.cmc_policy: auto" in header
    assert "analyzes new evaluation sequences" in header
    assert header.endswith("\n")
    assert all(line.startswith("# ") for line in header.splitlines())
    assert yaml.safe_load(header + yaml.safe_dump(config)) == config
    assert report.markdown().startswith("## Effective CMC Setup\n\n")


def test_saved_policy_is_evidence_for_the_bound_strategy(tmp_path) -> None:
    """Existing tuning artifacts explain an export without its original driver."""
    policy = SequenceCMCPolicy(sequences=(("a", False), ("b", True)))
    (tmp_path / "cmc-policy.yaml").write_text(yaml.safe_dump(policy.to_dict()))

    report = describe_cmc_setup({"calibration.cmc_policy": "auto", "use_cmc": False}, tune_dir=tmp_path)

    assert report is not None
    assert report.enabled == ("b",)
    assert report.disabled == ("a",)
    assert report.strategy == "auto"
    assert report.bound


@pytest.mark.parametrize("contents", [None, "not: [valid", "sequences: {a: maybe}"])
def test_missing_or_malformed_evidence_does_not_imply_disabled(tmp_path, contents) -> None:
    """An automatic binding remains automatic when sequence evidence is unavailable."""
    if contents is not None:
        (tmp_path / "cmc-policy.yaml").write_text(contents)

    report = describe_cmc_setup({"calibration.cmc_policy": "auto", "use_cmc": False}, tune_dir=tmp_path)

    assert report is not None
    assert report.enabled is report.disabled is None
    assert "sequence decisions unavailable" in report.yaml_header()
    assert "CMC disabled (" not in report.yaml_header()
    assert report.strategy == "auto"


def test_base_switch_without_policy_needs_no_override_report(tmp_path) -> None:
    """A normal tracker export should not advertise an absent sequence policy."""
    assert describe_cmc_setup({"use_cmc": False}, tune_dir=tmp_path) is None
    assert describe_cmc_setup({"use_cmc": True}) is None


def test_unbound_cli_auto_does_not_pretend_to_be_portable() -> None:
    """Comments cannot replace an executable automatic calibration binding."""
    report = describe_cmc_setup({"use_cmc": False}, args=SimpleNamespace(cmc_policy="auto"))

    assert report is not None
    assert not report.bound
    assert "Reuse requires --cmc-policy auto" in report.yaml_header()
    assert "Saved calibration.cmc_policy" not in report.yaml_header()
    assert "analyzes new evaluation sequences" in report.markdown()


def test_manual_policy_expands_default_for_current_sequences() -> None:
    """An explicit policy's default applies only to the selected sequence inventory."""
    policy = SequenceCMCPolicy(default=True, sequences=(("a", False),))
    args = SimpleNamespace(
        _cmc_setup=SimpleNamespace(policy=policy, strategy="fixed:example"),
        cmc_policy="camera.yaml",
        seq_info={"a": 100, "b": 200},
    )

    report = describe_cmc_setup({"use_cmc": False}, args=args)

    assert report is not None
    assert report.strategy == "manual"
    assert report.enabled == ("b",)
    assert report.disabled == ("a",)
    assert "Reuse requires --cmc-policy camera.yaml" in report.markdown()
    assert "analyzes new" not in report.markdown()


def test_off_policy_lists_current_sequences_without_claiming_automatic() -> None:
    """An empty off policy still resolves every known sequence to disabled."""
    args = SimpleNamespace(
        _cmc_setup=SimpleNamespace(policy=SequenceCMCPolicy(), strategy="off"),
        sequence_names=["a", "b"],
    )

    report = describe_cmc_setup({"calibration.cmc_policy": "off", "use_cmc": True}, args=args)

    assert report is not None
    assert report.enabled == ()
    assert report.disabled == ("a", "b")
    assert "Base use_cmc: true; the sequence policy overrides" in report.yaml_header()
    assert "analyzes new" not in report.yaml_header()


def test_recorded_policy_without_strategy_does_not_invent_automatic_binding(tmp_path) -> None:
    """Historical sequence decisions alone do not identify how they were selected."""
    policy = SequenceCMCPolicy(sequences=(("a", True),))
    (tmp_path / "cmc-policy.yaml").write_text(yaml.safe_dump(policy.to_dict()))

    report = describe_cmc_setup({"use_cmc": False}, tune_dir=tmp_path)

    assert report is not None
    assert report.strategy == "recorded sequence policy"
    assert not report.bound
    assert report.enabled == ("a",)
    assert "Reuse requires --cmc-policy PATH" in report.markdown()


@pytest.mark.parametrize(
    ("bound", "policy"),
    [
        ("off", SequenceCMCPolicy(sequences=(("a", True),))),
        ("off", SequenceCMCPolicy(default=True, sequences=(("a", False),))),
        ("auto", SequenceCMCPolicy(method="ecc", sequences=(("a", True),))),
        ("off", SequenceCMCPolicy(method="ecc", sequences=(("a", False),))),
    ],
)
def test_saved_policy_conflicting_with_binding_omits_decisions(tmp_path, bound, policy) -> None:
    """Stale decisions cannot override an explicit portable calibration strategy."""
    (tmp_path / "cmc-policy.yaml").write_text(yaml.safe_dump(policy.to_dict()))
    config = {"use_cmc": False, "cmc_method": "sof", "calibration.cmc_policy": bound}

    report = describe_cmc_setup(config, tune_dir=tmp_path)

    assert report is not None
    assert report.strategy == bound
    assert report.method == "sof"
    assert report.enabled is report.disabled is None
    assert "sequence decisions unavailable" in report.yaml_header()
    assert "CMC enabled (" not in report.markdown()
    assert "CMC disabled (" not in report.markdown()
