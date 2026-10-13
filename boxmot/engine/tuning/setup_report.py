"""Describe effective camera settings without changing reusable tracker options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boxmot.engine.config.cmc import SequenceCMCPolicy, load_cmc_policy


@dataclass(frozen=True)
class CMCSetupReport:
    """Present the selected strategy and observed sequence overrides together."""

    strategy: str
    method: str
    base_use_cmc: Any
    bound: bool
    enabled: tuple[str, ...] | None
    disabled: tuple[str, ...] | None
    requested: str | None = None

    def _lines(self) -> list[str]:
        """Share the same effective setup description across output formats."""
        base = str(self.base_use_cmc).lower() if isinstance(self.base_use_cmc, bool) else str(self.base_use_cmc)
        lines = [
            f"CMC strategy: {self.strategy}; method: {self.method}.",
            f"Base use_cmc: {base}; the sequence policy overrides this value during evaluation.",
        ]
        if self.enabled is None or self.disabled is None:
            lines.append("Tuning sequence decisions unavailable; the base value does not describe effective CMC.")
        else:
            lines.extend((
                f"CMC enabled ({len(self.enabled)}): {', '.join(self.enabled) or 'none'}.",
                f"CMC disabled ({len(self.disabled)}): {', '.join(self.disabled) or 'none'}.",
            ))
        if self.bound:
            lines.append(f"Saved calibration.cmc_policy: {self.strategy} applies when evaluating this profile.")
        else:
            requested = self.requested or (self.strategy if self.strategy in {"auto", "off"} else "PATH")
            lines.append(f"Reuse requires --cmc-policy {requested}; these annotations do not bind a runtime policy.")
        if self.strategy == "auto":
            lines.append("Automatic CMC analyzes new evaluation sequences; tuning decisions are observations only.")
        return lines

    def yaml_header(self) -> str:
        """Return YAML comments, preserving executable configuration semantics."""
        lines = ["Effective CMC setup", *self._lines()]
        return "".join(f"# {part}\n" for line in lines for part in line.splitlines())

    def markdown(self) -> str:
        """Return a standalone Markdown section describing effective CMC."""
        return "## Effective CMC Setup\n\n" + "\n\n".join(self._lines()) + "\n"


def describe_cmc_setup(
    config: dict, *, args: Any | None = None, tune_dir: Path | None = None
) -> CMCSetupReport | None:
    """Describe prepared decisions or recorded evidence, never infer them from defaults.

    ``config`` contains normalized flat tracker options. Prepared driver state
    takes precedence over saved policy files. Recorded sequence decisions are
    descriptive evidence, not a replacement for the portable automatic rule.
    """
    prepared = getattr(args, "_cmc_setup", None)
    policy = getattr(prepared, "policy", None)
    strategy = getattr(prepared, "strategy", None)
    bound = config.get("calibration.cmc_policy")
    requested = getattr(args, "cmc_policy", None)
    if not isinstance(policy, SequenceCMCPolicy):
        policy = None
        strategy = bound or requested
        policy_path = None if tune_dir is None else Path(tune_dir) / "cmc-policy.yaml"
        if policy_path is not None and policy_path.exists():
            try:
                policy = load_cmc_policy(policy_path)
            except (OSError, ValueError, UnicodeError):
                # Reporting must not turn absent or corrupt evidence into a
                # false claim that the base switch disabled every sequence.
                strategy = strategy or "recorded sequence policy"
            if policy is not None and bound in {"auto", "off"}:
                if policy.method != "sof" or (
                    bound == "off" and (policy.default or any(enabled for _, enabled in policy.sequences))
                ):
                    # A stale artifact must not contradict the executable
                    # calibration binding. Prepared driver state is authoritative.
                    policy = None
    if strategy is None:
        if policy is None:
            return None
        strategy = "recorded sequence policy"
    strategy = str(strategy)
    if strategy not in {"auto", "off", "recorded sequence policy"}:
        strategy = "manual"

    enabled = disabled = None
    if policy is not None:
        names = getattr(args, "sequence_names", None) or getattr(args, "seq_info", None)
        decisions = dict(policy.sequences)
        sequence_ids = sorted(names if names else decisions)
        if sequence_ids:
            enabled = tuple(name for name in sequence_ids if decisions.get(name, policy.default))
            disabled = tuple(name for name in sequence_ids if not decisions.get(name, policy.default))
    return CMCSetupReport(
        strategy=strategy,
        method=policy.method if policy is not None else str(config.get("cmc_method", "sof")),
        base_use_cmc=config.get("use_cmc", "unspecified"),
        bound=bound in {"auto", "off"} and bound == strategy,
        enabled=enabled,
        disabled=disabled,
        requested=None if requested is None else str(requested),
    )
