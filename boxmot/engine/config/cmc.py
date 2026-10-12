"""Sequence-specific camera-motion choices for cached tracking evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_config
from boxmot.trackers.common.motion.cmc.registry import available_cmc_methods
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_cmc_signature
from boxmot.trackers.common.specs import TrackerSpec

# Version both the sequence gate and the binding to the tracker's resolved SOF
# settings: fitted noise depends on the estimator's full-resolution warps.
CMC_ANALYSIS_RULE = "sof-motion-v3"


class _PolicyLoader(yaml.SafeLoader):
    """Reject repeated policy keys instead of silently replacing decisions."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping = super().construct_mapping(node, deep=deep)
        keys = [self.construct_object(key, deep=deep) for key, _value in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("CMC policy mappings must not contain duplicate keys.")
        return mapping


@dataclass(frozen=True, slots=True)
class SequenceCMCPolicy:
    """Choose CMC by sequence while preserving a tracker's other settings.

    Sequence IDs match the materialization exactly. ``default`` is used for
    sequences without an explicit entry; ``method`` applies to enabled ones.
    This policy records decisions from analysis and does not fit Kalman noise.
    """

    method: str = "sof"
    default: bool = False
    sequences: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        methods = available_cmc_methods()
        if not isinstance(self.method, str) or self.method not in methods:
            raise ValueError(f"CMC policy method must be one of: {', '.join(methods)}.")
        if not isinstance(self.default, bool):
            raise ValueError("CMC policy default must be a boolean.")
        if not isinstance(self.sequences, tuple):
            raise ValueError("CMC policy sequences must be an immutable tuple of sequence decisions.")
        names: set[str] = set()
        for entry in self.sequences:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each CMC policy sequence decision must be a (sequence ID, boolean) tuple.")
            name, enabled = entry
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("CMC policy sequence IDs must be non-empty strings without surrounding spaces.")
            if name in names:
                raise ValueError(f"Duplicate CMC policy sequence ID: {name!r}.")
            if not isinstance(enabled, bool):
                raise ValueError(f"CMC policy decision for {name!r} must be a boolean.")
            names.add(name)

    def _validate_tracker(self, spec: TrackerSpec) -> None:
        """Require a Python tracker with explicit enable and method controls."""
        if spec.backend != "python":
            raise ValueError("Sequence CMC policies require the Python tracker backend.")
        config_fields = {field.name for field in fields(get_tracker_config_class(spec.name))}
        if not {"use_cmc", "cmc_method"}.issubset(config_fields):
            raise ValueError(
                f"Tracker {spec.name!r} does not support sequence CMC policies: "
                "its configuration must expose use_cmc and cmc_method."
            )

    def validate(self, spec: TrackerSpec, sequence_ids: Iterable[str]) -> None:
        """Reject unsupported trackers and decisions for sequences absent from the build."""
        self._validate_tracker(spec)
        unknown = {name for name, _enabled in self.sequences} - set(sequence_ids)
        if unknown:
            raise ValueError(f"Unknown sequence IDs in CMC policy: {', '.join(sorted(unknown))}.")

    def apply(self, spec: TrackerSpec, sequence_id: str) -> TrackerSpec:
        """Return a sequence's tracker specification with only CMC options changed."""
        self._validate_tracker(spec)
        options = spec.option_dict
        options.update(
            use_cmc=dict(self.sequences).get(sequence_id, self.default),
            cmc_method=self.method,
        )
        return replace(spec, options=tuple(sorted(options.items())))

    def to_dict(self) -> dict[str, Any]:
        """Return the policy's serializable YAML schema."""
        return {"method": self.method, "default": self.default, "sequences": dict(self.sequences)}


def load_cmc_policy(path: Path | str) -> SequenceCMCPolicy:
    """Read a strict YAML policy without constructing trackers or CMC estimators."""
    policy_path = Path(path)
    try:
        with policy_path.open("r", encoding="utf-8") as handle:
            payload = yaml.load(handle, Loader=_PolicyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse CMC policy {policy_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"CMC policy {policy_path} must contain a YAML mapping.")
    unknown = set(payload) - {"method", "default", "sequences"}
    if unknown:
        raise ValueError(f"Unknown CMC policy fields: {', '.join(sorted(map(str, unknown)))}.")
    sequences = payload.get("sequences", {})
    if not isinstance(sequences, Mapping):
        raise ValueError("CMC policy sequences must be a mapping from sequence IDs to booleans.")
    return SequenceCMCPolicy(
        method=payload.get("method", "sof"),
        default=payload.get("default", False),
        sequences=tuple(sequences.items()),
    )


def validate_joint_cmc_selection(tracker: str, backend: str, *, select_kf: bool) -> None:
    """Reject unsupported joint calibration before image loading or fitting."""
    if not select_kf:
        raise ValueError("--select-cmc requires --select-kf.")
    if tracker not in KALMAN_TRACKER_NAMES or backend != "python":
        raise ValueError("--select-cmc requires a Python 2D Kalman tracker.")


def validate_cmc_analysis(tracker: str, backend: str) -> None:
    """Reject unsupported automatic camera analysis before reading image inputs."""
    if tracker not in KALMAN_TRACKER_NAMES or backend != "python":
        raise ValueError("Automatic CMC analysis requires a Python 2D Kalman tracker.")


def resolve_cmc_policy(args: Any, spec: TrackerSpec) -> SequenceCMCPolicy | Literal["auto"] | None:
    """Resolve camera behavior, leaving automatic sequence decisions to eval setup.

    Calibrated profiles bind the decision rule, masking threshold and SOF settings, so their
    noise remains reusable on new evaluation sequences. Training sequence IDs
    and artifact paths are provenance, never runtime camera decisions.
    """
    options = spec.option_dict
    bound = options.get("calibration.cmc_policy")
    explicit = getattr(args, "cmc_policy", None)
    if bound is not None:
        options = load_tracker_config(spec.name, None, options)
        if bound not in {"auto", "off"} or options.get("calibration.cmc_rule") != CMC_ANALYSIS_RULE:
            raise ValueError(
                "The calibrated CMC profile must specify a supported automatic/off decision rule; "
                "run joint selection again for the current SOF estimator."
            )
        if explicit is not None and str(explicit) != bound:
            raise ValueError("The CMC policy differs from this calibrated KF profile; run joint selection again.")
        if options.get("cmc_min_conf") != options.get("calibration.cmc_min_conf"):
            raise ValueError("The CMC detection-mask threshold differs from its calibrated KF profile.")
        if options.get("cmc_method") != "sof":
            raise ValueError("The CMC method differs from its calibrated KF profile.")
        if any(options.get(name) != value for name, value in calibration_cmc_signature(options).items()):
            raise ValueError(
                "The SOF estimator settings differ from this calibrated KF profile; run joint selection again."
            )
    requested = explicit if explicit is not None else bound
    if requested is None:
        return None
    if str(requested) == "auto":
        validate_cmc_analysis(spec.name, spec.backend)
        return "auto"
    if str(requested) == "off":
        return SequenceCMCPolicy()
    return load_cmc_policy(requested)
