"""Decide camera compensation once during evaluation or tuning setup."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import cv2
import yaml

from boxmot.engine.calibration.cmc import cmc_analysis_settings, prepare_cmc_candidates
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, resolve_cmc_policy
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf

if TYPE_CHECKING:
    from boxmot.engine.eval.session import ReplaySession


@dataclass(frozen=True)
class PreparedCMCSetup:
    """Carry setup decisions and their evidence across independent Ray actors."""

    input_key: str
    strategy: str
    policy: SequenceCMCPolicy
    report: bytes | None = None
    warps: bytes | None = None


def _analysis_inputs(args: Any, spec: TrackerSpec) -> dict[str, Any]:
    """Identify the selected build and settings that affect camera analysis."""
    options = load_tracker_config(spec.name, None, spec.option_dict)
    threshold = resolve_cmc_min_conf(options)
    build = Path(args.build_path).resolve()
    sequences = sorted(getattr(args, "sequence_names", None) or args.seq_info)
    return {
        "build": str(build),
        "manifest_sha256": hashlib.sha256((build / "manifest.json").read_bytes()).hexdigest(),
        "split": args.split,
        "sequences": [(name, args.seq_info[name]) for name in sequences],
        "tracker": spec.name,
        "backend": spec.backend,
        "geometry": spec.geometry,
        "class_ids": spec.class_ids,
        "class_names": spec.class_names,
        "rule": options.get("calibration.cmc_rule", CMC_ANALYSIS_RULE),
        # Automatic policies always analyze SOF, including trackers whose
        # configured runtime method is ECC until the sequence policy applies.
        "method": "sof",
        "settings": cmc_analysis_settings(options),
        "mask_min_conf": float(threshold),
    }


def _analysis_key(args: Any, spec: TrackerSpec, strategy: str) -> str:
    """Bind setup decisions to camera inputs and the requested strategy."""
    selection = {**_analysis_inputs(args, spec), "strategy": strategy}
    if strategy != "auto":
        # Explicit decisions do not depend on the image analyzer. Tuning its
        # settings must not invalidate an unbound disabled/manual policy.
        selection.pop("mask_min_conf")
        selection.pop("settings")
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()


def _strategy(policy: SequenceCMCPolicy | str) -> str:
    """Distinguish automatic, disabled, and explicit sequence policies."""
    if policy == "auto":
        return "auto"
    if policy == SequenceCMCPolicy():
        return "off"
    return "fixed:" + hashlib.sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()


def _publish_setup(directory: Path, prepared: PreparedCMCSetup) -> None:
    """Replace only this workflow's artifacts after a complete analysis exists."""
    directory.mkdir(parents=True, exist_ok=True)
    camera_dir = directory / "cmc"
    if prepared.report is not None:
        camera_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cmc-publish-", dir=directory) as temporary:
        staging = Path(temporary)
        for name, contents in (("analysis.json", prepared.report), ("warps.npz", prepared.warps)):
            if contents is None:
                (camera_dir / name).unlink(missing_ok=True)
                continue
            source = staging / name
            source.write_bytes(contents)
            source.replace(camera_dir / name)
        policy = staging / "cmc-policy.yaml"
        policy.write_text(yaml.safe_dump(prepared.policy.to_dict(), sort_keys=False), encoding="utf-8")
        policy.replace(directory / policy.name)


def _analyze(
    args: Any,
    spec: TrackerSpec,
    directory: Path,
    *,
    progress: Callable[[str], None] | None,
) -> PreparedCMCSetup:
    """Complete one image analysis before publishing or dispatching its result."""
    if progress is not None:
        progress("CMC setup: analyzing current sequences…")
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cmc-analysis-", dir=directory) as temporary:
        result = prepare_cmc_candidates(args, spec.option_dict, Path(temporary), progress=progress)
        return PreparedCMCSetup(
            _analysis_key(args, spec, "auto"),
            "auto",
            result.policy,
            result.report_path.read_bytes(),
            result.warps_path.read_bytes(),
        )


def _load_analysis(args: Any, spec: TrackerSpec, directory: Path) -> PreparedCMCSetup:
    """Validate saved camera evidence against the current build and estimator."""
    report_bytes = (directory / "analysis.json").read_bytes()
    warps = (directory / "warps.npz").read_bytes()
    try:
        report = json.loads(report_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Saved CMC setup analysis is invalid.") from exc
    inputs = _analysis_inputs(args, spec)
    expected = {
        "schema": "boxmot.cmc-analysis/v1",
        "status": "complete",
        "build_path": inputs["build"],
        "manifest_sha256": inputs["manifest_sha256"],
        "split": inputs["split"],
        "tracker": inputs["tracker"],
        "geometry": inputs["geometry"],
        "rule": inputs["rule"],
        "method": inputs["method"],
        "opencv_version": cv2.__version__,
        "settings": inputs["settings"],
        "mask_min_conf": inputs["mask_min_conf"],
        "class_ids": list(spec.class_ids),
        "total_frames": sum(count for _name, count in inputs["sequences"]),
        "warps_sha256": hashlib.sha256(warps).hexdigest(),
    }
    if (
        not isinstance(report, Mapping)
        or report.get("ground_truth_used") is not False
        or any(report.get(name) != value for name, value in expected.items())
    ):
        raise ValueError("Saved CMC setup analysis does not match the current inputs or warp artifact.")
    sequences = report.get("sequences")
    if not isinstance(sequences, Mapping) or set(sequences) != {name for name, _count in inputs["sequences"]}:
        raise ValueError("Saved CMC setup analysis does not match the selected sequences.")
    decisions = []
    for name, count in inputs["sequences"]:
        stats = sequences[name]
        if not isinstance(stats, Mapping) or stats.get("frames") != count or type(stats.get("enabled")) is not bool:
            raise ValueError("Saved CMC setup analysis has invalid sequence decisions or frame counts.")
        decisions.append((name, stats["enabled"]))
    policy = SequenceCMCPolicy(sequences=tuple(decisions))
    if report.get("policy") != policy.to_dict():
        raise ValueError("Saved CMC setup policy disagrees with its sequence analysis.")
    return PreparedCMCSetup(_analysis_key(args, spec, "auto"), "auto", policy, report_bytes, warps)


def _reuse_profile_analysis(
    args: Any,
    spec: TrackerSpec,
    *,
    progress: Callable[[str], None] | None,
) -> PreparedCMCSetup | None:
    """Find optional evidence associated with a supplied profile, never images.

    Source paths are hints, so a portable profile remains usable after moving
    or deleting its original run. Reuse assumes the materialized build and its
    referenced images remain immutable; input pixels are not decoded again.
    """
    directories: list[Path] = []
    profile = getattr(args, "tracker_config", None)
    if profile is not None:
        parent = Path(profile).resolve().parent
        directories.extend((parent / "cmc", parent.parent / "cmc"))
    source = spec.option_dict.get("calibration.source")
    if isinstance(source, str) and source:
        directories.append(Path(source).resolve().parent.parent / "cmc")
    for directory in dict.fromkeys(directories):
        if not (directory / "analysis.json").is_file():
            continue
        try:
            prepared = _load_analysis(args, spec, directory)
        except (OSError, ValueError) as exc:
            if progress is not None:
                progress(f"CMC setup: cannot reuse {directory}: {exc}")
            continue
        if progress is not None:
            progress(f"CMC setup: reusing saved analysis from {directory}…")
        return prepared
    return None


def prepare_tuning_cmc(
    args: Any,
    spec: TrackerSpec,
    output_dir: str | Path,
    *,
    analysis_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PreparedCMCSetup | None:
    """Resolve one driver-owned policy before Ray starts any tuning trials.

    ``analysis_dir`` may reference only a calibration bank whose artifact
    checksums the caller has already validated. Its report is additionally
    checked against this tuning run's camera inputs before it is reused. Without
    that strict handoff, a matching analysis beside the supplied tracker profile
    or in its source calibration bank avoids repeating camera estimation.
    """
    requested = resolve_cmc_policy(args, spec)
    if requested is None:
        return None
    directory = Path(output_dir)
    strategy = _strategy(requested)
    if requested == "auto":
        prepared = (
            _load_analysis(args, spec, Path(analysis_dir))
            if analysis_dir is not None
            else _reuse_profile_analysis(args, spec, progress=progress)
        )
        if prepared is None:
            prepared = _analyze(args, spec, directory, progress=progress)
    else:
        prepared = PreparedCMCSetup(_analysis_key(args, spec, strategy), strategy, requested)
    prepared.policy.validate(spec, args.seq_info)
    _publish_setup(directory, prepared)
    if progress is not None:
        decisions = dict(prepared.policy.sequences)
        enabled = sum(
            decisions.get(name, prepared.policy.default)
            for name in (getattr(args, "sequence_names", None) or args.seq_info)
        )
        progress(f"CMC setup: prepared {strategy} policy for all tuning trials ({enabled} sequences enabled).")
    return prepared


def prepare_eval_cmc(
    args: Any,
    spec: TrackerSpec,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    replay_session: ReplaySession | None = None,
) -> SequenceCMCPolicy | None:
    """Resolve sequence decisions before tracking, using no evaluation labels.

    Automatic analysis reads the current build's images and detections. Saved
    profiles contribute the rule and masking confidence, never sequence IDs.
    Tuning trials consume the driver's prepared decisions and never analyze
    images. Independent evaluation runs analyze their current inputs afresh.
    """
    policy = resolve_cmc_policy(args, spec)
    handoff = getattr(args, "_cmc_setup", None)
    if handoff is not None:
        if (
            not isinstance(handoff, PreparedCMCSetup)
            or policy is None
            or handoff.strategy != _strategy(policy)
            or handoff.input_key != _analysis_key(args, spec, handoff.strategy)
        ):
            raise ValueError("Trial CMC inputs differ from tuning setup; start a new tuning run.")
        handoff.policy.validate(spec, args.seq_info)
        _publish_setup(Path(output_dir), handoff)
        return handoff.policy
    if policy is None:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if policy == "auto":
        key = _analysis_key(args, spec, "auto")
        cache = None if replay_session is None else getattr(replay_session, "_cmc_analysis_cache", None)
        if replay_session is not None and cache is None:
            cache = replay_session._cmc_analysis_cache = {}
        prepared = None if cache is None else cache.get(key)
        if prepared is None:
            prepared = _analyze(args, spec, directory, progress=progress)
            if cache is not None:
                # A retained replay session needs only its latest input selection;
                # do not accumulate warp archives across arbitrary mask changes.
                cache.clear()
                cache[key] = prepared
        elif progress is not None:
            progress("CMC setup: reusing this session's analysis of the current inputs…")
        policy = prepared.policy
        policy.validate(spec, args.seq_info)
        if progress is not None:
            enabled = sum(value for _, value in policy.sequences)
            progress(f"CMC setup: enabled on {enabled}/{len(policy.sequences)} evaluation sequences.")
    else:
        prepared = PreparedCMCSetup("", _strategy(policy), policy)
    _publish_setup(directory, prepared)
    return policy
