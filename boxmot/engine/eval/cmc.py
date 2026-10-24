"""Decide camera compensation once during evaluation or tuning setup."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import yaml

from boxmot.engine.calibration.cmc import (
    cmc_analysis_inputs,
    prepare_cmc_candidates,
)
from boxmot.engine.calibration.cmc_validation import validate_cmc_artifacts
from boxmot.engine.config.cmc import SequenceCMCPolicy, resolve_cmc_policy
from boxmot.engine.eval.cmc_replay import CMCReplaySequence, cmc_tracker_inputs
from boxmot.trackers import TrackerSpec

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
    replay_sequences: tuple[CMCReplaySequence, ...] = ()


def _analysis_args(args: Any, spec: TrackerSpec) -> Any:
    """Give shared camera preparation the exact resolved tracker selection."""
    analysis_args = copy(args)
    for name, value in (
        ("tracker", spec.name),
        ("tracker_backend", spec.backend),
        ("geometry", spec.geometry),
        ("per_class", spec.per_class),
        ("tracker_class_ids", spec.class_ids),
        ("tracker_class_names", spec.class_names),
    ):
        setattr(analysis_args, name, value)
    return analysis_args


def _analysis_inputs(args: Any, spec: TrackerSpec) -> dict[str, Any]:
    """Identify the selected build and settings that affect camera analysis."""
    return cmc_analysis_inputs(_analysis_args(args, spec), spec.option_dict)


def _analysis_key(args: Any, spec: TrackerSpec, strategy: str) -> str:
    """Bind setup decisions to camera inputs and the requested strategy."""
    selection = {**_analysis_inputs(args, spec), "strategy": strategy}
    if strategy != "auto":
        # Explicit decisions do not depend on the image analyzer. Tuning its
        # settings must not invalidate an unbound disabled/manual policy.
        selection.pop("mask_min_conf")
        selection.pop("settings")
        selection.pop("runtime_mask")
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
    """Load or materialize camera analysis before dispatching its result."""
    if progress is not None:
        progress("CMC setup: checking materialized camera analysis…")
    directory.mkdir(parents=True, exist_ok=True)
    analysis_args = _analysis_args(args, spec)
    with TemporaryDirectory(prefix=".cmc-analysis-", dir=directory) as temporary:
        result = prepare_cmc_candidates(analysis_args, spec.option_dict, Path(temporary), progress=progress)
        return _validated_analysis(
            args,
            spec,
            result.report_path.read_bytes(),
            result.warps_path.read_bytes(),
        )


def _load_analysis(args: Any, spec: TrackerSpec, directory: Path) -> PreparedCMCSetup:
    """Validate saved camera evidence against the current build and estimator."""
    return _validated_analysis(
        args, spec, (directory / "analysis.json").read_bytes(), (directory / "warps.npz").read_bytes()
    )


def _validated_analysis(args: Any, spec: TrackerSpec, report_bytes: bytes, warps: bytes) -> PreparedCMCSetup:
    """Validate provenance and publish only immutable, frame-aligned transforms."""
    inputs = _analysis_inputs(args, spec)
    report, arrays = validate_cmc_artifacts(report_bytes, warps, inputs)
    decisions = tuple((name, report["sequences"][name]["enabled"]) for name, _count in inputs["sequences"])
    policy = SequenceCMCPolicy(sequences=decisions)
    replay_sequences = []
    tracker_inputs = json.dumps(cmc_tracker_inputs(spec), sort_keys=True)
    for name, _count in inputs["sequences"]:
        stats = report["sequences"][name]
        if not stats["enabled"]:
            continue
        indices = arrays[stats["archive_keys"]["frame_indices"]]
        transforms = arrays[stats["archive_keys"]["runtime_warps"]]
        replay_sequences.append(CMCReplaySequence(
            sequence_id=name,
            sample_ids=tuple(stats["sample_ids"]),
            frame_indices=tuple(int(index) for index in indices),
            warp_bytes=transforms.tobytes(order="C"),
            build=inputs["build"],
            split=args.split,
            source_signature=tuple((path, tuple(value)) for path, value in sorted(report["source_signature"].items())),
            image_root=stats["source_image_root"],
            image_refs=tuple(stats["image_refs"]),
            image_signature=tuple((path, tuple(value)) for path, value in sorted(stats["image_signature"].items())),
            tracker_inputs=tracker_inputs,
        ))
    return PreparedCMCSetup(
        _analysis_key(args, spec, "auto"), "auto", policy, report_bytes, warps, tuple(replay_sequences)
    )


def _reuse_profile_analysis(
    args: Any,
    spec: TrackerSpec,
    *,
    progress: Callable[[str], None] | None,
) -> PreparedCMCSetup | None:
    """Find optional evidence associated with a supplied profile, never images.

    Source paths are hints, so a portable profile remains usable after moving
    or deleting its original run. File mutation epochs validate the materialized
    build and referenced images without decoding their pixels again.
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
) -> PreparedCMCSetup | None:
    """Resolve sequence decisions before tracking, using no evaluation labels.

    Automatic analysis reads the current build's images and detections. Saved
    profiles contribute the rule and masking confidence, never sequence IDs.
    Tuning trials consume the driver's prepared decisions and transforms.
    Independent runs reuse matching camera artifacts stored with the materialized build.
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
        if handoff.strategy == "auto":
            if handoff.report is None or handoff.warps is None:
                raise ValueError("Trial CMC handoff is missing camera evidence; start a new tuning run.")
            handoff = _validated_analysis(args, spec, handoff.report, handoff.warps)
        handoff.policy.validate(spec, args.seq_info)
        _publish_setup(Path(output_dir), handoff)
        return handoff
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
        if prepared is not None:
            try:
                prepared = _validated_analysis(args, spec, prepared.report, prepared.warps)
            except (OSError, ValueError):
                prepared = None
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
    return prepared
