"""Persist independently fitted Kalman priors for categorical filter search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from boxmot.engine.calibration.kalman_artifacts import write_json, write_yaml
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.calibration_profile import (
    calibration_metadata,
    fixed_calibration_options,
    record_tuning_calibration,
    validate_calibration_metadata,
)
from boxmot.engine.tuning.kalman_refinement import is_kalman_option
from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_config, nest_tracker_options
from boxmot.trackers.common.motion.kalman_filters.profile import (
    KALMAN_CALIBRATION_PROFILES,
    apply_calibration_profile,
    calibration_cmc_signature,
    calibration_profile_signature,
)
from boxmot.trackers.common.motion.models import MotionModelKind, resolve_motion_model_kind

if TYPE_CHECKING:
    from boxmot.engine.calibration.kalman_data import CalibrationData


@dataclass(frozen=True)
class CalibrationBank:
    """One complete runtime configuration per filter and its original request."""

    baseline_type: str
    profiles: dict[str, dict[str, Any]]
    baseline_config: dict[str, Any]
    index_path: Path
    selected_type: str | None = None
    selection_path: Path | None = None
    selected_profile: str | None = None
    selected_cmc: str | None = None


def _same(left: Any, right: Any) -> bool:
    """Compare serialized options while keeping booleans distinct from numbers."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    """Read complete structured metadata with an actionable corruption error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Saved KF calibration bank metadata is missing or malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Saved KF calibration bank metadata must be a mapping: {path}")
    return payload


def _digest(path: Path) -> str:
    """Bind the index to the exact completed calibration artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_request(tracker_name: str, kind: str, baseline: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    """Use pristine noise for guarded comparisons, retaining authored policies."""
    if selected:
        from boxmot.engine.calibration.kalman_selection import selection_baseline_config

        return selection_baseline_config(tracker_name, kind, baseline)
    return {**baseline, "kalman.type": kind}


def _fit_profile(
    args: Any,
    tune_dir: Path,
    *,
    kind: str,
    request: Mapping[str, Any],
    profile: str,
    data: CalibrationData,
    camera_warps: Mapping[str, np.ndarray] | None,
    cmc_metadata: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and publish one recipe, retaining a validated default when fitting fails."""
    from boxmot.engine.calibration.kalman import calibrate_kalman

    directory = Path(tune_dir) / "kf-tuning"
    result = None
    try:
        result = calibrate_kalman(
            args,
            output_dir=tune_dir,
            progress=progress,
            tracker_options=request,
            calibration_data=data,
            profile_name=kind,
            **({"camera_warps": camera_warps} if cmc_metadata else {}),
        )
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        if profile != "default":
            raise
        # A rejected fit must not prevent use of a validated original prior.
        report_path = directory / kind / "calibration.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        signature = calibration_profile_signature(args.tracker, args.geometry, request)
        report = {
            "version": 3,
            "status": "complete",
            "method": "default_covariance_prior",
            **calibration_metadata(args),
            "baseline_config": request,
            "timing": {
                name: request[name]
                for name in ("kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s")
            },
            "filters": {
                str(track.class_id): {
                    "kind": kind,
                    "state_dimensions": signature["state_dimensions"],
                    "measurement_dimensions": signature["measurement_dimensions"],
                }
                for track in data.tracks
            },
            "calibration_error": str(exc),
            "ground_truth_sources": list(data.ground_truth_sources),
            "input_sources": list(data.input_sources),
        }
        write_json(report_path, report)
        if progress:
            progress(f"KF calibration: {kind} fit failed; retaining its validated default noise ({exc}).")
    if result is not None:
        report_path = result.report_path
        profile_args = copy(args)
        profile_args.tracker_config = result.config_path
        config = resolve_tracker_options(profile_args, include_defaults=True, stamp_timing=True)
        active_path = result.config_path
    if profile != "calibrated":
        selected_config = apply_calibration_profile(request, config if result is not None else request, profile)
        config = {
            **selected_config,
            **flatten_tracker_options(
                {
                    "calibration": {
                        **calibration_profile_signature(args.tracker, args.geometry, selected_config),
                        "dataset": args.dataset_id,
                        "split": args.split,
                        "source": str(report_path.resolve()),
                        "profile": profile,
                    }
                }
            ),
        }
        active_path = report_path.with_name(f"{profile}.yaml")
        write_yaml(active_path, {"tracker": args.tracker, **nest_tracker_options(config)})
    if cmc_metadata:
        config.update(cmc_metadata)
        # Every exported recipe, including the full fit behind a partial
        # winner, must retain its camera policy when evaluated separately.
        paths = {active_path}
        if result is not None:
            paths.add(result.config_path)
        for path in paths:
            authored = flatten_tracker_options(yaml.safe_load(path.read_text(encoding="utf-8")))
            authored.update(cmc_metadata)
            write_yaml(path, nest_tracker_options(authored))
    record_tuning_calibration(report_path, fixed_calibration_options(config))
    entry = {
        "profile": profile,
        "config": str(active_path.relative_to(directory)),
        "report": str(report_path.relative_to(directory)),
        "config_sha256": _digest(active_path),
        "report_sha256": _digest(report_path),
        "signature": calibration_profile_signature(args.tracker, args.geometry, config),
    }
    if result is not None:
        entry.update(
            calibrated_config=str(result.config_path.relative_to(directory)),
            calibrated_config_sha256=_digest(result.config_path),
        )
    return config, entry


def create_calibration_bank(
    args: Any,
    tune_dir: Path,
    *,
    runtime_config: Mapping[str, Any],
    kinds: Sequence[str],
    progress: Callable[[str], None] | None = None,
    select_best: bool = False,
    select_cmc: bool = False,
) -> CalibrationBank:
    """Match detections once, fit every requested filter, and publish a full bank.

    The completion index is written only after every independent filter profile
    is saved. Partial banks therefore cannot be mistaken for reusable priors.
    """
    from boxmot.engine.calibration.kalman import validate_kf_calibration
    from boxmot.engine.calibration.kalman_data import load_calibration_data
    from boxmot.engine.calibration.kalman_model import validate_calibration_options
    from boxmot.engine.eval.evaluator import _ensure_setup

    validate_kf_calibration(args.tracker, getattr(args, "tracker_backend", "python"))
    if select_cmc:
        from boxmot.engine.config.cmc import validate_joint_cmc_selection

        validate_joint_cmc_selection(args.tracker, getattr(args, "tracker_backend", "python"), select_kf=select_best)
    if select_best:
        from boxmot.engine.calibration.kalman_selection import validate_kf_selection

        validate_kf_selection(args.tracker, getattr(args, "tracker_backend", "python"))
    _ensure_setup(args)
    if isinstance(kinds, str) or not kinds:
        raise ValueError("A KF calibration bank requires a nonempty sequence of concrete filter types.")
    selected = tuple(MotionModelKind(kind).value for kind in kinds)
    if len(set(selected)) != len(selected):
        raise ValueError("KF calibration bank filter types must be unique.")
    resolved_args = copy(args)
    resolved_args.tracker_config = None
    baseline = resolve_tracker_options(
        resolved_args,
        {
            name: value
            for name, value in flatten_tracker_options(runtime_config).items()
            if not name.startswith("calibration.")
        },
        include_defaults=True,
        stamp_timing=True,
    )
    baseline_type = resolve_motion_model_kind(
        args.tracker, baseline.get("kalman.type"), is_obb=args.geometry == "obb"
    ).value
    baseline["kalman.type"] = baseline_type
    if baseline_type not in selected:
        raise ValueError("The original tuning baseline filter must be included in its calibration bank.")
    if getattr(args, "per_class", False):
        baseline["per_class"] = True
    if select_best:
        from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf

        baseline.update(
            use_cmc=False,
            cmc_min_conf=resolve_cmc_min_conf(baseline),
        )
        if select_cmc:
            baseline["cmc_method"] = "sof"
        elif progress is not None:
            progress(
                "KF selection: camera compensation is off and stays fixed; add --select-cmc to compare camera policies."
            )
    requests = {kind: _profile_request(args.tracker, kind, baseline, selected=select_best) for kind in selected}
    for request in requests.values():
        validate_calibration_options(args.tracker, request, geometry=args.geometry)
    directory = Path(tune_dir) / "kf-tuning"
    index_path = directory / "filters.json"
    if (
        index_path.exists()
        or (directory / "selection.json").exists()
        or any((directory / kind.value).exists() for kind in MotionModelKind)
    ):
        raise ValueError("A KF calibration bank already exists in this run; resume it or start a new tuning run.")
    resolved_args.variable_dt = baseline["kalman.variable_dt"]
    camera = None
    if select_cmc:
        from boxmot.engine.calibration.cmc import prepare_cmc_candidates

        camera = prepare_cmc_candidates(resolved_args, baseline, output_dir=directory / "cmc", progress=progress)
    if progress is not None:
        progress("KF calibration: matching cached detections to ground truth once for all filter types…")
    data = load_calibration_data(resolved_args, progress=progress)
    selection = None
    if select_best:
        from boxmot.engine.calibration.kalman_selection import select_kalman_model

        selection = select_kalman_model(
            data,
            tracker_name=args.tracker,
            geometry=args.geometry,
            configs=requests,
            baseline_type=baseline_type,
            progress=progress,
            **({"camera_candidates": {"off": None, "auto": camera.warps}} if camera is not None else {}),
        )
        if (
            selection.get("selected_type") not in requests
            or selection.get("selected_profile") not in KALMAN_CALIBRATION_PROFILES
        ):
            raise ValueError("KF selection did not identify a valid filter/profile pair.")
        if progress is not None:
            progress(
                f"KF selection: selected {selection['selected_type']} with {selection['selected_profile']} noise; "
                "saving reusable profiles…"
            )
    cmc_metadata, cmc_binding = {}, None
    selected_cmc = None if camera is None else selection.get("selected_cmc")
    selected_warps = None
    if camera is not None:
        from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy

        if selected_cmc not in {"off", "auto"}:
            raise ValueError("Joint selection did not identify a valid CMC policy.")
        selected_warps = camera.warps if selected_cmc == "auto" else None
        policy = camera.policy if selected_cmc == "auto" else SequenceCMCPolicy()
        policy_path = directory / "cmc-policy.yaml"
        write_yaml(policy_path, policy.to_dict())
        cmc_metadata = {
            "calibration.cmc_policy": selected_cmc,
            "calibration.cmc_rule": CMC_ANALYSIS_RULE,
            "calibration.cmc_min_conf": baseline["cmc_min_conf"],
            **calibration_cmc_signature(baseline),
        }
        cmc_binding = {
            "selected_policy": selected_cmc,
            "rule": CMC_ANALYSIS_RULE,
            "policy": "cmc-policy.yaml",
            "policy_sha256": _digest(policy_path),
            "analysis": str(camera.report_path.relative_to(directory)),
            "analysis_sha256": _digest(camera.report_path),
            "warps": str(camera.warps_path.relative_to(directory)),
            "warps_sha256": _digest(camera.warps_path),
        }
    profiles, entries = {}, {}
    for kind, request in requests.items():
        profile = selection["candidates"][kind]["selected_profile"] if selection is not None else "calibrated"
        if selection is not None and selection["candidates"][kind].get("status") == "invalid":
            continue
        if profile not in KALMAN_CALIBRATION_PROFILES:
            raise ValueError(f"KF selection did not produce a valid reusable profile for {kind}.")
        if progress is not None:
            progress(f"KF calibration: fitting {kind} ({len(profiles) + 1}/{len(selected)})…")
        profiles[kind], entries[kind] = _fit_profile(
            resolved_args,
            tune_dir,
            kind=kind,
            request=request,
            profile=profile,
            data=data,
            camera_warps=selected_warps,
            cmc_metadata=cmc_metadata,
            progress=progress,
        )
    index = {
        "version": 3,
        "status": "complete",
        **calibration_metadata(args),
        "baseline_type": baseline_type,
        "baseline_config": baseline,
        "filter_types": list(selected),
        "profiles": entries,
    }
    if cmc_binding is not None:
        index["cmc_selection"] = cmc_binding
    selected_type, selection_path, selected_profile = None, None, None
    if selection is not None:
        selected_type = selection["selected_type"]
        selected_profile = selection["selected_profile"]
        selection_path = directory / "selection.json"
        report = {
            **selection,
            "version": 4 if camera is not None else 3,
            "status": "complete",
            **calibration_metadata(args),
            "filter_types": list(selected),
            "baseline_type": baseline_type,
            "selected_config": entries[selected_type]["config"],
            "ground_truth_sources": list(data.ground_truth_sources),
            "input_sources": list(data.input_sources),
        }
        if cmc_binding is not None:
            report["cmc_selection"] = cmc_binding
        write_json(selection_path, report)
        index["selection"] = {
            "selected_type": selected_type,
            "selected_profile": selected_profile,
            "report": "selection.json",
            "report_sha256": _digest(selection_path),
        }
    write_json(index_path, index)
    if selection_path is not None:
        from boxmot.engine.tuning.analysis import generate_selection_analysis

        generate_selection_analysis(Path(tune_dir))
    return CalibrationBank(
        baseline_type, profiles, baseline, index_path, selected_type, selection_path, selected_profile, selected_cmc
    )


def _load_selection(args: Any, directory: Path, index: Mapping[str, Any]) -> tuple[str | None, Path | None, str | None]:
    """Validate and restore the chosen model without rerunning validation folds."""
    entry = index.get("selection")
    path = directory / "selection.json"
    if entry is None:
        if path.exists() or path.with_suffix(".tmp").exists():
            raise ValueError("Saved KF selection is incomplete: filters.json must bind selection.json.")
        return None, None, None
    if (
        not isinstance(entry, Mapping)
        or entry.get("report") != "selection.json"
        or not isinstance(entry.get("selected_type"), str)
        or entry.get("selected_type") not in index["profiles"]
        or entry.get("selected_profile") not in KALMAN_CALIBRATION_PROFILES
        or not path.is_file()
    ):
        raise ValueError("Saved KF selection is incomplete or has an invalid selected filter.")
    if entry.get("report_sha256") != _digest(path):
        raise ValueError("Saved KF selection report has changed since filter selection.")
    report = _read_json(path)
    selected_type = entry["selected_type"]
    joint = "cmc_selection" in index
    if (
        report.get("version") != (4 if joint else 3)
        or report.get("status") != "complete"
        or report.get("method")
        != ("grouped_cross_validated_cmc_box_profiles" if joint else "grouped_cross_validated_box_profiles")
        or report.get("selected_type") != selected_type
        or report.get("selected_profile") != entry["selected_profile"]
        or index["profiles"][selected_type].get("profile") != entry["selected_profile"]
        or report.get("selected_config") != index["profiles"][selected_type]["config"]
        or report.get("filter_types") != index["filter_types"]
        or report.get("baseline_type") != index["baseline_type"]
    ):
        raise ValueError("Saved KF selection report disagrees with its calibrated filter bank.")
    validate_calibration_metadata(args, report)
    candidates = report.get("candidates")
    if (
        not isinstance(candidates, Mapping)
        or set(candidates) != set(index["filter_types"])
        or any(
            not isinstance(candidates[kind], Mapping)
            or candidates[kind].get("status") != "valid"
            or candidates[kind].get("selected_profile") != profile.get("profile")
            for kind, profile in index["profiles"].items()
        )
        or any(
            not isinstance(candidate, Mapping) or candidate.get("status") != "invalid"
            for kind, candidate in candidates.items()
            if kind not in index["profiles"]
        )
    ):
        raise ValueError("Saved KF selection candidate profiles disagree with the saved filter bank.")
    if joint:
        binding = index["cmc_selection"]
        cameras = report.get("cmc_candidates", {})
        selected_cmc = binding["selected_policy"]
        if (
            not _same(report.get("cmc_selection"), binding)
            or report.get("selected_cmc") != selected_cmc
            or not isinstance(cameras, dict)
            or selected_cmc not in cameras
            or not _same(cameras[selected_cmc].get("candidates"), candidates)
        ):
            raise ValueError("Saved joint CMC/KF selection disagrees with its camera policy.")
    return selected_type, path, entry["selected_profile"]


def _load_cmc_selection(directory: Path, index: Mapping[str, Any]) -> str | None:
    """Bind a resumed fit to the exact camera policy and analyzed transitions."""
    from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE

    binding = index.get("cmc_selection")
    if binding is None:
        return None
    if (
        not isinstance(binding, Mapping)
        or binding.get("selected_policy") not in {"off", "auto"}
        or binding.get("rule") != CMC_ANALYSIS_RULE
    ):
        raise ValueError("Saved joint selection has an invalid CMC policy binding.")
    for field, relative in (
        ("policy", "cmc-policy.yaml"),
        ("analysis", "cmc/analysis.json"),
        ("warps", "cmc/warps.npz"),
    ):
        path = directory / relative
        if binding.get(field) != relative or not path.is_file() or binding.get(f"{field}_sha256") != _digest(path):
            raise ValueError(f"Saved joint selection CMC {field} artifact is missing or has changed.")
    return binding["selected_policy"]


def _validate_profile_artifacts(directory: Path, kind: str, entry: Mapping[str, Any], *, selected: bool) -> None:
    """Check saved profiles even when an inactive filter no longer runs here."""
    profile = entry.get("profile")
    if profile not in KALMAN_CALIBRATION_PROFILES or (not selected and profile != "calibrated"):
        raise ValueError(f"Saved KF calibration bank has an invalid profile choice for {kind}.")
    artifacts = [("config", f"{profile}.yaml"), ("report", "calibration.json")]
    if profile != "default" or "calibrated_config" in entry:
        artifacts.append(("calibrated_config", "calibrated.yaml"))
    for field, filename in artifacts:
        relative = f"{kind}/{filename}"
        if entry.get(field) != relative:
            raise ValueError(f"Saved KF calibration bank has an invalid {field} path for {kind}.")
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"Saved KF calibration bank is incomplete: missing {kind}/{filename}.")
        if entry.get(f"{field}_sha256") != _digest(path):
            raise ValueError(f"Saved KF calibration bank {kind}/{filename} has changed since calibration.")


def _load_profile(
    args: Any,
    directory: Path,
    kind: str,
    entry: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    selected: bool,
) -> dict[str, Any]:
    """Validate one independent profile before exposing any bank to trials."""
    profile = entry.get("profile")
    config_path, report_path = directory / entry["config"], directory / entry["report"]
    report = _read_json(report_path)
    expected_method = "supervised_covariance_moments" if "calibrated_config" in entry else "default_covariance_prior"
    if report.get("status") != "complete" or report.get("method") != expected_method:
        raise ValueError(f"Saved KF calibration bank profile {kind} is not a completed direct calibration.")
    validate_calibration_metadata(args, report)
    request = _profile_request(args.tracker, kind, baseline, selected=selected)
    if not _same(report.get("baseline_config"), request):
        raise ValueError(f"Saved KF calibration bank baseline options disagree for {kind}.")
    try:
        authored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ValueError(f"Saved KF calibration bank profile is malformed: {config_path}") from exc
    if not isinstance(authored, Mapping):
        raise ValueError(f"Saved KF calibration bank profile must be a mapping: {config_path}")
    authored = flatten_tracker_options(authored)
    profile_args = copy(args)
    profile_args.tracker_config, profile_args.variable_dt = config_path, None
    config = resolve_tracker_options(
        profile_args, include_defaults=True, stamp_timing=True
    )
    signature = calibration_profile_signature(args.tracker, args.geometry, config)
    if config.get("kalman.type") != kind or not _same(entry.get("signature"), signature):
        raise ValueError(f"Saved KF calibration bank signature disagrees with the {kind} profile.")
    if profile == "default" and (
        config.get("calibration.profile") != "default"
        or not _same(fixed_calibration_options(config), fixed_calibration_options(request))
    ):
        raise ValueError(f"Saved KF calibration bank default noise disagrees with the {kind} prior.")
    if profile in {"motion", "measurement"}:
        fitted_args = copy(profile_args)
        fitted_args.tracker_config = directory / entry["calibrated_config"]
        fitted = resolve_tracker_options(
            fitted_args,
            include_defaults=True,
            stamp_timing=True,
        )
        expected = apply_calibration_profile(request, fitted, profile)
        if (
            config.get("calibration.profile") != profile
            or fitted.get("kalman.type") != kind
            or not _same(fixed_calibration_options(config), fixed_calibration_options(expected))
        ):
            raise ValueError(f"Saved KF calibration bank {profile} noise disagrees with the {kind} recipe.")
    fixed = fixed_calibration_options(config)
    tuning = report.get("tuning")
    if (
        not isinstance(tuning, Mapping)
        or not _same(tuning.get("fixed_options"), fixed)
        or any(name not in authored for name in fixed)
    ):
        raise ValueError(f"Saved KF calibration bank fixed options disagree with the {kind} profile.")
    timing = {
        name: config[name] for name in ("kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s")
    }
    if not _same(report.get("timing"), timing):
        raise ValueError(f"Saved KF calibration bank timing disagrees with the {kind} profile.")
    if config.get("calibration.source") != str(report_path.resolve()):
        raise ValueError(f"Saved KF calibration bank provenance disagrees with the {kind} profile.")
    filters = report.get("filters")
    expected_model = {
        "kind": kind,
        "state_dimensions": signature["state_dimensions"],
        "measurement_dimensions": signature["measurement_dimensions"],
    }
    if (
        not isinstance(filters, Mapping)
        or not filters
        or any(not _same(model, expected_model) for model in filters.values())
    ):
        raise ValueError(f"Saved KF calibration bank filter dimensions disagree with the {kind} profile.")
    return config


def load_calibration_bank(
    args: Any,
    tune_dir: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CalibrationBank | None:
    """Restore compatible profiles without rematching detections or fitting.

    The original uncalibrated request provides resume defaults. Explicit changes
    to calibrated inputs are rejected; other tracker settings remain editable.
    Fixed selections require the winning filter to remain compatible. Inactive
    candidates still require intact artifacts but do not impose their revisions
    on the selected runtime.
    """
    directory = Path(tune_dir) / "kf-tuning"
    index_path = directory / "filters.json"
    if not index_path.exists():
        if (
            any((directory / kind.value).exists() for kind in MotionModelKind)
            or index_path.with_suffix(".tmp").exists()
            or (directory / "selection.json").exists()
        ):
            raise ValueError("Saved KF calibration bank is incomplete: filters.json is required.")
        return None
    if getattr(args, "tracker_backend", "python") != "python":
        raise ValueError("Resuming a calibrated tuning run requires the Python tracker backend.")
    index = _read_json(index_path)
    if index.get("version") != 3 or index.get("status") != "complete":
        raise ValueError(
            "Saved KF calibration bank is incomplete or has an unsupported version; start a new tuning run."
        )
    validate_calibration_metadata(args, index)
    baseline, entries = index.get("baseline_config"), index.get("profiles")
    baseline_type = index.get("baseline_type")
    selected = index.get("filter_types")
    allowed = {kind.value for kind in MotionModelKind}
    if (
        not isinstance(baseline, dict)
        or not isinstance(entries, dict)
        or not entries
        or any(kind not in allowed or not isinstance(entry, dict) for kind, entry in entries.items())
        or not isinstance(selected, list)
        or any(not isinstance(kind, str) for kind in selected)
        or len(selected) != len(set(selected))
        or not set(entries).issubset(selected)
        or ("selection" not in index and set(selected) != set(entries))
        or baseline_type not in selected
        or baseline.get("kalman.type") != baseline_type
    ):
        raise ValueError("Saved KF calibration bank is missing its original baseline or filter profiles.")
    selected_cmc = _load_cmc_selection(directory, index)
    selected_type, selection_path, selected_profile = _load_selection(args, directory, index)
    profiles = {}
    for kind, entry in entries.items():
        _validate_profile_artifacts(directory, kind, entry, selected=selected_type is not None)
        if selected_type is not None and kind != selected_type:
            signature = calibration_profile_signature(args.tracker, args.geometry, {**baseline, "kalman.type": kind})
            if not _same(entry.get("signature"), signature):
                # Selection is fixed on resume: inactive historical candidates
                # retain their artifacts, but cannot become runtime profiles.
                continue
        profiles[kind] = _load_profile(
            args, directory, kind, entry, baseline, selected=selected_type is not None
        )
    if selected_cmc is not None:
        from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE

        expected = {
            "calibration.cmc_policy": selected_cmc,
            "calibration.cmc_rule": CMC_ANALYSIS_RULE,
            "calibration.cmc_min_conf": baseline["cmc_min_conf"],
            **calibration_cmc_signature(baseline),
        }
        for profile in profiles.values():
            if any(not _same(profile.get(name), value) for name, value in expected.items()):
                raise ValueError("Saved KF profile is missing or disagrees with its selected CMC policy.")
    effective_args = copy(args)
    effective_overrides = flatten_tracker_options(overrides or {})
    if getattr(args, "tracker_config", None) is None:
        effective_overrides = {**baseline, **effective_overrides}
    effective = resolve_tracker_options(effective_args, effective_overrides, include_defaults=True, stamp_timing=True)
    effective["kalman.type"] = resolve_motion_model_kind(
        args.tracker, effective.get("kalman.type"), is_obb=args.geometry == "obb"
    ).value
    if selected_type is not None:
        # Selection fixes camera behavior independently of default-on trackers
        # and the original YAML's unspecified masking/method settings.
        authored = load_tracker_config(
            args.tracker, getattr(args, "tracker_config", None), overrides, include_defaults=False
        )
        for name in ("use_cmc", "cmc_method", "cmc_min_conf"):
            if name not in authored or (name == "cmc_min_conf" and authored[name] is None):
                effective[name] = baseline[name]
    if not _same(fixed_calibration_options(effective), fixed_calibration_options(baseline)):
        changed = sorted(
            name
            for name in set(fixed_calibration_options(effective)) | set(fixed_calibration_options(baseline))
            if not _same(effective.get(name), baseline.get(name))
        )
        raise ValueError(
            f"Cannot change KF calibration inputs when resuming: {', '.join(changed)}. Start a new tuning run."
        )
    runtime_options = {
        name: value
        for name, value in effective.items()
        if name != "kalman.type" and not is_kalman_option(name) and not name.startswith("calibration.")
    }
    profiles = {kind: {**profile, **runtime_options} for kind, profile in profiles.items()}
    return CalibrationBank(
        baseline_type,
        profiles,
        baseline,
        index_path,
        selected_type,
        selection_path,
        selected_profile,
        selected_cmc,
    )


__all__ = ("CalibrationBank", "create_calibration_bank", "load_calibration_bank")
