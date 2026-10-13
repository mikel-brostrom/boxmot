"""Select a box Kalman model using grouped, held-out prediction experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from boxmot.engine.calibration import kalman_folds, kalman_replay
from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack
from boxmot.engine.calibration.kalman_fit import apply_global_noise_fallback, fit_kalman_noise
from boxmot.engine.calibration.kalman_model import CalibrationModel
from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_NOISE_OPTIONS,
    KALMAN_TRACKER_NAMES,
    KALMAN_VECTOR_OPTIONS,
)
from boxmot.trackers.common.motion.kalman_filters.profile import (
    KALMAN_CALIBRATION_PROFILES,
    apply_calibration_profile,
)
from boxmot.trackers.common.motion.models import supported_motion_model_kinds

_TIE_TOLERANCE = 1e-9


def validate_kf_selection(tracker_name: str, backend: str = "python") -> None:
    """Reject unsupported selection before materialization or search startup."""
    if backend != "python" or tracker_name not in KALMAN_TRACKER_NAMES:
        raise ValueError("--select-kf requires a Python 2D tracker with a searchable Kalman filter.")


def selection_baseline_config(tracker_name: str, kind: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Prevent prior full-data calibration from leaking through fold fallbacks."""
    baseline = {
        name: value
        for name, value in flatten_tracker_options(options).items()
        if not name.startswith(("calibration.", "kalman.noise.by_class.")) and name not in KALMAN_VECTOR_OPTIONS
    }
    baseline.update(dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0))
    baseline["kalman.type"] = kind
    return load_tracker_config(tracker_name, None, baseline)


def _fit_fold(
    training: Sequence[CalibrationTrack],
    *,
    tracker_name: str,
    geometry: str,
    baseline: Mapping[str, Any],
    class_ids: Sequence[int],
    camera_warps: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit using training identities only, including pooled per-class fallbacks."""
    models = {class_id: CalibrationModel(tracker_name, geometry, baseline, cls_id=class_id) for class_id in class_ids}
    timed = baseline["kalman.variable_dt"]
    camera_options = {} if camera_warps is None else {"camera_warps": camera_warps}
    parameters, counts = fit_kalman_noise(training, models, baseline, variable_dt=timed, **camera_options)
    config = {**baseline, **{name: value["value"] for name, value in parameters.items()}}
    classes = {}
    if baseline.get("per_class", False):
        for class_id in class_ids:
            subset = tuple(track for track in training if track.class_id == class_id)
            estimates, statistics = fit_kalman_noise(subset, models, baseline, variable_dt=timed, **camera_options)
            estimates = apply_global_noise_fallback(estimates, parameters)
            for name, estimate in estimates.items():
                config[f"kalman.noise.by_class.{class_id}.{name.removeprefix('kalman.noise.')}"] = estimate["value"]
            classes[str(class_id)] = {"parameters": estimates, **statistics}
    return config, {"parameters": parameters, "classes": classes, **counts}


def _evaluate_profile(
    partitions: Sequence[tuple[Sequence[CalibrationTrack], Sequence[CalibrationTrack]]],
    *,
    profile: str,
    tracker_name: str,
    geometry: str,
    baseline: Mapping[str, Any],
    fitted_folds: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]] | str],
    progress: Callable[[str], None] | None,
    camera_warps: Mapping[str, np.ndarray] | None = None,
    primary_horizon: int | None = None,
    horizons: Sequence[int] | None = None,
    diagnostics_only: bool = False,
) -> dict[str, Any]:
    """Evaluate all held-out predictions, invalidating the profile on failure."""
    kind = baseline["kalman.type"]
    stage = "KF diagnostics" if diagnostics_only else "KF selection"
    records, fold_reports = [], []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index, (_, validation) in enumerate(partitions):
                label = f"{stage}: {kind}/{profile}, fold {index + 1}/{len(partitions)}"
                calibration_report = {}
                config = baseline
                if profile != "default":
                    fitted = fitted_folds[index]
                    if isinstance(fitted, str):
                        raise ValueError(fitted)
                    calibrated, calibration = fitted
                    config = apply_calibration_profile(baseline, calibrated, profile)
                    calibration_report["calibration"] = calibration
                if progress:
                    progress(f"{label} — evaluating covariance recipe…")
                evaluated = []
                replay_options = {} if horizons is None else {"horizons": horizons}
                for number, track in enumerate(validation):
                    if camera_warps is not None:
                        replay_options["camera_warps"] = camera_warps[track.sequence_id]
                    evaluated.append(
                        (
                            track,
                            kalman_replay.replay_track(
                                track,
                                tracker_name=tracker_name,
                                geometry=geometry,
                                config=config,
                                **replay_options,
                            ),
                        )
                    )
                    if progress and (number + 1) % 25 == 0:
                        progress(f"{label} — predicted {number + 1}/{len(validation)} trajectories…")
                records.extend(evaluated)
                fold_reports.append(
                    {
                        "index": index,
                        **calibration_report,
                        **kalman_replay.summarize_forecasts(evaluated, primary_horizon=primary_horizon),
                    }
                )
        return {
            "status": "valid",
            **kalman_replay.summarize_forecasts(records, primary_horizon=primary_horizon),
            "folds": fold_reports,
        }
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        if progress:
            outcome = "unavailable" if diagnostics_only else "ineligible"
            progress(f"{stage}: {kind}/{profile} is {outcome} ({exc}).")
        return {
            "status": "invalid",
            "mean_iou": None,
            "normalized_center_error": None,
            "error": str(exc),
            "folds": fold_reports,
        }


def _select_policy(
    data: CalibrationData,
    *,
    tracker_name: str,
    geometry: str,
    configs: Mapping[str, Mapping[str, Any]],
    baseline_type: str,
    progress: Callable[[str], None] | None = None,
    camera_warps: Mapping[str, np.ndarray] | None = None,
    primary_horizon: int | None = None,
) -> dict[str, Any]:
    """Compare original, partial, and complete calibration on held-out forecasts.

    At most three deterministic folds hold out entire sequences. With a single
    sequence, identities form groups instead; this weaker split shares camera
    conditions. Each KF/fold fits noise once, exclusively on training identities.
    Original, motion-only, measurement-only, and complete calibration reuse the
    same fit and held-out data. Each family retains its best recipe, preserving
    defaults then partial recipes on ties. The
    highest macro-averaged IoU wins; center error is diagnostic. Across families,
    ties within 1e-9 prefer the tracker baseline, then default profiles, then
    MotionModelKind order. The result is independent of tuning/search backends.
    """
    validate_kf_selection(tracker_name)
    if geometry not in {"aabb", "obb"}:
        raise ValueError("KF selection requires AABB or OBB geometry.")
    supported = supported_motion_model_kinds(is_obb=geometry == "obb")
    if not configs or set(configs) - {kind.value for kind in supported}:
        raise ValueError("KF selection requires supported box Kalman candidates.")
    tracks = kalman_folds.usable_tracks(data, geometry)
    grouping, partitions = kalman_folds.grouped_folds(tracks)
    baselines = {
        kind.value: selection_baseline_config(tracker_name, kind.value, configs[kind.value])
        for kind in supported
        if kind.value in configs
    }
    timing = {
        tuple(config[name] for name in ("kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s"))
        for config in baselines.values()
    }
    if len(timing) != 1:
        raise ValueError("KF selection candidates must share the same timing mode and reference interval.")
    if any(config["kalman.variable_dt"] for config in baselines.values()):
        for track in tracks:
            times = track.timestamps_s
            if (
                times is None
                or times.shape != track.frame_indices.shape
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError("Variable-time KF selection requires finite, strictly increasing capture timestamps.")
    folds = [
        {
            "index": index,
            "training_sequences": sorted({track.sequence_id for track in training}),
            "validation_sequences": sorted({track.sequence_id for track in validation}),
            "training_trajectories": [kalman_folds.track_reference(track) for track in training],
            "validation_trajectories": [kalman_folds.track_reference(track) for track in validation],
        }
        for index, (training, validation) in enumerate(partitions)
    ]
    classes = sorted({track.class_id for track in tracks})
    candidates = {}
    fits_by_kind = {}
    for kind, baseline in baselines.items():
        fitted_folds = []
        for index, (training, _) in enumerate(partitions):
            if progress:
                progress(f"KF selection: {kind}, fold {index + 1}/{len(partitions)} — fitting training noise once…")
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    fitted_folds.append(
                        _fit_fold(
                            training,
                            tracker_name=tracker_name,
                            geometry=geometry,
                            baseline=baseline,
                            class_ids=classes,
                            camera_warps=camera_warps,
                        )
                    )
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                fitted_folds.append(str(exc))
        fits_by_kind[kind] = fitted_folds
        profiles = {
            profile: _evaluate_profile(
                partitions,
                profile=profile,
                tracker_name=tracker_name,
                geometry=geometry,
                baseline=baseline,
                fitted_folds=fitted_folds,
                progress=progress,
                camera_warps=camera_warps,
                primary_horizon=primary_horizon,
                horizons=None if primary_horizon is None else (primary_horizon,),
            )
            for profile in KALMAN_CALIBRATION_PROFILES
        }
        valid_profiles = [name for name, result in profiles.items() if result["status"] == "valid"]
        if valid_profiles:
            best = max(profiles[name]["mean_iou"] for name in valid_profiles)
            selected_profile = next(
                name for name in valid_profiles if best - profiles[name]["mean_iou"] <= _TIE_TOLERANCE
            )
            candidates[kind] = {
                **profiles[selected_profile],
                "selected_profile": selected_profile,
                "profiles": profiles,
            }
        else:
            candidates[kind] = {
                "status": "invalid",
                "mean_iou": None,
                "normalized_center_error": None,
                "selected_profile": None,
                "error": "; ".join(f"{name}: {row['error']}" for name, row in profiles.items()),
                "folds": [],
                "profiles": profiles,
            }
    valid = [kind for kind in candidates if candidates[kind]["status"] == "valid"]
    if not valid:
        errors = "; ".join(f"{kind}: {row['error']}" for kind, row in candidates.items())
        raise ValueError(f"KF selection could not evaluate any candidate: {errors}")
    best_score = max(candidates[kind]["mean_iou"] for kind in valid)
    tied = [kind for kind in valid if best_score - candidates[kind]["mean_iou"] <= _TIE_TOLERANCE]
    selected = min(
        tied,
        key=lambda kind: (
            kind != baseline_type,
            candidates[kind]["selected_profile"] != "default",
            list(baselines).index(kind),
        ),
    )
    report = {
        "version": 3,
        "method": "grouped_cross_validated_box_profiles",
        "selected_type": selected,
        "selected_profile": candidates[selected]["selected_profile"],
        "baseline_type": baseline_type,
        "grouping": grouping,
        "folds": folds,
        "horizons": list(kalman_replay.DEFAULT_HORIZONS),
        "horizon_semantics": (
            "Delivered frames after a matched detection; intermediate observations withheld; "
            "stop at annotation discontinuities."
        ),
        "objective": "maximize_mean_iou",
        "aggregation": (
            "Forecasts within trajectory, trajectories within sequence, sequences within horizon, "
            "then represented horizons equally."
        ),
        "tie_break": "baseline_then_default_profile_then_candidate_order",
        "tie_tolerance": _TIE_TOLERANCE,
        "candidate_order": list(baselines),
        "profile_order": list(KALMAN_CALIBRATION_PROFILES),
        "excluded_trajectories": len(data.tracks) - len(tracks),
        "candidates": candidates,
        "limitations": [
            "Oracle GT identities establish detection correspondence; this is a motion benchmark, "
            "not HOTA or full tracking accuracy.",
            "GT is used for training-fold noise fitting and held-out scoring, "
            "never for held-out state updates or initialization.",
            "A single-sequence trajectory split shares camera and scene conditions "
            "and does not establish sequence generalization.",
            "Prediction uses image coordinates and each filter's intrinsic recovery and abnormal-motion "
            "suppression where supported, without camera-motion compensation, association thresholds, "
            "or tracker lifecycles.",
            "Detector confidence enters XYSCR state and StrongSORT measurement noise; "
            "geometric scores do not evaluate confidence-based association benefits.",
            "Matched detections inherit the calibration IoU gate; "
            "false positives and detector association mistakes are not modeled.",
            "Fold covariance scales start from unit priors, are fitted on training groups only, "
            "and use pooled training-class fallbacks when needed.",
            "Each model compares original, motion-only, measurement-only, and complete calibration "
            "using the same fold fit and predictions; fitted groups are retained only when their recipe "
            "improves held-out mean IoU beyond the tie tolerance.",
            "Selection scores use held-out folds to choose a model and noise profile "
            "and are not an unbiased final test-set tracking metric.",
        ],
    }
    if primary_horizon is not None:
        if progress:
            progress(f"KF selection: evaluating gap diagnostics for winner {selected}/{report['selected_profile']}…")
        diagnostics = _evaluate_profile(
            partitions,
            profile=report["selected_profile"],
            tracker_name=tracker_name,
            geometry=geometry,
            baseline=baselines[selected],
            fitted_folds=fits_by_kind[selected],
            progress=progress,
            camera_warps=camera_warps,
            primary_horizon=primary_horizon,
            horizons=kalman_replay.DEFAULT_HORIZONS,
            diagnostics_only=True,
        )
        report.update(
            version=4,
            objective="maximize_one_frame_mean_iou",
            primary_horizon=primary_horizon,
            horizons=[primary_horizon],
            diagnostic_horizons=[horizon for horizon in kalman_replay.DEFAULT_HORIZONS if horizon != primary_horizon],
            diagnostics={
                "selected_type": selected,
                "selected_profile": report["selected_profile"],
                "requested_horizons": list(kalman_replay.DEFAULT_HORIZONS),
                "affects_selection": False,
                "horizons": {},
                **diagnostics,
            },
            horizon_semantics=(
                "One-frame priors are scored after prediction and the current image's camera transform, "
                "before the current detection update. Three/five-frame metrics replay causal image-assisted "
                "detection gaps: intermediate detections are withheld, while each delivered image supplies "
                "its previous-to-current transform. They are diagnostics, not future-image forecasts, "
                "and do not enter selection. Only each policy's winning KF/profile receives these "
                "diagnostics; diagnostic failures cannot invalidate its one-frame ranking. "
                "Stop at annotation discontinuities."
            ),
            aggregation=(
                "One-frame predictions within trajectory, trajectories within sequence, "
                "then sequences equally. Longer detection-gap horizons are diagnostic only."
            ),
        )
        report["limitations"][3] = (
            "Prediction follows this candidate's camera and filter policies, including intrinsic recovery "
            "and abnormal-motion suppression where supported, without association thresholds or tracker lifecycles. "
            "Precomputed camera transforms and sequence-level automatic-policy statistics come from "
            "images/detection masks independently of held-out GT. Withheld detections refer to KF updates."
        )
        report["limitations"][-2] = (
            "Each model compares original, motion-only, measurement-only, and complete calibration "
            "using the same fold fit under this camera policy; fitted groups are retained only when "
            "their recipe improves held-out one-frame mean IoU beyond the tie tolerance."
        )
    return report


def select_kalman_model(
    data: CalibrationData,
    *,
    tracker_name: str,
    geometry: str,
    configs: Mapping[str, Mapping[str, Any]],
    baseline_type: str,
    progress: Callable[[str], None] | None = None,
    camera_candidates: Mapping[str, Mapping[str, np.ndarray] | None] | None = None,
) -> dict[str, Any]:
    """Select held-out KF/noise recipes, optionally jointly with camera policies.

    Without camera candidates the original equal-horizon selection is retained.
    Joint selection fits every policy/KF on identical training folds under its
    own previous-to-current camera transforms. One-frame prior IoU is the sole
    objective; only each policy's winning KF/profile receives three/five-frame
    image-assisted detection-gap diagnostics, reusing its fold fits. Diagnostic
    failures cannot invalidate the primary candidate. Off policies win ties,
    then the baseline KF and simpler
    noise profile. The winning policy's ordinary KF reports remain available
    through ``candidates``; ``cmc_candidates`` contains every policy's report.
    """
    options = {
        "tracker_name": tracker_name,
        "geometry": geometry,
        "configs": configs,
        "baseline_type": baseline_type,
    }
    if camera_candidates is None:
        return _select_policy(data, **options, progress=progress)
    validate_kf_selection(tracker_name)
    if geometry not in {"aabb", "obb"}:
        raise ValueError("KF selection requires AABB or OBB geometry.")
    candidates = kalman_folds.validate_camera_candidates(camera_candidates, kalman_folds.usable_tracks(data, geometry))
    reports = {}
    for name, camera_warps in candidates.items():

        def camera_progress(message: str, *, policy: str = name) -> None:
            """Keep long-running model/fold updates attributed to their policy."""
            if progress:
                progress(f"CMC {policy}: {message}")

        try:
            reports[name] = {
                **_select_policy(
                    data,
                    **options,
                    progress=camera_progress if progress else None,
                    camera_warps=camera_warps,
                    primary_horizon=1,
                ),
                "status": "valid",
            }
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            reports[name] = {"status": "invalid", "error": str(exc)}
    valid = [name for name, report in reports.items() if report["status"] == "valid"]
    if not valid:
        errors = "; ".join(f"{name}: {row['error']}" for name, row in reports.items())
        raise ValueError(f"Joint CMC/KF selection could not evaluate any policy: {errors}")

    def winner(name: str) -> dict[str, Any]:
        """Read a policy's selected family/profile score without duplicating it."""
        report = reports[name]
        return report["candidates"][report["selected_type"]]

    best = max(winner(name)["mean_iou"] for name in valid)
    tied = [name for name in valid if best - winner(name)["mean_iou"] <= _TIE_TOLERANCE]
    selected = min(
        tied,
        key=lambda name: (
            candidates[name] is not None,
            reports[name]["selected_type"] != baseline_type,
            reports[name]["selected_profile"] != "default",
            list(candidates).index(name),
        ),
    )
    return {
        **reports[selected],
        "method": "grouped_cross_validated_cmc_box_profiles",
        "selected_cmc": selected,
        "cmc_candidate_order": list(candidates),
        "cmc_candidates": reports,
        "tie_break": "cmc_off_then_baseline_then_default_profile_then_candidate_order",
    }
