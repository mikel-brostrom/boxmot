"""KF noise estimation helpers for the eval/tune pipelines."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from boxmot.configs.benchmark import load_benchmark_cfg
from boxmot.utils import logger as LOGGER

# Mapping from tracker name to KF parameterization type
_TRACKER_KF_MAP: dict[str, str] = {
    "botsort": "xywh",
    "bytetrack": "xyah",
    "strongsort": "xyah",
    "deepocsort": "xysr",
    "ocsort": "xysr",
    "hybridsort": "xysr",
    "boosttrack": "xyhr",
    "occluboost": "xyhr",
}


def tracker_kf_type(tracker_name: str) -> str | None:
    """Return the KF parameterization for a tracker, or None if it has no KF."""
    return _TRACKER_KF_MAP.get(tracker_name.lower())


def resolve_kf_train_root(args: argparse.Namespace) -> Path | None:
    """Resolve a train-split GT path for KF tuning.

    When the user is evaluating on a non-train split (e.g. ``--split test``),
    we prefer to tune KF noise from the *train* split so that the tuning data
    is independent of the evaluation data.  Falls back to ``args.source``
    (the eval split) when no separate train split can be resolved.
    """
    eval_split = getattr(args, "split", None) or ""
    benchmark_id = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "data", None)
    )
    if not benchmark_id or eval_split == "train":
        return None

    try:
        cfg = load_benchmark_cfg(benchmark_id)
    except Exception:
        return None

    all_splits = cfg.get("splits") or {}
    train_entry = all_splits.get("train")
    if not train_entry:
        return None

    # train_entry can be a string (path) or dict with "path" key
    if isinstance(train_entry, dict):
        train_rel = str(train_entry.get("path") or "train")
    else:
        train_rel = str(train_entry)

    source_root = Path(str(cfg.get("path") or ""))
    if source_root and source_root.parts:
        train_root = source_root / train_rel
    else:
        # Derive from eval source: walk up from eval path and replace split subpath
        eval_rel = str(all_splits.get(eval_split) or eval_split)
        if isinstance(all_splits.get(eval_split), dict):
            eval_rel = str(all_splits[eval_split].get("path") or eval_split)
        eval_source = Path(args.source).resolve()
        # Strip the eval-relative suffix to get dataset root, then append train
        eval_rel_parts = Path(eval_rel).parts
        dataset_root = eval_source
        for _ in eval_rel_parts:
            dataset_root = dataset_root.parent
        train_root = dataset_root / train_rel

    if train_root.exists():
        return train_root

    return None


def run_kf_tuning(
    args: argparse.Namespace,
    kf_type: str,
    verbose: bool = False,
    capture: bool = False,
) -> tuple[dict | None, str]:
    """Run KF noise estimation when both GT and cached dets exist.

    Returns ``(result_dict, log_text)``.  When *capture* is True the
    verbose output is redirected into *log_text* instead of printing to
    stdout.

    Strategy for train/test separation:
    - If a train split has BOTH GT and cached dets, use them (proper separation).
    - Otherwise fall back to the eval split for both GT and dets so that
      sequence names are coherent between dets and GT.
    """
    from tools.analysis.mot_ds_kf_tuning import main as kf_tune_main

    if args.source is None:
        LOGGER.warning("KF tuning skipped: no GT source path available.")
        return None, ""

    # Resolve dets folder helper
    cache_project = Path(getattr(args, "cache_project", args.project))
    benchmark = getattr(args, "benchmark", None)
    eval_split = getattr(args, "split", None)
    detector_key = args.detector[0].stem if hasattr(args.detector[0], "stem") else str(args.detector[0])

    def _dets_path_for_split(split_name: str | None) -> Path:
        base = cache_project / "dets_n_embs"
        if benchmark:
            base = base / benchmark
        if split_name:
            base = base / split_name
        return base / detector_key / "dets"

    # Try train split first (proper train/test separation)
    train_root = resolve_kf_train_root(args)
    if train_root is not None:
        train_dets = _dets_path_for_split("train")
        if train_dets.exists() and any(train_dets.glob("*.npy")):
            gt_root = train_root
            dets_root = train_dets
            if not capture:
                LOGGER.info(
                    f"KF tuning: using train split for both GT and dets "
                    f"(eval split: '{eval_split}')"
                )
        else:
            # Train dets not available — can't use train GT because sequences
            # won't match test dets.  Fall back to eval split.
            if not capture:
                LOGGER.info(
                    f"KF tuning: train split GT found but no train dets at "
                    f"{train_dets}, falling back to eval split"
                )
            gt_root = Path(args.source)
            dets_root = _dets_path_for_split(eval_split)
    else:
        gt_root = Path(args.source)
        dets_root = _dets_path_for_split(eval_split)

    if not dets_root.exists() or not any(dets_root.glob("*.npy")):
        LOGGER.warning(f"KF tuning skipped: no cached detections at {dets_root}")
        return None, ""

    if not capture:
        LOGGER.info(f"[bold]KF Tuning[/bold] ({kf_type}): GT={gt_root}, dets={dets_root}")
    buf = io.StringIO() if capture else None
    cm = contextlib.redirect_stdout(buf) if buf is not None else contextlib.nullcontext()

    # Enable per-class KF tuning only when the dataset has multiple eval classes
    per_class_kf = bool(getattr(args, "per_class_kf", True))
    gt_class_offset = 0  # offset to convert GT class IDs → detector class IDs
    if per_class_kf:
        # Auto-detect: check the benchmark config's eval class count
        benchmark_id = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", None)
            or getattr(args, "data", None)
        )
        if benchmark_id:
            try:
                _cfg = load_benchmark_cfg(benchmark_id)
                names_dict = _cfg.get("names") or {}
                n_eval_classes = len(names_dict)
                if n_eval_classes <= 1:
                    per_class_kf = False
                elif names_dict:
                    # GT class IDs are typically 1-indexed; detectors use 0-indexed.
                    # Compute offset so per-class keys align with detector output.
                    gt_class_offset = min(int(k) for k in names_dict.keys())
            except Exception:
                pass
    try:
        with cm:
            result = kf_tune_main(
                train_root=gt_root,
                kf_type=kf_type,
                dets_root=dets_root,
                use_temp_gt=bool(getattr(args, "use_temp_gt", False)),
                verbose=verbose or capture,
                per_class=per_class_kf,
            )
        # Re-index per-class keys from GT class IDs to detector (0-indexed) class IDs
        if gt_class_offset and "per_class" in result:
            reindexed = {
                int(k) - gt_class_offset: v
                for k, v in result["per_class"].items()
            }
            result["per_class"] = reindexed
        if not capture:
            LOGGER.info(
                f"[bold]KF Tuning result:[/bold] "
                f"_std_weight_position={result['std_weight_position']:.6f}, "
                f"_std_weight_velocity={result['std_weight_velocity']:.6f}"
            )
        log_text = buf.getvalue().rstrip() if buf else ""
        return result, log_text
    except Exception as e:
        LOGGER.warning(f"KF tuning failed: {e}")
        log_text = buf.getvalue().rstrip() if buf else ""
        return None, log_text
