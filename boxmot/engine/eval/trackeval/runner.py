from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from boxmot.configs.benchmark import load_benchmark_cfg
from boxmot.data.benchmark import (
    COCO_CLASSES,
    load_benchmark_cfg_from_args,
    resolve_obb_class_ids_to_eval,
    resolve_obb_classes_to_eval,
)
from boxmot.utils import NUM_THREADS
from boxmot.utils import logger as LOGGER

_TRACKEVAL_STDERR_NOISE_PATTERNS = (
    "resource_tracker",
    "multiprocessing/resource_tracker.py",
    "warnings.warn('resource_tracker:",
    "missing the following timesteps",
    "raw_gt_data = self._load_raw_file",
    "raw_tracker_data = self._load_raw_file",
)


def _should_use_parallel_trackeval() -> bool:
    # The subprocess env (_trackeval_subprocess_env) already suppresses the
    # spurious resource_tracker semaphore warnings on macOS + Python 3.12+,
    # so parallelism is safe on all platforms.
    return True


def _trackeval_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # Suppress noisy UserWarnings from trackeval (resource_tracker + missing timesteps)
    warning_filters = [
        "ignore:resource_tracker:UserWarning",
        "ignore:.*missing the following timesteps.*:UserWarning",
    ]
    existing = env.get("PYTHONWARNINGS", "")
    for wf in warning_filters:
        if wf not in existing.split(","):
            existing = ",".join(filter(None, [existing, wf]))
    env["PYTHONWARNINGS"] = existing
    return env


def _filter_trackeval_stderr(stderr: str) -> str:
    if not stderr:
        return ""

    kept_lines = [
        line
        for line in stderr.splitlines()
        if line and not any(pattern in line for pattern in _TRACKEVAL_STDERR_NOISE_PATTERNS)
    ]
    return "\n".join(kept_lines).strip()


def build_dataset_eval_settings(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> dict:
    """Derive benchmark-specific evaluation settings."""
    cfg = {}
    try:
        benchmark_id = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", None)
        )
        if benchmark_id:
            cfg = load_benchmark_cfg(benchmark_id)
    except FileNotFoundError:
        cfg = {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(f"Error loading benchmark config: {exc}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    distractor_cfg = bench_cfg.get("distractor_classes") if isinstance(bench_cfg, dict) else None
    ignore_dataset_ids = bench_cfg.get("ignore_dataset_ids") if isinstance(bench_cfg, dict) else None

    layout_name = str(cfg.get("layout") or bench_cfg.get("layout") or "").lower() if isinstance(cfg, dict) else ""

    gt_loc_format = "{gt_folder}/{seq}/gt/gt_temp.txt"
    is_visdrone = (
        layout_name == "visdrone"
        or "visdrone" in getattr(args, "benchmark", "").lower()
        or "visdrone" in str(gt_folder).lower()
    )
    if is_visdrone:
        gt_loc_format = "{gt_folder}/{seq}.txt"

    benchmark_name = getattr(args, "benchmark", "")

    if ignore_dataset_ids is not None:
        distractor_ids = [int(class_id) for class_id in ignore_dataset_ids]
    elif isinstance(distractor_cfg, dict) and distractor_cfg:
        distractor_ids = [int(k) for k in distractor_cfg.keys()]
    else:
        distractor_ids = []

    if getattr(args, "remapped_class_ids", None):
        return {
            "classes_to_eval": args.remapped_class_names,
            "class_ids": args.remapped_class_ids,
            "distractor_ids": distractor_ids,
            "gt_loc_format": gt_loc_format,
            "benchmark_name": benchmark_name,
            "seq_info": seq_info,
        }

    classes_to_eval: list[str] = []
    class_ids: list[int] = []

    if hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(i)] for i in class_indices]
        class_ids = [int(i) + 1 for i in class_indices]

    if isinstance(eval_classes_cfg, dict) and eval_classes_cfg:
        ordered = sorted(((int(k), v) for k, v in eval_classes_cfg.items()), key=lambda kv: kv[0])
        if class_ids:
            class_ids = [k for k, _ in ordered if k in class_ids]
            classes_to_eval = [v for k, v in ordered if k in class_ids]
        else:
            class_ids = [k for k, _ in ordered]
            classes_to_eval = [v for k, v in ordered]

    if not classes_to_eval:
        classes_to_eval = ["person"]
    if not class_ids:
        class_ids = [1]

    seen: set[str] = set()
    pairs: list[tuple[str, int]] = []
    for name, class_id in zip(classes_to_eval, class_ids):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, class_id))

    return {
        "classes_to_eval": [name for name, _ in pairs],
        "class_ids": [class_id for _, class_id in pairs],
        "distractor_ids": distractor_ids,
        "gt_loc_format": gt_loc_format,
        "benchmark_name": benchmark_name,
        "seq_info": seq_info,
    }


def trackeval_aabb(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list[str] = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """Execute the standard MOT TrackEval adapter."""
    del save_dir

    if not seq_info:
        seq_names = [seq_path.parent.name if seq_path.name == "img1" else seq_path.name for seq_path in seq_paths]
        seq_info = {name: None for name in seq_names}

    seq_info_args = []
    for name in sorted(seq_info.keys()):
        length = seq_info[name]
        seq_info_args.append(f"{name}:{length}" if length else name)

    dataset_settings = build_dataset_eval_settings(args, gt_folder, seq_info)
    classes_to_eval = dataset_settings["classes_to_eval"]
    class_ids = dataset_settings["class_ids"]
    distractor_ids = dataset_settings["distractor_ids"]
    gt_loc_format = dataset_settings["gt_loc_format"]
    benchmark_name = dataset_settings["benchmark_name"]
    use_parallel = _should_use_parallel_trackeval()
    parallel_cores = min(NUM_THREADS, len(seq_info)) if use_parallel else 1

    cmd_args = [
        sys.executable,
        "-W", "ignore::UserWarning",
        "-m",
        "boxmot.engine.eval.trackeval.scripts.mot_challenge",
        "--GT_FOLDER",
        str(gt_folder),
        "--BENCHMARK",
        benchmark_name,
        "--TRACKERS_FOLDER",
        str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL",
        args.exp_dir.name,
        "--SPLIT_TO_EVAL",
        args.split,
        "--METRICS",
        *metrics,
        "--TRACKER_SUB_FOLDER",
        "",
        "--NUM_PARALLEL_CORES",
        str(parallel_cores),
        "--SKIP_SPLIT_FOL",
        "True",
        "--GT_LOC_FORMAT",
        gt_loc_format,
        "--CLASSES_TO_EVAL",
        *classes_to_eval,
        "--CLASS_IDS",
        *[str(class_id) for class_id in class_ids],
        "--DISTRACTOR_CLASS_IDS",
        *[str(class_id) for class_id in distractor_ids],
        "--SEQ_INFO",
        *seq_info_args,
        "--USE_PARALLEL",
        "True" if use_parallel else "False",
    ]

    proc = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_trackeval_subprocess_env(),
    )
    stdout, stderr = proc.communicate()
    stderr = _filter_trackeval_stderr(stderr)

    if stderr:
        LOGGER.warning(f"TrackEval stderr:\n{stderr}")
    return stdout


def _load_obb_gt_matrix(source: Path) -> np.ndarray:
    """Load OBB GT in the TrackEval 13-column corner format."""
    data = np.loadtxt(source, delimiter=",")
    if data.size == 0:
        return np.empty((0, 13), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 13:
        return data.astype(np.float32)
    raise ValueError(
        f"Unsupported OBB GT format in {source}: expected 13 columns in corner format, got {data.shape[1]}"
    )


def _prepare_obb_eval_bridge(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> tuple[Path, Path]:
    """Create the flat GT/image layout expected by the OBB TrackEval adapter."""
    bridge_root = args.exp_dir / "trackeval_obb"
    gt_bridge = bridge_root / "gt"
    img_bridge = bridge_root / "img"
    gt_bridge.mkdir(parents=True, exist_ok=True)
    img_bridge.mkdir(parents=True, exist_ok=True)

    for seq_name in seq_info:
        seq_dir = Path(args.source) / seq_name
        src_img_dir = (seq_dir / "img1" if (seq_dir / "img1").exists() else seq_dir).resolve()
        bridge_img_dir = img_bridge / seq_name
        if os.path.lexists(bridge_img_dir) and bridge_img_dir.is_symlink():
            if os.readlink(bridge_img_dir) == str(src_img_dir):
                pass  # symlink already correct
            else:
                bridge_img_dir.unlink()
                os.symlink(src_img_dir, bridge_img_dir, target_is_directory=True)
        elif not bridge_img_dir.exists():
            try:
                os.symlink(src_img_dir, bridge_img_dir, target_is_directory=True)
            except OSError:
                shutil.copytree(src_img_dir, bridge_img_dir, dirs_exist_ok=True)

        # Some OBB benchmarks store GT as flat files in a sibling "mot/" directory
        # (e.g. <root>/test/mot/data23-1.txt alongside <root>/test/npy/data23-1/)
        mot_sibling = Path(args.source).parent / "mot" / f"{seq_name}.txt"

        gt_out = gt_bridge / f"{seq_name}.txt"

        raw_candidates = [
            mot_sibling,
            seq_dir / "gt" / "gt_temp.txt",
            gt_folder / seq_name / "gt" / "gt_temp.txt",
            seq_dir / "gt" / "gt.txt",
            gt_folder / seq_name / "gt" / "gt.txt",
            seq_dir / "gt" / "gt_obb_raw_temp.txt",
            gt_folder / seq_name / "gt" / "gt_obb_raw_temp.txt",
            seq_dir / "gt" / "gt_obb_temp.txt",
            gt_folder / seq_name / "gt" / "gt_obb_temp.txt",
            seq_dir / "gt" / "gt_obb_raw.txt",
            gt_folder / seq_name / "gt" / "gt_obb_raw.txt",
            seq_dir / "gt" / "gt_obb.txt",
            gt_folder / seq_name / "gt" / "gt_obb.txt",
        ]
        source_gt = None
        normalized_gt = None
        for candidate in raw_candidates:
            if not candidate.exists():
                continue
            try:
                # Skip rewrite if bridge GT is newer than source
                if gt_out.exists() and gt_out.stat().st_mtime >= candidate.stat().st_mtime:
                    source_gt = candidate
                    break
                normalized_gt = _load_obb_gt_matrix(candidate)
                source_gt = candidate
                break
            except ValueError:
                continue

        if source_gt is None:
            raise FileNotFoundError(
                f"No OBB GT file found for sequence {seq_name}. "
                "Expected gt.txt/gt_temp.txt or gt_obb*.txt in 13-column corner format."
            )

        # normalized_gt is None when cache hit (bridge GT already up-to-date)
        if normalized_gt is not None:
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            if normalized_gt.size == 0:
                gt_out.write_text("")
            else:
                np.savetxt(gt_out, normalized_gt, delimiter=",", fmt="%g")

    return gt_bridge, img_bridge


def trackeval_obb(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list[str] = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """Evaluate OBB tracking results via BoxMOT's custom OBB TrackEval runner."""
    del save_dir, seq_paths
    if not seq_info:
        raise ValueError("seq_info is required for OBB TrackEval")

    bench_cfg = load_benchmark_cfg_from_args(args).get("benchmark", {})
    classes_to_eval = resolve_obb_classes_to_eval(args, bench_cfg)
    class_ids = resolve_obb_class_ids_to_eval(args, bench_cfg)
    gt_bridge, img_bridge = _prepare_obb_eval_bridge(args, gt_folder, seq_info)

    use_parallel = _should_use_parallel_trackeval()
    parallel_cores = min(NUM_THREADS, len(seq_info)) if use_parallel else 1

    cmd_args = [
        sys.executable,
        "-W", "ignore::UserWarning",
        "-m",
        "boxmot.engine.eval.trackeval.scripts.mot_challenge_obb",
        "--GT_FOLDER",
        str(gt_bridge),
        "--IMG_FOLDER",
        str(img_bridge),
        "--TRACKERS_FOLDER",
        str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL",
        args.exp_dir.name,
        "--TRACKER_SUB_FOLDER",
        "",
        "--OUTPUT_SUB_FOLDER",
        "",
        "--SPLIT_TO_EVAL",
        str(getattr(args, "split", "train")),
        "--METRICS",
        *metrics,
        "--PRINT_CONFIG",
        "False",
        "--PRINT_ONLY_COMBINED",
        "False",
        "--USE_PARALLEL",
        "True" if use_parallel else "False",
        "--NUM_PARALLEL_CORES",
        str(parallel_cores),
    ]
    if classes_to_eval:
        cmd_args.extend(["--CLASSES_TO_EVAL", *classes_to_eval])
    if class_ids:
        cmd_args.extend(["--CLASS_IDS", *[str(class_id) for class_id in class_ids]])

    proc = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_trackeval_subprocess_env(),
    )
    stdout, stderr = proc.communicate()
    stderr = _filter_trackeval_stderr(stderr)

    if stderr:
        LOGGER.warning(f"OBB TrackEval stderr:\n{stderr}")
    return stdout


__all__ = [
    "_load_obb_gt_matrix",
    "_prepare_obb_eval_bridge",
    "build_dataset_eval_settings",
    "trackeval_aabb",
    "trackeval_obb",
]
