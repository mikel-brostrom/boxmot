from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from boxmot.detectors import default_conf, default_imgsz, get_runtime_detector_cfg
from boxmot.utils import TRACKEVAL, logger as LOGGER
from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    apply_reid_runtime_defaults,
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    get_benchmark_detector_cfg,
    get_benchmark_reid_cfg,
    load_benchmark_cfg,
    load_runtime_reid_component_cfg,
    resolve_required_reid_model,
    resolve_required_yolo_model,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.download import download_trackeval
from boxmot.utils.misc import resolve_model_path


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_benchmark_cfg_from_args(args: argparse.Namespace) -> dict:
    benchmark = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
    )
    if not benchmark:
        return {}

    try:
        return load_benchmark_cfg(benchmark) or {}
    except FileNotFoundError:
        return {}


def resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    eval_box_type = getattr(args, "eval_box_type", None)
    if eval_box_type:
        return str(eval_box_type).lower()

    benchmark_cfg = (bench_cfg or {}).get("benchmark", {})
    box_type = benchmark_cfg.get("box_type")
    return str(box_type).lower() if box_type else "aabb"


def configure_benchmark_runtime(
    args: argparse.Namespace,
    *,
    load_benchmark_cfg_fn: Callable[[argparse.Namespace], dict] = load_benchmark_cfg_from_args,
    should_use_benchmark_detector_fn: Callable[[argparse.Namespace, dict], bool] = should_use_benchmark_detector,
    should_use_benchmark_reid_fn: Callable[[argparse.Namespace, dict], bool] = should_use_benchmark_reid,
    ensure_benchmark_detector_model_fn: Callable[[dict], Optional[Path]] = ensure_benchmark_detector_model,
    ensure_benchmark_reid_model_fn: Callable[[dict], Optional[Path]] = ensure_benchmark_reid_model,
) -> tuple[dict, dict, dict]:
    """Apply benchmark-driven detector and ReID defaults to the current args namespace."""
    benchmark_bundle = load_benchmark_cfg_fn(args)
    benchmark_cfg = benchmark_bundle.get("benchmark", {})

    use_benchmark_detector = should_use_benchmark_detector_fn(args, benchmark_bundle)
    use_benchmark_reid = should_use_benchmark_reid_fn(args, benchmark_bundle)
    benchmark_detector_cfg = get_benchmark_detector_cfg(benchmark_bundle) if use_benchmark_detector else {}

    required_yolo_model = resolve_required_yolo_model(benchmark_bundle)
    if required_yolo_model and use_benchmark_detector:
        required_model = (
            ensure_benchmark_detector_model_fn(benchmark_bundle)
            or resolve_model_path(required_yolo_model)
        )
        if args.yolo_model[0] != required_model:
            LOGGER.info(f"Using benchmark-default detector: {required_model}")
        args.yolo_model = [required_model]

    required_reid_model = resolve_required_reid_model(benchmark_bundle)
    if required_reid_model and use_benchmark_reid:
        required_model = (
            ensure_benchmark_reid_model_fn(benchmark_bundle)
            or resolve_model_path(required_reid_model)
        )
        if args.reid_model[0] != required_model:
            LOGGER.info(f"Using benchmark-default ReID: {required_model}")
        args.reid_model = [required_model]

    runtime_reid_cfg = (
        get_benchmark_reid_cfg(benchmark_bundle)
        if use_benchmark_reid
        else load_runtime_reid_component_cfg(args.reid_model[0])
    )
    apply_reid_runtime_defaults(args, {"reid": runtime_reid_cfg}, use_config=bool(runtime_reid_cfg))

    dataset_detector_cfg = get_runtime_detector_cfg(args.yolo_model[0], benchmark_detector_cfg)
    args.dataset_detector_cfg = dataset_detector_cfg or None

    if not getattr(args, "eval_box_type", None):
        box_type = benchmark_cfg.get("box_type") or dataset_detector_cfg.get("box_type")
        if box_type:
            args.eval_box_type = str(box_type).lower()

    if args.imgsz is None:
        args.imgsz = (
            list(dataset_detector_cfg["imgsz"])
            if "imgsz" in dataset_detector_cfg
            else default_imgsz(args.yolo_model[0])
        )

    if args.conf is None:
        args.conf = (
            float(dataset_detector_cfg["conf"])
            if "conf" in dataset_detector_cfg
            else default_conf(args.yolo_model[0])
        )

    return benchmark_bundle, benchmark_cfg, dataset_detector_cfg


def resolve_obb_eval_class_pairs(args: argparse.Namespace, bench_cfg: dict) -> list[tuple[str, int]]:
    """Resolve OBB class names and their actual zero-based MMOT class IDs."""
    eval_classes_cfg = bench_cfg.get("eval_classes") or {}
    ordered_pairs = [
        (str(name).lower(), int(class_id) - 1)
        for class_id, name in sorted(eval_classes_cfg.items(), key=lambda kv: int(kv[0]))
    ]

    if not ordered_pairs and getattr(args, "remapped_class_ids", None) and getattr(args, "remapped_class_names", None):
        return [
            (str(name).lower(), int(class_id))
            for name, class_id in zip(args.remapped_class_names, args.remapped_class_ids)
        ]

    translated_names = getattr(args, "translated_benchmark_class_names", None)
    if translated_names:
        wanted = {str(name).lower() for name in translated_names}
        return [(name, class_id) for name, class_id in ordered_pairs if name in wanted]

    remapped_names = getattr(args, "remapped_class_names", None)
    if remapped_names:
        wanted = {str(name).lower() for name in remapped_names}
        return [(name, class_id) for name, class_id in ordered_pairs if name in wanted]

    class_indices = getattr(args, "classes", None)
    if class_indices is not None:
        wanted = {int(idx) for idx in class_indices}
        return [(name, class_id) for idx, (name, class_id) in enumerate(ordered_pairs) if idx in wanted]

    return ordered_pairs


def resolve_obb_classes_to_eval(args: argparse.Namespace, bench_cfg: dict) -> list[str]:
    """Resolve class names for the OBB TrackEval runner."""
    return [name for name, _ in resolve_obb_eval_class_pairs(args, bench_cfg)]


def resolve_obb_class_ids_to_eval(args: argparse.Namespace, bench_cfg: dict) -> list[int]:
    """Resolve zero-based class IDs for the OBB TrackEval runner."""
    return [class_id for _, class_id in resolve_obb_eval_class_pairs(args, bench_cfg)]


def build_gt_class_remap(
    bench_cfg: dict,
    det_cfg: Optional[dict],
    benchmark_name: str = "",
    model_stem: str = "",
) -> Optional[tuple[dict, list[int], list[str]]]:
    """Build a GT class remap so gt_temp.txt class IDs match tracker output."""
    eval_classes_cfg = bench_cfg.get("eval_classes")
    class_mapping = bench_cfg.get("class_mapping")

    if det_cfg is None:
        if class_mapping:
            LOGGER.error(
                "class_mapping is defined in the benchmark config but no detector class metadata was "
                f"found for model '{model_stem}'. "
                "Use the benchmark-default detector or remove class_mapping to use default evaluation."
            )
        return None

    det_classes = det_cfg.get("classes", {})
    if not det_classes:
        LOGGER.warning(f"Detector config for '{model_stem}' has no 'classes' field. Skipping remap.")
        return None

    det_name_to_id = {str(v): int(k) for k, v in det_classes.items()}

    if not class_mapping:
        remap_logging = len(eval_classes_cfg) > 1

        if remap_logging:
            LOGGER.warning(
                f"No class_mapping found for benchmark '{benchmark_name}'. "
                "Using positional auto-mapping: first N benchmark classes -> first N detector classes."
            )

        bench_ordered = sorted((int(k), str(v)) for k, v in eval_classes_cfg.items())
        det_ordered = sorted((int(k), str(v)) for k, v in det_classes.items())
        n_pairs = min(len(bench_ordered), len(det_ordered))

        remap: dict[int, int] = {}
        seen_det_ids: list[int] = []
        seen_det_names: list[str] = []
        rows: list[tuple[str, str]] = []
        for index in range(n_pairs):
            bench_id, bench_name = bench_ordered[index]
            det_id, det_name = det_ordered[index]
            new_gt_id = det_id + 1
            remap[bench_id] = new_gt_id
            rows.append((bench_name, det_name))
            if new_gt_id not in seen_det_ids:
                seen_det_ids.append(new_gt_id)
                seen_det_names.append(det_name)

        if remap_logging:
            LOGGER.opt(colors=True).info("<yellow>Auto class mapping (positional):</yellow>")
            for bench_name, det_name in rows:
                LOGGER.opt(colors=True).info(f"  <yellow>{bench_name:<22}</yellow> -> <cyan>{det_name}</cyan>")
            LOGGER.opt(colors=True).info(
                "  <yellow>GT class IDs remapped:</yellow> "
                + ", ".join(f"{bench_id}->{remap[bench_id]}" for bench_id in sorted(remap))
            )
            LOGGER.opt(colors=True).info(
                "  <yellow>Evaluating detector classes:</yellow> "
                + ", ".join(f"{name} ({class_id})" for name, class_id in zip(seen_det_names, seen_det_ids))
            )
        return remap, seen_det_ids, seen_det_names

    if not eval_classes_cfg:
        LOGGER.warning("class_mapping is set but eval_classes is missing in benchmark config. Skipping remap.")
        return None

    bench_name_to_id = {str(v): int(k) for k, v in eval_classes_cfg.items()}

    remap: dict[int, int] = {}
    det_classes_used: dict[str, int] = {}
    skipped: list[str] = []
    for benchmark_class_name, detector_class_name in class_mapping.items():
        benchmark_class_name = str(benchmark_class_name)
        detector_class_name = str(detector_class_name)
        if benchmark_class_name not in bench_name_to_id:
            skipped.append(f"benchmark class '{benchmark_class_name}' not in eval_classes")
            continue
        if detector_class_name not in det_name_to_id:
            skipped.append(f"detector class '{detector_class_name}' not in detector config")
            continue
        bench_id = bench_name_to_id[benchmark_class_name]
        det_id = det_name_to_id[detector_class_name]
        remap[bench_id] = det_id + 1
        det_classes_used[detector_class_name] = det_id + 1

    for message in skipped:
        LOGGER.warning(f"class_mapping: skipping - {message}")

    if not remap:
        LOGGER.warning("class_mapping produced no valid entries. Skipping remap.")
        return None

    new_entries = sorted(det_classes_used.items(), key=lambda item: item[1])
    new_class_ids = [class_id for _, class_id in new_entries]
    new_class_names = [name for name, _ in new_entries]

    model_label = f" -> {model_stem}" if model_stem else ""
    LOGGER.opt(colors=True).info(f"<cyan>Class mapping ({benchmark_name}{model_label}):</cyan>")
    for benchmark_class_name, detector_class_name in class_mapping.items():
        benchmark_class_name = str(benchmark_class_name)
        detector_class_name = str(detector_class_name)
        if benchmark_class_name in bench_name_to_id and detector_class_name in det_name_to_id:
            LOGGER.opt(colors=True).info(
                f"  <blue>{benchmark_class_name:<22}</blue> -> <cyan>{detector_class_name}</cyan>"
            )
    LOGGER.opt(colors=True).info(
        "  <cyan>GT class IDs remapped:</cyan> "
        + ", ".join(f"{bench_id}->{remap[bench_id]}" for bench_id in sorted(remap))
    )
    LOGGER.opt(colors=True).info(
        "  <cyan>Evaluating detector classes:</cyan> "
        + ", ".join(f"{name} ({class_id})" for name, class_id in zip(new_class_names, new_class_ids))
    )
    return remap, new_class_ids, new_class_names


def apply_gt_class_remap(
    source: Path,
    remap: dict[int, int],
    distractor_ids: Optional[list[int]] = None,
) -> None:
    """Rewrite every gt_temp.txt under *source* using *remap*."""
    distractor_set = set(distractor_ids or [])
    keep_ids = set(remap.keys()) | distractor_set

    gt_files = list(source.glob("*/gt/gt_temp.txt"))
    if not gt_files:
        LOGGER.warning(f"apply_gt_class_remap: no gt_temp.txt files found under {source}")
        return

    for gt_file in gt_files:
        try:
            data = np.loadtxt(gt_file, delimiter=",")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"apply_gt_class_remap: could not read {gt_file}: {exc}")
            continue

        if data.size == 0:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        class_col = data[:, 7].astype(int)
        data = data[np.isin(class_col, list(keep_ids))]

        if data.size == 0:
            np.savetxt(gt_file, data, delimiter=",")
            continue

        class_col = data[:, 7].astype(int)
        for old_id, new_id in remap.items():
            data[class_col == old_id, 7] = new_id

        np.savetxt(gt_file, data, delimiter=",", fmt="%g")


def _write_filtered_eval_gt(src: Path, dst: Path, keep_ids: Optional[set[int]] = None) -> None:
    """Copy a GT-like CSV file to *dst*, optionally filtering by frame IDs."""
    try:
        data = np.loadtxt(src, delimiter=",")
    except ValueError:
        data = np.empty((0, 0), dtype=np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if data.size == 0:
        dst.write_text("")
        return

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if keep_ids:
        data = data[np.isin(data[:, 0].astype(int), list(keep_ids))]

    if data.size == 0:
        dst.write_text("")
        return

    np.savetxt(dst, data, delimiter=",", fmt="%g")


def prepare_aabb_eval_gt(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> Path:
    """Create a run-local AABB GT tree so evaluation does not mutate source datasets."""
    bridge_root = args.exp_dir / "trackeval_gt"
    kept_by_seq = getattr(args, "seq_frame_nums", {}) or {}
    uses_flat_annotations = all((gt_folder / f"{seq}.txt").exists() for seq in seq_info)

    for seq_name in seq_info:
        keep_ids = set(kept_by_seq.get(seq_name, [])) or None
        if uses_flat_annotations:
            src_gt = gt_folder / f"{seq_name}.txt"
            dst_gt = bridge_root / f"{seq_name}.txt"
        else:
            seq_gt_dir = gt_folder / seq_name / "gt"
            src_gt = seq_gt_dir / "gt.txt"
            if not src_gt.exists():
                src_gt = seq_gt_dir / "gt_temp.txt"
            if not src_gt.exists():
                raise FileNotFoundError(f"Missing GT file for sequence {seq_name} under {seq_gt_dir}")
            dst_gt = bridge_root / seq_name / "gt" / "gt_temp.txt"

        _write_filtered_eval_gt(src_gt, dst_gt, keep_ids)

    remap = getattr(args, "gt_class_remap", None)
    if remap and not uses_flat_annotations:
        distractor_ids = getattr(args, "gt_class_distractor_ids", None)
        apply_gt_class_remap(bridge_root, remap, distractor_ids)

    return bridge_root


def eval_init(
    args: argparse.Namespace,
    trackeval_dest: Path = TRACKEVAL,
    branch: str = "main",
    overwrite: bool = False,
) -> None:
    """
    Common initialization: download TrackEval and benchmark data, then canonicalize paths.
    """
    download_trackeval(dest=trackeval_dest, branch=branch, overwrite=overwrite)
    apply_benchmark_config(args, overwrite=overwrite)

    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)


__all__ = [
    "COCO_CLASSES",
    "apply_gt_class_remap",
    "build_gt_class_remap",
    "configure_benchmark_runtime",
    "eval_init",
    "load_benchmark_cfg_from_args",
    "prepare_aabb_eval_gt",
    "resolve_eval_box_type",
    "resolve_obb_class_ids_to_eval",
    "resolve_obb_classes_to_eval",
    "resolve_obb_eval_class_pairs",
]
