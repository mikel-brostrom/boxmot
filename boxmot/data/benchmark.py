from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from rich.markup import escape as _escape_markup

from boxmot.utils import logger as LOGGER

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




def _ordered_benchmark_eval_class_names(bench_cfg: dict) -> list[str]:
    """Return benchmark eval class names in config order without splitting embedded whitespace."""
    if not isinstance(bench_cfg, dict):
        return []

    eval_classes_cfg = bench_cfg.get("eval_classes")
    if isinstance(eval_classes_cfg, dict) and eval_classes_cfg:
        return [str(name) for _, name in sorted(eval_classes_cfg.items(), key=lambda kv: int(kv[0]))]
    if isinstance(eval_classes_cfg, (list, tuple)):
        return [str(name) for name in eval_classes_cfg]
    return []


def resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    eval_box_type = getattr(args, "eval_box_type", None)
    if eval_box_type:
        return str(eval_box_type).lower()

    benchmark_cfg = (bench_cfg or {}).get("benchmark", {})
    box_type = benchmark_cfg.get("box_type")
    return str(box_type).lower() if box_type else "aabb"






def resolve_obb_eval_class_pairs(args: argparse.Namespace, bench_cfg: dict) -> list[tuple[str, int]]:
    """Resolve OBB class names and their actual zero-based MMOT class IDs."""
    class_bridge = bench_cfg.get("class_bridge") or []
    if class_bridge:
        ordered_pairs = [
            (str(entry.get("name", entry.get("dataset_id"))).lower(), int(entry["detector_id"]))
            for entry in class_bridge
            if isinstance(entry, dict) and entry.get("detector_id") is not None
        ]
    else:
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

    if not ordered_pairs:
        return []

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
    """Resolve class names for OBB evaluation."""
    return [name for name, _ in resolve_obb_eval_class_pairs(args, bench_cfg)]


def resolve_obb_class_ids_to_eval(args: argparse.Namespace, bench_cfg: dict) -> list[int]:
    """Resolve zero-based class IDs for OBB evaluation."""
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
    class_bridge = bench_cfg.get("class_bridge") or []

    if class_bridge:
        remap: dict[int, int] = {}
        used_detector_classes: dict[int, str] = {}
        det_classes = det_cfg.get("classes", {}) if det_cfg else {}
        det_classes_by_id = {int(key): str(value) for key, value in det_classes.items()}

        for entry in class_bridge:
            if not isinstance(entry, dict):
                continue
            if entry.get("dataset_id") is None or entry.get("detector_id") is None:
                LOGGER.warning(f"class_bridge: skipping incomplete entry {entry}")
                continue

            dataset_id = int(entry["dataset_id"])
            detector_id = int(entry["detector_id"])
            detector_name = str(entry.get("detector_name") or entry.get("name") or detector_id)

            configured_detector_name = det_classes_by_id.get(detector_id)
            if configured_detector_name is not None and configured_detector_name != detector_name:
                LOGGER.warning(
                    "class_bridge detector metadata differs from detector config: "
                    f"id {detector_id} is '{configured_detector_name}', bridge says '{detector_name}'"
                )

            remap[dataset_id] = detector_id + 1
            used_detector_classes[detector_id + 1] = detector_name

        if not remap:
            LOGGER.warning("class_bridge produced no valid entries. Skipping remap.")
            return None

        new_entries = sorted(used_detector_classes.items())
        new_class_ids = [class_id for class_id, _ in new_entries]
        new_class_names = [name for _, name in new_entries]

        model_label = f" -> {model_stem}" if model_stem else ""
        LOGGER.info(f"[cyan]Class bridge ({_escape_markup(str(benchmark_name))}{_escape_markup(model_label)}):[/cyan]")
        for entry in class_bridge:
            if not isinstance(entry, dict) or entry.get("dataset_id") is None or entry.get("detector_id") is None:
                continue
            dataset_name = str(entry.get("name") or entry["dataset_id"])
            detector_name = str(entry.get("detector_name") or dataset_name)
            LOGGER.info(
                f"  [blue]{_escape_markup(dataset_name):<22}[/blue] "
                f"dataset:{int(entry['dataset_id'])} -> "
                f"[cyan]{_escape_markup(detector_name)}[/cyan] detector:{int(entry['detector_id'])}"
            )
        LOGGER.info(
            "  [cyan]GT class IDs remapped:[/cyan] "
            + ", ".join(f"{bench_id}->{remap[bench_id]}" for bench_id in sorted(remap))
        )
        return remap, new_class_ids, new_class_names

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

    det_name_to_id = {str(value): int(key) for key, value in det_classes.items()}

    if not class_mapping:
        remap_logging = len(eval_classes_cfg) > 1

        if remap_logging:
            LOGGER.warning(
                f"No class_mapping found for benchmark '{benchmark_name}'. "
                "Using positional auto-mapping: first N benchmark classes -> first N detector classes."
            )

        bench_ordered = sorted((int(key), str(value)) for key, value in eval_classes_cfg.items())
        det_ordered = sorted((int(key), str(value)) for key, value in det_classes.items())
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
            LOGGER.info("[yellow]Auto class mapping (positional):[/yellow]")
            for bench_name, det_name in rows:
                LOGGER.info(
                    f"  [yellow]{_escape_markup(str(bench_name)):<22}[/yellow] -> "
                    f"[cyan]{_escape_markup(str(det_name))}[/cyan]"
                )
            LOGGER.info(
                "  [yellow]GT class IDs remapped:[/yellow] "
                + ", ".join(f"{bench_id}->{remap[bench_id]}" for bench_id in sorted(remap))
            )
            LOGGER.info(
                "  [yellow]Evaluating detector classes:[/yellow] "
                + ", ".join(f"{name} ({class_id})" for name, class_id in zip(seen_det_names, seen_det_ids))
            )
        return remap, seen_det_ids, seen_det_names

    if not eval_classes_cfg:
        LOGGER.warning("class_mapping is set but eval_classes is missing in benchmark config. Skipping remap.")
        return None

    bench_name_to_id = {str(value): int(key) for key, value in eval_classes_cfg.items()}

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
    LOGGER.info(f"[cyan]Class mapping ({_escape_markup(str(benchmark_name))}{_escape_markup(model_label)}):[/cyan]")
    for benchmark_class_name, detector_class_name in class_mapping.items():
        benchmark_class_name = str(benchmark_class_name)
        detector_class_name = str(detector_class_name)
        if benchmark_class_name in bench_name_to_id and detector_class_name in det_name_to_id:
            LOGGER.info(
                f"  [blue]{_escape_markup(benchmark_class_name):<22}[/blue] -> "
                f"[cyan]{_escape_markup(detector_class_name)}[/cyan]"
            )
    LOGGER.info(
        "  [cyan]GT class IDs remapped:[/cyan] "
        + ", ".join(f"{bench_id}->{remap[bench_id]}" for bench_id in sorted(remap))
    )
    LOGGER.info(
        "  [cyan]Evaluating detector classes:[/cyan] "
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
        except Exception as exc:
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
    bridge_root = args.exp_dir / "motmetrics_gt"
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




__all__ = [
    "COCO_CLASSES",
    "_ordered_benchmark_eval_class_names",
    "apply_gt_class_remap",
    "build_gt_class_remap",
    "prepare_aabb_eval_gt",
    "resolve_eval_box_type",
    "resolve_obb_class_ids_to_eval",
    "resolve_obb_classes_to_eval",
    "resolve_obb_eval_class_pairs",
]
