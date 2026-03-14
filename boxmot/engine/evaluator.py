# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import cv2
import os
import shutil
import torch
import sys
import concurrent.futures
from contextlib import nullcontext

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import BENCHMARK_CONFIGS, NUM_THREADS, ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, TRACKEVAL
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.misc import increment_path, prompt_overwrite, resolve_model_path
from boxmot.utils.timing import TimingStats, wrap_tracker_reid
from typing import Optional, List, Dict, Generator, Union

from boxmot.utils.dataloaders.dataset import MOTDataset
from boxmot.postprocessing.gsi import gsi

from boxmot.engine.inference import DetectorReIDPipeline, prepare_detections
from boxmot.detectors import default_imgsz, default_conf
from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    ensure_benchmark_detector_model,
    get_benchmark_detector_cfg,
    load_benchmark_cfg,
    resolve_required_yolo_model,
    should_use_benchmark_detector,
)
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format, write_mot_results, xywha_to_corners
from boxmot.utils.download import download_trackeval

checker = RequirementsChecker()
checker.check_packages(('ultralytics', ))  # install


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
SUMMARY_INT_COLUMNS = {"IDSW", "IDs"}
SUMMARY_AGGREGATE_LABELS = {
    "cls_comb_det_av": "Class Avg (Det)",
    "cls_comb_cls_av": "Class Avg (Cls)",
    "HUMAN": "Human (Super)",
    "VEHICLE": "Vehicle (Super)",
    "BIKE": "Bike (Super)",
    "all": "All Classes",
}
def _load_benchmark_cfg(args: argparse.Namespace) -> dict:
    benchmark = getattr(args, "benchmark_id", None) or getattr(args, "dataset_id", None) or getattr(args, "benchmark", None)
    if not benchmark:
        return {}
    try:
        return load_benchmark_cfg(benchmark) or {}
    except FileNotFoundError:
        return {}


def _resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    eval_box_type = getattr(args, "eval_box_type", None)
    if eval_box_type:
        return str(eval_box_type).lower()

    benchmark_cfg = (bench_cfg or {}).get("benchmark", {})
    box_type = benchmark_cfg.get("box_type")
    return str(box_type).lower() if box_type else "aabb"


def _configure_benchmark_runtime(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Apply benchmark-driven detector defaults to the current args namespace."""
    benchmark_bundle = _load_benchmark_cfg(args)
    benchmark_cfg = benchmark_bundle.get("benchmark", {})

    use_benchmark_detector = should_use_benchmark_detector(args, benchmark_bundle)
    dataset_detector_cfg: dict = {}
    if use_benchmark_detector:
        dataset_detector_cfg = get_benchmark_detector_cfg(benchmark_bundle)
        if dataset_detector_cfg:
            args.dataset_detector_cfg = dataset_detector_cfg
    else:
        args.dataset_detector_cfg = None

    required_yolo_model = resolve_required_yolo_model(benchmark_bundle)
    if required_yolo_model and use_benchmark_detector:
        required_model = ensure_benchmark_detector_model(benchmark_bundle) or resolve_model_path(required_yolo_model)
        if args.yolo_model[0] != required_model:
            LOGGER.info(f"Using benchmark-default detector: {required_model}")
        args.yolo_model = [required_model]

    if benchmark_cfg.get("box_type") and not getattr(args, "eval_box_type", None):
        args.eval_box_type = str(benchmark_cfg["box_type"]).lower()

    if args.imgsz is None:
        if "imgsz" in dataset_detector_cfg:
            args.imgsz = list(dataset_detector_cfg["imgsz"])
        else:
            args.imgsz = default_imgsz(args.yolo_model[0])

    if args.conf is None:
        if "conf" in dataset_detector_cfg:
            args.conf = float(dataset_detector_cfg["conf"])
        else:
            args.conf = default_conf(args.yolo_model[0])

    return benchmark_bundle, benchmark_cfg, dataset_detector_cfg


def build_detection_class_remap(
    bench_cfg: dict,
    det_cfg: Optional[dict],
    benchmark_name: str = "",
    model_stem: str = "",
) -> dict[int, int]:
    """Build a detector-class remap for OBB benchmarks whose GT classes are zero-based."""
    eval_classes_cfg = bench_cfg.get("eval_classes") or {}
    det_classes = (det_cfg or {}).get("classes", {})
    class_mapping = bench_cfg.get("class_mapping") or {}

    if not eval_classes_cfg or not det_classes:
        return {}

    det_name_to_id = {str(v).lower(): int(k) for k, v in det_classes.items()}
    remap: dict[int, int] = {}

    if class_mapping:
        bench_name_to_zero_based_id = {str(v).lower(): int(k) - 1 for k, v in eval_classes_cfg.items()}
        for bench_name_i, det_name_i in class_mapping.items():
            bench_key = str(bench_name_i).lower()
            det_key = str(det_name_i).lower()
            if bench_key not in bench_name_to_zero_based_id or det_key not in det_name_to_id:
                continue
            remap[det_name_to_id[det_key]] = bench_name_to_zero_based_id[bench_key]
        return remap

    if len(eval_classes_cfg) > 1:
        LOGGER.warning(
            f"No class_mapping found for OBB benchmark '{benchmark_name}'"
            f" ({model_stem}). Using positional detector-class remap."
        )

    bench_ordered = sorted((int(k) - 1, str(v).lower()) for k, v in eval_classes_cfg.items())
    det_ordered = sorted((int(k), str(v).lower()) for k, v in det_classes.items())
    for (bench_id, _), (det_id, _) in zip(bench_ordered, det_ordered):
        remap[det_id] = bench_id
    return remap


def translate_detection_classes(
    dets: np.ndarray,
    embs: np.ndarray,
    remap: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Translate detector class IDs in-place-compatible arrays and drop unmapped rows."""
    if dets.size == 0 or not remap:
        return dets, embs

    class_col = 6 if dets.shape[1] >= 7 else 5
    class_ids = dets[:, class_col].astype(int)
    keep_mask = np.isin(class_ids, list(remap.keys()))

    translated_dets = dets[keep_mask].copy()
    translated_embs = embs[keep_mask].copy() if embs.size else embs
    translated_classes = translated_dets[:, class_col].astype(int)
    translated_dets[:, class_col] = np.asarray([remap[cid] for cid in translated_classes], dtype=np.float32)
    return translated_dets, translated_embs


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
) -> Optional[tuple]:
    """Build a GT class-ID remapping so that gt_temp.txt class IDs match tracker output.

    The tracker writes MOT-format files using ``det_class_id + 1`` (see
    ``mot_utils.convert_to_mot_format``).  This function produces a remap dict
    ``{bench_gt_id: det_id + 1}`` together with the resulting detector class lists
    that should be passed to TrackEval.

    Four cases:
      1. No det_cfg, no class_mapping  → return None (evaluate as-is, current behaviour)
      2. No det_cfg, class_mapping set → LOGGER.error; return None
      3. det_cfg set, no class_mapping → positional auto-mapping with WARNING
      4. det_cfg set, class_mapping set → full semantic mapping

    Returns:
        ``(remap_dict, new_class_ids, new_class_names)`` or ``None``.
    """
    eval_classes_cfg = bench_cfg.get("eval_classes")  # {1: person, 2: bus_small, ...}
    class_mapping = bench_cfg.get("class_mapping")    # {bus_small: Bus, ...} or None

    if det_cfg is None:
        if class_mapping:
            LOGGER.error(
                "class_mapping is defined in the benchmark config but no detector class metadata was "
                f"found for model '{model_stem}'. "
                "Use the benchmark-default detector or remove class_mapping to use default evaluation."
            )
        # Cases 1 & 2: no remapping
        return None

    det_classes = det_cfg.get("classes", {})  # {0: Moto, 1: Car, ...}
    if not det_classes:
        LOGGER.warning(f"Detector config for '{model_stem}' has no 'classes' field. Skipping remap.")
        return None

    det_name_to_id: dict = {str(v): int(k) for k, v in det_classes.items()}

    if not class_mapping:
        # Case 3: positional auto-mapping

        # If there is only one class, don't log info
        remap_logging = len(eval_classes_cfg) > 1
        
        if remap_logging:
            LOGGER.warning(
                f"No class_mapping found for benchmark '{benchmark_name}'. "
                "Using positional auto-mapping: first N benchmark classes → first N detector classes."
            )
        bench_ordered = sorted((int(k), str(v)) for k, v in eval_classes_cfg.items())
        det_ordered   = sorted((int(k), str(v)) for k, v in det_classes.items())
        n = min(len(bench_ordered), len(det_ordered))

        remap: dict = {}
        seen_det_ids: list = []
        seen_det_names: list = []
        rows = []
        for i in range(n):
            bench_id, bench_name_i = bench_ordered[i]
            det_id,   det_name_i   = det_ordered[i]
            new_gt_id = det_id + 1
            remap[bench_id] = new_gt_id
            rows.append((bench_name_i, det_name_i))
            if new_gt_id not in seen_det_ids:
                seen_det_ids.append(new_gt_id)
                seen_det_names.append(det_name_i)

        if remap_logging:
            LOGGER.opt(colors=True).info("<yellow>Auto class mapping (positional):</yellow>")
            for bench_name_i, det_name_i in rows:
                LOGGER.opt(colors=True).info(f"  <yellow>{bench_name_i:<22}</yellow> → <cyan>{det_name_i}</cyan>")
            LOGGER.opt(colors=True).info(
                f"  <yellow>GT class IDs remapped:</yellow> "
                + ", ".join(f"{b}→{remap[b]}" for b in sorted(remap))
            )
            LOGGER.opt(colors=True).info(
                "  <yellow>Evaluating detector classes:</yellow> "
                + ", ".join(f"{n} ({i})" for n, i in zip(seen_det_names, seen_det_ids))
            )
        return remap, seen_det_ids, seen_det_names

    # Case 4: full semantic mapping
    if not eval_classes_cfg:
        LOGGER.warning("class_mapping is set but eval_classes is missing in benchmark config. Skipping remap.")
        return None

    bench_name_to_id: dict = {str(v): int(k) for k, v in eval_classes_cfg.items()}

    remap = {}
    det_classes_used: dict = {}  # det_name → (det_id + 1)
    skipped = []
    for bname, dname in class_mapping.items():
        bname, dname = str(bname), str(dname)
        if bname not in bench_name_to_id:
            skipped.append(f"benchmark class '{bname}' not in eval_classes")
            continue
        if dname not in det_name_to_id:
            skipped.append(f"detector class '{dname}' not in detector config")
            continue
        bench_id = bench_name_to_id[bname]
        det_id   = det_name_to_id[dname]
        remap[bench_id] = det_id + 1
        det_classes_used[dname] = det_id + 1

    if skipped:
        for msg in skipped:
            LOGGER.warning(f"class_mapping: skipping — {msg}")

    if not remap:
        LOGGER.warning("class_mapping produced no valid entries. Skipping remap.")
        return None

    # Sort by detector class ID
    new_entries = sorted(det_classes_used.items(), key=lambda x: x[1])
    new_class_ids   = [nid  for _, nid  in new_entries]
    new_class_names = [name for name, _ in new_entries]

    model_label = f" → {model_stem}" if model_stem else ""
    LOGGER.opt(colors=True).info(
        f"<cyan>Class mapping ({benchmark_name}{model_label}):</cyan>"
    )
    for bname, dname in class_mapping.items():
        bname, dname = str(bname), str(dname)
        if bname in bench_name_to_id and dname in det_name_to_id:
            LOGGER.opt(colors=True).info(
                f"  <blue>{bname:<22}</blue> → <cyan>{dname}</cyan>"
            )
    LOGGER.opt(colors=True).info(
        "  <cyan>GT class IDs remapped:</cyan> "
        + ", ".join(f"{b}→{remap[b]}" for b in sorted(remap))
    )
    LOGGER.opt(colors=True).info(
        "  <cyan>Evaluating detector classes:</cyan> "
        + ", ".join(f"{n} ({i})" for n, i in zip(new_class_names, new_class_ids))
    )
    return remap, new_class_ids, new_class_names


def apply_gt_class_remap(
    source: Path,
    remap: dict,
    distractor_ids: Optional[List[int]] = None,
) -> None:
    """Rewrite every gt_temp.txt under *source* using *remap*.

    Column 7 (0-indexed) of the MOTChallenge gt format holds the class ID.
    Rows whose class ID is in *remap* are remapped; rows in *distractor_ids*
    are left untouched; all other rows are removed.

    Args:
        source: Root sequence directory (each seq has a ``gt/gt_temp.txt``).
        remap: ``{old_class_id: new_class_id}`` mapping.
        distractor_ids: Class IDs that should be kept but not remapped (e.g. ignore regions).
    """
    distractor_set = set(distractor_ids or [])
    keep_ids = set(remap.keys()) | distractor_set

    gt_files = list(source.glob("*/gt/gt_temp.txt"))
    if not gt_files:
        LOGGER.warning(f"apply_gt_class_remap: no gt_temp.txt files found under {source}")
        return

    for gt_file in gt_files:
        try:
            data = np.loadtxt(gt_file, delimiter=',')
        except Exception as e:
            LOGGER.warning(f"apply_gt_class_remap: could not read {gt_file}: {e}")
            continue

        if data.size == 0:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        class_col = data[:, 7].astype(int)
        mask = np.isin(class_col, list(keep_ids))
        data = data[mask]

        if data.size == 0:
            np.savetxt(gt_file, data, delimiter=',')
            continue

        # Remap eval-class rows; leave distractor rows as-is
        class_col = data[:, 7].astype(int)
        for old_id, new_id in remap.items():
            data[class_col == old_id, 7] = new_id

        np.savetxt(gt_file, data, delimiter=',', fmt="%g")


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


def _prepare_aabb_eval_gt(
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


def eval_init(args,
              trackeval_dest: Path = TRACKEVAL,
              branch: str = "main",
              overwrite: bool = False
    ) -> None:
    """
    Common initialization: download TrackEval and (if needed) the MOT-challenge
    data for benchmark runs, then canonicalize args.source.
    Modifies args in place.
    """
    # 1) download the TrackEval code
    download_trackeval(dest=trackeval_dest, branch=branch, overwrite=overwrite)

    # 2) if a benchmark YAML was provided via args.data, download and rewire args.source/split
    apply_benchmark_config(args, overwrite=overwrite)

    # 3) finally, make source an absolute Path everywhere
    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)


def _match_header_class_name(raw_name: str, known_classes: Optional[list[str]] = None) -> str:
    """Resolve a TrackEval header suffix to a class name, preserving hyphenated names."""
    if known_classes:
        exact_matches = [name for name in known_classes if raw_name.endswith(name)]
        if exact_matches:
            return max(exact_matches, key=len)

        normalized = raw_name.lower()
        folded_matches = [name for name in known_classes if normalized.endswith(name.lower())]
        if folded_matches:
            return max(folded_matches, key=len)

    if "-" in raw_name:
        return raw_name.split("-")[-1]
    return raw_name or "default"


def parse_mot_results(results: str, seq_names=None, known_classes: Optional[list[str]] = None) -> dict:
    """
    Extracts COMBINED HOTA, MOTA, IDF1, AssA, AssRe, IDSW, and IDs from MOTChallenge evaluation output.
    Returns a dictionary keyed by class name.

    Args:
        results: Raw stdout string from TrackEval subprocess.
        seq_names: Optional collection of known sequence names.  When provided,
            longest-prefix matching is used to correctly split sequence-name from
            metric values even when the name exceeds TrackEval's 35-char column
            (which causes the name to run directly into the first value with no
            whitespace separator).
        known_classes: Optional list of expected class/group names. When provided,
            longest-suffix matching is used to preserve hyphenated class names
            such as ``awning-bike``.
    """
    metric_specs = {
        'HOTA':   ('HOTA:',      {'HOTA': 0, 'AssA': 2, 'AssRe': 5}),
        'MOTA':   ('CLEAR:',     {'MOTA': 0, 'IDSW': 12}),
        'IDF1':   ('Identity:',  {'IDF1': 0}),
        'IDs':    ('Count:',     {'IDs': 2}),
    }

    int_fields = {'IDSW', 'IDs'}
    parsed_results = {}

    # Pre-sort known names longest-first so the first match is the correct one
    sorted_names = sorted(seq_names, key=len, reverse=True) if seq_names else None

    lines = results.splitlines()
    current_class = None
    current_metric_type = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for header lines
        is_header = False
        for metric_name, (prefix, _) in metric_specs.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                header_token = "Dets" if metric_name == "IDs" else metric_name

                # Formats seen in practice:
                #   "HOTA: tracker-classHOTA ..."
                #   "HOTA: tracker-class HOTA ..."
                content = line[len(prefix):].strip()
                tokens = content.split()
                if tokens:
                    first_word = tokens[0]
                    if len(tokens) > 1 and tokens[1] == header_token:
                        tracker_class = first_word
                    elif first_word.endswith(header_token):
                        tracker_class = first_word[:-len(header_token)]
                    else:
                        tracker_class = first_word

                    current_class = _match_header_class_name(tracker_class, known_classes)

                    if current_class not in parsed_results:
                        parsed_results[current_class] = {"per_sequence": {}}
                break

        if is_header:
            continue

        # Check for data rows (COMBINED or sequence names)
        if current_class and current_metric_type:
            _, field_map = metric_specs[current_metric_type]

            # Resolve row name and the remaining value tokens.
            # TrackEval formats the name column as %-35s; names longer than 35
            # chars overflow directly into the first value with no whitespace.
            # When we know the valid names we use longest-prefix matching to
            # correctly split the line regardless of name length.
            if line.startswith('COMBINED'):
                fields = line.split()
                row_name = 'COMBINED'
                values = fields[1:]
            elif sorted_names is not None:
                row_name = None
                for name in sorted_names:
                    if line.startswith(name):
                        row_name = name
                        values = line[len(name):].split()
                        break
                if row_name is None:
                    continue  # unrecognised row, skip
            else:
                fields = line.split()
                if len(fields) < 2:
                    continue
                row_name = fields[0]
                values = fields[1:]

            if not values:
                continue

            if row_name == 'COMBINED':
                for key, idx in field_map.items():
                    if idx < len(values):
                        val = values[idx]
                        parsed_results[current_class][key] = max(0, int(val) if key in int_fields else float(val))
            else:
                if row_name not in parsed_results[current_class]['per_sequence']:
                    parsed_results[current_class]['per_sequence'][row_name] = {}
                for key, idx in field_map.items():
                    if idx < len(values):
                        val = values[idx]
                        parsed_results[current_class]['per_sequence'][row_name][key] = max(0, int(val) if key in int_fields else float(val))

    return parsed_results


def _filter_obb_trackeval_results(
    parsed_results: dict,
    args: argparse.Namespace,
    bench_cfg: dict,
) -> tuple[dict, bool]:
    """Keep selected OBB classes and append aggregate MMOT rows when relevant."""
    if not parsed_results:
        return parsed_results, False

    selected_classes = resolve_obb_classes_to_eval(args, bench_cfg)
    ordered: dict = {}

    for cls_name in selected_classes:
        actual_key = cls_name if cls_name in parsed_results else next(
            (key for key in parsed_results if key.lower() == cls_name.lower()),
            None,
        )
        if actual_key is not None:
            ordered[actual_key] = parsed_results[actual_key]

    if len(ordered) > 1:
        for name in ["cls_comb_det_av", "cls_comb_cls_av", "HUMAN", "VEHICLE", "BIKE", "all"]:
            if name in parsed_results and name not in ordered:
                ordered[name] = parsed_results[name]

    if ordered:
        return ordered, len(selected_classes) == 1 and len(ordered) == 1

    preferred_order = ["cls_comb_det_av", "cls_comb_cls_av", "HUMAN", "VEHICLE", "BIKE", "all"]
    fallback = {name: parsed_results[name] for name in preferred_order if name in parsed_results}
    if fallback:
        return fallback, "cls_comb_det_av" in fallback and len(fallback) == 1

    return parsed_results, len(parsed_results) == 1


def _display_summary_name(name: str) -> str:
    """Return a human-readable label for a parsed summary key."""
    return SUMMARY_AGGREGATE_LABELS.get(name, name)


def _format_summary_values(metrics: dict) -> list[str]:
    """Format the subset of TrackEval metrics shown in the console summary."""
    values: list[str] = []
    for key in SUMMARY_COLUMNS:
        value = metrics.get(key, 0)
        if key in SUMMARY_INT_COLUMNS:
            values.append(f"{int(value):>10}")
        else:
            values.append(f"{float(value):>10.2f}")
    return values


def _summary_sort_keys(parsed_results: dict, args: argparse.Namespace, cfg: dict) -> tuple[list[str], list[str]]:
    """Split parsed results into primary class rows and aggregate rows."""
    if not parsed_results:
        return [], []

    eval_box_type = _resolve_eval_box_type(args, cfg)
    if eval_box_type != "obb":
        return list(parsed_results.keys()), []

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    primary_keys: list[str] = []
    seen: set[str] = set()
    for cls_name in resolve_obb_classes_to_eval(args, bench_cfg):
        actual_key = cls_name if cls_name in parsed_results else next(
            (key for key in parsed_results if key.lower() == cls_name.lower()),
            None,
        )
        if actual_key is not None and actual_key not in seen:
            primary_keys.append(actual_key)
            seen.add(actual_key)

    aggregate_keys = [key for key in parsed_results if key not in seen]
    if not primary_keys:
        return aggregate_keys, []
    return primary_keys, aggregate_keys


def _known_trackeval_class_names(args: argparse.Namespace, cfg: dict) -> list[str]:
    """Return expected class/group names for TrackEval header parsing."""
    known: list[str] = []

    if getattr(args, "remapped_class_names", None):
        known.extend([str(name) for name in args.remapped_class_names])

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    if isinstance(eval_classes_cfg, dict):
        known.extend([str(name) for _, name in sorted(eval_classes_cfg.items(), key=lambda kv: int(kv[0]))])

    known.extend(["cls_comb_cls_av", "cls_comb_det_av", "HUMAN", "VEHICLE", "BIKE", "all"])

    deduped: list[str] = []
    seen: set[str] = set()
    for name in known:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _print_summary_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, dict, bool]],
    total_w: int,
    name_w: int,
) -> None:
    """Render a compact metric table to the logger."""
    if not rows:
        return

    header_values = [name_header, *SUMMARY_COLUMNS]
    header_fmt = f"{{:<{name_w}}} " + " ".join(["{:>10}"] * len(SUMMARY_COLUMNS))
    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold><cyan>{title:^{total_w}}</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>{header_fmt.format(*header_values)}</bold>")
    LOGGER.opt(colors=True).info("<blue>" + "-" * total_w + "</blue>")

    for row_name, metrics, highlight in rows:
        name_col = f"{row_name:<{name_w}}"
        vals_str = " ".join(_format_summary_values(metrics))
        if highlight:
            LOGGER.opt(colors=True).info(f"<bold>{name_col} <cyan>{vals_str}</cyan></bold>")
        else:
            LOGGER.opt(colors=True).info(f"{name_col} <blue>{vals_str}</blue>")

    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")


# ---------------------------
# Batched det+emb generation
# ---------------------------
def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _list_sequence_frames(img_dir: Path) -> list[Path]:
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))


def _sequence_name_from_img_dir(img_dir: Path) -> str:
    return img_dir.parent.name if img_dir.name == "img1" else img_dir.name


def _read_image_cv2(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return im


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths = []
    seq_info: dict[str, int] = {}
    for seq_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        img_dir = _sequence_img_dir(seq_dir)
        frame_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not frame_files:
            continue
        seq_paths.append(img_dir)
        seq_info[seq_dir.name] = len(frame_files)
    return seq_paths, seq_info


def _clear_device_cache(device: str) -> None:
    dev_lower = str(device).lower()
    if dev_lower.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _count_data_lines(path: Path, skip_header: bool = False) -> int:
    """Count non-header lines in a txt file, tolerating missing files."""
    try:
        with open(path, "r") as fh:
            if skip_header:
                return sum(1 for line in fh if not line.startswith("#"))
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


def _count_embedding_rows(path: Path) -> int:
    """Count rows in a text embedding cache."""
    return _count_data_lines(path, skip_header=True)


def _max_frame_id(path: Path) -> int:
    """Return the maximum frame id (first column) in a dets txt, skipping headers."""
    max_f = 0
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    f_val = int(float(parts[0]))
                except Exception:
                    continue
                if f_val > max_f:
                    max_f = f_val
    except FileNotFoundError:
        return 0
    return max_f


def _saved_detection_column_count(path: Path) -> int:
    """Return the number of columns in the first non-header detections row."""
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return len(line.replace(",", " ").split())
    except FileNotFoundError:
        return 0
    return 0


def _serialize_eval_detections(dets: np.ndarray, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Serialize detector output for cache files and return the boxes used for ReID crops."""
    if dets.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    if dets.shape[1] == 7:
        frame_col = np.full((dets.shape[0], 1), float(frame_id), dtype=np.float32)
        exported = np.concatenate([frame_col, dets], axis=1).astype(np.float32)
        reid_boxes = dets[:, :5].astype(np.float32)
        return exported, reid_boxes

    if dets.shape[1] == 6:
        frame_col = np.full((dets.shape[0], 1), float(frame_id), dtype=np.float32)
        boxes = dets[:, :4].astype(np.float32)
        confs = dets[:, 4:5].astype(np.float32)
        clss = dets[:, 5:6].astype(np.float32)
        exported = np.concatenate([frame_col, boxes, confs, clss], axis=1).astype(np.float32)
        return exported, boxes

    raise ValueError(f"Unsupported detection shape for serialization: {dets.shape}")


@torch.inference_mode()
def generate_dets_embs_batched(args: argparse.Namespace, y: Path, source_root: Path, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Generate detections and embeddings in batches for evaluation.
    
    Args:
        args: CLI arguments.
        y: Path to YOLO model weights.
        source_root: Root path containing sequence folders.
        timing_stats: Optional TimingStats for timing instrumentation.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    batch_size = int(getattr(args, "batch_size", 16))
    read_threads_val = getattr(args, "read_threads", None)
    if read_threads_val is None:
        read_threads_val = min(8, (os.cpu_count() or 8))
    read_threads = int(read_threads_val)
    auto_batch = bool(getattr(args, "auto_batch", True))
    resume = bool(getattr(args, "resume", True))

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    expected_det_cols = 8 if str(getattr(args, "eval_box_type", "")).lower() == "obb" else 7

    benchmark = getattr(args, "benchmark", None)
    dets_base = Path(args.project) / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    dets_folder = dets_base / y.stem / "dets"
    embs_root = dets_base / y.stem / "embs"

    mot_folder_paths = sorted([p for p in Path(source_root).iterdir() if p.is_dir()])

    seq_states = {}
    det_fhs = {}
    emb_fhs = {r.stem: {} for r in args.reid_model}
    total_frames = 0
    initial_done = 0

    for seq_dir in mot_folder_paths:
        img_dir = _sequence_img_dir(seq_dir)
        frames = _list_sequence_frames(img_dir)
        if not frames:
            continue

        seq_name = _sequence_name_from_img_dir(img_dir)

        dets_path = dets_folder / f"{seq_name}.txt"
        processed = 0

        emb_paths = {}
        any_emb_cached = False
        for r in args.reid_model:
            ep_txt = embs_root / r.stem / f"{seq_name}.txt"
            emb_paths[r.stem] = ep_txt
            if ep_txt.exists():
                any_emb_cached = True

        expected_files = False
        rows_match = False
        det_rows = 0
        det_max_frame = 0
        emb_rows: dict[str, int] = {}

        if resume:
            det_rows = _count_data_lines(dets_path, skip_header=True)
            det_max_frame = _max_frame_id(dets_path)
            det_col_count = _saved_detection_column_count(dets_path)
            emb_rows = {
                stem: _count_embedding_rows(ep)
                for stem, ep in emb_paths.items()
            }
            expected_files = dets_path.exists() and all(ep.exists() for ep in emb_paths.values())
            rows_match = len(set([det_rows, *emb_rows.values()])) == 1 if expected_files else False
            schema_match = det_col_count in (0, expected_det_cols)
            if expected_files and not schema_match:
                LOGGER.warning(
                    f"Cached detection schema mismatch for {seq_name}: "
                    f"found {det_col_count} columns, expected {expected_det_cols}. Resetting cached data."
                )
                for p in [dets_path, *emb_paths.values()]:
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                processed = 0
                expected_files = False
                rows_match = False
            elif expected_files and rows_match and det_rows > 0:
                processed = min(det_max_frame, len(frames))
            elif expected_files and not rows_match:
                LOGGER.warning(
                    f"Cached det/emb rows mismatch for {seq_name}; resetting cached data."
                )
                for p in [dets_path, *emb_paths.values()]:
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                processed = 0

        if resume and processed >= len(frames) and dets_path.exists() and all(ep.exists() for ep in emb_paths.values()):
            if expected_files and rows_match and det_rows:
                LOGGER.info(
                    f"Skipping {seq_name} (cached complete; {processed}/{len(frames)} frames)."
                )
            else:
                LOGGER.info(f"Skipping {seq_name} (resume: already complete).")
            initial_done += len(frames)
            continue

        if resume and 0 < processed < len(frames):
            LOGGER.info(f"Resuming {seq_name}: cached {processed}/{len(frames)} frames.")

        if (not resume) and dets_path.exists() and any_emb_cached:
            if not prompt_overwrite('Detections and Embeddings', dets_path, args.ci):
                LOGGER.debug(f"Skipping {seq_name} (cached).")
                continue

        dets_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'ab' if (resume and dets_path.exists()) else 'wb'
        det_fhs[seq_name] = open(dets_path, mode, buffering=1024 * 1024)
        if mode == 'wb' or dets_path.stat().st_size == 0:
            np.savetxt(det_fhs[seq_name], [], fmt='%f', header=str(img_dir))

        for r in args.reid_model:
            ep = emb_paths[r.stem]
            ep.parent.mkdir(parents=True, exist_ok=True)
            emb_mode = 'a' if (resume and ep.exists()) else 'w'
            emb_fhs[r.stem][seq_name] = open(ep, emb_mode, buffering=1024 * 1024)

        seq_states[seq_name] = {"frames": frames, "i": processed, "img_dir": img_dir}
        total_frames += len(frames)
        initial_done += processed

    if not seq_states:
        LOGGER.info("No sequences to process (all cached or no images).")
        return

    # Use unified DetectorReIDPipeline with timing for both detection and ReID
    pipeline = DetectorReIDPipeline(
        detector_path=y,
        reid_paths=args.reid_model,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half,
        timing_stats=timing_stats,
    )

    # Warmup the model
    pipeline.warmup()

    if auto_batch:
        batch_size = pipeline.autotune_batch_size(batch_size)
        args.batch_size = batch_size

    use_cuda = str(args.device).startswith("cuda")
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_cuda and getattr(args, "half", False))
        else nullcontext()
    )

    seq_names = list(seq_states.keys())
    rr = 0

    pbar = tqdm(total=total_frames, desc=f"Batched YOLO+ReID ({y.name}, bs={batch_size})", unit="frame")
    reid_pbar = tqdm(total=0, desc="ReID embeddings", unit="det", dynamic_ncols=True)
    if initial_done:
        pbar.update(initial_done)

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=read_threads) as pool, amp_ctx:
            alive = True
            while alive:
                batch_items = []
                tried = 0
                while len(batch_items) < batch_size and tried < len(seq_names):
                    seq_name = seq_names[rr % len(seq_names)]
                    rr += 1
                    tried += 1

                    st = seq_states[seq_name]
                    if st["i"] >= len(st["frames"]):
                        continue
                    frame_id = st["i"] + 1
                    img_path = st["frames"][st["i"]]
                    st["i"] += 1
                    batch_items.append((seq_name, frame_id, img_path))

                if not batch_items:
                    alive = False
                    break

                futures = [pool.submit(_read_image_cv2, p) for _, _, p in batch_items]
                imgs = [f.result() for f in futures]

                yolo_results = None
                while True:
                    try:
                        yolo_results = pipeline.predict_batch(
                            images=imgs[:batch_size],
                            conf=0.01,  # Always collect all detections; conf filtering happens at tracking stage
                            iou=args.iou,
                            agnostic_nms=args.agnostic_nms,
                            classes=args.classes,
                        )
                        break
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        if batch_size == 1:
                            raise

                        _clear_device_cache(args.device)

                        for seq_name, _, _ in batch_items:
                            seq_states[seq_name]["i"] -= 1

                        new_bs = max(1, batch_size // 2)
                        LOGGER.warning(f"YOLO predict OOM at batch size {batch_size}; retrying with {new_bs}.")
                        batch_size = new_bs
                        args.batch_size = batch_size
                        yolo_results = None
                        break

                if yolo_results is None:
                    continue

                det_counts = [len(r.dets) for r in yolo_results]
                emb_dims: dict[str, int] = {}
                LOGGER.info(
                    f"YOLO batch frames={len(batch_items)} | dets/frame={det_counts} | total_dets={sum(det_counts)}"
                )
                touched: set[str] = set()

                for (seq_name, frame_id, _), r, img in zip(batch_items, yolo_results, imgs):
                    dets = prepare_detections(r, img)

                    if len(dets) == 0:
                        if timing_stats:
                            timing_stats.frames += 1
                        pbar.update(1)
                        continue

                    dets_np, det_boxes_np = _serialize_eval_detections(dets, frame_id)
                    np.savetxt(det_fhs[seq_name], dets_np, fmt="%f")

                    all_embs = pipeline.get_all_reid_features(det_boxes_np, img)
                    for reid_name, embs in all_embs.items():
                        if embs.shape[0] != det_boxes_np.shape[0]:
                            raise RuntimeError(
                                f"Embedding count mismatch: dets={det_boxes_np.shape[0]} embs={embs.shape[0]}"
                            )
                        if embs.ndim >= 2 and reid_name not in emb_dims:
                            emb_dims[reid_name] = embs.shape[1]
                        np.savetxt(emb_fhs[reid_name][seq_name], embs.astype(np.float32), fmt="%f")

                    reid_pbar.update(det_boxes_np.shape[0])

                    if timing_stats:
                        timing_stats.frames += 1

                    pbar.update(1)
                    touched.add(seq_name)

                if emb_dims:
                    LOGGER.info(
                        "ReID embedding dims per model: "
                        + ", ".join([f"{k}={v}" for k, v in emb_dims.items()])
                    )
                else:
                    LOGGER.info("ReID embedding dims per model: n/a (no detections)")

                for seq_name in touched:
                    try:
                        det_fhs[seq_name].flush()
                        for per_reid in emb_fhs.values():
                            if seq_name in per_reid:
                                per_reid[seq_name].flush()
                    except Exception:
                        pass

                del yolo_results, imgs
                _clear_device_cache(args.device)

    finally:
        pbar.close()
        reid_pbar.close()
        for fh in det_fhs.values():
            try:
                fh.close()
            except Exception:
                pass
        for per_reid in emb_fhs.values():
            for fh in per_reid.values():
                try:
                    fh.close()
                except Exception:
                    pass


def run_generate_dets_embs(args: argparse.Namespace, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Generate detections and embeddings for all sequences.
    
    Args:
        args: CLI arguments.
        timing_stats: Optional TimingStats for timing instrumentation.
    """
    if getattr(args, "data", None) and getattr(args, "source", None) is None:
        apply_benchmark_config(args, overwrite=False)

    _configure_benchmark_runtime(args)
    source_root = Path(args.source)

    args.batch_size = int(getattr(args, "batch_size", 16))
    if getattr(args, "read_threads", None) is None:
        args.read_threads = min(8, (os.cpu_count() or 8))
    if not hasattr(args, "auto_batch"):
        args.auto_batch = True
    if not hasattr(args, "resume"):
        args.resume = True

    for y in args.yolo_model:
        LOGGER.info(f"Generating dets+embs (batched single-process): {y.name}")
        generate_dets_embs_batched(args, y, source_root, timing_stats=timing_stats)

def build_dataset_eval_settings(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> dict:
    """Derive benchmark-specific evaluation settings (classes, ids, distractors, gt path format).

    This centralizes logic for MOT-style datasets and non-MOT layouts such as VisDrone.
    """

    cfg = {}
    try:
        benchmark_id = getattr(args, "benchmark_id", None) or getattr(args, "dataset_id", None) or getattr(args, "benchmark", None)
        if benchmark_id:
            cfg = load_benchmark_cfg(benchmark_id)
    except FileNotFoundError:
        cfg = {}
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Error loading benchmark config: {e}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    distractor_cfg = bench_cfg.get("distractor_classes") if isinstance(bench_cfg, dict) else None

    # GT path format (VisDrone uses flat txt under annotations)
    gt_loc_format = "{gt_folder}/{seq}/gt/gt_temp.txt"
    is_visdrone = "visdrone" in getattr(args, "benchmark", "").lower() or "visdrone" in str(gt_folder).lower()
    if is_visdrone:
        gt_loc_format = "{gt_folder}/{seq}.txt"

    benchmark_name = getattr(args, "benchmark", "")

    # If GT files were already remapped to detector class IDs, use those directly.
    if getattr(args, "remapped_class_ids", None):
        distractor_ids: list[int] = []
        if isinstance(distractor_cfg, dict) and len(distractor_cfg) > 0:
            distractor_ids = [int(k) for k in distractor_cfg.keys()]
        return {
            "classes_to_eval": args.remapped_class_names,
            "class_ids": args.remapped_class_ids,
            "distractor_ids": distractor_ids,
            "gt_loc_format": gt_loc_format,
            "benchmark_name": benchmark_name,
            "seq_info": seq_info,
        }

    classes_to_eval = []
    class_ids = []

    # Filter classes by user provided classes
    if hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(i)] for i in class_indices]
        class_ids = [int(i) + 1 for i in class_indices]

    # Match classes by benchmark config
    if isinstance(eval_classes_cfg, dict) and len(eval_classes_cfg) > 0:
        ordered = sorted(((int(k), v) for k, v in eval_classes_cfg.items()), key=lambda kv: kv[0])
        if class_ids:
            class_ids = [k for k, _ in ordered if class_ids and k in class_ids]
            classes_to_eval = [v for k, v in ordered if class_ids and k in class_ids]
        else:
            class_ids = [k for k, _ in ordered]
            classes_to_eval = [v for k, v in ordered]

    # Default classes
    if not classes_to_eval:
        classes_to_eval = ["person"]
    if not class_ids:
        class_ids = [1]

    # Distractors
    distractor_ids: list[int] = []
    if isinstance(distractor_cfg, dict) and len(distractor_cfg) > 0:
        distractor_ids = [int(k) for k in distractor_cfg.keys()]

    # Remove duplicates while preserving order
    seen = set()
    pairs = []
    for name, cid in zip(classes_to_eval, class_ids):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, cid))
    classes_to_eval = [name for name, _ in pairs]
    class_ids = [cid for _, cid in pairs]

    return {
        "classes_to_eval": classes_to_eval,
        "class_ids": class_ids,
        "distractor_ids": distractor_ids,
        "gt_loc_format": gt_loc_format,
        "benchmark_name": benchmark_name,
        "seq_info": seq_info,
    }


def trackeval(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

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

    cmd_args = [
        sys.executable, ROOT / 'boxmot' / 'utils' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", benchmark_name,
        "--TRACKERS_FOLDER", str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL", args.exp_dir.name,
        "--SPLIT_TO_EVAL", args.split,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_LOC_FORMAT", gt_loc_format,
        "--CLASSES_TO_EVAL", *classes_to_eval,
        "--CLASS_IDS", *[str(i) for i in class_ids],
        "--DISTRACTOR_CLASS_IDS", *[str(i) for i in distractor_ids],
        "--SEQ_INFO", *seq_info_args
    ]

    p = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        LOGGER.warning(f"TrackEval stderr:\n{stderr}")
    return stdout


def trackeval_aabb(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """Compatibility wrapper for the existing AABB TrackEval path."""
    return trackeval(args, seq_paths, save_dir, gt_folder, metrics=metrics, seq_info=seq_info)


def _load_obb_gt_matrix(source: Path) -> np.ndarray:
    """Load OBB GT and normalize it into the MMOT TrackEval 13-column corner format."""
    data = np.loadtxt(source, delimiter=",")
    if data.size == 0:
        return np.empty((0, 13), dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 13:
        return data.astype(np.float32)
    elif data.shape[1] == 10:
        corners = xywha_to_corners(data[:, 2:7]).astype(np.float32)
        return np.concatenate([data[:, 0:2], corners, data[:, 7:10]], axis=1)

    raise ValueError(f"Unsupported OBB GT format in {source}: expected 10 or 13 columns, got {data.shape[1]}")


def _prepare_obb_eval_bridge(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> tuple[Path, Path]:
    """Create the flat GT/image layout expected by the OBB TrackEval adapter."""
    bridge_root = args.exp_dir / "trackeval_mmot_rgb"
    gt_bridge = bridge_root / "gt"
    img_bridge = bridge_root / "img"
    gt_bridge.mkdir(parents=True, exist_ok=True)
    img_bridge.mkdir(parents=True, exist_ok=True)

    for seq_name in seq_info:
        seq_dir = Path(args.source) / seq_name
        src_img_dir = (seq_dir / "img1" if (seq_dir / "img1").exists() else seq_dir).resolve()
        bridge_img_dir = img_bridge / seq_name
        if os.path.lexists(bridge_img_dir):
            if bridge_img_dir.is_symlink():
                bridge_img_dir.unlink()
        if not bridge_img_dir.exists():
            try:
                os.symlink(src_img_dir, bridge_img_dir, target_is_directory=True)
            except OSError:
                shutil.copytree(src_img_dir, bridge_img_dir, dirs_exist_ok=True)

        raw_candidates = [
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
                normalized_gt = _load_obb_gt_matrix(candidate)
                source_gt = candidate
                break
            except ValueError:
                continue
        if source_gt is None:
            raise FileNotFoundError(
                f"No OBB GT file found for sequence {seq_name}. "
                "Expected gt.txt/gt_temp.txt in corner format or legacy gt_obb*.txt."
            )
        gt_out = gt_bridge / f"{seq_name}.txt"
        gt_out.parent.mkdir(parents=True, exist_ok=True)
        if normalized_gt is None or normalized_gt.size == 0:
            gt_out.write_text("")
        else:
            np.savetxt(gt_out, normalized_gt, delimiter=",", fmt="%g")

    return gt_bridge, img_bridge


def trackeval_obb(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """Evaluate OBB tracking results via BoxMOT's custom OBB TrackEval runner."""
    del save_dir, seq_paths
    if not seq_info:
        raise ValueError("seq_info is required for OBB TrackEval")

    bench_cfg = _load_benchmark_cfg(args).get("benchmark", {})
    classes_to_eval = resolve_obb_classes_to_eval(args, bench_cfg)
    class_ids = resolve_obb_class_ids_to_eval(args, bench_cfg)
    gt_bridge, img_bridge = _prepare_obb_eval_bridge(args, gt_folder, seq_info)

    cmd_args = [
        sys.executable,
        "-m",
        "boxmot.utils.run_mmot_rgb",
        "--GT_FOLDER", str(gt_bridge),
        "--IMG_FOLDER", str(img_bridge),
        "--TRACKERS_FOLDER", str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL", args.exp_dir.name,
        "--TRACKER_SUB_FOLDER", "",
        "--OUTPUT_SUB_FOLDER", "",
        "--SPLIT_TO_EVAL", str(getattr(args, "split", "train")),
        "--METRICS", *metrics,
        "--PRINT_CONFIG", "False",
        "--PRINT_ONLY_COMBINED", "False",
        "--USE_PARALLEL", "False",
    ]
    if classes_to_eval:
        cmd_args.extend(["--CLASSES_TO_EVAL", *classes_to_eval])
    if class_ids:
        cmd_args.extend(["--CLASS_IDS", *[str(class_id) for class_id in class_ids]])

    p = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    stdout, stderr = p.communicate()
    if stderr:
        LOGGER.warning(f"OBB TrackEval stderr:\n{stderr}")
    return stdout



def process_sequence(seq_name: str,
                     mot_root: str,
                     project_root: str,
                     model_name: str,
                     reid_name: str,
                     tracking_method: str,
                     exp_folder: str,
                     target_fps: Optional[int],
                     device: str,
                     cfg_dict: Optional[Dict] = None,
                     dataset_name: Optional[str] = None,
                     conf_threshold: float = 0.0,
                     ):
    """
    Process a single sequence: run tracker on pre-computed detections/embeddings.
    
    Returns:
        Tuple of (seq_name, kept_frame_ids, timing_dict) where timing_dict contains
        'track_time_ms' and 'num_frames' for aggregating timing stats.
    """
    import time

    # Embeddings are pre-computed: tracker association (Kalman + Hungarian) is CPU-only.
    # Loading tracker's internal ReID model on GPU would waste VRAM across NUM_THREADS workers.
    tracker_device = select_device("cpu")
    tracker = create_tracker(
        tracker_type=tracking_method,
        tracker_config=TRACKER_CONFIGS / (tracking_method + ".yaml"),
        reid_weights=Path(reid_name + '.pt'),
        device=tracker_device,
        half=False,
        per_class=False,
        evolve_param_dict=cfg_dict,
    )

    # load with the user’s FPS
    # runs/dets_n_embs/<dataset_name>/ when dataset_name is set
    det_emb_root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        det_emb_root = det_emb_root / dataset_name
    dataset = MOTDataset(
        mot_root=mot_root,
        det_emb_root=str(det_emb_root),
        model_name=model_name,
        reid_name=reid_name,
        target_fps=target_fps
    )
    sequence = dataset.get_sequence(seq_name)

    all_tracks = []
    kept_frame_ids = []
    total_track_time_ms = 0.0
    num_frames = 0
    
    for frame in sequence:
        fid  = int(frame['frame_id'])
        dets = frame['dets']
        embs = frame['embs']
        img  = frame['img']

        kept_frame_ids.append(fid)
        num_frames += 1

        if dets.size and embs.size:
            # Filter by confidence threshold before passing to tracker.
            # Detections are saved with conf=0.01 (all detections); conf_threshold
            # comes from the detector config or CLI and is applied here.
            if conf_threshold > 0:
                conf_col = 5 if dets.shape[1] == 7 else 4
                mask = dets[:, conf_col] >= conf_threshold
                dets = dets[mask]
                embs = embs[mask]

        if dets.size and embs.size:
            if dets.shape[0] != embs.shape[0]:
                msg = (
                    f"Detection/embedding count mismatch for {seq_name} frame {fid}: "
                    f"dets={dets.shape[0]} embs={embs.shape[0]}"
                )
                LOGGER.error(msg)
                raise ValueError(msg)

            # Time the tracker update (association only, embeddings pre-computed)
            t0 = time.perf_counter()
            tracks = tracker.update(dets, img, embs)
            total_track_time_ms += (time.perf_counter() - t0) * 1000
            
            if tracks.size:
                if tracks.ndim == 1:
                    tracks = tracks.reshape(1, -1)
                if tracks.shape[1] >= 9:
                    all_tracks.append(convert_to_mmot_obb_format(tracks, fid))
                else:
                    all_tracks.append(convert_to_mot_format(tracks, fid))

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)
    
    timing_dict = {
        'track_time_ms': total_track_time_ms,
        'num_frames': num_frames,
    }
    return seq_name, kept_frame_ids, timing_dict


from boxmot.utils import configure_logging as _configure_logging

def _worker_init():
    # each spawned process needs its own sinks
    _configure_logging()

def run_generate_mot_results(args: argparse.Namespace, evolve_config: dict = None, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Run tracker on pre-computed detections/embeddings and generate MOT result files.
    
    Args:
        args: CLI arguments.
        evolve_config: Optional config dict for hyperparameter tuning.
        timing_stats: Optional TimingStats to record tracking/association time.
    """
    # Prepare experiment folder: runs/mot/<dataset_name>/model_reid_tracker when benchmark is set
    base = args.project / "mot"
    if getattr(args, "benchmark", None):
        base = base / args.benchmark
    base = base / f"{args.yolo_model[0].stem}_{args.reid_model[0].stem}_{args.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = exp_dir

    # Just collect sequence names by scanning directory names
    sequence_names = []
    for d in Path(args.source).iterdir():
        if not d.is_dir():
            continue
        img_dir = d / "img1" if (d / "img1").exists() else d
        if any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
            sequence_names.append(d.name)
    sequence_names.sort()

    # Build task arguments (include dataset_name for det_emb_root path)
    dataset_name = getattr(args, "benchmark", None)
    # conf_threshold comes from the detector config YAML (resolved in main() or by the caller).
    # Falls back to default_conf() if somehow not set — never silently disables filtering.
    conf_threshold = getattr(args, "conf", None)
    if conf_threshold is None:
        conf_threshold = default_conf(args.yolo_model[0])
    task_args = [
        (
            seq,
            str(args.source),
            str(args.project),
            args.yolo_model[0].stem,
            args.reid_model[0].stem,
            args.tracking_method,
            str(exp_dir),
            getattr(args, "fps", None),
            args.device,
            evolve_config,
            dataset_name,
            conf_threshold,
        )
        for seq in sequence_names
    ]

    seq_frame_nums = {}
    total_track_time_ms = 0.0
    total_track_frames = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
        futures = {
            executor.submit(process_sequence, *args): args[0] for args in task_args
        }

        for fut in concurrent.futures.as_completed(futures):
            seq = futures[fut]
            try:
                seq_name, kept_ids, timing_dict = fut.result()
                seq_frame_nums[seq_name] = kept_ids
                # Aggregate timing from worker process
                total_track_time_ms += timing_dict.get('track_time_ms', 0)
                total_track_frames += timing_dict.get('num_frames', 0)
            except Exception:
                LOGGER.exception(f"Error processing {seq}")

    args.seq_frame_nums = seq_frame_nums
    
    # Record aggregated tracking time in timing_stats
    if timing_stats is not None:
        timing_stats.totals['track'] += total_track_time_ms
        # Also update frame count if not already set (from batch mode)
        if timing_stats.frames == 0 and total_track_frames > 0:
            timing_stats.frames = total_track_frames
        # Log summary
        if total_track_frames > 0:
            avg_track = total_track_time_ms / total_track_frames
            LOGGER.opt(colors=True).info(
                f"<bold>Tracking:</bold> {total_track_frames} frames, "
                f"total: <cyan>{total_track_time_ms:.1f}ms</cyan>, "
                f"avg: <cyan>{avg_track:.2f}ms/frame</cyan>"
            )

    # Optional GSI postprocessing
    if getattr(args, "postprocessing", "none") == "gsi":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GSI postprocessing...")
        from boxmot.postprocessing.gsi import gsi
        gsi(mot_results_folder=exp_dir)

    elif getattr(args, "postprocessing", "none") == "gbrc":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GBRC postprocessing...")
        from boxmot.postprocessing.gbrc import gbrc
        gbrc(mot_results_folder=exp_dir)


def run_trackeval(args: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        args (Namespace): Parsed command line arguments.
        verbose (bool): Whether to print results summary. Default True.
    """
    seq_paths, seq_info = _collect_seq_info(args.source)
    annotations_dir = args.source.parent / "annotations"
    gt_folder = annotations_dir if annotations_dir.exists() else args.source

    if not seq_paths:
        raise ValueError(f"No sequences with images found under {args.source}")

    if annotations_dir.exists():
        for seq_name in list(seq_info.keys()):
            ann_file = annotations_dir / f"{seq_name}.txt"
            if not ann_file.exists():
                continue
            try:
                with open(ann_file, "r") as f:
                    max_frame = 0
                    for line in f:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except Exception:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference")
    # runs/<dataset_name>/<name> when benchmark is set
    if getattr(args, "benchmark", None):
        save_dir = Path(args.project) / args.benchmark / args.name
    else:
        save_dir = Path(args.project) / args.name

    cfg = _load_benchmark_cfg(args)
    if not cfg:
        # Try to load config from benchmark name first, then fallback to source parent name
        cfg_name = getattr(args, "benchmark_id", None) or getattr(args, "dataset_id", None) or getattr(args, 'benchmark', str(args.source.parent.name))
        try:
            cfg = load_benchmark_cfg(cfg_name)
        except FileNotFoundError:
            # If config not found, try to find it by checking if source path ends with a known config name.
            # This handles cases where source is a custom path.
            found = False
            for config_file in BENCHMARK_CONFIGS.glob("*.yaml"):
                if config_file.stem in str(args.source):
                    cfg = load_benchmark_cfg(config_file.stem)
                    found = True
                    break
            
            if not found:
                LOGGER.warning(f"Could not find benchmark config for {cfg_name}. Class filtering might be incorrect.")
                cfg = {}

    if _resolve_eval_box_type(args, cfg) == "obb":
        trackeval_results = trackeval_obb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    else:
        gt_folder = _prepare_aabb_eval_gt(args, gt_folder, seq_info)
        trackeval_results = trackeval_aabb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)

    parsed_results = parse_mot_results(
        trackeval_results,
        seq_names=set(seq_info.keys()),
        known_classes=_known_trackeval_class_names(args, cfg),
    )
    eval_box_type = _resolve_eval_box_type(args, cfg)

    # Filter parsed_results based on user provided classes (args.classes)
    single_class_mode = False

    if eval_box_type == "obb":
        parsed_results, single_class_mode = _filter_obb_trackeval_results(parsed_results, args, cfg.get("benchmark", {}))
    # Priority 0: GT was already remapped to detector classes — use those names directly.
    # TrackEval lowercases class names, so compare case-insensitively.
    elif getattr(args, "remapped_class_names", None):
        remapped_lower = {n.lower() for n in args.remapped_class_names}
        parsed_results = {k: v for k, v in parsed_results.items() if k.lower() in remapped_lower}
        if len(args.remapped_class_names) == 1:
            single_class_mode = True
    # Priority 1: Benchmark config classes (overrides user classes)
    elif "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = None

        if isinstance(bench_cfg, dict):
            if "eval_classes" in bench_cfg:
                bench_classes = [v for _, v in sorted(bench_cfg["eval_classes"].items(), key=lambda kv: int(kv[0]))]
            elif "classes" in bench_cfg:
                bench_classes = bench_cfg["classes"].split()

        if bench_classes:
            parsed_results = {k: v for k, v in parsed_results.items() if k in bench_classes}
            if len(bench_classes) == 1:
                single_class_mode = True
    # Priority 2: User provided classes
    elif hasattr(args, 'classes') and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        user_classes = [COCO_CLASSES[int(i)] for i in class_indices]
        parsed_results = {k: v for k, v in parsed_results.items() if k in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    if single_class_mode and len(parsed_results) > 0:
        final_results = list(parsed_results.values())[0]
    else:
        final_results = parsed_results
    
    # Print results summary
    if verbose:
        LOGGER.info("")
        primary_keys, aggregate_keys = _summary_sort_keys(parsed_results, args, cfg)
        single_sequence = len(seq_info) == 1

        all_names = [
            _display_summary_name(name)
            for name in [*primary_keys, *aggregate_keys]
        ]
        for class_metrics in parsed_results.values():
            all_names.extend(class_metrics.get("per_sequence", {}).keys())
        all_names.extend([f"COMBINED ({_display_summary_name(name)})" for name in primary_keys])

        nw = max(18, max((len(name) for name in all_names), default=18) + 2)
        total_w = nw + 1 + 76

        LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")
        LOGGER.opt(colors=True).info(f"<bold><cyan>{'📊 RESULTS SUMMARY':^{total_w}}</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")

        if len(primary_keys) > 1:
            class_rows = [
                (_display_summary_name(name), parsed_results[name], False)
                for name in primary_keys
            ]
            _print_summary_table("Per-Class Combined Metrics", "Class", class_rows, total_w, nw)

            if aggregate_keys:
                aggregate_rows = [
                    (_display_summary_name(name), parsed_results[name], False)
                    for name in aggregate_keys
                ]
                _print_summary_table("Aggregate Groups", "Group", aggregate_rows, total_w, nw)

            if not single_sequence:
                for cls in primary_keys:
                    per_sequence_rows = [
                        (seq_name, seq_metrics, False)
                        for seq_name, seq_metrics in sorted(parsed_results[cls].get("per_sequence", {}).items())
                    ]
                    per_sequence_rows.append(
                        (f"COMBINED ({_display_summary_name(cls)})", parsed_results[cls], True)
                    )
                    _print_summary_table(
                        f"Per-Sequence Details: {_display_summary_name(cls)}",
                        "Sequence",
                        per_sequence_rows,
                        total_w,
                        nw,
                    )
        else:
            detail_keys = primary_keys or aggregate_keys or list(parsed_results.keys())
            for cls in detail_keys:
                per_sequence_rows = [
                    (seq_name, seq_metrics, False)
                    for seq_name, seq_metrics in sorted(parsed_results[cls].get("per_sequence", {}).items())
                ]
                if not single_sequence or not per_sequence_rows:
                    per_sequence_rows.append(
                        (f"COMBINED ({_display_summary_name(cls)})", parsed_results[cls], True)
                    )
                _print_summary_table(
                    _display_summary_name(cls),
                    "Sequence",
                    per_sequence_rows,
                    total_w,
                    nw,
                )

    if getattr(args, "ci", False):
        with open(args.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))
    
    return final_results


def apply_class_remap(args, det_cfg: dict) -> None:
    """Remap GT class IDs to match detector output (step 3.5).

    Loads benchmark config, builds the class mapping, and stores remap metadata
    so TrackEval can work on a run-local GT copy.
    """
    bench_cfg: dict = {}
    benchmark_id = getattr(args, "benchmark_id", None) or getattr(args, "dataset_id", None) or getattr(args, "benchmark", None)
    if benchmark_id:
        try:
            bench_cfg = (load_benchmark_cfg(benchmark_id) or {}).get("benchmark", {})
        except Exception:
            pass

    if str(bench_cfg.get("box_type", "")).lower() == "obb":
        return

    remap_result = build_gt_class_remap(
        bench_cfg, det_cfg,
        benchmark_name=getattr(args, "benchmark", ""),
        model_stem=args.yolo_model[0].stem,
    )
    if remap_result is not None:
        remap_dict, new_class_ids, new_class_names = remap_result
        distractor_ids = [int(k) for k in bench_cfg.get("distractor_classes", {}).keys()]
        args.gt_class_remap = remap_dict
        args.gt_class_distractor_ids = distractor_ids
        args.remapped_class_ids = new_class_ids
        args.remapped_class_names = [n.lower() for n in new_class_names]


def main(args):
    args.yolo_model = [resolve_model_path(model) for model in args.yolo_model]
    args.reid_model = [resolve_model_path(model) for model in args.reid_model]

    # Step 1: Download TrackEval and resolve benchmark config before detector defaults.
    LOGGER.opt(colors=True).info("<cyan>[1/4]</cyan> Setting up TrackEval...")
    eval_init(args)

    _, benchmark_cfg, dataset_detector_cfg = _configure_benchmark_runtime(args)

    # Benchmark detector settings drive imgsz/conf/class remapping when they are active.
    _det_cfg = dict(dataset_detector_cfg or {})

    # Print evaluation pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT Evaluation Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Benchmark:</bold> <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Image size:</bold> <cyan>{getattr(args, 'imgsz', None)}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # Initialize timing stats for the evaluation pipeline
    timing_stats = TimingStats()

    # Step 2: Generate detections and embeddings (with timing)
    LOGGER.opt(colors=True).info("<cyan>[2/4]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args, timing_stats=timing_stats)

    # Step 3: Generate MOT results (with tracking timing)
    LOGGER.opt(colors=True).info("<cyan>[3/4]</cyan> Running tracker...")
    run_generate_mot_results(args, timing_stats=timing_stats)

    # Step 3.5: Prepare GT class remapping metadata for TrackEval.
    apply_class_remap(args, _det_cfg)

    # Step 4: Evaluate with TrackEval
    LOGGER.opt(colors=True).info("<cyan>[4/4]</cyan> Evaluating results...")
    results = run_trackeval(args)
    
    # Print timing summary if we collected timing data
    if timing_stats.frames > 0:
        timing_stats.print_summary()
    
    # Only plot if we have results for a single class or handle multi-class plotting differently
    # For now, let's just skip the radar chart if we have multiple classes or complex structure
    # Or pick the first class found?
    # The original code expected a flat dict of metrics. Now we have {class: {metric: val}}
    
    # Let's try to plot for each class or just skip for now to avoid breaking
    # If 'pedestrian' is in results, use that, otherwise use the first key
    
    # Check if results is flat (single class backward compatibility) or nested
    is_flat = False
    if results and isinstance(list(results.values())[0], (int, float)):
        is_flat = True
        metrics_data = results
        plot_class = 'single_class'
    else:
        plot_class = 'pedestrian'
        if plot_class not in results and len(results) > 0:
            plot_class = list(results.keys())[0]
        metrics_data = results.get(plot_class, {})

    if metrics_data:
        plotter = MetricsPlotter(args.exp_dir)
        
        # Filter only the metrics we want to plot
        plot_metrics = ['HOTA', 'MOTA', 'IDF1']
        plot_values = [metrics_data.get(m, 0) for m in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracking_method: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=['20', '40', '60', '80', '100']
        )
        

if __name__ == "__main__":
    main()
