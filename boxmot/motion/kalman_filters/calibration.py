"""KF noise estimation helpers for the eval/tune pipelines."""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.markup import escape as _escape_markup
from scipy.optimize import linear_sum_assignment

from boxmot.configs.benchmark import load_benchmark_cfg
from boxmot.utils import logger as LOGGER

KF_TYPES = ("xywh", "xyah", "xysr", "xyhr")

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


def _cxywh_to_measurement(cxywh: np.ndarray, kf_type: str) -> np.ndarray:
    """Convert ``(cx, cy, w, h)`` boxes to a KF measurement vector."""
    cx, cy, w, h = cxywh[..., 0], cxywh[..., 1], cxywh[..., 2], cxywh[..., 3]
    if kf_type == "xywh":
        return np.stack([cx, cy, w, h], axis=-1)
    if kf_type == "xyah":
        aspect = w / np.maximum(h, 1e-6)
        return np.stack([cx, cy, aspect, h], axis=-1)
    if kf_type == "xysr":
        scale = w * h
        ratio = w / np.maximum(h, 1e-6)
        return np.stack([cx, cy, scale, ratio], axis=-1)
    if kf_type == "xyhr":
        ratio = w / np.maximum(h, 1e-6)
        return np.stack([cx, cy, h, ratio], axis=-1)
    raise ValueError(f"Unknown kf_type: {kf_type}")


def _measurement_labels(kf_type: str) -> list[str]:
    """Return human-readable labels for the KF measurement dimensions."""
    labels = {
        "xywh": ["cx", "cy", "w", "h"],
        "xyah": ["cx", "cy", "a", "h"],
        "xysr": ["cx", "cy", "s", "r"],
        "xyhr": ["cx", "cy", "h", "r"],
    }
    try:
        return labels[kf_type]
    except KeyError as exc:
        raise ValueError(f"Unknown kf_type: {kf_type}") from exc


def _get_dim_x(kf_type: str) -> int:
    """Return the state dimension for a KF parameterization."""
    if kf_type == "xysr":
        return 7
    if kf_type in KF_TYPES:
        return 8
    raise ValueError(f"Unknown kf_type: {kf_type}")


def _get_dim_z(kf_type: str) -> int:
    """Return the AABB measurement dimension for a KF parameterization."""
    if kf_type not in KF_TYPES:
        raise ValueError(f"Unknown kf_type: {kf_type}")
    return 4


def _obb_to_cxywh(gt: np.ndarray) -> np.ndarray:
    """Convert OBB GT rows to MOT-like ``(frame, id, x, y, w, h, ...)`` rows."""
    corners_x = gt[:, [2, 4, 6, 8]]
    corners_y = gt[:, [3, 5, 7, 9]]
    x_min = corners_x.min(axis=1)
    x_max = corners_x.max(axis=1)
    y_min = corners_y.min(axis=1)
    y_max = corners_y.max(axis=1)
    width = x_max - x_min
    height = y_max - y_min

    result = np.column_stack([gt[:, 0], gt[:, 1], x_min, y_min, width, height])
    if gt.shape[1] > 10:
        result = np.column_stack([result, gt[:, 10:]])
    return result


def load_gt_data(
    seq_dir: Path,
    annotations_dir: Path | None = None,
    use_temp_gt: bool = False,
) -> np.ndarray:
    """Load MOT/VisDrone/MMOT ground truth for one sequence."""
    gt: np.ndarray | None = None

    if annotations_dir is not None and annotations_dir.exists():
        ann_file = annotations_dir / f"{seq_dir.name}.txt"
        if ann_file.exists():
            gt = np.atleast_2d(np.loadtxt(ann_file, delimiter=","))

    if gt is None:
        gt_file = seq_dir / "gt" / ("gt_temp.txt" if use_temp_gt else "gt.txt")
        if gt_file.exists():
            gt = np.atleast_2d(np.loadtxt(gt_file, delimiter=","))

    if gt is None:
        raise FileNotFoundError(f"No GT file found for sequence {seq_dir.name}")

    if gt.shape[1] >= 13:
        gt = _obb_to_cxywh(gt)

    return gt


def build_tracks_from_sequence(
    seq_dir: Path,
    kf_type: str = "xywh",
    annotations_dir: Path | None = None,
    use_temp_gt: bool = False,
    min_detections: int = 5,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], np.ndarray, np.ndarray]:
    """Build KF measurement/state tracks from one sequence's GT."""
    dim_z = _get_dim_z(kf_type)

    orig_gt = load_gt_data(seq_dir, annotations_dir, use_temp_gt)

    tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    all_ws: list[np.ndarray] = []
    all_hs: list[np.ndarray] = []

    for obj_id in np.unique(orig_gt[:, 1].astype(int)):
        sel = orig_gt[orig_gt[:, 1] == obj_id]
        sel = sel[np.argsort(sel[:, 0].astype(int))]
        cls_id = int(np.median(sel[:, 7])) if sel.shape[1] > 7 else 0

        cxywh = np.column_stack(
            [
                sel[:, 2] + sel[:, 4] / 2,
                sel[:, 3] + sel[:, 5] / 2,
                sel[:, 4],
                sel[:, 5],
            ]
        )
        z_seq = _cxywh_to_measurement(cxywh, kf_type)
        v_z = np.vstack(([np.zeros(dim_z)], np.diff(z_seq, axis=0)))
        if kf_type == "xysr":
            x_seq = np.column_stack([z_seq, v_z[:, :3]])
        else:
            x_seq = np.hstack([z_seq, v_z])

        if len(z_seq) >= min_detections:
            tracks.append((z_seq, x_seq, cls_id))

        all_ws.append(sel[:, 4])
        all_hs.append(sel[:, 5])

    if not tracks:
        raise RuntimeError(f"No object with >= {min_detections} detections in {seq_dir}")

    return tracks, np.concatenate(all_ws), np.concatenate(all_hs)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for ``xyxy`` boxes."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _xyxy_to_cxywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``xyxy`` boxes to ``(cx, cy, w, h)``."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return np.column_stack([cx, cy, width, height])


def estimate_R_from_detections(
    gt_root: Path,
    dets_root: Path,
    kf_type: str = "xywh",
    iou_threshold: float = 0.5,
    use_temp_gt: bool = False,
    annotations_dir: Path | None = None,
    verbose: bool = True,
    per_class: bool = False,
) -> tuple[np.ndarray, float, float] | tuple[np.ndarray, float, float, dict[int, np.ndarray]]:
    """Estimate measurement noise by matching cached detections to GT boxes."""
    all_residuals: list[np.ndarray] = []
    all_cls_ids: list[int] = []
    all_ws: list[np.ndarray] = []
    all_hs: list[np.ndarray] = []

    det_files = sorted(f for f in dets_root.glob("*.npy") if not f.name.startswith("._"))
    if not det_files:
        raise FileNotFoundError(f"No .npy detection files found in {dets_root}")

    for det_file in det_files:
        seq_name = det_file.stem
        seq_dir = gt_root / seq_name
        if not seq_dir.is_dir():
            if verbose:
                print(f"  Skipping {seq_name}: no matching GT directory")
            continue

        if verbose:
            print(f"  Matching dets to GT: {seq_name}")

        dets = np.load(det_file)
        is_obb_dets = dets.shape[1] >= 8

        try:
            gt_data = load_gt_data(seq_dir, annotations_dir, use_temp_gt)
        except FileNotFoundError as exc:
            if verbose:
                print(f"    Skipping: {exc}")
            continue

        all_ws.append(gt_data[:, 4])
        all_hs.append(gt_data[:, 5])

        for frame_id in np.unique(gt_data[:, 0].astype(int)):
            gt_frame = gt_data[gt_data[:, 0].astype(int) == frame_id]
            gt_xyxy = np.column_stack(
                [
                    gt_frame[:, 2],
                    gt_frame[:, 3],
                    gt_frame[:, 2] + gt_frame[:, 4],
                    gt_frame[:, 3] + gt_frame[:, 5],
                ]
            )

            det_frame = dets[dets[:, 0].astype(int) == frame_id]
            if len(det_frame) == 0:
                continue

            if is_obb_dets:
                cx, cy = det_frame[:, 1], det_frame[:, 2]
                width, height = det_frame[:, 3], det_frame[:, 4]
                det_xyxy = np.column_stack(
                    [
                        cx - width / 2,
                        cy - height / 2,
                        cx + width / 2,
                        cy + height / 2,
                    ]
                )
            else:
                det_xyxy = det_frame[:, 1:5]

            iou = _iou_matrix(det_xyxy, gt_xyxy)
            if iou.size == 0:
                continue

            row_ind, col_ind = linear_sum_assignment(1 - iou)
            for det_idx, gt_idx in zip(row_ind, col_ind):
                if iou[det_idx, gt_idx] < iou_threshold:
                    continue

                det_cxywh = _xyxy_to_cxywh(det_xyxy[det_idx:det_idx + 1])
                gt_cxywh = _xyxy_to_cxywh(gt_xyxy[gt_idx:gt_idx + 1])
                det_z = _cxywh_to_measurement(det_cxywh, kf_type)[0]
                gt_z = _cxywh_to_measurement(gt_cxywh, kf_type)[0]
                all_residuals.append(det_z - gt_z)
                if per_class:
                    cls_id = int(gt_frame[gt_idx, 7]) if gt_frame.shape[1] > 7 else 0
                    all_cls_ids.append(cls_id)

    if not all_residuals:
        raise RuntimeError("No det-GT matches found. Check IoU threshold and paths.")

    residuals = np.array(all_residuals)
    labels = _measurement_labels(kf_type)
    if verbose:
        print(f"\n  Total det-GT matches: {len(residuals)}")
        print(f"  Mean residual ({', '.join(labels)}): {residuals.mean(axis=0)}")
        print(f"  Std  residual ({', '.join(labels)}): {residuals.std(axis=0)}")

    R_hat = np.cov(residuals, rowvar=False)
    mean_w = np.concatenate(all_ws).mean() if all_ws else 1.0
    mean_h = np.concatenate(all_hs).mean() if all_hs else 1.0

    if not per_class:
        return R_hat, mean_w, mean_h

    cls_ids = np.array(all_cls_ids)
    per_class_R: dict[int, np.ndarray] = {}
    for cls_id in np.unique(cls_ids):
        cls_residuals = residuals[cls_ids == cls_id]
        if len(cls_residuals) < 3:
            continue
        per_class_R[int(cls_id)] = np.cov(cls_residuals, rowvar=False)
        if verbose:
            diag = np.diag(per_class_R[int(cls_id)])
            print(f"  [class {cls_id}] {len(cls_residuals)} det-GT matches, R diag={diag}")

    return R_hat, mean_w, mean_h, per_class_R


def _estimate_process_noise(
    tracks: list[tuple[np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate diagonal position and velocity process noise from GT tracks."""
    pos_residuals: list[np.ndarray] = []
    accel_samples: list[np.ndarray] = []

    for z_seq, _x_seq, _cls_id in tracks:
        if len(z_seq) < 3:
            continue

        velocity = np.diff(z_seq, axis=0)
        z_pred = z_seq[1:-1] + velocity[:-1]
        pos_residuals.append(z_seq[2:] - z_pred)
        accel_samples.append(np.diff(z_seq, n=2, axis=0))

    if not pos_residuals:
        raise RuntimeError("No valid tracks with >= 3 detections found.")

    return np.var(np.vstack(pos_residuals), axis=0), np.var(np.vstack(accel_samples), axis=0)


def estimate_kf_noise(
    train_root: Path,
    kf_type: str = "xywh",
    dets_root: Path | None = None,
    use_temp_gt: bool = True,
    min_detections: int = 5,
    iou_threshold: float = 0.5,
    verbose: bool = True,
    per_class: bool = False,
) -> dict:
    """Estimate KF Q/R matrices and std-weight parameters from MOT-like data."""
    dim_x = _get_dim_x(kf_type)
    dim_z = _get_dim_z(kf_type)
    log = print if verbose else (lambda *a, **kw: None)

    log(f"KF type: {kf_type} (state dim={dim_x}, meas dim={dim_z})")
    labels = _measurement_labels(kf_type)

    annotations_dir = train_root.parent / "annotations" if (train_root.parent / "annotations").exists() else None
    if train_root.name == "sequences":
        seq_root = train_root
    elif (train_root / "sequences").exists():
        seq_root = train_root / "sequences"
        annotations_dir = train_root / "annotations"
    else:
        seq_root = train_root

    if annotations_dir is None:
        mot_dir = train_root.parent / "mot" if train_root.name == "npy" else train_root / "mot"
        if mot_dir.exists():
            annotations_dir = mot_dir

    log(f"Dataset root: {train_root}")
    log(f"Sequences dir: {seq_root}")
    if annotations_dir:
        log(f"Annotations dir: {annotations_dir}")

    all_tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    all_ws: list[np.ndarray] = []
    all_hs: list[np.ndarray] = []

    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        log(f"Processing sequence: {seq_dir.name}")
        try:
            tracks, ws, hs = build_tracks_from_sequence(
                seq_dir,
                kf_type=kf_type,
                annotations_dir=annotations_dir,
                use_temp_gt=use_temp_gt,
                min_detections=min_detections,
            )
        except FileNotFoundError as exc:
            log(f"  Skipping: {exc}")
            continue
        except Exception as exc:
            log(f"  Error: {exc}")
            continue

        all_tracks.extend(tracks)
        all_ws.append(ws)
        all_hs.append(hs)

    if not all_tracks:
        raise RuntimeError("No valid tracks found in any sequence. Check dataset path and format.")

    all_ws_flat = np.concatenate(all_ws)
    all_hs_flat = np.concatenate(all_hs)
    mean_w = all_ws_flat.mean()
    mean_h = all_hs_flat.mean()
    log(f"Mean box width: {mean_w:.2f}, height: {mean_h:.2f}")

    Q_pos_diag, Q_vel_diag = _estimate_process_noise(all_tracks)

    Q_hat = np.zeros((dim_x, dim_x), dtype=float)
    n_pos = min(len(Q_pos_diag), dim_z)
    n_vel = min(len(Q_vel_diag), dim_x - dim_z)
    for idx in range(n_pos):
        Q_hat[idx, idx] = Q_pos_diag[idx]
    for idx in range(n_vel):
        Q_hat[dim_z + idx, dim_z + idx] = Q_vel_diag[idx]

    log(f"\n-- Independent Q estimates ({kf_type}) --")
    log(f"  Q position diagonal: {Q_pos_diag}")
    log(f"  Q velocity diagonal: {Q_vel_diag}")
    log(f"  Q ({dim_x}x{dim_x}):\n", Q_hat)

    R_hat = np.diag(Q_pos_diag[:dim_z])
    per_class_R_from_dets: dict[int, np.ndarray] = {}
    if dets_root is not None:
        log(f"\n-- Estimating R from det-vs-GT matching (IoU >= {iou_threshold}) --")
        r_result = estimate_R_from_detections(
            gt_root=seq_root,
            dets_root=dets_root,
            kf_type=kf_type,
            iou_threshold=iou_threshold,
            use_temp_gt=use_temp_gt,
            annotations_dir=annotations_dir,
            verbose=verbose,
            per_class=per_class,
        )
        if per_class:
            R_hat, _det_mean_w, _det_mean_h, per_class_R_from_dets = r_result
        else:
            R_hat, _det_mean_w, _det_mean_h = r_result
        log(f"Estimated R from detections ({dim_z}x{dim_z}):\n", R_hat)
    else:
        log(f"Estimated R from GT proxy ({dim_z}x{dim_z}):\n", R_hat)

    var_R = np.diag(R_hat)
    std_wpos = np.sqrt(np.abs(var_R).mean()) / mean_h
    std_wvel = np.sqrt(np.abs(Q_vel_diag).mean()) / mean_h

    source = "det-vs-GT" if dets_root else "GT-only"
    kf_class_name = {
        "xywh": "KalmanFilterXYWH",
        "xyah": "KalmanFilterXYAH",
        "xysr": "KalmanFilterXYSR",
        "xyhr": "KalmanFilterXYHR",
    }[kf_type]

    log(f"\n-- {kf_class_name} weights ({source}) --")
    log(f"  Measurement labels: {labels}")
    log(f"  R diagonal (abs variances): {np.abs(var_R)}")
    log(f"  Q position diagonal: {Q_pos_diag}")
    log(f"  Q velocity diagonal: {Q_vel_diag}")
    log(f"  Mean box size: {mean_w:.1f} x {mean_h:.1f}")
    log(f"-> _std_weight_position = {std_wpos:.6f}")
    log(f"-> _std_weight_velocity = {std_wvel:.6f}")

    if kf_type == "xyhr":
        log("\n  Note: KalmanFilterXYHR uses ConstantNoiseXYHR (BoostTrack model).")
        log("  These weights would replace the constant-noise policy if desired.")

    result = {
        "kf_type": kf_type,
        "kf_class": kf_class_name,
        "std_weight_position": float(std_wpos),
        "std_weight_velocity": float(std_wvel),
        "Q": Q_hat,
        "R": R_hat,
        "Q_vel_diag": Q_vel_diag,
        "source": source,
        "mean_w": float(mean_w),
        "mean_h": float(mean_h),
    }

    if not per_class:
        return result

    class_tracks: defaultdict[int, list[tuple[np.ndarray, np.ndarray, int]]] = defaultdict(list)
    for track in all_tracks:
        class_tracks[track[2]].append(track)

    per_class_results: dict[int, dict] = {}
    for cls_id in sorted(class_tracks):
        cls_tracks = class_tracks[cls_id]
        if len(cls_tracks) < 3:
            log(f"\n  [class {cls_id}] Skipping: only {len(cls_tracks)} tracks (need >= 3)")
            continue

        try:
            cls_Q_pos, cls_Q_vel = _estimate_process_noise(cls_tracks)
        except RuntimeError:
            log(f"\n  [class {cls_id}] Skipping: insufficient track lengths")
            continue

        cls_Q = np.zeros((dim_x, dim_x), dtype=float)
        for idx in range(min(len(cls_Q_pos), dim_z)):
            cls_Q[idx, idx] = cls_Q_pos[idx]
        for idx in range(min(len(cls_Q_vel), dim_x - dim_z)):
            cls_Q[dim_z + idx, dim_z + idx] = cls_Q_vel[idx]

        if cls_id in per_class_R_from_dets:
            cls_R = per_class_R_from_dets[cls_id]
            r_source = "det-vs-GT"
        else:
            cls_R = np.diag(cls_Q_pos[:dim_z])
            r_source = "GT-proxy"

        per_class_results[cls_id] = {
            "Q": cls_Q,
            "R": cls_R,
            "Q_vel_diag": cls_Q_vel,
            "n_tracks": len(cls_tracks),
        }
        log(
            f"\n  [class {cls_id}] {len(cls_tracks)} tracks, "
            f"Q_pos={cls_Q_pos}, Q_vel={cls_Q_vel}, R={r_source}"
        )

    result["per_class"] = per_class_results
    log(f"\n-- Per-class KF tuning: {len(per_class_results)} classes estimated --")

    return result


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
            # Train dets not available; train GT sequence names may not match
            # eval dets, so fall back to the eval split.
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
        LOGGER.info(
            f"[bold]KF Tuning[/bold] ({_escape_markup(str(kf_type))}): "
            f"GT={_escape_markup(str(gt_root))}, dets={_escape_markup(str(dets_root))}"
        )
    buf = io.StringIO() if capture else None
    cm = contextlib.redirect_stdout(buf) if buf is not None else contextlib.nullcontext()

    # Enable per-class KF tuning only when the dataset has multiple eval classes
    per_class_kf = bool(getattr(args, "per_class_kf", True))
    gt_class_offset = 0  # offset to convert GT class IDs to detector class IDs
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
            result = estimate_kf_noise(
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
