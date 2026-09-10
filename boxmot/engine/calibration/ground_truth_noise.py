"""Estimate Kalman noise from MOT-like ground-truth dataset directories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

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
    return _TRACKER_KF_MAP.get(tracker_name)


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
    use_temp_gt: bool = True,
    min_detections: int = 5,
    verbose: bool = True,
    per_class: bool = False,
) -> dict:
    """Estimate KF Q/R matrices and weights from MOT-like ground truth."""
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
    log(f"Estimated R from GT proxy ({dim_z}x{dim_z}):\n", R_hat)

    var_R = np.diag(R_hat)
    std_wpos = np.sqrt(np.abs(var_R).mean()) / mean_h
    std_wvel = np.sqrt(np.abs(Q_vel_diag).mean()) / mean_h

    source = "GT-only"
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
