#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.linalg import pinv

from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH


def build_tracks_from_sequence(
    seq_dir: Path,
    use_temp_gt: bool = False,
    min_detections: int = 5,
):
    """
    Load ground-truth from a single MOT sequence folder,
    build 8-D state/4-D measurement tracks for each object,
    and return (tracks, widths, heights).
    """
    # load GT
    gt_file = seq_dir / "gt" / ("gt_temp.txt" if use_temp_gt else "gt.txt")
    orig_gt = np.loadtxt(gt_file, delimiter=',')
    # filter distractors
    MOT_DISTRACTOR_IDS = []
    mask = ~np.isin(orig_gt[:,1].astype(int), MOT_DISTRACTOR_IDS)
    orig_gt = orig_gt[mask]

    dt = 1.0
    tracks = []
    all_ws = []
    all_hs = []

    for obj_id in np.unique(orig_gt[:,1].astype(int)):
        sel = orig_gt[orig_gt[:,1] == obj_id]
        sel = sel[np.argsort(sel[:,0].astype(int))]
        # centers
        ctrs = np.stack([
            sel[:,2] + sel[:,4]/2,
            sel[:,3] + sel[:,5]/2
        ], axis=1)
        # widths/heights
        wh = sel[:,4:6]
        # velocities
        v_ctr = np.vstack(([[0,0]], np.diff(ctrs, axis=0)/dt))
        v_wh  = np.vstack(([[0,0]], np.diff(wh,  axis=0)/dt))
        # true 8-D state: [x,y,w,h,vx,vy,vw,vh]
        x_seq = np.hstack((ctrs, wh, v_ctr, v_wh))
        # measurements are [x,y,w,h]
        z_seq = np.hstack((ctrs, wh))

        if len(z_seq) >= min_detections:
            tracks.append((z_seq, x_seq))

        # collect box sizes for mean computation
        all_ws.append(sel[:,4])
        all_hs.append(sel[:,5])

    if not tracks:
        raise RuntimeError(f"No object with >= {min_detections} detections in {seq_dir}")

    widths = np.concatenate(all_ws)
    heights = np.concatenate(all_hs)
    return tracks, widths, heights


def main(
    train_root: Path,
    use_temp_gt: bool = True,
    min_detections: int = 5,
):
    # prepare dynamic model
    D = 8
    F = np.eye(D)
    for i in range(4):
        F[i, i+4] = 1.0  # dt=1
    H = np.zeros((4, D))
    H[0,0] = H[1,1] = H[2,2] = H[3,3] = 1

    # aggregate across all sequences
    all_tracks = []
    all_ws = []
    all_hs = []

    for seq_dir in sorted(train_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        print(f"Processing sequence: {seq_dir.name}")
        tracks, ws, hs = build_tracks_from_sequence(
            seq_dir, use_temp_gt=use_temp_gt, min_detections=min_detections)
        all_tracks.extend(tracks)
        all_ws.append(ws)
        all_hs.append(hs)

    # flatten widths/heights
    all_ws = np.concatenate(all_ws)
    all_hs = np.concatenate(all_hs)
    mean_w = all_ws.mean()
    mean_h = all_hs.mean()
    print(f"Mean box width: {mean_w:.2f}, height: {mean_h:.2f}")

    # method-of-moments estimation
    def estimate_noise_covariances(tracks, F, H):
        dim_x = F.shape[0]
        dim_z = H.shape[0]
        sum_innov = np.zeros((dim_z, dim_z))
        sum_proc  = np.zeros((dim_x, dim_x))
        count = 0

        for z_seq, x_true_seq in tracks:
            x_est = x_true_seq[0].copy()
            P = np.eye(dim_x) * 1e-3
            prev_x, prev_P = None, None
            for z in z_seq:
                # 1) Predict
                x_pred = F @ x_est
                P_pred = F @ P @ F.T
                # 2) Innovation
                nu = z - (H @ x_pred)
                sum_innov += np.outer(nu, nu) - (H @ P_pred @ H.T)
                count += 1
                # 3) Update
                S = H @ P_pred @ H.T
                K = P_pred @ H.T @ pinv(S)
                x_est = x_pred + K @ nu
                P = (np.eye(dim_x) - K @ H) @ P_pred
                # 4) Process noise
                if prev_x is not None:
                    w = x_est - (F @ prev_x)
                    sum_proc += np.outer(w, w) - (F @ prev_P @ F.T)
                prev_x, prev_P = x_est.copy(), P.copy()

        if count == 0:
            raise RuntimeError("No valid innovation samples found.")
        R_hat = sum_innov / count
        Q_hat = sum_proc  / count
        return Q_hat, R_hat

    Q_hat, R_hat = estimate_noise_covariances(all_tracks, F, H)
    print("Estimated R (4×4):\n", R_hat)
    print("Estimated Q (8×8):\n", Q_hat)

    # derive std weights
    var_R = np.diag(R_hat)           # [Var(x), Var(y), Var(w), Var(h)]
    var_Q = np.diag(Q_hat)[4:8]      # [Var(vx), Var(vy), Var(vw), Var(vh)]
    mean_box = (mean_w + mean_h) / 2

    std_pos_xy = np.sqrt(var_R[0]) / mean_box
    std_pos_wh = np.sqrt(var_R[2]) / mean_box
    std_wpos  = float((std_pos_xy + std_pos_wh) / 2)

    std_vel_xy = np.sqrt(var_Q[0]) / mean_box
    std_vel_wh = np.sqrt(var_Q[2]) / mean_box
    std_wvel  = float((std_vel_xy + std_vel_wh) / 2)

    # convert absolute‐unit std_wpos into fraction of mean box height
    # (KalmanFilterXYWH expects a relative weight)
    mean_height = mean_h         # ≈172.91 from the pooled data
    std_pos_abs  = std_wpos      # ≈3.2864 (absolute std in px)
    std_pos_frac = std_pos_abs / mean_height  # ≈0.0190

    print(f"→ std_weight_position = {std_pos_frac:.4f}")
    print(f"→ std_weight_velocity = {std_wvel:.4f}")

    # configure KalmanFilter
    kf = KalmanFilterXYWH()
    kf._std_weight_position = std_pos_frac
    kf._std_weight_velocity = std_wvel
    print("KalmanFilterXYWH configured with pooled data-driven weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate Q/R and std_weight_* across all MOT17-ablation train sequences"
    )
    parser.add_argument(
        "--train_root", 
        type=Path,
        default=Path("./boxmot/engine/TrackEval/data/MOT17-ablation/train"),
        help="Root folder containing all MOT17-ablation train sequences"
    )
    parser.add_argument(
        "--use_temp_gt", action="store_true",
        help="Use gt_temp.txt instead of gt.txt"
    )
    parser.add_argument(
        "--min_detections", type=int, default=5,
        help="Minimum detections per track to include"
    )
    args = parser.parse_args()
    main(args.train_root, args.use_temp_gt, args.min_detections)
