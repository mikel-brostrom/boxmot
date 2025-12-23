#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from boxmot.utils import TRACKEVAL


def plot_gt_boxes_with_trajectories(
    seq_dir: Union[str, Path],
    use_temp_gt: bool = True,
    pad: int = 0
):
    """
    Plot all ground-truth boxes for a MOT17 sequence, coloring each object ID
    with a unique color and drawing its trajectory through the centers of its boxes.

    Args:
        seq_dir (str | Path): Path to the sequence folder (containing img1/ and gt/).
        use_temp_gt (bool): If True, plot gt/gt_temp.txt (filtered by FPS).
                            Otherwise plot gt/gt.txt.
        pad (int): Extra padding (in pixels) to add around the outermost boxes.
    """
    seq_dir = Path(seq_dir)
    # --- 1) grab image size from first frame ---
    img_files = sorted((seq_dir / "img1").glob("*.jpg"))
    if not img_files:
        raise RuntimeError(f"No images found in {seq_dir / 'img1'}")
    first_img = img_files[0]
    img = cv2.imread(str(first_img))
    if img is None:
        raise RuntimeError(f"Could not load image {first_img}")
    height, width = img.shape[:2]

    # --- 2) load the GT file ---
    gt_file = seq_dir / "gt" / ("gt_temp.txt" if use_temp_gt else "gt.txt")
    orig_gt = np.loadtxt(gt_file, delimiter=",")
    MOT_DISTRACTOR_IDS = [2, 7, 8, 12, 13]  # e.g., person_on_vehicle, static_person, etc.
    class_ids = orig_gt[:, 1].astype(int)
    mask = ~np.isin(class_ids, MOT_DISTRACTOR_IDS)
    orig_gt = orig_gt[mask]
    # we'll unpack the first 6 columns for plotting

    # --- 3) compute the full extents of all boxes ---
    boxes = orig_gt[:, 2:6]  # x, y, w, h
    xs      = boxes[:, 0]
    ys      = boxes[:, 1]
    rights  = xs + boxes[:, 2]
    bottoms = ys + boxes[:, 3]

    xmin = min(xs.min(), 0) - pad
    ymin = min(ys.min(), 0) - pad
    xmax = max(rights.max(), width) + pad
    ymax = max(bottoms.max(), height) + pad

    # --- 4) set up Matplotlib figure ---
    fig, ax = plt.subplots(
        figsize=(10, 10 * (ymax - ymin) / (xmax - xmin))
    )
    ax.set_xlim(xmin, xmax)
    # invert y so origin is top-left
    ax.set_ylim(ymax, ymin)
    ax.set_aspect('equal')
    ax.set_title(
        f"GT boxes & trajectories for sequence {seq_dir.name}\n"
        f"Image size: {width}×{height}, total boxes: {len(boxes)}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- 5) draw the original image and its border ---
    ax.imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        extent=(0, width, height, 0),
        zorder=0
    )
    img_border = patches.Rectangle(
        (0, 0), width, height,
        linewidth=1, edgecolor='gray', linestyle='--',
        facecolor='none', zorder=1
    )
    ax.add_patch(img_border)

    # --- 6) plot each box in a unique color per ID ---
    ids = orig_gt[:, 1].astype(int)
    unique_ids = np.unique(ids)
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i / len(unique_ids)) for i in range(len(unique_ids))]
    id2color = {obj_id: colors[i] for i, obj_id in enumerate(unique_ids)}

    for row in orig_gt:
        frame, obj_id, x, y, w_box, h_box = row[:6]
        obj_id = int(obj_id)
        color = id2color[obj_id]
        outside = (
            x < 0 or y < 0 or
            x + w_box > width or
            y + h_box > height
        )
        rect = patches.Rectangle(
            (x, y),
            w_box, h_box,
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            linestyle='--' if outside else '-',
            zorder=2
        )
        ax.add_patch(rect)

    # annotate count of out‐of‐frame boxes
    n_outside = sum(
        (row[2] < 0 or row[3] < 0 or
         row[2] + row[4] > width or
         row[3] + row[5] > height)
        for row in orig_gt
    )
    ax.text(
        xmin + 0.01*(xmax - xmin),
        ymax - 0.02*(ymax - ymin),
        f"Outside boxes: {n_outside} / {len(boxes)}",
        color='red', fontsize=12, va='top', zorder=3
    )

    # --- 7) plot each ID’s trajectory via box centers ---
    for obj_id in unique_ids:
        sel = orig_gt[orig_gt[:, 1] == obj_id]
        # center = (x + w/2, y + h/2)
        centers = np.stack([
            sel[:, 2] + sel[:, 4] / 2,
            sel[:, 3] + sel[:, 5] / 2
        ], axis=1)
        # sort by frame index
        order = np.argsort(sel[:, 0].astype(int))
        centers = centers[order]
        ax.plot(
            centers[:, 0], centers[:, 1],
            '-', linewidth=1,
            color=id2color[obj_id],
            zorder=3
        )

    # (optional) legend if number of IDs is small
    # ax.legend(
    #     [f"ID {i}" for i in unique_ids],
    #     loc='upper right',
    #     fontsize='small',
    #     ncol=2,
    #     framealpha=0.5
    # )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MOT17 ground-truth boxes and trajectories"
    )
    parser.add_argument(
        "--seq_dir",
        default=TRACKEVAL / "MOT17-ablation/train/MOT17-09",
        help="Path to the MOT17 sequence folder (must contain img1/ and gt/ subfolders)"
    )
    parser.add_argument(
        "--use_temp_gt",
        action="store_true",
        help="Plot gt/gt_temp.txt instead of gt/gt.txt"
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Extra padding (pixels) around the outermost boxes"
    )
    args = parser.parse_args()

    plot_gt_boxes_with_trajectories(
        args.seq_dir,
        use_temp_gt=args.use_temp_gt,
        pad=args.pad
    )
