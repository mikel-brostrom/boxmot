from __future__ import annotations

# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch

from boxmot.trackers.common.geometry.obb import xywha_to_corners
from boxmot.trackers.results import TrackResults
from boxmot.utils import logger as LOGGER

MOT_ROW_FORMAT = "%d,%d,%d,%d,%d,%d,%.6f,%d,%d"
MMOT_ROW_FORMAT = "%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d"


def _xyxy_to_ltwh(boxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert ``[x1, y1, x2, y2]`` boxes to ``[x1, y1, w, h]``."""
    if isinstance(boxes, torch.Tensor):
        converted = boxes.clone()
    else:
        converted = np.array(boxes, copy=True)
    converted[..., 2] = converted[..., 2] - converted[..., 0]
    converted[..., 3] = converted[..., 3] - converted[..., 1]
    return converted


def _build_val_half_split(seq_dirs: list[Path], dst_dir: Path) -> None:
    """Copy sequences to *dst_dir*, keeping only the second half of frames.

    For each sequence, this:
    - Copies the directory structure
    - Trims gt/gt.txt and det/det.txt to frames > N//2+1
    - Removes images from the first half
    - Re-indexes remaining frames starting from 1
    - Updates seqinfo.ini with the new sequence length

    This is the standard ByteTrack ablation protocol (val-half).
    """
    for src_seq in seq_dirs:
        seq_name = src_seq.name
        dst_seq = dst_dir / seq_name
        if dst_seq.exists():
            continue

        # Copy entire sequence
        for item in src_seq.rglob("*"):
            target = dst_seq / item.relative_to(src_seq)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes())

        gt_path = dst_seq / "gt" / "gt.txt"
        if not gt_path.exists():
            LOGGER.warning(f"Skipping `{dst_seq}` – no gt.txt found")
            continue

        # ByteTrack split: keep frames > N//2+1 (1-indexed)
        df = pd.read_csv(gt_path, header=None)
        max_frame = int(df[0].max())
        split_frame = max_frame // 2 + 1
        val_length = max_frame - split_frame

        if split_frame >= max_frame:
            continue

        LOGGER.info(f"{seq_name}: keeping frames {split_frame + 1}-{max_frame} ({val_length} frames)")

        # Filter and re-index gt
        df = df[df[0] > split_frame].copy()
        df[0] = df[0] - split_frame
        df.to_csv(gt_path, header=False, index=False)

        # Filter and re-index det
        det_path = dst_seq / "det" / "det.txt"
        if det_path.exists():
            det_df = pd.read_csv(det_path, header=None)
            det_df = det_df[det_df[0] > split_frame].copy()
            det_df[0] = det_df[0] - split_frame
            det_df.to_csv(det_path, header=False, index=False)

        # Delete first-half images
        img_folder = dst_seq / "img1"
        for img in img_folder.glob("*.jpg"):
            if int(img.stem) <= split_frame:
                img.unlink()

        # Rename remaining to 000001…
        remaining = sorted(img_folder.glob("*.jpg"))
        for idx, img in enumerate(remaining, start=1):
            img.rename(img_folder / f"{idx:06}.jpg")

        # Update seqinfo.ini
        ini_path = dst_seq / "seqinfo.ini"
        if ini_path.exists():
            text = ini_path.read_text()
            text = re.sub(r"seqLength=\d+", f"seqLength={val_length}", text)
            ini_path.write_text(text)


def split_dataset(src_fldr: Path) -> Tuple[Path, str]:
    """
    Copies the dataset and keeps only the validation half, matching ByteTrack's split:
        train_half: [0, num_images // 2]        (0-indexed, discarded)
        val_half:   [num_images // 2 + 1, num_images - 1]  (0-indexed, kept)

    Updates img1/, gt/gt.txt, det/det.txt, and seqinfo.ini for each sequence.

    Args:
        src_fldr (Path): Source folder (e.g. /…/MOT20/train or /…/MOT20/test)

    Returns:
        dst_fldr (Path): The root of the new val-half split (e.g. …/MOT20-ablation/train)
        new_benchmark_name (str): e.g. "MOT20-ablation"
    """
    src_fldr = Path(src_fldr)

    # --- detect the "MOTxx" part in the path ---
    m = re.search(r"(MOT\d+)", str(src_fldr))
    if not m:
        raise ValueError(f"Could not find MOT benchmark in path: {src_fldr}")
    benchmark = m.group(1)

    # build the new benchmark name
    new_benchmark_name = f"{benchmark}-ablation"
    dst_fldr = Path(str(src_fldr).replace(benchmark, new_benchmark_name))

    # copy entire folder tree if not already done
    if not dst_fldr.exists():
        for item in src_fldr.rglob("*"):
            target = dst_fldr / item.relative_to(src_fldr)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.write_bytes(item.read_bytes())

    # iterate every sequence under dst_fldr
    for seq_path in dst_fldr.iterdir():
        if not seq_path.is_dir():
            continue

        gt_path = seq_path / "gt" / "gt.txt"
        if not gt_path.exists():
            LOGGER.warning(f"Skipping `{seq_path}` – no gt.txt found")
            continue

        # ByteTrack split: train_half = [0, N//2], val_half = [N//2+1, N-1] (0-indexed)
        # In 1-indexed frames: split_frame = N//2 + 1, keep frames > split_frame
        df = pd.read_csv(gt_path, header=None)
        max_frame = int(df[0].max())
        split_frame = max_frame // 2 + 1
        val_length = max_frame - split_frame

        if split_frame >= max_frame:
            LOGGER.info(f"`{seq_path}` already ≤ split size, skipping.")
            continue

        LOGGER.info(f"{seq_path.name}: keeping frames {split_frame + 1}-{max_frame}")

        # filter and re-index gt
        df = df[df[0] > split_frame].copy()
        df[0] = df[0] - split_frame
        df.to_csv(gt_path, header=False, index=False)

        # filter and re-index det
        det_path = seq_path / "det" / "det.txt"
        if det_path.exists():
            det_df = pd.read_csv(det_path, header=None)
            det_df = det_df[det_df[0] > split_frame].copy()
            det_df[0] = det_df[0] - split_frame
            det_df.to_csv(det_path, header=False, index=False)

        # delete early images
        img_folder = seq_path / "img1"
        for img in img_folder.glob("*.jpg"):
            if int(img.stem) <= split_frame:
                img.unlink()

        # rename rest to 000001…000xxx
        remaining = sorted(img_folder.glob("*.jpg"))
        for idx, img in enumerate(remaining, start=1):
            img.rename(img_folder / f"{idx:06}.jpg")

        # update seqinfo.ini
        ini_path = seq_path / "seqinfo.ini"
        if ini_path.exists():
            text = ini_path.read_text()
            text = re.sub(r"seqLength=\d+", f"seqLength={val_length}", text)
            ini_path.write_text(text)

        LOGGER.info(f"{seq_path.name}: now {val_length} images")

    return dst_fldr, new_benchmark_name


def convert_to_mot_format(results: Any | np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Converts tracking results for a single frame into MOT challenge format.

    This function supports inputs as either a custom object with a 'boxes' attribute or a numpy array.
    For custom object inputs, 'boxes' should contain 'id', 'xyxy', 'conf', and 'cls' sub-attributes.
    For numpy array inputs, the expected format per row is:
    ``(xmin, ymin, xmax, ymax, id, conf, cls[, det_ind])``.

    Parameters:
    - results (Union[Results, np.ndarray]): Tracking results for the current frame.
    - frame_idx (int): The zero-based index of the frame being processed.

    Returns:
    - np.ndarray: An array containing the MOT formatted results for the frame.
    """

    if isinstance(results, np.ndarray):
        if results.size == 0:
            return np.empty((0, 9), dtype=np.float32)

        tr = TrackResults(results)
        tlwh = _xyxy_to_ltwh(tr.xyxy)
        frame_idx_column = np.full((len(tr), 1), frame_idx, dtype=np.int32)
        det_ind = tr.det_ind.reshape(-1, 1).astype(np.int32)
        return np.column_stack(
            (
                frame_idx_column,  # frame index
                tr.id.reshape(-1, 1).astype(np.int32),  # track id
                tlwh.round().astype(np.int32),  # top,left,width,height
                tr.conf.reshape(-1, 1),  # confidence (float)
                (tr.cls + 1).reshape(-1, 1).astype(np.int32),  # class
                det_ind,  # detection index
            )
        )

    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 9), dtype=np.float32)

    num_detections = len(boxes)
    frame_indices = torch.full((num_detections, 1), frame_idx + 1, dtype=torch.int32)
    det_inds = torch.full((num_detections, 1), -1, dtype=torch.int32)

    track_ids = torch.as_tensor(boxes.id).reshape(-1, 1).to(dtype=torch.int32)
    tlwh = _xyxy_to_ltwh(torch.as_tensor(boxes.xyxy)).to(dtype=torch.int32)
    conf = torch.as_tensor(boxes.conf).reshape(-1, 1).to(dtype=torch.float32)
    cls = torch.as_tensor(boxes.cls).reshape(-1, 1).to(dtype=torch.int32) + 1
    mot_results = torch.cat(
        [
            frame_indices,  # frame index
            track_ids,  # track id
            tlwh,  # top,left,width,height
            conf,  # confidence (float)
            cls,  # class
            det_inds,  # detection index
        ],
        dim=1,
    )

    return mot_results.numpy()


def convert_to_mmot_obb_format(results: np.ndarray, frame_idx: int) -> np.ndarray:
    """Convert OBB tracker output ``[cx, cy, w, h, angle, id, conf, cls, det_ind]`` to MMOT format."""
    if results.size == 0:
        return np.empty((0, 13), dtype=np.float32)

    if results.ndim == 1:
        results = results.reshape(1, -1)

    tr = TrackResults(results)
    if not tr.is_obb:
        raise ValueError(f"Expected OBB tracking results with at least 9 columns, got {results.shape[1]}")

    frame_col = np.full((len(tr), 1), frame_idx, dtype=np.float32)
    track_ids = tr.id.reshape(-1, 1).astype(np.float32)
    corners = xywha_to_corners(tr.xywha).astype(np.float32)
    conf = tr.conf.reshape(-1, 1).astype(np.float32)
    cls = tr.cls.reshape(-1, 1).astype(np.float32)
    det_ind = tr.det_ind.reshape(-1, 1).astype(np.float32)
    return np.concatenate((frame_col, track_ids, corners, conf, cls, det_ind), axis=1)


def format_frame_tagged_tracks_for_mot(entries: np.ndarray) -> np.ndarray:
    """Convert GTA ``[frame, *tracker_output]`` rows to canonical MOT/MMOT rows.

    GTA stores interpolated rows in the tracker's native output schema so it
    can preserve oriented geometry.  This helper deliberately routes every
    frame through the same exporters used for normal tracker output instead
    of treating those native rows as already serialized MOT records.
    """
    rows = np.asarray(entries, dtype=np.float32)
    if rows.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] not in (9, 10):
        raise ValueError(
            "Frame-tagged rows must contain a frame plus an 8-column AABB or "
            f"9-column OBB tracker output, got shape {rows.shape}"
        )

    frame_values = rows[:, 0]
    if not np.isfinite(frame_values).all():
        raise ValueError("Frame IDs must be finite integers")
    frame_ids = frame_values.astype(np.int64)
    if not np.array_equal(frame_values, frame_ids):
        raise ValueError("Frame IDs must be finite integers")

    formatted = []
    for frame_id in np.unique(frame_ids):
        tracks = rows[frame_ids == frame_id, 1:]
        if tracks.shape[1] == 9:
            formatted.append(convert_to_mmot_obb_format(tracks, int(frame_id)))
        else:
            formatted.append(convert_to_mot_format(tracks, int(frame_id)))
    return np.vstack(formatted)


def write_mot_results(txt_path: Path, mot_results: np.ndarray) -> None:
    """
    Writes the MOT challenge formatted results to a text file.

    Parameters:
    - txt_path (Path): The path to the text file where results are saved.
    - mot_results (np.ndarray): An array containing the MOT formatted results.

    Note: The text file will be created if it does not exist, and the directory
    path to the file will be created as well if necessary.
    """
    if mot_results is not None:
        # Ensure the parent directory of the txt_path exists
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the file exists before opening
        txt_path.touch(exist_ok=True)

        if mot_results.size != 0:
            if mot_results.ndim == 1:
                mot_results = mot_results.reshape(1, -1)
            # Open the file in append mode and save the MOT results
            with open(str(txt_path), "a") as file:
                if mot_results.shape[1] == 9:
                    np.savetxt(file, mot_results, fmt=MOT_ROW_FORMAT)
                elif mot_results.shape[1] == 13:
                    np.savetxt(file, mot_results, fmt=MMOT_ROW_FORMAT)
                else:
                    raise ValueError(
                        "MOT output must contain 9-column AABB or 13-column MMOT rows, "
                        f"got shape {mot_results.shape}"
                    )
