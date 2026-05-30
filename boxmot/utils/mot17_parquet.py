"""MOT17 parquet dataset setup.

Downloads deduplicated images + parquet annotations from ``Lekim89/mot17-parquet``
and reconstructs the MOTChallenge-compatible layout expected by the eval pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boxmot.utils import logger as LOGGER

PARQUET_REPO = "Lekim89/mot17-parquet"

DETECTORS = ("DPM", "FRCNN", "SDP")


def _ablation_start(seq_length: int) -> int:
    """Original frame number where ablation (second half) begins.

    Matches the ByteTrack / existing Lekim89/MOT17 convention:
    ``seqLength // 2 + 2``, and ablation frames are renumbered starting from 1.
    """
    return seq_length // 2 + 2


def _require_pd():
    import pandas as pd
    return pd


def _require_hf():
    from huggingface_hub import hf_hub_download, snapshot_download
    return snapshot_download, hf_hub_download


def setup_mot17_from_parquet(
    dest: Path,
    split: str = "ablation",
    detector: str = "FRCNN",
    *,
    overwrite: bool = False,
    status_fn: Any = None,
) -> None:
    """Download from the parquet repo and create MOTChallenge layout.

    Creates the expected structure::

        dest/<split>/
            MOT17-02-FRCNN/
                img1/000001.jpg -> symlink (renumbered for ablation)
                gt/gt.txt       (renumbered frames)
                det/det.txt     (renumbered frames)
                seqinfo.ini

    Also creates public-detector .npy caches under::

        runs/dets_n_embs/mot17/<split>/<detector_id>/dets/MOT17-02-FRCNN.npy
    """
    detector = detector.upper()
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, got '{detector}'")

    split_dir = dest / split
    marker = split_dir / ".parquet_setup_complete"
    if not overwrite and marker.exists():
        LOGGER.debug(f"MOT17 parquet setup already done: {split_dir}")
        return

    # If split directory already has sequences with images (legacy download), skip
    if not overwrite and split_dir.is_dir():
        for seq_dir in split_dir.iterdir():
            if seq_dir.is_dir() and (seq_dir / "img1").exists():
                LOGGER.debug(f"MOT17 split already populated: {split_dir}")
                marker.touch()
                return

    pd = _require_pd()
    snapshot_download, hf_hub_download = _require_hf()

    msg = f"Setting up MOT17 {split} ({detector}) from parquet..."
    if status_fn:
        status_fn(msg)
    else:
        LOGGER.info(msg)

    # Determine which image split to download from the parquet repo.
    # ablation uses train images; test uses test images.
    img_split = "train" if split in ("train", "ablation", "val") else "test"

    # 1. Download images (deduplicated)
    _download_images(dest, img_split, status_fn=status_fn)

    # 2. Download parquet files we need
    parquet_cache = dest / ".parquet_cache"
    parquet_cache.mkdir(parents=True, exist_ok=True)

    # Seqinfo
    seqinfo_path = _ensure_parquet(parquet_cache, "data/seqinfo/seqinfo.parquet")
    seqinfo_df = pd.read_parquet(seqinfo_path)

    # GT (only for splits that have GT)
    # For ablation, read the full train GT and filter ourselves (need original frames for renumbering)
    if split == "ablation":
        gt_parquet_name = "data/gt/train-00000-of-00001.parquet"
    else:
        gt_parquet_name = f"data/gt/{split}-00000-of-00001.parquet"
    gt_path = _ensure_parquet(parquet_cache, gt_parquet_name, required=False)
    gt_df = pd.read_parquet(gt_path) if gt_path else None

    # Public detections - same logic: for ablation read full train and filter
    if split == "ablation":
        det_parquet_name = f"data/detections/{detector.lower()}/train-00000-of-00001.parquet"
    else:
        det_parquet_name = f"data/detections/{detector.lower()}/{split}-00000-of-00001.parquet"
    det_path = _ensure_parquet(parquet_cache, det_parquet_name, required=False)
    det_df = pd.read_parquet(det_path) if det_path else None

    # 3. Build MOTChallenge layout
    # Filter seqinfo to the relevant sequences
    seq_info = seqinfo_df[seqinfo_df["split"] == img_split]

    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in seq_info.iterrows():
        seq_name = row["sequence"]  # e.g. MOT17-02
        seq_full = f"{seq_name}-{detector}"  # e.g. MOT17-02-FRCNN
        seq_dir = split_dir / seq_full
        total_len = int(row["seq_length"])

        # Compute frame range for this split
        if split == "ablation":
            abl_start = _ablation_start(total_len)
            frame_start = abl_start
            frame_end = total_len
            abl_length = frame_end - frame_start + 1
        else:
            frame_start = 1
            frame_end = total_len
            abl_length = total_len

        seq_dir.mkdir(parents=True, exist_ok=True)

        # img1/ - for ablation, create individual symlinks with renumbered names
        img1_dir = seq_dir / "img1"
        shared_img_dir = dest / "images" / img_split / seq_name / "img1"
        if not img1_dir.exists():
            if split == "ablation":
                # Create renumbered symlinks: 000001.jpg -> original_frame.jpg
                img1_dir.mkdir(parents=True, exist_ok=True)
                for new_frame_idx in range(1, abl_length + 1):
                    orig_frame = frame_start + new_frame_idx - 1
                    src = shared_img_dir / f"{orig_frame:06d}.jpg"
                    dst = img1_dir / f"{new_frame_idx:06d}.jpg"
                    if src.exists() and not dst.exists():
                        dst.symlink_to(src.resolve())
            else:
                # For train/test, symlink the whole directory
                if shared_img_dir.exists():
                    img1_dir.symlink_to(shared_img_dir.resolve())
                else:
                    LOGGER.warning(f"Image dir not found: {shared_img_dir}")

        # seqinfo.ini
        _write_seqinfo(seq_dir / "seqinfo.ini", seq_full, row, abl_length)

        # gt/gt.txt (with renumbered frames for ablation)
        if gt_df is not None:
            seq_gt = gt_df[gt_df["sequence"] == seq_name].copy()
            if split == "ablation":
                seq_gt = seq_gt[(seq_gt["frame"] >= frame_start) & (seq_gt["frame"] <= frame_end)]
                seq_gt = seq_gt.copy()
                seq_gt["frame"] = seq_gt["frame"] - frame_start + 1
            if not seq_gt.empty:
                gt_dir = seq_dir / "gt"
                gt_dir.mkdir(parents=True, exist_ok=True)
                _write_gt(gt_dir / "gt.txt", seq_gt)

        # det/det.txt (with renumbered frames for ablation)
        if det_df is not None:
            seq_det = det_df[det_df["sequence"] == seq_name].copy()
            if split == "ablation":
                seq_det = seq_det[(seq_det["frame"] >= frame_start) & (seq_det["frame"] <= frame_end)]
                seq_det = seq_det.copy()
                seq_det["frame"] = seq_det["frame"] - frame_start + 1
            if not seq_det.empty:
                det_dir = seq_dir / "det"
                det_dir.mkdir(parents=True, exist_ok=True)
                _write_det(det_dir / "det.txt", seq_det)

    # 4. Create .npy detection cache for public detections
    if det_df is not None:
        _create_det_npy_cache(dest, split, detector, det_df, seq_info)

    marker.touch()
    LOGGER.info(f"MOT17 parquet setup complete: {split_dir}")


def _download_images(dest: Path, img_split: str, status_fn: Any = None) -> None:
    """Download deduplicated images for the given split."""
    images_dir = dest / "images" / img_split
    marker = images_dir / ".hf_download_complete"
    if marker.exists():
        return

    snapshot_download, _ = _require_hf()

    msg = f"Downloading MOT17 images ({img_split})..."
    if status_fn:
        status_fn(msg)
    else:
        LOGGER.info(msg)

    snapshot_download(
        repo_id=PARQUET_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[f"images/{img_split}/**"],
    )

    images_dir.mkdir(parents=True, exist_ok=True)
    marker.touch()


def _ensure_parquet(cache_dir: Path, repo_path: str, required: bool = True) -> Path | None:
    """Download a single parquet file from the repo if not cached."""
    local = cache_dir / repo_path
    if local.exists():
        return local

    try:
        _, hf_hub_download = _require_hf()
        downloaded = hf_hub_download(
            repo_id=PARQUET_REPO,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(cache_dir),
        )
        return Path(downloaded)
    except Exception as e:
        if required:
            raise
        LOGGER.debug(f"Optional parquet not found: {repo_path} ({e})")
        return None


def _write_seqinfo(path: Path, seq_name: str, row: Any, seq_length: int) -> None:
    """Write a MOTChallenge seqinfo.ini file with case-preserving keys."""
    lines = [
        "[Sequence]",
        f"name={seq_name}",
        f"imDir=img1",
        f"frameRate={int(row['fps'])}",
        f"seqLength={seq_length}",
        f"imWidth={int(row['width'])}",
        f"imHeight={int(row['height'])}",
        f"imExt=.jpg",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_gt(path: Path, df: Any) -> None:
    """Write GT in MOTChallenge format: frame,id,bb_left,bb_top,bb_w,bb_h,conf,cls,vis."""
    cols = ["frame", "track_id", "bbox_left", "bbox_top", "bbox_width", "bbox_height",
            "conf", "class_id", "visibility"]
    df[cols].to_csv(path, index=False, header=False, float_format="%.6g")


def _write_det(path: Path, df: Any) -> None:
    """Write detections in MOTChallenge format: frame,-1,bb_left,bb_top,bb_w,bb_h,score,-1,-1,-1."""
    pd = _require_pd()
    out = pd.DataFrame({
        "frame": df["frame"],
        "id": -1,
        "bbox_left": df["bbox_left"],
        "bbox_top": df["bbox_top"],
        "bbox_width": df["bbox_width"],
        "bbox_height": df["bbox_height"],
        "score": df["score"],
        "_7": -1,
        "_8": -1,
        "_9": -1,
    })
    out.to_csv(path, index=False, header=False, float_format="%.6g")


def _create_det_npy_cache(
    dest: Path,
    split: str,
    detector: str,
    det_df: Any,
    seq_info: Any,
) -> None:
    """Create .npy detection cache in the runs/ layout.

    Format: (N, 7) float32 = [frame_id, x1, y1, x2, y2, conf, cls]
    Bbox is converted from xywh (MOTChallenge) to xyxy.
    Frames are renumbered to 1-based for ablation split.
    """
    det_key = f"mot17_public_{detector.lower()}"
    cache_dir = Path("runs") / "dets_n_embs" / "mot17" / split / det_key / "dets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build seq_length lookup
    seq_lengths = {}
    for _, row in seq_info.iterrows():
        seq_lengths[row["sequence"]] = int(row["seq_length"])

    for seq_name in sorted(det_df["sequence"].unique()):
        seq_full = f"{seq_name}-{detector}"
        npy_path = cache_dir / f"{seq_full}.npy"
        if npy_path.exists():
            continue

        seq_det = det_df[det_df["sequence"] == seq_name].copy().sort_values("frame")
        total_len = seq_lengths.get(seq_name, 0)

        # For ablation: filter to second half and renumber
        if split == "ablation" and total_len > 0:
            abl_start = _ablation_start(total_len)
            seq_det = seq_det[(seq_det["frame"] >= abl_start) & (seq_det["frame"] <= total_len)]
            seq_det = seq_det.copy()
            seq_det["frame"] = seq_det["frame"] - abl_start + 1

        if seq_det.empty:
            continue

        # Convert xywh -> xyxy
        frame_ids = seq_det["frame"].values.astype(np.float32)
        x1 = seq_det["bbox_left"].values.astype(np.float32)
        y1 = seq_det["bbox_top"].values.astype(np.float32)
        w = seq_det["bbox_width"].values.astype(np.float32)
        h = seq_det["bbox_height"].values.astype(np.float32)
        x2 = x1 + w
        y2 = y1 + h
        conf = seq_det["score"].values.astype(np.float32)
        cls = np.zeros_like(conf)  # class 0 = person

        arr = np.stack([frame_ids, x1, y1, x2, y2, conf, cls], axis=1)
        np.save(npy_path, arr)

    LOGGER.info(f"Public detection cache: {cache_dir}")
