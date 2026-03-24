from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for raw-sequence to MOT17 conversion."""
    parser = argparse.ArgumentParser(
        description="Convert raw `.npy` + per-frame polygon `.txt` sequences into MOT17-style folders.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more raw sequence directories, e.g. data44-3 data42-3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/MOT17-mini/train"),
        help="Destination root where MOT-style sequences will be written.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Channel indices to extract from the source `.npy` arrays for RGB output.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Frame rate written to seqinfo.ini.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination sequence directories if they already exist.",
    )
    return parser.parse_args()


def aabb_from_polygon_row(parts: Sequence[str]) -> tuple[float, float, float, float]:
    """Convert x1,y1,...,x4,y4 polygon coordinates into axis-aligned x,y,w,h."""
    xs = [float(parts[i]) for i in (2, 4, 6, 8)]
    ys = [float(parts[i]) for i in (3, 5, 7, 9)]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return x1, y1, x2 - x1, y2 - y1


def iter_annotation_rows(txt_paths: Iterable[Path]) -> Iterable[list[str]]:
    """Yield parsed annotation rows from the raw per-frame text files."""
    for txt_path in txt_paths:
        with txt_path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = [value.strip() for value in line.split(",")]
                if len(parts) != 13:
                    raise ValueError(
                        f"Unexpected annotation format in {txt_path}: {line}"
                    )
                yield parts


def write_sequence(
    source_dir: Path,
    output_root: Path,
    channels: Sequence[int],
    frame_rate: int,
    overwrite: bool,
) -> Path:
    """Convert a single raw sequence directory into MOT17-style layout."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    npy_files = sorted(source_dir.glob("*.npy"))
    txt_files = sorted(source_dir.glob("*.txt"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {source_dir}")
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")

    first = np.load(npy_files[0])
    if first.ndim != 3:
        raise ValueError(f"Unexpected image shape in {npy_files[0]}: {first.shape}")

    height, width, channel_count = first.shape
    if max(channels) >= channel_count:
        raise ValueError(
            f"Sequence {source_dir.name} has {channel_count} channels, cannot select {list(channels)}"
        )

    destination = output_root / source_dir.name
    img_dir = destination / "img1"
    gt_dir = destination / "gt"
    det_dir = destination / "det"
    seqinfo_path = destination / "seqinfo.ini"

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)

    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    for npy_path in npy_files:
        frame = np.load(npy_path)
        if frame.ndim != 3:
            raise ValueError(f"Unexpected frame shape in {npy_path}: {frame.shape}")
        rgb = np.ascontiguousarray(frame[:, :, list(channels)])
        image_path = img_dir / f"{npy_path.stem}.jpg"
        if not cv2.imwrite(str(image_path), rgb):
            raise RuntimeError(f"Failed writing {image_path}")

    gt_rows = []
    det_rows = []
    for parts in iter_annotation_rows(txt_files):
        frame_id = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = aabb_from_polygon_row(parts)
        confidence = float(parts[10])
        cls = int(float(parts[11]))
        truncation = float(parts[12])

        gt_rows.append(
            f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls},{max(0.0, 1.0 - truncation):.6f}"
        )
        det_score = confidence if confidence >= 0 else 1.0
        det_rows.append(
            f"{frame_id},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{det_score:.6f}"
        )

    (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + ("\n" if gt_rows else ""))
    (det_dir / "det.txt").write_text("\n".join(det_rows) + ("\n" if det_rows else ""))
    seqinfo_path.write_text(
        "[Sequence]\n"
        f"name={source_dir.name}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={len(npy_files)}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )
    return destination


def main() -> None:
    """Run the raw-to-MOT17 conversion CLI."""
    args = parse_args()
    for source in args.sources:
        destination = write_sequence(
            source_dir=Path(source),
            output_root=args.output_root,
            channels=args.channels,
            frame_rate=args.frame_rate,
            overwrite=args.overwrite,
        )
        print(destination)


if __name__ == "__main__":
    main()