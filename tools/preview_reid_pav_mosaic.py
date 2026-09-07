"""Render deterministic Market-1501 PAV-Mosaic training previews."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from boxmot.reid.datasets.market1501 import Market1501
from boxmot.reid.datasets.pav_mosaic import PoseAlignedViewMosaic
from boxmot.reid.datasets.transforms import (
    IdentityPreservingBackgroundMosaic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/reid/Market-1501-v15.09.15"),
        help="Original Market-1501 dataset root.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("Market-1501-pav-metadata"),
        help="Pose, person-mask, and optional bag-mask metadata root.",
    )
    parser.add_argument(
        "--background-mask-dir",
        type=Path,
        default=None,
        help=(
            "Optional high-confidence primary/all_people mask root. When "
            "provided, context mosaic is applied before PAV as in training."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reid_pav_examples"),
        help="Empty output directory for examples and the contact sheet.",
    )
    parser.add_argument("--count", type=int, default=12, help="Number of PAV mosaics to render.")
    parser.add_argument("--seed", type=int, default=29, help="Deterministic preview seed.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=16,
        help="Candidate augmentations tried per identity; the clearest one is retained.",
    )
    parser.add_argument(
        "--candidate-identities",
        type=int,
        default=80,
        help="Maximum number of same-ID, cross-camera groups to inspect.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=(384, 128),
        metavar=("HEIGHT", "WIDTH"),
        help="Rendered model input size.",
    )
    parser.add_argument(
        "--selection",
        choices=("representative", "largest-change"),
        default="largest-change",
        help=("Keep the first valid draw per identity or the largest pixel change across all attempts."),
    )
    parser.add_argument("--max-parts", type=int, default=3)
    parser.add_argument(
        "--max-foreground-replacement",
        type=float,
        default=0.45,
    )
    parser.add_argument("--cross-camera-rate", type=float, default=1.0)
    parser.add_argument("--different-pose-rate", type=float, default=1.0)
    parser.add_argument(
        "--min-keypoint-confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument("--feather", type=float, default=0.8)
    return parser.parse_args()


def _comparison_image(
    original: Image.Image,
    augmented: Image.Image,
    *,
    augmentation_name: str,
    label: str,
) -> Image.Image:
    header_height = 22
    canvas = Image.new(
        "RGB",
        (original.width + augmented.width, original.height + header_height),
        (32, 32, 32),
    )
    canvas.paste(original, (0, header_height))
    canvas.paste(augmented, (original.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), "original", fill=(255, 255, 255))
    draw.text(
        (original.width + 4, 5),
        f"{augmentation_name} | {label}",
        fill=(255, 255, 255),
    )
    return canvas


def _contact_sheet(comparisons: list[Image.Image], columns: int = 3) -> Image.Image:
    tile_width = max(image.width for image in comparisons)
    tile_height = max(image.height for image in comparisons)
    rows = (len(comparisons) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (16, 16, 16))
    for index, comparison in enumerate(comparisons):
        x = index % columns * tile_width
        y = index // columns * tile_height
        sheet.paste(comparison, (x, y))
    return sheet


def _difference_score(original: Image.Image, augmented: Image.Image) -> float:
    left = np.asarray(original, dtype=np.int16)
    right = np.asarray(augmented, dtype=np.int16)
    return float(np.abs(left - right).mean())


def main() -> int:
    args = parse_args()
    if min(args.count, args.attempts, args.candidate_identities, *args.imgsz) < 1:
        raise ValueError("count, attempts, candidate identities, and image size must be positive")
    if not 1 <= args.max_parts <= 7:
        raise ValueError("max parts must be in [1, 7]")
    if not 0 < args.max_foreground_replacement <= 1:
        raise ValueError("maximum foreground replacement must be in (0, 1]")
    if not 0 <= args.cross_camera_rate <= 1:
        raise ValueError("cross-camera rate must be in [0, 1]")
    if not 0 <= args.different_pose_rate <= 1:
        raise ValueError("different-pose rate must be in [0, 1]")
    if not 0 <= args.min_keypoint_confidence <= 1:
        raise ValueError("minimum keypoint confidence must be in [0, 1]")
    if args.feather < 0:
        raise ValueError("feather must be non-negative")

    data_root = args.data_dir.expanduser().resolve()
    metadata_root = args.metadata_dir.expanduser().resolve()
    background_mask_root = None if args.background_mask_dir is None else args.background_mask_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output}")

    dataset = Market1501(str(data_root), relabel_train=False)
    samples = dataset.train.samples
    pav = PoseAlignedViewMosaic(
        samples,
        image_root=dataset.root,
        metadata_root=metadata_root,
        probability=1.0,
        max_parts=args.max_parts,
        max_foreground_replacement=args.max_foreground_replacement,
        cross_camera_rate=args.cross_camera_rate,
        different_pose_rate=args.different_pose_rate,
        min_keypoint_confidence=args.min_keypoint_confidence,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
        feather=args.feather,
    )
    pav.set_epoch(1)
    context_mosaic = None
    background_eligible: set[int] | None = None
    if background_mask_root is not None:
        primary_mask_root = background_mask_root / "primary"
        donor_mask_root = background_mask_root / "all_people"
        if not primary_mask_root.is_dir() or not donor_mask_root.is_dir():
            raise FileNotFoundError(
                f"Background mosaic requires primary/ and all_people/ mask trees under {background_mask_root}"
            )
        context_mosaic = IdentityPreservingBackgroundMosaic(
            samples,
            image_root=dataset.root,
            primary_mask_root=primary_mask_root,
            donor_mask_root=donor_mask_root,
            probability=1.0,
            start_epoch=0,
            ramp_end_epoch=0,
            min_foreground_ratio=0.2,
            max_foreground_ratio=0.9,
            feather=1.5,
            dilation=2,
        )
        context_mosaic.set_epoch(1)
        dataset_root = dataset.root.resolve()
        background_eligible = {
            index
            for index, sample in enumerate(samples)
            if (primary_mask_root / Path(sample.img_path).resolve().relative_to(dataset_root))
            .with_suffix(".png")
            .is_file()
        }

    indices_by_pid: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        record = pav._record(index)
        if isinstance(record, dict) and record.get("person_mask"):
            indices_by_pid[sample.pid].append(index)
    candidate_groups = [
        indices
        for indices in indices_by_pid.values()
        if len(indices) >= 2 and len({samples[index].camid for index in indices}) >= 2
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidate_groups)
    candidate_groups = candidate_groups[: args.candidate_identities]

    examples: list[tuple[float, Image.Image, Image.Image, dict]] = []
    for group_order, group in enumerate(candidate_groups):
        anchor_candidates = [
            index
            for index in group
            if (background_eligible is None or index in background_eligible)
            and any(samples[other].camid != samples[index].camid for other in group)
        ]
        if not anchor_candidates:
            continue
        anchor_index = rng.choice(anchor_candidates)
        sample = samples[anchor_index]
        with Image.open(sample.img_path) as handle:
            original = handle.convert("RGB")

        best_augmented: Image.Image | None = None
        best_score = -1.0
        for attempt in range(args.attempts):
            random.seed(args.seed + group_order * 1009 + attempt)
            pav_input = original
            if context_mosaic is not None:
                pav_input = context_mosaic(original, anchor_index)
                if np.array_equal(
                    np.asarray(pav_input),
                    np.asarray(original),
                ):
                    continue
            augmented, applied = pav.apply_with_status(
                pav_input,
                anchor_index,
            )
            if not applied:
                continue
            score = _difference_score(original, augmented)
            if args.selection == "representative":
                best_augmented = augmented
                best_score = score
                break
            if score > best_score:
                best_augmented = augmented
                best_score = score
        if best_augmented is None:
            continue

        donor_cameras = sorted({samples[index].camid + 1 for index in group if samples[index].camid != sample.camid})
        examples.append(
            (
                best_score,
                original,
                best_augmented,
                {
                    "filename": Path(sample.img_path).name,
                    "pid": sample.pid,
                    "camera": sample.camid + 1,
                    "available_cross_camera_donors": donor_cameras,
                    "mean_absolute_pixel_change": round(best_score, 3),
                },
            )
        )

    if args.selection == "largest-change":
        examples.sort(key=lambda item: item[0], reverse=True)
    examples = examples[: args.count]
    if len(examples) != args.count:
        raise RuntimeError(
            f"Could only produce {len(examples)}/{args.count} valid PAV previews "
            f"from {len(candidate_groups)} cross-camera identity groups"
        )

    target_size = (args.imgsz[1], args.imgsz[0])
    originals_dir = output / "original"
    augmentation_name = "PAV+context" if context_mosaic is not None else "PAV"
    pav_dir = output / ("pav_context" if context_mosaic is not None else "pav")
    comparisons_dir = output / "comparison"
    for directory in (originals_dir, pav_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparisons: list[Image.Image] = []
    manifest_examples: list[dict] = []
    for index, (_, original_native, augmented_native, metadata) in enumerate(examples, start=1):
        original = original_native.resize(target_size, Image.Resampling.BILINEAR)
        augmented = augmented_native.resize(target_size, Image.Resampling.BILINEAR)
        stem = Path(metadata["filename"]).stem
        output_name = f"{index:02d}_{stem}.jpg"
        original.save(originals_dir / output_name, quality=95)
        augmented.save(pav_dir / output_name, quality=95)
        comparison = _comparison_image(
            original,
            augmented,
            augmentation_name=augmentation_name,
            label=f"pid={metadata['pid']} cam={metadata['camera']}",
        )
        comparison.save(comparisons_dir / output_name, quality=95)
        comparisons.append(comparison)
        manifest_examples.append({"output": output_name, **metadata})

    sheet = _contact_sheet(comparisons)
    sheet.save(output / "contact_sheet.jpg", quality=95)
    manifest = {
        "data_root": str(dataset.root.resolve()),
        "metadata_root": str(metadata_root),
        "metadata_records_at_render": len(pav.records),
        "background_mask_root": (None if background_mask_root is None else str(background_mask_root)),
        "seed": args.seed,
        "rendered_size": list(args.imgsz),
        "selection": {
            "attempts_per_identity": args.attempts,
            "candidate_identities": len(candidate_groups),
            "mode": args.selection,
            "ranking": (
                "mean absolute pixel change"
                if args.selection == "largest-change"
                else "deterministically shuffled identities; first valid draw"
            ),
        },
        "pav": {
            "probability": 1.0,
            "max_parts": args.max_parts,
            "max_foreground_replacement": args.max_foreground_replacement,
            "cross_camera_rate": args.cross_camera_rate,
            "different_pose_rate": args.different_pose_rate,
            "min_keypoint_confidence": args.min_keypoint_confidence,
            "feather": args.feather,
        },
        "background_mosaic": (
            None
            if context_mosaic is None
            else {
                "preview_probability": 1.0,
                "training_probability": 0.20,
                "min_foreground_ratio": 0.2,
                "max_foreground_ratio": 0.9,
                "feather": 1.5,
                "dilation": 2,
                "order": "before PAV",
            }
        ),
        "examples": manifest_examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(examples)} PAV previews to {output}")
    print(f"Contact sheet: {output / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
