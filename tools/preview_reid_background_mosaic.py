"""Render deterministic Market-1501 background-mosaic training previews."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from boxmot.reid.datasets.market1501 import Market1501
from boxmot.reid.datasets.transforms import IdentityPreservingBackgroundMosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Market-1501-v15.09.15"),
        help="Original Market-1501 dataset root.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("Market-1501-mosaic-highconf-masks"),
        help="High-confidence mask root generated with --masks-only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reid_mosaic_examples_highconf"),
        help="Empty output directory for examples and the contact sheet.",
    )
    parser.add_argument("--count", type=int, default=12, help="Number of mosaics to render.")
    parser.add_argument("--seed", type=int, default=19, help="Deterministic preview seed.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=(384, 128),
        metavar=("HEIGHT", "WIDTH"),
        help="Rendered model input size.",
    )
    return parser.parse_args()


def _generation_report(mask_root: Path) -> tuple[Path, dict]:
    suffix = "-masks"
    if not mask_root.name.endswith(suffix):
        raise ValueError(f"Mask directory must end with {suffix!r}: {mask_root}")
    report = mask_root.parent / f"{mask_root.name.removesuffix(suffix)}-masking-report.json"
    if not report.is_file():
        raise FileNotFoundError(f"Missing mask-generation report: {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    confidence = float(payload.get("settings", {}).get("conf", -1))
    if confidence < 0.5:
        raise ValueError(f"Preview requires masks generated at conf>=0.50, got {confidence:g}")
    if payload.get("masks_only") is not True:
        raise ValueError("Preview requires a dedicated --masks-only generation report")
    if payload.get("mask_layout", {}).get("primary") != "primary":
        raise ValueError("Preview requires the dual primary/all_people mask layout")
    if payload.get("mask_layout", {}).get("all_people") != "all_people":
        raise ValueError("Preview requires the dual primary/all_people mask layout")
    return report, payload


def _comparison_image(
    original: Image.Image,
    augmented: Image.Image,
    *,
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
    draw.text((original.width + 4, 5), f"mosaic | {label}", fill=(255, 255, 255))
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


def main() -> int:
    args = parse_args()
    if args.count < 1 or min(args.imgsz) < 1:
        raise ValueError("--count and --imgsz values must be positive")

    data_root = args.data_dir.expanduser().resolve()
    mask_root = args.mask_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output}")

    report_path, report = _generation_report(mask_root)
    dataset = Market1501(str(data_root), relabel_train=False)
    samples = dataset.train.samples
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=dataset.root,
        primary_mask_root=mask_root / "primary",
        donor_mask_root=mask_root / "all_people",
        probability=1.0,
        start_epoch=0,
        ramp_end_epoch=0,
        min_foreground_ratio=0.2,
        max_foreground_ratio=0.9,
        feather=1.5,
        dilation=2,
    )
    mosaic.set_epoch(1)

    candidate_indices = [
        index
        for index, sample in enumerate(samples)
        if (
            mask_root
            / "primary"
            / Path(sample.img_path).resolve().relative_to(dataset.root.resolve())
        ).with_suffix(".png").is_file()
    ]
    random.Random(args.seed).shuffle(candidate_indices)

    target_size = (args.imgsz[1], args.imgsz[0])
    examples: list[tuple[Image.Image, Image.Image, dict]] = []
    for candidate_order, sample_index in enumerate(candidate_indices):
        sample = samples[sample_index]
        with Image.open(sample.img_path) as handle:
            original = handle.convert("RGB")

        augmented = original
        for attempt in range(12):
            random.seed(args.seed + candidate_order * 101 + attempt)
            augmented = mosaic(original, sample_index)
            if not np.array_equal(np.asarray(augmented), np.asarray(original)):
                break
        else:
            continue

        examples.append(
            (
                original.resize(target_size, Image.Resampling.BILINEAR),
                augmented.resize(target_size, Image.Resampling.BILINEAR),
                {
                    "filename": Path(sample.img_path).name,
                    "pid": sample.pid,
                    "camera": sample.camid + 1,
                },
            )
        )
        if len(examples) == args.count:
            break

    if len(examples) != args.count:
        raise RuntimeError(
            f"Could only produce {len(examples)}/{args.count} valid mosaic previews"
        )

    originals_dir = output / "original"
    mosaics_dir = output / "mosaic"
    comparisons_dir = output / "comparison"
    for directory in (originals_dir, mosaics_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparisons = []
    manifest_examples = []
    for index, (original, augmented, metadata) in enumerate(examples, start=1):
        stem = Path(metadata["filename"]).stem
        output_name = f"{index:02d}_{stem}.jpg"
        original.save(originals_dir / output_name, quality=95)
        augmented.save(mosaics_dir / output_name, quality=95)
        comparison = _comparison_image(
            original,
            augmented,
            label=f"pid={metadata['pid']} cam={metadata['camera']}",
        )
        comparison.save(comparisons_dir / output_name, quality=95)
        comparisons.append(comparison)
        manifest_examples.append({"output": output_name, **metadata})

    sheet = _contact_sheet(comparisons)
    sheet.save(output / "contact_sheet.jpg", quality=95)
    manifest = {
        "data_root": str(dataset.root.resolve()),
        "mask_root": str(mask_root),
        "primary_mask_root": str(mask_root / "primary"),
        "donor_mask_root": str(mask_root / "all_people"),
        "mask_report": str(report_path),
        "mask_generation_confidence": report["settings"]["conf"],
        "seed": args.seed,
        "rendered_size": list(args.imgsz),
        "mosaic": {
            "probability": 1.0,
            "min_foreground_ratio": 0.2,
            "max_foreground_ratio": 0.9,
            "feather": 1.5,
            "dilation": 2,
        },
        "examples": manifest_examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(examples)} mosaic previews to {output}")
    print(f"Contact sheet: {output / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
