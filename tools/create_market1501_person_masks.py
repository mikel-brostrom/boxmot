"""Create a gray-background Market-1501 dataset using person segmentation.

The output retains the original Market-1501 directory structure and filenames.
Foreground RGB pixels from the selected person and their nearby bags are
preserved; all other pixels are replaced by a configurable neutral gray.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
COCO_BAG_CLASSES = (24, 26, 28)  # backpack, handbag, suitcase
MARKET_ROOT_NAMES = (
    "Market-1501-v15.09.15",
    "Market-1501",
    "market1501",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment the primary person in every Market-1501 image and replace "
            "the background with neutral gray."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Market-1501-v15.09.15"),
        help="Market-1501 root or a parent containing the dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Market-1501-gray"),
        help="Destination dataset root. Original Market-1501 directories are created directly inside it.",
    )
    parser.add_argument(
        "--model",
        default="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
        help="Ultralytics segmentation checkpoint. The default is downloaded automatically.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device accepted by Ultralytics, for example 0, mps, or cpu.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Segmentation inference image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per inference batch.")
    parser.add_argument("--conf", type=float, default=0.05, help="Minimum person confidence.")
    parser.add_argument("--iou", type=float, default=0.7, help="Segmentation NMS IoU threshold.")
    parser.add_argument(
        "--person-class",
        type=int,
        default=0,
        help="Person class index in the segmentation model.",
    )
    parser.add_argument(
        "--bag-classes",
        type=int,
        nargs="*",
        default=COCO_BAG_CLASSES,
        help=(
            "Bag class indices to preserve when their masks are near the selected person. "
            "Defaults to COCO backpack, handbag, and suitcase (24, 26, 28). "
            "Pass --bag-classes with no values to disable."
        ),
    )
    parser.add_argument(
        "--bag-proximity",
        type=float,
        default=0.05,
        help=(
            "Maximum person-to-bag mask gap as a fraction of the longer image dimension. "
            "Nearby bag masks are merged without dilating the saved foreground."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to binarize predicted masks.",
    )
    parser.add_argument(
        "--gray",
        type=int,
        default=127,
        help="Background gray value in [0, 255].",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="Optional foreground-mask dilation radius in pixels.",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=0.0,
        help="Optional Gaussian edge feathering sigma in pixels.",
    )
    parser.add_argument(
        "--on-missing",
        choices=("fail", "copy", "gray"),
        default="fail",
        help=(
            "Behavior when no person mask is found: fail leaves the output image absent, "
            "copy preserves it, and gray replaces the complete image."
        ),
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help=(
            "Also save binary PNG masks beside the cloned dataset in a separate "
            "<dataset>-person-masks directory."
        ),
    )
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help=(
            "Generate separate primary/ and all_people/ bounding_box_train mask "
            "trees for augmentation use. No gray dataset is written, --save-masks "
            "is implied, and images without an accepted primary person are omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images that already exist in the destination.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit for a smoke run; zero processes the full dataset.",
    )
    return parser.parse_args()


def resolve_market_root(source: Path) -> Path:
    """Resolve either an exact Market root or its parent directory."""
    source = source.expanduser().resolve()
    candidates = (source, *(source / name for name in MARKET_ROOT_NAMES))
    for candidate in candidates:
        if (candidate / "bounding_box_train").is_dir() and (candidate / "query").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find Market-1501 under {source}. Expected bounding_box_train/ and query/."
    )


def discover_images(root: Path) -> list[Path]:
    """Return every image in the Market-1501 tree in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_paths(source: Path, output: Path) -> None:
    """Prevent accidental source overwrite or recursive output discovery."""
    output = output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError(f"Output must be outside the source dataset: source={source}, output={output}")


def resolve_output_root(output: Path) -> Path:
    """Resolve the exact destination dataset root requested by the user."""
    return output.expanduser().resolve()


def select_mosaic_person_masks(
    result: Any,
    image_shape: tuple[int, int],
    *,
    person_class: int,
    mask_threshold: float,
    bag_classes: tuple[int, ...] = (),
    bag_proximity: float = 0.05,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return primary-anchor and all-people donor-cleanup masks."""
    if result.boxes is None or result.masks is None:
        return None
    if bag_proximity < 0:
        raise ValueError("bag_proximity must be non-negative")

    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    height, width = image_shape
    image_center = np.array((width / 2.0, height / 2.0), dtype=np.float32)
    image_diagonal = max(float(np.hypot(width, height)), 1.0)

    best_mask: np.ndarray | None = None
    best_score = -1.0
    person_masks: list[np.ndarray] = []
    for index in np.flatnonzero(classes == person_class):
        mask = masks[index]
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        binary_mask = mask >= mask_threshold
        area = float(binary_mask.sum())
        if area == 0:
            continue
        person_masks.append(binary_mask)

        x1, y1, x2, y2 = boxes[index]
        box_center = np.array(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float32)
        normalized_distance = float(np.linalg.norm(box_center - image_center) / image_diagonal)
        centrality = 1.0 / (1.0 + 4.0 * normalized_distance * normalized_distance)
        score = float(confidences[index]) * np.sqrt(area) * centrality
        if score > best_score:
            best_score = score
            best_mask = binary_mask

    if best_mask is None:
        return None
    all_people_mask = np.logical_or.reduce(person_masks)
    if not bag_classes:
        return best_mask, all_people_mask

    proximity_pixels = int(round(max(height, width) * bag_proximity))
    if proximity_pixels > 0:
        kernel_size = 2 * proximity_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        person_neighborhood = cv2.dilate(best_mask.astype(np.uint8), kernel).astype(bool)
        all_people_neighborhood = cv2.dilate(
            all_people_mask.astype(np.uint8),
            kernel,
        ).astype(bool)
    else:
        person_neighborhood = best_mask
        all_people_neighborhood = all_people_mask

    bag_class_array = np.asarray(bag_classes, dtype=np.int64)
    for index in np.flatnonzero(np.isin(classes, bag_class_array)):
        bag_mask = masks[index]
        if bag_mask.shape != (height, width):
            bag_mask = cv2.resize(bag_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        binary_bag_mask = bag_mask >= mask_threshold
        if binary_bag_mask.any() and np.any(binary_bag_mask & person_neighborhood):
            best_mask = best_mask | binary_bag_mask
        if binary_bag_mask.any() and np.any(binary_bag_mask & all_people_neighborhood):
            all_people_mask = all_people_mask | binary_bag_mask
    return best_mask, all_people_mask


def gray_background(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    gray: int,
    dilate: int = 0,
    feather: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite an RGB/BGR image over neutral gray and return the effective mask."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel image, got {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")

    effective_mask = mask.astype(np.uint8)
    if dilate > 0:
        kernel_size = 2 * dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        effective_mask = cv2.dilate(effective_mask, kernel)

    gray_image = np.full_like(image, gray)
    if feather > 0:
        alpha = cv2.GaussianBlur(effective_mask.astype(np.float32), (0, 0), feather)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        output = image.astype(np.float32) * alpha + gray_image.astype(np.float32) * (1.0 - alpha)
        return np.rint(output).clip(0, 255).astype(np.uint8), effective_mask.astype(bool)

    output = np.where(effective_mask[..., None].astype(bool), image, gray_image)
    return output, effective_mask.astype(bool)


def clone_structure_and_non_images(source: Path, output: Path) -> None:
    """Clone the source directory tree and all files that are not images."""
    for source_path in source.rglob("*"):
        destination = output / source_path.relative_to(source)
        if source_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not source_path.is_file() or source_path.suffix.lower() in IMAGE_SUFFIXES:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)


def verify_clone_layout(source: Path, output: Path) -> None:
    """Require the cloned dataset to have exactly the source paths."""
    source_files = {
        path.relative_to(source)
        for path in source.rglob("*")
        if path.is_file()
    }
    output_files = {
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file()
    }
    source_dirs = {
        path.relative_to(source)
        for path in source.rglob("*")
        if path.is_dir()
    }
    output_dirs = {
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_dir()
    }
    missing_files = sorted(source_files - output_files)
    extra_files = sorted(output_files - source_files)
    missing_dirs = sorted(source_dirs - output_dirs)
    extra_dirs = sorted(output_dirs - source_dirs)
    if missing_files or extra_files or missing_dirs or extra_dirs:
        raise RuntimeError(
            "Output is not an exact Market-1501 layout clone: "
            f"missing_files={len(missing_files)}, extra_files={len(extra_files)}, "
            f"missing_dirs={len(missing_dirs)}, extra_dirs={len(extra_dirs)}"
        )


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image atomically using quality-preserving codec settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    parameters: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif path.suffix.lower() == ".png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(temporary), image, parameters):
        raise OSError(f"Failed to write image: {temporary}")
    os.replace(temporary, path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask as an 8-bit PNG."""
    write_image(path.with_suffix(".png"), mask.astype(np.uint8) * 255)


def chunks(paths: list[Path], size: int):
    """Yield deterministic fixed-size path batches."""
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.imgsz < 1:
        raise ValueError("--batch-size and --imgsz must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or not 0 <= args.mask_threshold <= 1:
        raise ValueError("--conf, --iou, and --mask-threshold must be in [0, 1]")
    if not 0 <= args.gray <= 255:
        raise ValueError("--gray must be in [0, 255]")
    if args.dilate < 0 or args.feather < 0 or args.bag_proximity < 0:
        raise ValueError("--dilate, --feather, and --bag-proximity must be non-negative")

    source = resolve_market_root(args.source)
    output = resolve_output_root(args.output)
    validate_paths(source, output)
    images = discover_images(source)
    if args.masks_only:
        images = discover_images(source / "bounding_box_train")
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No supported images found under {source}")

    if not args.masks_only:
        output.mkdir(parents=True, exist_ok=True)
        clone_structure_and_non_images(source, output)
    mask_output = output.parent / (
        f"{output.name}-masks"
        if args.masks_only
        else f"{output.name}-person-masks"
    )
    primary_mask_output = mask_output / "primary" if args.masks_only else mask_output
    all_people_mask_output = mask_output / "all_people" if args.masks_only else None
    save_masks = args.save_masks or args.masks_only

    from ultralytics import YOLO

    model = YOLO(args.model)
    completed = 0
    skipped_existing = 0
    missing: list[str] = []

    pending = []
    for source_path in images:
        relative_path = source_path.relative_to(source)
        destination = output / relative_path
        primary_mask_destination = (primary_mask_output / relative_path).with_suffix(".png")
        all_people_mask_destination = (
            None
            if all_people_mask_output is None
            else (all_people_mask_output / relative_path).with_suffix(".png")
        )
        masks_complete = primary_mask_destination.is_file() and (
            all_people_mask_destination is None or all_people_mask_destination.is_file()
        )
        output_complete = masks_complete if args.masks_only else destination.exists()
        if output_complete and not args.overwrite:
            if not args.masks_only:
                shutil.copystat(source_path, destination)
            skipped_existing += 1
        else:
            pending.append(source_path)

    with tqdm(
        total=len(images),
        initial=skipped_existing,
        desc=(
            "Generating Market-1501 mosaic masks"
            if args.masks_only
            else "Masking Market-1501"
        ),
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(written=completed, missing=0, existing=skipped_existing)
        for batch_paths in chunks(pending, args.batch_size):
            predict_kwargs = {
                "source": [str(path) for path in batch_paths],
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "classes": sorted({args.person_class, *args.bag_classes}),
                "batch": args.batch_size,
                "retina_masks": True,
                "verbose": False,
            }
            if args.device is not None:
                predict_kwargs["device"] = args.device
            results = model.predict(**predict_kwargs)
            if len(results) != len(batch_paths):
                raise RuntimeError(
                    f"Segmentation returned {len(results)} results for {len(batch_paths)} inputs"
                )

            for source_path, result in zip(batch_paths, results, strict=True):
                relative_path = source_path.relative_to(source)
                destination = output / relative_path
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Failed to read image: {source_path}")
                selected_masks = select_mosaic_person_masks(
                    result,
                    image.shape[:2],
                    person_class=args.person_class,
                    mask_threshold=args.mask_threshold,
                    bag_classes=tuple(args.bag_classes),
                    bag_proximity=args.bag_proximity,
                )
                if selected_masks is None:
                    missing.append(relative_path.as_posix())
                    if args.masks_only:
                        continue
                    if args.on_missing == "fail":
                        continue
                    if args.on_missing == "copy":
                        output_image = image
                        effective_mask = np.ones(image.shape[:2], dtype=bool)
                    else:
                        output_image = np.full_like(image, args.gray)
                        effective_mask = np.zeros(image.shape[:2], dtype=bool)
                else:
                    primary_mask, all_people_mask = selected_masks
                    output_image, effective_mask = gray_background(
                        image,
                        primary_mask,
                        gray=args.gray,
                        dilate=args.dilate,
                        feather=args.feather,
                    )
                    _, effective_all_people_mask = gray_background(
                        image,
                        all_people_mask,
                        gray=args.gray,
                        dilate=args.dilate,
                        feather=args.feather,
                    )
                if not args.masks_only:
                    write_image(destination, output_image)
                    shutil.copystat(source_path, destination)
                if save_masks:
                    write_mask(primary_mask_output / relative_path, effective_mask)
                    if all_people_mask_output is not None:
                        write_mask(
                            all_people_mask_output / relative_path,
                            effective_all_people_mask,
                        )
                completed += 1
            progress.update(len(batch_paths))
            progress.set_postfix(
                written=completed,
                missing=len(missing),
                existing=skipped_existing,
                refresh=False,
            )

    report = {
        "source": str(source),
        "output": str(output),
        "model": args.model,
        "masks_only": args.masks_only,
        "scope": ["bounding_box_train"] if args.masks_only else ["all"],
        "mask_layout": (
            {
                "root": str(mask_output),
                "primary": "primary",
                "all_people": "all_people",
            }
            if args.masks_only
            else {"root": str(mask_output), "primary": "."}
        ),
        "settings": {
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "conf": args.conf,
            "iou": args.iou,
            "person_class": args.person_class,
            "bag_classes": args.bag_classes,
            "bag_proximity": args.bag_proximity,
            "mask_threshold": args.mask_threshold,
            "gray": args.gray,
            "dilate": args.dilate,
            "feather": args.feather,
            "on_missing": args.on_missing,
        },
        "images_selected": len(images),
        "images_written": completed,
        "images_skipped_existing": skipped_existing,
        "missing_masks": missing,
    }
    report_path = output.parent / f"{output.name}-masking-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"Finished: written={completed}, existing={skipped_existing}, "
        f"missing_masks={len(missing)}, report={report_path}"
    )
    if missing and args.masks_only:
        print(
            "High-confidence mask-only generation intentionally omitted "
            f"{len(missing)} images; mosaic will leave those anchors unchanged."
        )
    elif missing and args.on_missing == "fail":
        print(
            "Conversion is incomplete because --on-missing=fail. "
            "Review masking_report.json and retry with a lower --conf or another fallback."
        )
        return 2
    if args.max_images == 0 and not args.masks_only:
        verify_clone_layout(source, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
