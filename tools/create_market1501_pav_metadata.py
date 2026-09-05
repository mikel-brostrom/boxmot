"""Generate high-confidence pose and foreground metadata for PAV-Mosaic."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from tools.create_market1501_person_masks import (
    COCO_BAG_CLASSES,
    chunks,
    discover_images,
    resolve_market_root,
    write_mask,
)

DEFAULT_POSE_MODEL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt"
DEFAULT_SEG_MODEL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized COCO keypoints plus high-confidence person and "
            "bag masks for Market-1501 PAV-Mosaic training."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Market-1501-v15.09.15"),
        help="Market-1501 root or a parent containing it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Market-1501-pav-metadata"),
        help="Destination metadata root.",
    )
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--seg-model", default=DEFAULT_SEG_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--seg-conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--person-class", type=int, default=0)
    parser.add_argument("--bag-classes", type=int, nargs="*", default=COCO_BAG_CLASSES)
    parser.add_argument("--bag-proximity", type=float, default=0.05)
    parser.add_argument(
        "--min-primary-area",
        type=float,
        default=0.35,
        help="Reject a primary pose whose box covers less than this image fraction.",
    )
    parser.add_argument(
        "--max-primary-center-distance",
        type=float,
        default=0.35,
        help="Maximum primary-box center distance from image center, normalized to the half-diagonal.",
    )
    parser.add_argument(
        "--min-primary-score-margin",
        type=float,
        default=1.25,
        help="Minimum best/runner-up primary-pose score ratio when multiple people are detected.",
    )
    parser.add_argument(
        "--min-pose-seg-iou",
        type=float,
        default=0.50,
        help="Minimum box IoU between the primary pose and its person segmentation.",
    )
    parser.add_argument(
        "--min-pose-mask-agreement",
        type=float,
        default=0.75,
        help="Minimum fraction of reliable pose keypoints inside the matched person mask.",
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.50,
        help="Keypoint confidence used by pose-mask agreement checks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class PoseSelection:
    """Primary pose plus crowd-ambiguity diagnostics."""

    keypoints: np.ndarray
    box: np.ndarray
    confidence: float
    candidate_count: int
    primary_score: float
    runner_up_score: float | None
    score_margin: float | None
    area_fraction: float
    center_distance: float


@dataclass(frozen=True)
class SegmentationSelection:
    """Pose-matched person mask plus instance-matching diagnostics."""

    person_mask: np.ndarray | None
    bag_mask: np.ndarray
    person_candidate_count: int
    bag_candidate_count: int
    matched_box: np.ndarray | None
    matched_confidence: float | None
    matched_iou: float | None
    matched_overlap: float | None


def _box_area(box: np.ndarray) -> float:
    """Return the non-negative area of an xyxy box."""
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    intersection_width = max(
        min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])),
        0.0,
    )
    intersection_height = max(
        min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])),
        0.0,
    )
    intersection = intersection_width * intersection_height
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / max(union, 1e-9)


def _box_overlap_over_smaller(left: np.ndarray, right: np.ndarray) -> float:
    """Return box intersection divided by the smaller box area."""
    intersection_width = max(
        min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])),
        0.0,
    )
    intersection_height = max(
        min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])),
        0.0,
    )
    intersection = intersection_width * intersection_height
    return intersection / max(min(_box_area(left), _box_area(right)), 1e-9)


def _box_quality(box: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """Return box area fraction and center distance normalized to half-diagonal."""
    height, width = shape
    area_fraction = _box_area(box) / max(float(height * width), 1.0)
    image_center = np.array((width / 2.0, height / 2.0), dtype=np.float32)
    box_center = np.array(
        ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
        dtype=np.float32,
    )
    half_diagonal = max(float(np.hypot(width, height)) * 0.5, 1.0)
    center_distance = float(np.linalg.norm(box_center - image_center) / half_diagonal)
    return area_fraction, center_distance


def _centrality_score(box: np.ndarray, confidence: float, shape: tuple[int, int]) -> float:
    height, width = shape
    center = np.array((width / 2.0, height / 2.0), dtype=np.float32)
    box_center = np.array(
        ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
        dtype=np.float32,
    )
    diagonal = max(float(np.hypot(width, height)), 1.0)
    distance = float(np.linalg.norm(box_center - center) / diagonal)
    area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)
    return confidence * np.sqrt(area) / (1.0 + 4.0 * distance * distance)


def select_primary_pose(
    result: Any,
    image_shape: tuple[int, int],
) -> PoseSelection | None:
    """Return keypoints, bounding box, and detector confidence for the anchor."""
    if result.boxes is None or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    points = result.keypoints.xy.detach().cpu().numpy()
    keypoint_conf = getattr(result.keypoints, "conf", None)
    if keypoint_conf is None:
        point_confidences = np.ones(points.shape[:2], dtype=np.float32)
    else:
        point_confidences = keypoint_conf.detach().cpu().numpy()
    if not len(boxes) or points.ndim != 3 or points.shape[1:] != (17, 2):
        return None
    scores = np.asarray(
        [
            _centrality_score(box, float(confidence), image_shape)
            for box, confidence in zip(boxes, confidences, strict=True)
        ],
        dtype=np.float64,
    )
    ranked_indices = np.argsort(scores)[::-1]
    index = int(ranked_indices[0])
    runner_up_score = float(scores[ranked_indices[1]]) if len(ranked_indices) > 1 else None
    score_margin = float(scores[index] / max(runner_up_score, 1e-12)) if runner_up_score is not None else None
    keypoints = np.concatenate(
        (points[index], point_confidences[index, :, None]),
        axis=1,
    ).astype(np.float32)
    box = boxes[index].astype(np.float32)
    area_fraction, center_distance = _box_quality(box, image_shape)
    return PoseSelection(
        keypoints=keypoints,
        box=box,
        confidence=float(confidences[index]),
        candidate_count=len(boxes),
        primary_score=float(scores[index]),
        runner_up_score=runner_up_score,
        score_margin=score_margin,
        area_fraction=area_fraction,
        center_distance=center_distance,
    )


def select_primary_segmentation(
    result: Any,
    image_shape: tuple[int, int],
    *,
    pose_box: np.ndarray,
    person_class: int,
    bag_classes: tuple[int, ...],
    bag_proximity: float,
    mask_threshold: float,
) -> SegmentationSelection:
    """Return the pose-matched person mask and nearby bag-mask union."""
    height, width = image_shape
    empty = np.zeros((height, width), dtype=bool)
    if result.boxes is None or result.masks is None:
        return SegmentationSelection(
            person_mask=None,
            bag_mask=empty,
            person_candidate_count=0,
            bag_candidate_count=0,
            matched_box=None,
            matched_confidence=None,
            matched_iou=None,
            matched_overlap=None,
        )
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    person_indices = np.flatnonzero(classes == person_class)
    bag_indices = np.flatnonzero(np.isin(classes, np.asarray(bag_classes)))
    if not person_indices.size:
        return SegmentationSelection(
            person_mask=None,
            bag_mask=empty,
            person_candidate_count=0,
            bag_candidate_count=len(bag_indices),
            matched_box=None,
            matched_confidence=None,
            matched_iou=None,
            matched_overlap=None,
        )

    def binary_mask(index: int) -> np.ndarray:
        mask = masks[index]
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return mask >= mask_threshold

    matches = []
    for raw_index in person_indices:
        index = int(raw_index)
        iou = _box_iou(pose_box, boxes[index])
        overlap = _box_overlap_over_smaller(pose_box, boxes[index])
        geometry_score = 0.75 * iou + 0.25 * overlap
        matches.append(
            (
                geometry_score,
                float(confidences[index]),
                index,
                iou,
                overlap,
            )
        )
    _, matched_confidence, person_index, matched_iou, matched_overlap = max(
        matches,
        key=lambda item: (item[0], item[1]),
    )
    person_mask = binary_mask(person_index)
    bag_mask = empty.copy()
    if not bag_classes:
        return SegmentationSelection(
            person_mask=person_mask,
            bag_mask=bag_mask,
            person_candidate_count=len(person_indices),
            bag_candidate_count=0,
            matched_box=boxes[person_index].astype(np.float32),
            matched_confidence=matched_confidence,
            matched_iou=matched_iou,
            matched_overlap=matched_overlap,
        )
    proximity = int(round(max(height, width) * bag_proximity))
    if proximity:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * proximity + 1, 2 * proximity + 1),
        )
        neighborhood = cv2.dilate(person_mask.astype(np.uint8), kernel).astype(bool)
    else:
        neighborhood = person_mask
    for index in bag_indices:
        candidate = binary_mask(int(index))
        if np.any(candidate & neighborhood):
            bag_mask |= candidate
    return SegmentationSelection(
        person_mask=person_mask,
        bag_mask=bag_mask,
        person_candidate_count=len(person_indices),
        bag_candidate_count=len(bag_indices),
        matched_box=boxes[person_index].astype(np.float32),
        matched_confidence=matched_confidence,
        matched_iou=matched_iou,
        matched_overlap=matched_overlap,
    )


def pose_mask_agreement(
    keypoints: np.ndarray,
    person_mask: np.ndarray,
    min_keypoint_confidence: float,
) -> float:
    """Return the fraction of reliable pose keypoints inside a person mask."""
    height, width = person_mask.shape
    reliable = (
        (keypoints[:, 2] >= min_keypoint_confidence)
        & (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] <= width - 1)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] <= height - 1)
    )
    if not reliable.any():
        return 0.0
    x = np.rint(keypoints[reliable, 0]).astype(np.int64)
    y = np.rint(keypoints[reliable, 1]).astype(np.int64)
    return float(person_mask[y, x].mean())


def _normalized_record(
    keypoints: np.ndarray,
    box: np.ndarray,
    confidence: float,
    shape: tuple[int, int],
) -> dict:
    height, width = shape
    normalized_points = keypoints.copy()
    normalized_points[:, 0] /= max(width, 1)
    normalized_points[:, 1] /= max(height, 1)
    normalized_box = box / np.array((width, height, width, height), dtype=np.float32)
    return {
        "image_size": [height, width],
        "pose_confidence": confidence,
        "bbox": normalized_box.round(7).tolist(),
        "keypoints": normalized_points.round(7).tolist(),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp{path.suffix}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _generation_settings(args: argparse.Namespace) -> dict:
    """Return every inference and cleanup setting that affects the output."""
    return {
        "imgsz": args.imgsz,
        "pose_conf": args.pose_conf,
        "seg_conf": args.seg_conf,
        "iou": args.iou,
        "mask_threshold": args.mask_threshold,
        "person_class": args.person_class,
        "bag_classes": list(args.bag_classes),
        "bag_proximity": args.bag_proximity,
        "min_primary_area": args.min_primary_area,
        "max_primary_center_distance": args.max_primary_center_distance,
        "min_primary_score_margin": args.min_primary_score_margin,
        "min_pose_seg_iou": args.min_pose_seg_iou,
        "min_pose_mask_agreement": args.min_pose_mask_agreement,
        "keypoint_conf": args.keypoint_conf,
    }


def _candidate_counts(
    result: Any,
    *,
    person_class: int,
    bag_classes: tuple[int, ...],
) -> tuple[int, int]:
    """Return detected person and bag counts without requiring masks."""
    if result.boxes is None:
        return 0, 0
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    return (
        int((classes == person_class).sum()),
        int(np.isin(classes, np.asarray(bag_classes)).sum()),
    )


def _pose_only_has_multiple_people(
    pose_candidate_count: int,
    segmentation_candidate_count: int,
) -> bool:
    """Return whether either detector found multiple plausible people."""
    return max(pose_candidate_count, segmentation_candidate_count) > 1


def _pose_quality(pose: PoseSelection) -> dict:
    """Serialize primary-pose measurements into a JSON-safe mapping."""
    return {
        "pose_candidates": pose.candidate_count,
        "primary_score": round(pose.primary_score, 7),
        "runner_up_score": (round(pose.runner_up_score, 7) if pose.runner_up_score is not None else None),
        "primary_score_margin": (round(pose.score_margin, 7) if pose.score_margin is not None else None),
        "primary_area_fraction": round(pose.area_fraction, 7),
        "primary_center_distance": round(pose.center_distance, 7),
    }


def _segmentation_quality(
    segmentation: SegmentationSelection,
    agreement: float | None,
) -> dict:
    """Serialize pose/segmentation matching measurements."""
    return {
        "person_segmentation_candidates": (segmentation.person_candidate_count),
        "bag_candidates": segmentation.bag_candidate_count,
        "matched_segmentation_confidence": (
            round(segmentation.matched_confidence, 7) if segmentation.matched_confidence is not None else None
        ),
        "pose_segmentation_iou": (round(segmentation.matched_iou, 7) if segmentation.matched_iou is not None else None),
        "pose_segmentation_overlap": (
            round(segmentation.matched_overlap, 7) if segmentation.matched_overlap is not None else None
        ),
        "pose_mask_agreement": (round(agreement, 7) if agreement is not None else None),
        "has_person_mask": bool(segmentation.person_mask is not None and segmentation.person_mask.any()),
    }


def _manifest_payload(
    *,
    source: Path,
    pose_model: str,
    seg_model: str,
    settings: dict,
    records: dict,
) -> dict:
    """Build the versioned manifest persisted during and after generation."""
    return {
        "version": 2,
        "source": str(source),
        "pose_model": pose_model,
        "seg_model": seg_model,
        "settings": settings,
        "images": records,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.imgsz < 1:
        raise ValueError("--batch-size and --imgsz must be positive")
    unit_interval_settings = {
        "pose_conf": args.pose_conf,
        "seg_conf": args.seg_conf,
        "iou": args.iou,
        "mask_threshold": args.mask_threshold,
        "min_primary_area": args.min_primary_area,
        "max_primary_center_distance": args.max_primary_center_distance,
        "min_pose_seg_iou": args.min_pose_seg_iou,
        "min_pose_mask_agreement": args.min_pose_mask_agreement,
        "keypoint_conf": args.keypoint_conf,
    }
    invalid_unit_settings = {key: value for key, value in unit_interval_settings.items() if not 0 <= value <= 1}
    if invalid_unit_settings:
        raise ValueError(
            f"confidence, overlap, area, and distance settings must be in [0, 1], got {invalid_unit_settings}"
        )
    if args.bag_proximity < 0:
        raise ValueError("--bag-proximity must be non-negative")
    if args.min_primary_score_margin < 1:
        raise ValueError("--min-primary-score-margin must be at least 1")

    source = resolve_market_root(args.source)
    output = args.output.expanduser().resolve()
    images = discover_images(source / "bounding_box_train")
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No training images found under {source}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "metadata.json"
    state_path = output / "generation-state.json"
    settings = _generation_settings(args)
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_settings = existing.get("settings")
        if existing_settings and existing_settings != settings:
            raise ValueError(
                "Output contains metadata generated with different settings; "
                "choose a new --output directory or pass --overwrite"
            )
        records = dict(existing.get("images", {}))
    else:
        records = {}
    if state_path.is_file() and not args.overwrite:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        decisions = dict(state.get("decisions", {}))
    else:
        decisions = {}
    for key, record in records.items():
        decisions.setdefault(
            key,
            {
                "status": "accepted",
                "reasons": [],
                "quality": record.get("quality", {}),
            },
        )

    from ultralytics import YOLO

    pose_model = YOLO(args.pose_model)
    seg_model = YOLO(args.seg_model)
    pending = [path for path in images if args.overwrite or path.relative_to(source).as_posix() not in decisions]
    completed = 0
    rejected = 0
    with tqdm(
        total=len(images),
        initial=len(images) - len(pending),
        desc="Generating Market-1501 PAV metadata",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for batch_index, batch_paths in enumerate(chunks(pending, args.batch_size), start=1):
            common = {
                "source": [str(path) for path in batch_paths],
                "imgsz": args.imgsz,
                "iou": args.iou,
                "batch": args.batch_size,
                "verbose": False,
            }
            if args.device is not None:
                common["device"] = args.device
            pose_results = pose_model.predict(conf=args.pose_conf, **common)
            seg_results = seg_model.predict(
                conf=args.seg_conf,
                classes=sorted({args.person_class, *args.bag_classes}),
                retina_masks=True,
                **common,
            )
            if len(pose_results) != len(batch_paths) or len(seg_results) != len(batch_paths):
                raise RuntimeError("YOLO pose/segmentation result count does not match input batch")

            for path, pose_result, seg_result in zip(
                batch_paths,
                pose_results,
                seg_results,
                strict=True,
            ):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Failed to read image: {path}")
                relative = path.relative_to(source)
                key = relative.as_posix()
                selected_pose = select_primary_pose(pose_result, image.shape[:2])
                seg_person_count, seg_bag_count = _candidate_counts(
                    seg_result,
                    person_class=args.person_class,
                    bag_classes=tuple(args.bag_classes),
                )
                if selected_pose is None:
                    decisions[key] = {
                        "status": "rejected",
                        "reasons": ["missing_pose"],
                        "quality": {
                            "pose_candidates": 0,
                            "person_segmentation_candidates": (seg_person_count),
                            "bag_candidates": seg_bag_count,
                        },
                    }
                    rejected += 1
                    continue

                rejection_reasons = []
                if selected_pose.area_fraction < args.min_primary_area:
                    rejection_reasons.append("primary_pose_too_small")
                if selected_pose.center_distance > args.max_primary_center_distance:
                    rejection_reasons.append("primary_pose_off_center")
                if (
                    selected_pose.candidate_count > 1
                    and selected_pose.score_margin is not None
                    and selected_pose.score_margin < args.min_primary_score_margin
                ):
                    rejection_reasons.append("ambiguous_primary_pose")
                if rejection_reasons:
                    decisions[key] = {
                        "status": "rejected",
                        "reasons": rejection_reasons,
                        "quality": {
                            **_pose_quality(selected_pose),
                            "person_segmentation_candidates": (seg_person_count),
                            "bag_candidates": seg_bag_count,
                        },
                    }
                    rejected += 1
                    continue

                segmentation = select_primary_segmentation(
                    seg_result,
                    image.shape[:2],
                    pose_box=selected_pose.box,
                    person_class=args.person_class,
                    bag_classes=tuple(args.bag_classes),
                    bag_proximity=args.bag_proximity,
                    mask_threshold=args.mask_threshold,
                )
                agreement = None
                if segmentation.person_mask is not None and segmentation.person_mask.any():
                    if segmentation.matched_iou is None or segmentation.matched_iou < args.min_pose_seg_iou:
                        rejection_reasons.append("pose_segmentation_iou_below_threshold")
                    agreement = pose_mask_agreement(
                        selected_pose.keypoints,
                        segmentation.person_mask,
                        args.keypoint_conf,
                    )
                    if agreement < args.min_pose_mask_agreement:
                        rejection_reasons.append("pose_mask_agreement_below_threshold")
                elif _pose_only_has_multiple_people(
                    selected_pose.candidate_count,
                    seg_person_count,
                ):
                    rejection_reasons.append("pose_only_with_multiple_people")

                quality = {
                    **_pose_quality(selected_pose),
                    **_segmentation_quality(segmentation, agreement),
                    "person_segmentation_candidates": seg_person_count,
                    "bag_candidates": seg_bag_count,
                }
                if rejection_reasons:
                    decisions[key] = {
                        "status": "rejected",
                        "reasons": rejection_reasons,
                        "quality": quality,
                    }
                    rejected += 1
                    continue

                record = _normalized_record(
                    selected_pose.keypoints,
                    selected_pose.box,
                    selected_pose.confidence,
                    image.shape[:2],
                )
                record["quality"] = quality
                if segmentation.person_mask is not None and segmentation.person_mask.any():
                    person_path = (Path("person") / relative).with_suffix(".png")
                    write_mask(
                        output / person_path,
                        segmentation.person_mask,
                    )
                    record["person_mask"] = person_path.as_posix()
                if segmentation.bag_mask.any():
                    bag_path = (Path("bags") / relative).with_suffix(".png")
                    write_mask(output / bag_path, segmentation.bag_mask)
                    record["bag_mask"] = bag_path.as_posix()
                records[key] = record
                decisions[key] = {
                    "status": "accepted",
                    "reasons": [],
                    "quality": quality,
                }
                completed += 1
            progress.update(len(batch_paths))
            progress.set_postfix(
                accepted=completed,
                rejected=rejected,
                refresh=False,
            )
            if batch_index % 50 == 0:
                _write_json_atomic(
                    manifest_path,
                    _manifest_payload(
                        source=source,
                        pose_model=args.pose_model,
                        seg_model=args.seg_model,
                        settings=settings,
                        records=records,
                    ),
                )
                _write_json_atomic(
                    state_path,
                    {"version": 2, "decisions": decisions},
                )

    manifest = _manifest_payload(
        source=source,
        pose_model=args.pose_model,
        seg_model=args.seg_model,
        settings=settings,
        records=records,
    )
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        state_path,
        {"version": 2, "decisions": decisions},
    )
    rejections = {key: decision for key, decision in decisions.items() if decision.get("status") == "rejected"}
    rejection_reason_counts = Counter(
        reason for decision in rejections.values() for reason in decision.get("reasons", ())
    )
    report = {
        **{key: value for key, value in manifest.items() if key != "images"},
        "images_selected": len(images),
        "images_with_pose": len(records),
        "images_with_person_mask": sum(bool(record.get("person_mask")) for record in records.values()),
        "images_pose_only": sum(not bool(record.get("person_mask")) for record in records.values()),
        "images_rejected": len(rejections),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "rejections": rejections,
    }
    _write_json_atomic(output / "generation-report.json", report)
    print(f"Finished: accepted={len(records)}, rejected={len(rejections)}, new={completed}, metadata={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
