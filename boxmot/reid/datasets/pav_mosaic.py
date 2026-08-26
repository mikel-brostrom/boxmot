"""Pose-aligned, same-identity semantic-part mosaic augmentation."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.pose_metadata import load_metadata_images

COCO_KEYPOINT_COUNT = 17
PAV_PARTS = (
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "upper_legs",
    "lower_legs",
    "bag",
)


class IndexedTransformCompose:
    """Compose sample-index-aware transforms while preserving applied status."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = tuple(transforms)

    def set_epoch(self, epoch: int) -> None:
        """Forward the active epoch to every stateful transform."""
        for transform in self.transforms:
            setter = getattr(transform, "set_epoch", None)
            if callable(setter):
                setter(epoch)

    def apply_with_status(
        self,
        image: Image.Image,
        index: int,
    ) -> tuple[Image.Image, bool]:
        """Apply each transform and report whether any changed the sample."""
        changed = False
        for transform in self.transforms:
            apply_with_status = getattr(transform, "apply_with_status", None)
            if callable(apply_with_status):
                image, transform_changed = apply_with_status(image, index)
                changed = changed or bool(transform_changed)
            else:
                transformed = transform(image, index)
                changed = changed or transformed is not image
                image = transformed
        return image, changed

    def __call__(self, image: Image.Image, index: int) -> Image.Image:
        return self.apply_with_status(image, index)[0]


class PoseAlignedViewMosaic:
    """Warp semantic parts from high-confidence same-ID donor observations."""

    _PART_KEYPOINTS = {
        "head": (0, 1, 2, 3, 4),
        "torso": (5, 6, 11, 12),
        "left_arm": (5, 7, 9),
        "right_arm": (6, 8, 10),
        "upper_legs": (11, 12, 13, 14),
        "lower_legs": (13, 14, 15, 16),
        "bag": (5, 6, 11, 12),
    }
    _LIMB_SEGMENTS = {
        "left_arm": ((5, 7), (7, 9)),
        "right_arm": ((6, 8), (8, 10)),
        "upper_legs": ((11, 13), (12, 14)),
        "lower_legs": ((13, 15), (14, 16)),
    }

    def __init__(
        self,
        samples: Sequence[ReIDSample],
        *,
        image_root: str | Path,
        metadata_root: str | Path,
        probability: float = 0.25,
        max_parts: int = 3,
        max_foreground_replacement: float = 0.45,
        cross_camera_rate: float = 0.8,
        different_pose_rate: float = 0.5,
        min_keypoint_confidence: float = 0.5,
        warmup_epochs: int = 40,
        decay_start_epoch: int = 170,
        decay_end_epoch: int = 200,
        final_probability_scale: float = 0.5,
        feather: float = 0.8,
        image_cache_size: int = 512,
        mask_cache_size: int = 2048,
    ) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("PAV mosaic probability must be in [0, 1]")
        if max_parts < 1 or max_parts > len(PAV_PARTS):
            raise ValueError(f"PAV max_parts must be in [1, {len(PAV_PARTS)}]")
        if not 0 < max_foreground_replacement <= 1:
            raise ValueError("PAV maximum foreground replacement must be in (0, 1]")
        if not 0 <= cross_camera_rate <= 1 or not 0 <= different_pose_rate <= 1:
            raise ValueError("PAV donor preference rates must be in [0, 1]")
        if not 0 <= min_keypoint_confidence <= 1:
            raise ValueError("PAV minimum keypoint confidence must be in [0, 1]")
        if warmup_epochs < 0 or not 0 <= decay_start_epoch <= decay_end_epoch:
            raise ValueError("invalid PAV warmup/decay epoch schedule")
        if not 0 <= final_probability_scale <= 1 or feather < 0:
            raise ValueError("invalid PAV final probability scale or feather")
        if image_cache_size < 0 or mask_cache_size < 0:
            raise ValueError("PAV cache sizes must be non-negative")

        self.samples = tuple(samples)
        self.image_root = Path(image_root).expanduser().resolve()
        self.metadata_root = Path(metadata_root).expanduser().resolve()
        manifest_path = self.metadata_root / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"PAV metadata manifest does not exist: {manifest_path}")
        records = load_metadata_images(manifest_path)
        if not isinstance(records, dict):
            raise ValueError(f"PAV metadata manifest has no image mapping: {manifest_path}")
        self.records = records
        self.probability = float(probability)
        self.max_parts = int(max_parts)
        self.max_foreground_replacement = float(max_foreground_replacement)
        self.cross_camera_rate = float(cross_camera_rate)
        self.different_pose_rate = float(different_pose_rate)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.warmup_epochs = int(warmup_epochs)
        self.decay_start_epoch = int(decay_start_epoch)
        self.decay_end_epoch = int(decay_end_epoch)
        self.final_probability_scale = float(final_probability_scale)
        self.feather = float(feather)
        self.image_cache_size = int(image_cache_size)
        self.mask_cache_size = int(mask_cache_size)
        self.epoch = 0
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._mask_cache: OrderedDict[
            tuple[str, int, int],
            np.ndarray | None,
        ] = OrderedDict()

        self._sample_keys = tuple(self._sample_key(sample) for sample in self.samples)
        self._indices_by_pid: dict[int, list[int]] = {}
        for index, sample in enumerate(self.samples):
            if self._sample_keys[index] in self.records:
                self._indices_by_pid.setdefault(sample.pid, []).append(index)
        self._normalized_keypoints = tuple(
            self._normalized_record_keypoints(self._record(index))
            for index in range(len(self.samples))
        )
        self._part_indices = {
            part: part_index for part_index, part in enumerate(PAV_PARTS)
        }
        self._visibility_by_index = np.zeros(
            (len(self.samples), len(PAV_PARTS)),
            dtype=np.float32,
        )
        for index in range(len(self.samples)):
            record = self._record(index) or {}
            for part, part_index in self._part_indices.items():
                self._visibility_by_index[index, part_index] = self._visibility(
                    record,
                    part,
                )
        self._donors_by_pid_part: dict[tuple[int, str], tuple[int, ...]] = {}
        for pid, indices in self._indices_by_pid.items():
            for part in PAV_PARTS:
                self._donors_by_pid_part[(pid, part)] = tuple(
                    index
                    for index in indices
                    if self._visibility_for_index(index, part) > 0
                    and (
                        part != "bag"
                        or bool((self._record(index) or {}).get("bag_mask"))
                    )
                )

    def _sample_key(self, sample: ReIDSample) -> str:
        path = Path(sample.img_path).expanduser().resolve()
        try:
            return path.relative_to(self.image_root).as_posix()
        except ValueError:
            return ""

    def set_epoch(self, epoch: int) -> None:
        """Set the one-based epoch controlling warm-up and final decay."""
        self.epoch = int(epoch)

    def effective_probability(self) -> float:
        """Return scheduled probability with warm-up and late-training decay."""
        if self.warmup_epochs > 0:
            scale = min(max(self.epoch, 0) / self.warmup_epochs, 1.0)
        else:
            scale = 1.0
        if self.epoch > self.decay_start_epoch and self.decay_end_epoch > self.decay_start_epoch:
            progress = min(
                (self.epoch - self.decay_start_epoch)
                / (self.decay_end_epoch - self.decay_start_epoch),
                1.0,
            )
            scale *= 1.0 - progress * (1.0 - self.final_probability_scale)
        return self.probability * scale

    def _record(self, index: int) -> dict | None:
        key = self._sample_keys[index]
        record = self.records.get(key)
        return record if isinstance(record, dict) else None

    @staticmethod
    def _normalized_record_keypoints(record: dict | None) -> np.ndarray | None:
        if record is None:
            return None
        points = np.asarray(record.get("keypoints", ()), dtype=np.float32)
        if points.shape != (COCO_KEYPOINT_COUNT, 3):
            return None
        points = points.copy()
        points.setflags(write=False)
        return points

    @staticmethod
    def _keypoints(record: dict, size: tuple[int, int]) -> np.ndarray | None:
        points = PoseAlignedViewMosaic._normalized_record_keypoints(record)
        if points is None:
            return None
        width, height = size
        points = points.copy()
        points[:, 0] *= width
        points[:, 1] *= height
        return points

    def _keypoints_for_index(
        self,
        index: int,
        size: tuple[int, int],
    ) -> np.ndarray | None:
        points = self._normalized_keypoints[index]
        if points is None:
            return None
        width, height = size
        scaled = points.copy()
        scaled[:, 0] *= width
        scaled[:, 1] *= height
        return scaled

    def _visibility(self, record: dict, part: str) -> float:
        points = np.asarray(record.get("keypoints", ()), dtype=np.float32)
        if points.shape != (COCO_KEYPOINT_COUNT, 3):
            return 0.0
        confidences = points[list(self._PART_KEYPOINTS[part]), 2]
        if part == "head":
            visible = confidences[confidences >= self.min_keypoint_confidence]
            return float(visible.mean()) if visible.size >= 2 else 0.0
        if part in {"torso", "bag"}:
            required = 4
        elif part in {"upper_legs", "lower_legs"}:
            required = 3
        else:
            required = 2
        visible = confidences[confidences >= self.min_keypoint_confidence]
        return float(visible.mean()) if visible.size >= required else 0.0

    def _visibility_for_index(self, index: int, part: str) -> float:
        return float(
            self._visibility_by_index[index, self._part_indices[part]]
        )

    @staticmethod
    def _pose_distance_points(
        left_points: np.ndarray | None,
        right_points: np.ndarray | None,
        min_keypoint_confidence: float,
    ) -> float:
        if left_points is None or right_points is None:
            return 0.0
        visible = (
            (left_points[:, 2] >= min_keypoint_confidence)
            & (right_points[:, 2] >= min_keypoint_confidence)
        )
        if int(visible.sum()) < 4:
            return 0.0
        return float(
            np.linalg.norm(
                left_points[visible, :2] - right_points[visible, :2],
                axis=1,
            ).mean()
        )

    def _pose_distance(self, left: dict, right: dict) -> float:
        return self._pose_distance_points(
            self._normalized_record_keypoints(left),
            self._normalized_record_keypoints(right),
            self.min_keypoint_confidence,
        )

    def _pose_distance_indices(self, left: int, right: int) -> float:
        return self._pose_distance_points(
            self._normalized_keypoints[left],
            self._normalized_keypoints[right],
            self.min_keypoint_confidence,
        )

    def _select_donor(
        self,
        anchor_index: int,
        part: str,
        used: set[int],
    ) -> int | None:
        anchor = self.samples[anchor_index]
        if self._record(anchor_index) is None:
            return None
        candidates = [
            index
            for index in self._donors_by_pid_part.get((anchor.pid, part), ())
            if index != anchor_index
        ]
        unused = [index for index in candidates if index not in used]
        if unused:
            candidates = unused
        if not candidates:
            return None
        cross_camera = [
            index for index in candidates if self.samples[index].camid != anchor.camid
        ]
        same_camera = [
            index for index in candidates if self.samples[index].camid == anchor.camid
        ]
        if cross_camera and same_camera:
            candidates = (
                cross_camera
                if random.random() < self.cross_camera_rate
                else same_camera
            )
        prefer_pose_diversity = random.random() < self.different_pose_rate

        def score(index: int) -> float:
            value = self._visibility_for_index(index, part)
            if prefer_pose_diversity:
                value += 0.25 * self._pose_distance_indices(
                    anchor_index,
                    index,
                )
            return value + random.random() * 1e-4

        return max(candidates, key=score)

    def _load_mask(
        self,
        record: dict,
        key: str,
        size: tuple[int, int],
    ) -> np.ndarray | None:
        relative_path = record.get(key)
        if not relative_path:
            return None
        path = self.metadata_root / str(relative_path)
        width, height = size
        cache_key = (str(path), width, height)
        if cache_key in self._mask_cache:
            mask = self._mask_cache.pop(cache_key)
            self._mask_cache[cache_key] = mask
            return mask
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            result = None
        else:
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            result = mask >= 128
            result.setflags(write=False)
        if self.mask_cache_size > 0:
            self._mask_cache[cache_key] = result
            while len(self._mask_cache) > self.mask_cache_size:
                self._mask_cache.popitem(last=False)
        return result

    def _load_donor_image(self, index: int) -> Image.Image:
        path = self.samples[index].img_path
        if path in self._image_cache:
            image = self._image_cache.pop(path)
            self._image_cache[path] = image
            return image
        with Image.open(path) as donor_handle:
            image = donor_handle.convert("RGB")
        if self.image_cache_size > 0:
            self._image_cache[path] = image
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return image

    @staticmethod
    def _segment_affine(
        source_a: np.ndarray,
        source_b: np.ndarray,
        target_a: np.ndarray,
        target_b: np.ndarray,
    ) -> np.ndarray | None:
        source_vector = source_b - source_a
        target_vector = target_b - target_a
        source_length = float(np.linalg.norm(source_vector))
        target_length = float(np.linalg.norm(target_vector))
        if source_length < 2 or target_length < 2:
            return None
        scale = target_length / source_length
        if not 0.5 <= scale <= 2.0:
            return None
        source_normal = np.array((-source_vector[1], source_vector[0]), dtype=np.float32)
        source_normal /= source_length
        target_normal = np.array((-target_vector[1], target_vector[0]), dtype=np.float32)
        target_normal /= target_length
        source_mid = (source_a + source_b) * 0.5
        target_mid = (target_a + target_b) * 0.5
        source_control = np.float32(
            [source_a, source_b, source_mid + source_normal * source_length * 0.25]
        )
        target_control = np.float32(
            [target_a, target_b, target_mid + target_normal * target_length * 0.25]
        )
        return cv2.getAffineTransform(source_control, target_control)

    @staticmethod
    def _warp(
        donor: np.ndarray,
        source_mask: np.ndarray,
        matrix: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = target_size
        warped = cv2.warpAffine(
            donor,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = cv2.warpAffine(
            source_mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return warped, warped_mask.astype(bool)

    @staticmethod
    def _shoulder_width(points: np.ndarray) -> float:
        return max(float(np.linalg.norm(points[5, :2] - points[6, :2])), 4.0)

    def _limb_layer(
        self,
        part: str,
        donor: np.ndarray,
        donor_points: np.ndarray,
        anchor_points: np.ndarray,
        donor_foreground: np.ndarray | None,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        output = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        output_mask = np.zeros((target_size[1], target_size[0]), dtype=bool)
        width_scale = 0.18 if "arm" in part else 0.24
        thickness = max(int(round(self._shoulder_width(donor_points) * width_scale)), 2)
        for start, end in self._LIMB_SEGMENTS[part]:
            if min(donor_points[start, 2], donor_points[end, 2]) < self.min_keypoint_confidence:
                continue
            if min(anchor_points[start, 2], anchor_points[end, 2]) < self.min_keypoint_confidence:
                continue
            source_mask = np.zeros(donor.shape[:2], dtype=np.uint8)
            cv2.line(
                source_mask,
                tuple(np.rint(donor_points[start, :2]).astype(int)),
                tuple(np.rint(donor_points[end, :2]).astype(int)),
                1,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            if donor_foreground is not None:
                source_mask &= donor_foreground.astype(np.uint8)
            matrix = self._segment_affine(
                donor_points[start, :2],
                donor_points[end, :2],
                anchor_points[start, :2],
                anchor_points[end, :2],
            )
            if matrix is None:
                continue
            warped, mask = self._warp(donor, source_mask, matrix, target_size)
            output[mask] = warped[mask]
            output_mask |= mask
        return (output, output_mask) if output_mask.any() else None

    def _torso_layer(
        self,
        donor: np.ndarray,
        donor_points: np.ndarray,
        anchor_points: np.ndarray,
        donor_foreground: np.ndarray | None,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        indices = (5, 6, 11, 12)
        if any(
            donor_points[index, 2] < self.min_keypoint_confidence
            or anchor_points[index, 2] < self.min_keypoint_confidence
            for index in indices
        ):
            return None
        source_polygon = np.rint(donor_points[list(indices), :2]).astype(np.int32)
        source_mask = np.zeros(donor.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(source_mask, source_polygon[[0, 1, 3, 2]], 1)
        if donor_foreground is not None:
            source_mask &= donor_foreground.astype(np.uint8)
        source_control = np.float32(
            [donor_points[5, :2], donor_points[6, :2], (donor_points[11, :2] + donor_points[12, :2]) / 2]
        )
        target_control = np.float32(
            [anchor_points[5, :2], anchor_points[6, :2], (anchor_points[11, :2] + anchor_points[12, :2]) / 2]
        )
        source_scale = float(np.linalg.norm(source_control[1] - source_control[0]))
        target_scale = float(np.linalg.norm(target_control[1] - target_control[0]))
        if source_scale < 2 or not 0.5 <= target_scale / source_scale <= 2.0:
            return None
        matrix = cv2.getAffineTransform(source_control, target_control)
        return self._warp(donor, source_mask, matrix, target_size)

    def _head_layer(
        self,
        donor: np.ndarray,
        donor_points: np.ndarray,
        anchor_points: np.ndarray,
        donor_foreground: np.ndarray | None,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        donor_visible = donor_points[:5, 2] >= self.min_keypoint_confidence
        anchor_visible = anchor_points[:5, 2] >= self.min_keypoint_confidence
        if int(donor_visible.sum()) < 2 or int(anchor_visible.sum()) < 2:
            return None
        donor_center = donor_points[:5, :2][donor_visible].mean(axis=0)
        anchor_center = anchor_points[:5, :2][anchor_visible].mean(axis=0)
        donor_width = self._shoulder_width(donor_points) * 0.72
        anchor_width = self._shoulder_width(anchor_points) * 0.72
        source_mask = np.zeros(donor.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            source_mask,
            tuple(np.rint(donor_center).astype(int)),
            (max(int(donor_width / 2), 2), max(int(donor_width * 0.65), 3)),
            0,
            0,
            360,
            1,
            -1,
        )
        if donor_foreground is not None:
            source_mask &= donor_foreground.astype(np.uint8)
        matrix = self._segment_affine(
            donor_center,
            donor_center + np.array((donor_width, 0), dtype=np.float32),
            anchor_center,
            anchor_center + np.array((anchor_width, 0), dtype=np.float32),
        )
        if matrix is None:
            return None
        return self._warp(donor, source_mask, matrix, target_size)

    def _bag_layer(
        self,
        donor: np.ndarray,
        donor_record: dict,
        donor_points: np.ndarray,
        anchor_points: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        control_indices = (5, 6, 11, 12)
        if any(
            donor_points[index, 2] < self.min_keypoint_confidence
            or anchor_points[index, 2] < self.min_keypoint_confidence
            for index in control_indices
        ):
            return None
        bag_mask = self._load_mask(
            donor_record,
            "bag_mask",
            (donor.shape[1], donor.shape[0]),
        )
        if bag_mask is None or not bag_mask.any():
            return None
        source_control = np.float32(
            [donor_points[5, :2], donor_points[6, :2], (donor_points[11, :2] + donor_points[12, :2]) / 2]
        )
        target_control = np.float32(
            [anchor_points[5, :2], anchor_points[6, :2], (anchor_points[11, :2] + anchor_points[12, :2]) / 2]
        )
        first = source_control[1] - source_control[0]
        second = source_control[2] - source_control[0]
        determinant = float(first[0] * second[1] - first[1] * second[0])
        if abs(determinant) < 1:
            return None
        source_scale = float(np.linalg.norm(first))
        target_scale = float(np.linalg.norm(target_control[1] - target_control[0]))
        if source_scale < 2 or not 0.5 <= target_scale / source_scale <= 2.0:
            return None
        matrix = cv2.getAffineTransform(source_control, target_control)
        return self._warp(donor, bag_mask, matrix, target_size)

    def _part_layer(
        self,
        part: str,
        donor_image: Image.Image,
        donor_record: dict,
        donor_points: np.ndarray,
        anchor_points: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        donor = np.asarray(donor_image.convert("RGB"), dtype=np.uint8)
        donor_foreground = self._load_mask(
            donor_record,
            "person_mask",
            (donor.shape[1], donor.shape[0]),
        )
        if part in self._LIMB_SEGMENTS:
            return self._limb_layer(
                part,
                donor,
                donor_points,
                anchor_points,
                donor_foreground,
                target_size,
            )
        if part == "torso":
            return self._torso_layer(
                donor,
                donor_points,
                anchor_points,
                donor_foreground,
                target_size,
            )
        if part == "head":
            return self._head_layer(
                donor,
                donor_points,
                anchor_points,
                donor_foreground,
                target_size,
            )
        return self._bag_layer(
            donor,
            donor_record,
            donor_points,
            anchor_points,
            target_size,
        )

    @staticmethod
    def _photometric_donor(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

    def apply_with_status(
        self,
        image: Image.Image,
        index: int,
    ) -> tuple[Image.Image, bool]:
        """Return a pose-aligned semantic composite and whether it was applied."""
        if random.random() >= self.effective_probability():
            return image, False
        anchor_record = self._record(index)
        if anchor_record is None:
            return image, False
        anchor_points = self._keypoints_for_index(index, image.size)
        if anchor_points is None:
            return image, False

        anchor_foreground = self._load_mask(anchor_record, "person_mask", image.size)
        if anchor_foreground is None:
            anchor_foreground = np.zeros((image.height, image.width), dtype=np.uint8)
            for part in ("torso", *self._LIMB_SEGMENTS):
                if part == "torso":
                    layer = self._torso_layer(
                        np.zeros((image.height, image.width, 3), dtype=np.uint8),
                        anchor_points,
                        anchor_points,
                        None,
                        image.size,
                    )
                else:
                    layer = self._limb_layer(
                        part,
                        np.zeros((image.height, image.width, 3), dtype=np.uint8),
                        anchor_points,
                        anchor_points,
                        None,
                        image.size,
                    )
                if layer is not None:
                    anchor_foreground |= layer[1]
            anchor_foreground = anchor_foreground.astype(bool)
        if not anchor_foreground.any():
            return image, False
        clip_mask = cv2.dilate(
            anchor_foreground.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        ).astype(bool)

        available_parts = [
            part
            for part in PAV_PARTS
            if self._visibility_for_index(index, part) > 0
            and (part != "bag" or any(
                bool((self._record(donor) or {}).get("bag_mask"))
                for donor in self._indices_by_pid.get(self.samples[index].pid, ())
                if donor != index
            ))
        ]
        random.shuffle(available_parts)
        target_count = random.randint(1, min(self.max_parts, len(available_parts))) if available_parts else 0
        if target_count == 0:
            return image, False

        composite = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        replaced = np.zeros(anchor_foreground.shape, dtype=bool)
        used_donors: set[int] = set()
        applied_parts = 0
        foreground_area = max(int(anchor_foreground.sum()), 1)
        for part in available_parts:
            donor_index = self._select_donor(index, part, used_donors)
            if donor_index is None:
                continue
            donor_record = self._record(donor_index)
            if donor_record is None:
                continue
            donor_image = self._photometric_donor(
                self._load_donor_image(donor_index)
            )
            donor_points = self._keypoints_for_index(
                donor_index,
                donor_image.size,
            )
            if donor_points is None:
                continue
            layer = self._part_layer(
                part,
                donor_image,
                donor_record,
                donor_points,
                anchor_points,
                image.size,
            )
            if layer is None:
                continue
            warped, part_mask = layer
            if part != "bag":
                part_mask &= clip_mask
            if not part_mask.any():
                continue
            candidate_replaced = replaced | part_mask
            if float(candidate_replaced.sum() / foreground_area) > self.max_foreground_replacement:
                continue
            if self.feather > 0:
                alpha = cv2.GaussianBlur(
                    part_mask.astype(np.float32),
                    (0, 0),
                    self.feather,
                )
                # Feather inward so the effective blend support still obeys
                # the foreground clip and replacement-area contract.
                alpha *= part_mask
                alpha = np.clip(alpha, 0.0, 1.0)[..., None]
                composite = np.rint(
                    warped.astype(np.float32) * alpha
                    + composite.astype(np.float32) * (1.0 - alpha)
                ).clip(0, 255).astype(np.uint8)
            else:
                composite[part_mask] = warped[part_mask]
            replaced = candidate_replaced
            used_donors.add(donor_index)
            applied_parts += 1
            if applied_parts >= target_count:
                break

        if applied_parts == 0:
            return image, False
        return Image.fromarray(composite, mode="RGB"), True

    def __call__(self, image: Image.Image, index: int) -> Image.Image:
        return self.apply_with_status(image, index)[0]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.probability}, "
            f"parts=1-{self.max_parts}, metadata={self.metadata_root})"
        )
