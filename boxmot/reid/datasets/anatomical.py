"""Training-only anatomical targets derived from pose and person masks."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.pose_metadata import load_metadata_images

COCO_KEYPOINT_COUNT = 17
COCO_KEYPOINT_FLIP_PERMUTATION = (
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
)
ANATOMICAL_PARTS = (
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
)
ANATOMICAL_PART_KEYPOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
)
ANATOMICAL_FLIP_PERMUTATION = (0, 1, 3, 2, 5, 4)
ANATOMICAL_CANONICAL_GRID_SIZE = (4, 2)
ANATOMICAL_CANONICAL_CELLS = (
    ANATOMICAL_CANONICAL_GRID_SIZE[0]
    * ANATOMICAL_CANONICAL_GRID_SIZE[1]
)


class PoseAnatomicalTargetProvider:
    """Rasterize soft anatomical masks from cached COCO keypoints.

    The provider intentionally uses only training metadata. The returned masks
    are transformed together with the RGB image and consumed by auxiliary
    losses; they are never required by the inference model.
    """

    _PART_KEYPOINTS = dict(
        zip(
            ANATOMICAL_PARTS,
            ANATOMICAL_PART_KEYPOINTS,
            strict=True,
        )
    )
    _LIMB_SEGMENTS = {
        "left_arm": ((5, 7), (7, 9)),
        "right_arm": ((6, 8), (8, 10)),
        "left_leg": ((11, 13), (13, 15)),
        "right_leg": ((12, 14), (14, 16)),
    }

    def __init__(
        self,
        samples: Sequence[ReIDSample],
        *,
        image_root: str | Path,
        metadata_root: str | Path,
        person_mask_dir: str | Path | None = None,
        min_keypoint_confidence: float = 0.5,
        pose_only_reliability: float = 0.35,
        feather: float = 1.0,
        compact_nonsemantic: bool = False,
    ) -> None:
        if not 0 <= min_keypoint_confidence <= 1:
            raise ValueError("minimum keypoint confidence must be in [0, 1]")
        if not 0 <= pose_only_reliability <= 1:
            raise ValueError("pose-only reliability must be in [0, 1]")
        if feather < 0:
            raise ValueError("anatomical mask feather must be non-negative")

        self.samples = tuple(samples)
        self.image_root = Path(image_root).expanduser().resolve()
        self.metadata_root = Path(metadata_root).expanduser().resolve()
        self.person_mask_dir = (
            None
            if person_mask_dir is None
            else Path(person_mask_dir).expanduser().resolve()
        )
        if (
            self.person_mask_dir is not None
            and not self.person_mask_dir.is_dir()
        ):
            raise FileNotFoundError(
                "External person-mask directory does not exist: "
                f"{self.person_mask_dir}"
            )
        manifest_path = self.metadata_root / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Anatomical metadata manifest does not exist: {manifest_path}"
            )
        records = load_metadata_images(manifest_path)
        if not isinstance(records, dict):
            raise ValueError(
                f"Anatomical metadata manifest has no image mapping: {manifest_path}"
            )
        self.records = records
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.pose_only_reliability = float(pose_only_reliability)
        self.feather = float(feather)
        self.compact_nonsemantic = bool(compact_nonsemantic)
        self._sample_keys = tuple(self._sample_key(sample) for sample in self.samples)
        matched_records = tuple(
            record
            for key in self._sample_keys
            if key and isinstance((record := self.records.get(key)), dict)
        )
        self.matched_record_count = len(matched_records)
        self.pose_record_count = sum(
            np.asarray(record.get("keypoints", ())).shape
            == (COCO_KEYPOINT_COUNT, 3)
            for record in matched_records
        )
        self.qualified_pose_record_count = sum(
            self._record_has_qualified_pose(record)
            for record in matched_records
        )
        resolved_masks = tuple(
            self._person_mask_path(
                (
                    record
                    if isinstance(
                        (record := self.records.get(key)),
                        dict,
                    )
                    else {}
                ),
                key,
            )
            for key in self._sample_keys
        )
        nonempty_masks = tuple(
            self._mask_has_foreground(path)
            for path in resolved_masks
        )
        self.person_mask_record_count = sum(
            path is not None and path.is_file()
            for path in resolved_masks
        )
        self.nonempty_person_mask_record_count = sum(nonempty_masks)
        self.pose_person_mask_record_count = sum(
            isinstance((record := self.records.get(key)), dict)
            and self._record_has_qualified_pose(record)
            and mask_has_foreground
            for key, mask_has_foreground in zip(
                self._sample_keys,
                nonempty_masks,
                strict=True,
            )
        )
        self.effective_supervision_record_count = (
            self.qualified_pose_record_count
            if self.pose_only_reliability > 0
            else self.pose_person_mask_record_count
        )
        self.external_person_mask_record_count = sum(
            path is not None
            and self.person_mask_dir is not None
            and path.is_relative_to(self.person_mask_dir)
            and path.is_file()
            for path in resolved_masks
        )
        self.accessory_mask_record_count = sum(
            isinstance((record := self.records.get(key)), dict)
            and bool(record.get("bag_mask"))
            and (self.metadata_root / str(record["bag_mask"])).is_file()
            for key in self._sample_keys
        )
        declared_masks = tuple(
            (
                self.metadata_root / str(record["person_mask"]),
                self._person_mask_path(record, key),
            )
            for key in self._sample_keys
            if isinstance((record := self.records.get(key)), dict)
            and record.get("person_mask")
        )
        missing_masks = tuple(
            declared
            for declared, resolved in declared_masks
            if not declared.is_file()
            and (resolved is None or not resolved.is_file())
        )
        self.missing_person_mask_count = len(missing_masks)
        self.first_missing_person_mask = (
            missing_masks[0]
            if missing_masks
            else None
        )
        # Sparse six-part masks compress to a few KiB per image. Canonical
        # sampling grids contain only 96 coordinates, so caching them is much
        # cheaper than retaining one full-resolution mask per canonical cell.
        self._target_cache: dict[
            tuple[int, int, int],
            tuple[object, ...] | None,
        ] = {}

    def compact_target(
        self,
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Drop dense targets after aligned transforms when they are unused.

        Non-semantic pose-concatenation supervision needs the transformed
        per-part presence bit, canonical grids, pose and reliability metadata,
        but it does not consume full-resolution masks. Compaction deliberately
        happens after spatial augmentation so visibility and presence retain
        exactly the dense-target geometry.
        """
        if not self.compact_nonsemantic:
            return target
        if "mask_present" not in target:
            raise RuntimeError(
                "Anatomical targets must be spatially aligned before compaction"
            )
        compact = dict(target)
        for key in (
            "masks",
            "foreground_mask",
            "accessory_mask",
            "accessory_visibility",
            "accessory_reliability",
            "accessory_valid",
        ):
            compact.pop(key, None)
        return compact

    def _sample_key(self, sample: ReIDSample) -> str:
        path = Path(sample.img_path).expanduser().resolve()
        try:
            return path.relative_to(self.image_root).as_posix()
        except ValueError:
            return ""

    def _person_mask_path(
        self,
        record: dict,
        sample_key: str,
    ) -> Path | None:
        """Resolve the external high-confidence mask before metadata masks."""
        if self.person_mask_dir is not None and sample_key:
            relative = Path(sample_key).with_suffix(".png")
            for candidate in (
                self.person_mask_dir / relative,
                self.person_mask_dir / relative.name,
            ):
                if candidate.is_file():
                    return candidate
        relative_path = record.get("person_mask")
        if relative_path:
            return self.metadata_root / str(relative_path)
        return None

    def _record_has_qualified_pose(self, record: dict) -> bool:
        """Return whether a record can supervise at least one body part."""
        points = np.asarray(record.get("keypoints", ()))
        if points.shape != (COCO_KEYPOINT_COUNT, 3):
            return False
        return any(
            self._visibility(points, part) > 0
            for part in ANATOMICAL_PARTS
        )

    @staticmethod
    def _mask_has_foreground(path: Path | None) -> bool:
        """Reject missing, unreadable, and all-background person masks."""
        if path is None or not path.is_file():
            return False
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return mask is not None and bool(np.any(mask >= 128))

    def _person_mask(
        self,
        record: dict,
        sample_key: str,
        size: tuple[int, int],
    ) -> np.ndarray | None:
        width, height = size
        mask_path = self._person_mask_path(record, sample_key)
        if mask_path is None:
            return None
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            return None
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask >= 128).astype(np.float32)

    def _accessory_mask(
        self,
        record: dict,
        size: tuple[int, int],
    ) -> np.ndarray | None:
        """Load an optional carried object, including pixels off the body."""
        relative_path = record.get("bag_mask")
        if not relative_path:
            return None
        mask = cv2.imread(
            str(self.metadata_root / str(relative_path)),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            return None
        width, height = size
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask >= 128).astype(np.float32)
        return mask if mask.any() else None

    @staticmethod
    def _empty_target(width: int, height: int) -> dict[str, torch.Tensor]:
        return {
            "masks": torch.zeros(
                (len(ANATOMICAL_PARTS), height, width),
                dtype=torch.float32,
            ),
            "foreground_mask": torch.zeros(
                (1, height, width),
                dtype=torch.float32,
            ),
            "accessory_mask": torch.zeros(
                (1, height, width),
                dtype=torch.float32,
            ),
            "canonical_grid": torch.zeros(
                (
                    len(ANATOMICAL_PARTS),
                    *ANATOMICAL_CANONICAL_GRID_SIZE,
                    2,
                ),
                dtype=torch.float32,
            ),
            "canonical_grid_valid": torch.zeros(
                (
                    len(ANATOMICAL_PARTS),
                    *ANATOMICAL_CANONICAL_GRID_SIZE,
                ),
                dtype=torch.bool,
            ),
            "canonical_grid_pose_valid": torch.zeros(
                (
                    len(ANATOMICAL_PARTS),
                    *ANATOMICAL_CANONICAL_GRID_SIZE,
                ),
                dtype=torch.bool,
            ),
            "pose_keypoints": torch.zeros(
                (COCO_KEYPOINT_COUNT, 3),
                dtype=torch.float32,
            ),
            "visibility": torch.zeros(len(ANATOMICAL_PARTS)),
            "reliability": torch.zeros(len(ANATOMICAL_PARTS)),
            "pose_reliability": torch.zeros(len(ANATOMICAL_PARTS)),
            "pose_mask_agreement": torch.tensor(0.0),
            "accessory_visibility": torch.tensor(0.0),
            "accessory_reliability": torch.tensor(0.0),
            "accessory_valid": torch.tensor(False),
            "pose_valid": torch.tensor(False),
            "mask_valid": torch.tensor(False),
            "valid": torch.tensor(False),
        }

    def _cached_target(
        self,
        cache_key: tuple[int, int, int],
        *,
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor] | None:
        if cache_key not in self._target_cache:
            return None
        cached = self._target_cache[cache_key]
        if cached is None:
            return self._empty_target(width, height)
        (
            mask_payload,
            foreground_payload,
            accessory_payload,
            grid_payload,
            grid_valid_payload,
            pose_grid_valid_payload,
            pose_keypoint_payload,
            visibility,
            reliability,
            pose_reliability,
            pose_mask_agreement,
            mask_valid,
            accessory_visibility,
            accessory_valid,
        ) = cached
        masks = np.frombuffer(
            zlib.decompress(mask_payload),
            dtype=np.uint8,
        ).reshape(
            len(ANATOMICAL_PARTS),
            height,
            width,
        )
        foreground = np.frombuffer(
            zlib.decompress(foreground_payload),
            dtype=np.uint8,
        ).reshape(1, height, width)
        accessory = np.frombuffer(
            zlib.decompress(accessory_payload),
            dtype=np.uint8,
        ).reshape(1, height, width)
        canonical_grid = np.frombuffer(
            grid_payload,
            dtype=np.float16,
        ).reshape(
            len(ANATOMICAL_PARTS),
            *ANATOMICAL_CANONICAL_GRID_SIZE,
            2,
        )
        canonical_grid_valid = np.frombuffer(
            grid_valid_payload,
            dtype=np.uint8,
        ).reshape(
            len(ANATOMICAL_PARTS),
            *ANATOMICAL_CANONICAL_GRID_SIZE,
        )
        canonical_grid_pose_valid = np.frombuffer(
            pose_grid_valid_payload,
            dtype=np.uint8,
        ).reshape(
            len(ANATOMICAL_PARTS),
            *ANATOMICAL_CANONICAL_GRID_SIZE,
        )
        pose_keypoints = np.frombuffer(
            pose_keypoint_payload,
            dtype=np.float16,
        ).reshape(COCO_KEYPOINT_COUNT, 3)
        pose_valid = any(value > 0 for value in pose_reliability)
        return {
            "masks": torch.from_numpy(masks.copy()).float().div_(255.0),
            "foreground_mask": (
                torch.from_numpy(foreground.copy()).float().div_(255.0)
            ),
            "accessory_mask": (
                torch.from_numpy(accessory.copy()).float().div_(255.0)
            ),
            "canonical_grid": torch.from_numpy(
                canonical_grid.copy()
            ).float(),
            "canonical_grid_valid": torch.from_numpy(
                canonical_grid_valid.copy()
            ).bool(),
            "canonical_grid_pose_valid": torch.from_numpy(
                canonical_grid_pose_valid.copy()
            ).bool(),
            "pose_keypoints": torch.from_numpy(
                pose_keypoints.copy()
            ).float(),
            "visibility": torch.tensor(visibility, dtype=torch.float32),
            "reliability": torch.tensor(
                reliability,
                dtype=torch.float32,
            ),
            "pose_reliability": torch.tensor(
                pose_reliability,
                dtype=torch.float32,
            ),
            "pose_mask_agreement": torch.tensor(
                pose_mask_agreement,
                dtype=torch.float32,
            ),
            "accessory_visibility": torch.tensor(
                accessory_visibility,
                dtype=torch.float32,
            ),
            "accessory_reliability": torch.tensor(
                accessory_visibility,
                dtype=torch.float32,
            ),
            "accessory_valid": torch.tensor(accessory_valid),
            "pose_valid": torch.tensor(pose_valid),
            "mask_valid": torch.tensor(mask_valid),
            "valid": torch.tensor(mask_valid),
        }

    @staticmethod
    def _keypoints(
        record: dict,
        size: tuple[int, int],
    ) -> np.ndarray | None:
        points = np.asarray(record.get("keypoints", ()), dtype=np.float32)
        if points.shape != (COCO_KEYPOINT_COUNT, 3):
            return None
        width, height = size
        points = points.copy()
        points[:, 0] *= width
        points[:, 1] *= height
        return points

    @staticmethod
    def _discard_out_of_bounds_confidence(
        points: np.ndarray,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Clear confidence for joints that cannot supervise image features."""
        width, height = size
        in_bounds = (
            (points[:, 0] >= 0)
            & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height - 1)
        )
        sanitized = points.copy()
        sanitized[~in_bounds, 2] = 0.0
        return sanitized

    def _visibility(self, points: np.ndarray, part: str) -> float:
        confidences = points[list(self._PART_KEYPOINTS[part]), 2]
        reliable = np.where(
            confidences >= self.min_keypoint_confidence,
            confidences,
            0.0,
        )
        return float(np.clip(reliable.mean(), 0.0, 1.0))

    def _pose_mask_agreement(
        self,
        points: np.ndarray,
        foreground: np.ndarray | None,
    ) -> float:
        """Measure estimator agreement without rejecting pose-only samples."""
        if foreground is None:
            return 1.0
        height, width = foreground.shape
        reliable = (
            (points[:, 2] >= self.min_keypoint_confidence)
            & (points[:, 0] >= 0)
            & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height - 1)
        )
        if not reliable.any():
            return 0.0
        x = np.rint(points[reliable, 0]).astype(np.int64)
        y = np.rint(points[reliable, 1]).astype(np.int64)
        return float((foreground[y, x] >= 0.5).mean())

    def _shoulder_width(self, points: np.ndarray, width: int) -> float:
        fallback = max(width * 0.12, 2.0)
        if min(points[5, 2], points[6, 2]) < self.min_keypoint_confidence:
            return fallback
        shoulders = float(np.linalg.norm(points[5, :2] - points[6, :2]))
        return max(shoulders, fallback)

    def _draw_head(
        self,
        mask: np.ndarray,
        points: np.ndarray,
        shoulder_width: float,
    ) -> None:
        indices = np.asarray(self._PART_KEYPOINTS["head"])
        visible = points[
            indices[points[indices, 2] >= self.min_keypoint_confidence],
            :2,
        ]
        if len(visible) < 2:
            return
        center = visible.mean(axis=0)
        spread = np.ptp(visible, axis=0) if len(visible) > 1 else np.zeros(2)
        radius_x = max(float(spread[0]) * 0.7, shoulder_width * 0.32, 2.0)
        radius_y = max(float(spread[1]) * 0.9, shoulder_width * 0.42, 3.0)
        cv2.ellipse(
            mask,
            tuple(np.rint(center).astype(int)),
            (int(round(radius_x)), int(round(radius_y))),
            0,
            0,
            360,
            1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    def _draw_torso(
        self,
        mask: np.ndarray,
        points: np.ndarray,
    ) -> None:
        indices = (5, 6, 12, 11)
        if any(
            points[index, 2] < self.min_keypoint_confidence
            for index in indices
        ):
            return
        polygon = np.rint(
            points[[5, 6, 12, 11], :2]
        ).astype(np.int32)
        cv2.fillConvexPoly(
            mask,
            polygon,
            1.0,
            lineType=cv2.LINE_AA,
        )

    def _draw_limb(
        self,
        mask: np.ndarray,
        points: np.ndarray,
        part: str,
        shoulder_width: float,
    ) -> None:
        thickness_scale = 0.24 if "arm" in part else 0.34
        thickness = max(int(round(shoulder_width * thickness_scale)), 2)
        for start, end in self._LIMB_SEGMENTS[part]:
            if min(points[start, 2], points[end, 2]) < (
                self.min_keypoint_confidence
            ):
                continue
            cv2.line(
                mask,
                tuple(np.rint(points[start, :2]).astype(int)),
                tuple(np.rint(points[end, :2]).astype(int)),
                1.0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def _normalize_grid(
        grid: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert image pixel centers to grid_sample coordinates."""
        normalized = grid.copy()
        normalized[..., 0] = (
            2.0 * (normalized[..., 0] + 0.5) / width - 1.0
        )
        normalized[..., 1] = (
            2.0 * (normalized[..., 1] + 0.5) / height - 1.0
        )
        return normalized

    def _head_grid(
        self,
        points: np.ndarray,
        shoulder_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build an orientation-normalized 4x2 face grid."""
        rows, columns = ANATOMICAL_CANONICAL_GRID_SIZE
        grid = np.zeros((rows, columns, 2), dtype=np.float32)
        valid = np.zeros((rows, columns), dtype=bool)
        reliable = points[:, 2] >= self.min_keypoint_confidence
        left_indices = [index for index in (1, 3) if reliable[index]]
        right_indices = [index for index in (2, 4) if reliable[index]]
        head_indices = [
            index
            for index in self._PART_KEYPOINTS["head"]
            if reliable[index]
        ]
        if len(head_indices) < 2:
            return grid, valid

        bilateral_face_axis = bool(left_indices and right_indices)
        if bilateral_face_axis:
            left_center = points[left_indices, :2].mean(axis=0)
            right_center = points[right_indices, :2].mean(axis=0)
        elif reliable[5] and reliable[6]:
            # Cropped/occluded faces often lose one eye/ear side. The
            # shoulder axis still preserves anatomical left-to-right order.
            left_center = points[5, :2]
            right_center = points[6, :2]
        else:
            return grid, valid
        horizontal = right_center - left_center
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-6:
            return grid, valid
        horizontal /= horizontal_norm
        vertical = np.asarray(
            (-horizontal[1], horizontal[0]),
            dtype=np.float32,
        )
        if vertical[1] < 0:
            vertical *= -1

        visible = points[head_indices, :2]
        center = visible.mean(axis=0)
        horizontal_extent = max(
            (
                horizontal_norm * 0.70
                if bilateral_face_axis
                else shoulder_width * 0.32
            ),
            shoulder_width * 0.32,
            2.0,
        )
        projected_vertical = (visible - center) @ vertical
        vertical_extent = max(
            float(np.ptp(projected_vertical)) * 0.90,
            shoulder_width * 0.42,
            3.0,
        )
        row_offsets = (
            (np.arange(rows, dtype=np.float32) + 0.5) / rows
            - 0.5
        ) * 2.0
        column_offsets = (
            (np.arange(columns, dtype=np.float32) + 0.5) / columns
            - 0.5
        ) * 2.0
        grid = (
            center[None, None]
            + row_offsets[:, None, None]
            * vertical[None, None]
            * vertical_extent
            + column_offsets[None, :, None]
            * horizontal[None, None]
            * horizontal_extent
        ).astype(np.float32)
        valid.fill(True)
        return grid, valid

    def _torso_grid(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bilinearly map a canonical 4x2 grid into the torso quadrilateral."""
        rows, columns = ANATOMICAL_CANONICAL_GRID_SIZE
        grid = np.zeros((rows, columns, 2), dtype=np.float32)
        valid = np.zeros((rows, columns), dtype=bool)
        indices = (5, 6, 11, 12)
        if any(
            points[index, 2] < self.min_keypoint_confidence
            for index in indices
        ):
            return grid, valid

        left_shoulder = points[5, :2]
        right_shoulder = points[6, :2]
        left_hip = points[11, :2]
        right_hip = points[12, :2]
        for row in range(rows):
            vertical = (row + 0.5) / rows
            left = (
                (1.0 - vertical) * left_shoulder
                + vertical * left_hip
            )
            right = (
                (1.0 - vertical) * right_shoulder
                + vertical * right_hip
            )
            for column in range(columns):
                horizontal = (column + 0.5) / columns
                grid[row, column] = (
                    (1.0 - horizontal) * left
                    + horizontal * right
                )
        valid.fill(True)
        return grid, valid

    def _limb_grid(
        self,
        points: np.ndarray,
        part: str,
        shoulder_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align grid rows proximal-to-distal along a two-segment limb."""
        rows, columns = ANATOMICAL_CANONICAL_GRID_SIZE
        grid = np.zeros((rows, columns, 2), dtype=np.float32)
        valid = np.zeros((rows, columns), dtype=bool)
        thickness_scale = 0.24 if "arm" in part else 0.34
        radius = max(shoulder_width * thickness_scale * 0.5, 1.0)
        row = 0
        for start, end in self._LIMB_SEGMENTS[part]:
            segment_valid = min(
                points[start, 2],
                points[end, 2],
            ) >= self.min_keypoint_confidence
            if not segment_valid:
                row += 2
                continue
            start_point = points[start, :2]
            end_point = points[end, :2]
            direction = end_point - start_point
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                row += 2
                continue
            direction /= length
            transverse = np.asarray(
                (-direction[1], direction[0]),
                dtype=np.float32,
            )
            for segment_row in range(2):
                longitudinal = (segment_row + 0.5) / 2.0
                center = (
                    (1.0 - longitudinal) * start_point
                    + longitudinal * end_point
                )
                for column in range(columns):
                    transverse_offset = (
                        (column + 0.5) / columns - 0.5
                    ) * 2.0
                    grid[row, column] = (
                        center
                        + transverse_offset * transverse * radius
                    )
                    valid[row, column] = True
                row += 1
        return grid, valid

    @staticmethod
    def _foreground_grid_validity(
        grid: np.ndarray,
        foreground: np.ndarray,
    ) -> np.ndarray:
        """Return cells whose bilinear sample lies on visible foreground."""
        height, width = foreground.shape
        in_bounds = (
            (grid[..., 0] >= 0)
            & (grid[..., 0] <= width - 1)
            & (grid[..., 1] >= 0)
            & (grid[..., 1] <= height - 1)
        )
        sampled = cv2.remap(
            foreground,
            grid[..., 0].astype(np.float32),
            grid[..., 1].astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return in_bounds & (sampled >= 0.5)

    def __call__(
        self,
        index: int,
        size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Return masks, visible fractions, pose reliability, and validity."""
        width, height = size
        cache_key = (int(index), int(width), int(height))
        cached = self._cached_target(
            cache_key,
            width=width,
            height=height,
        )
        if cached is not None:
            return cached
        sample_key = self._sample_keys[index]
        record = self.records.get(sample_key)
        if not isinstance(record, dict):
            record = {}
        foreground = self._person_mask(record, sample_key, size)
        mask_valid = foreground is not None
        accessory = self._accessory_mask(
            record,
            size,
        )
        accessory_valid = accessory is not None
        accessory_visibility = float(accessory_valid)
        points = self._keypoints(record, size)
        if points is None:
            if not mask_valid and not accessory_valid:
                self._target_cache[cache_key] = None
                return self._empty_target(width, height)
            target = self._empty_target(width, height)
            if mask_valid:
                target["foreground_mask"] = torch.from_numpy(
                    foreground[None].copy()
                ).float()
            if accessory_valid:
                target["accessory_mask"] = torch.from_numpy(
                    accessory[None].copy()
                ).float()
                target["accessory_visibility"] = torch.tensor(1.0)
                target["accessory_reliability"] = torch.tensor(1.0)
                target["accessory_valid"] = torch.tensor(True)
            target["mask_valid"] = torch.tensor(mask_valid)
            target["valid"] = torch.tensor(
                mask_valid or accessory_valid
            )
            return target
        points = self._discard_out_of_bounds_confidence(points, size)

        shoulder_width = self._shoulder_width(points, width)
        masks = np.zeros(
            (len(ANATOMICAL_PARTS), height, width),
            dtype=np.float32,
        )
        canonical_grid = np.zeros(
            (
                len(ANATOMICAL_PARTS),
                *ANATOMICAL_CANONICAL_GRID_SIZE,
                2,
            ),
            dtype=np.float32,
        )
        pose_grid_valid = np.zeros(
            (
                len(ANATOMICAL_PARTS),
                *ANATOMICAL_CANONICAL_GRID_SIZE,
            ),
            dtype=bool,
        )
        visibility = np.zeros(len(ANATOMICAL_PARTS), dtype=np.float32)
        for part_index, part in enumerate(ANATOMICAL_PARTS):
            if part == "head":
                self._draw_head(
                    masks[part_index],
                    points,
                    shoulder_width,
                )
                part_grid, pose_valid = self._head_grid(
                    points,
                    shoulder_width,
                )
            elif part == "torso":
                self._draw_torso(
                    masks[part_index],
                    points,
                )
                part_grid, pose_valid = self._torso_grid(points)
            else:
                self._draw_limb(
                    masks[part_index],
                    points,
                    part,
                    shoulder_width,
                )
                part_grid, pose_valid = self._limb_grid(
                    points,
                    part,
                    shoulder_width,
                )
            unoccluded_mass = float(masks[part_index].sum())
            if mask_valid:
                # Pose supplies topology while the person mask softly favors
                # visible foreground. The residual support avoids turning
                # segmentation holes into hard-negative anatomical targets.
                masks[part_index] *= 0.15 + 0.85 * foreground
                visible_mass = float(masks[part_index].sum())
                if unoccluded_mass > 0:
                    visibility[part_index] = np.clip(
                        visible_mass / unoccluded_mass,
                        0.0,
                        1.0,
                    )
                if self.feather > 0:
                    masks[part_index] = cv2.GaussianBlur(
                        masks[part_index],
                        (0, 0),
                        sigmaX=self.feather,
                        sigmaY=self.feather,
                    )
                    maximum = float(masks[part_index].max())
                    if maximum > 0:
                        masks[part_index] /= maximum
            else:
                if self.feather > 0:
                    masks[part_index] = cv2.GaussianBlur(
                        masks[part_index],
                        (0, 0),
                        sigmaX=self.feather,
                        sigmaY=self.feather,
                    )
                maximum = float(masks[part_index].max())
                if maximum > 0:
                    masks[part_index] /= maximum

            canonical_grid[part_index] = part_grid
            pose_grid_valid[part_index] = pose_valid

        grid_rows, grid_columns = ANATOMICAL_CANONICAL_GRID_SIZE
        reshaped_grid = canonical_grid.reshape(
            len(ANATOMICAL_PARTS) * grid_rows,
            grid_columns,
            2,
        )
        pose_grid_in_bounds = self._foreground_grid_validity(
            reshaped_grid,
            np.ones((height, width), dtype=np.float32),
        ).reshape(
            len(ANATOMICAL_PARTS),
            grid_rows,
            grid_columns,
        )
        canonical_grid_pose_valid = (
            pose_grid_valid & pose_grid_in_bounds
        )
        foreground_grid_valid = (
            self._foreground_grid_validity(
                reshaped_grid,
                foreground,
            ).reshape(
                len(ANATOMICAL_PARTS),
                grid_rows,
                grid_columns,
            )
            if mask_valid
            else np.zeros_like(pose_grid_valid)
        )
        canonical_grid_valid = (
            pose_grid_valid & foreground_grid_valid
        )
        canonical_grid = self._normalize_grid(
            canonical_grid,
            width=width,
            height=height,
        )
        pose_keypoints = np.zeros(
            (COCO_KEYPOINT_COUNT, 3),
            dtype=np.float32,
        )
        pose_keypoints[:, :2] = self._normalize_grid(
            points[:, :2],
            width=width,
            height=height,
        )
        pose_in_bounds = (
            (points[:, 0] >= 0)
            & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height - 1)
        )
        pose_keypoints[:, 2] = np.where(
            pose_in_bounds
            & (points[:, 2] >= self.min_keypoint_confidence),
            points[:, 2],
            0.0,
        )

        pose_reliability = np.asarray(
            [self._visibility(points, part) for part in ANATOMICAL_PARTS],
            dtype=np.float32,
        )
        masks_present = (
            masks.reshape(len(ANATOMICAL_PARTS), -1).max(axis=1) > 0
        )
        if not mask_valid:
            visibility = pose_reliability.copy()
        visibility *= masks_present
        pose_mask_agreement = self._pose_mask_agreement(
            points,
            foreground,
        )
        reliability = (
            pose_reliability
            * masks_present
            * (
                (0.25 + 0.75 * pose_mask_agreement)
                if mask_valid
                else self.pose_only_reliability
            )
        )
        quantized_masks = np.rint(masks.clip(0, 1) * 255.0).astype(np.uint8)
        quantized_foreground = np.rint(
            (
                foreground
                if foreground is not None
                else np.zeros((height, width), dtype=np.float32)
            ).clip(0, 1)
            * 255.0
        ).astype(np.uint8)[None]
        quantized_accessory = np.rint(
            (
                accessory
                if accessory is not None
                else np.zeros((height, width), dtype=np.float32)
            ).clip(0, 1)
            * 255.0
        ).astype(np.uint8)[None]
        cached_grid = canonical_grid.astype(np.float16)
        cached_grid_valid = canonical_grid_valid.astype(np.uint8)
        cached_pose_grid_valid = canonical_grid_pose_valid.astype(np.uint8)
        cached_pose_keypoints = pose_keypoints.astype(np.float16)
        visibility_tuple = tuple(float(value) for value in visibility)
        reliability_tuple = tuple(float(value) for value in reliability)
        pose_reliability_tuple = tuple(
            float(value) for value in pose_reliability
        )
        pose_valid = any(value > 0 for value in pose_reliability_tuple)
        self._target_cache[cache_key] = (
            zlib.compress(quantized_masks.tobytes(), level=1),
            zlib.compress(quantized_foreground.tobytes(), level=1),
            zlib.compress(quantized_accessory.tobytes(), level=1),
            cached_grid.tobytes(),
            cached_grid_valid.tobytes(),
            cached_pose_grid_valid.tobytes(),
            cached_pose_keypoints.tobytes(),
            visibility_tuple,
            reliability_tuple,
            pose_reliability_tuple,
            pose_mask_agreement,
            mask_valid,
            accessory_visibility,
            accessory_valid,
        )
        return {
            "masks": torch.from_numpy(quantized_masks).float().div_(255.0),
            "foreground_mask": (
                torch.from_numpy(quantized_foreground).float().div_(255.0)
            ),
            "accessory_mask": (
                torch.from_numpy(quantized_accessory).float().div_(255.0)
            ),
            "canonical_grid": torch.from_numpy(
                cached_grid.astype(np.float32)
            ),
            "canonical_grid_valid": torch.from_numpy(
                cached_grid_valid
            ).bool(),
            "canonical_grid_pose_valid": torch.from_numpy(
                cached_pose_grid_valid
            ).bool(),
            "pose_keypoints": torch.from_numpy(
                cached_pose_keypoints.astype(np.float32)
            ),
            "visibility": torch.tensor(visibility_tuple, dtype=torch.float32),
            "reliability": torch.tensor(
                reliability_tuple,
                dtype=torch.float32,
            ),
            "pose_reliability": torch.tensor(
                pose_reliability_tuple,
                dtype=torch.float32,
            ),
            "pose_mask_agreement": torch.tensor(
                pose_mask_agreement,
                dtype=torch.float32,
            ),
            "accessory_visibility": torch.tensor(
                accessory_visibility,
                dtype=torch.float32,
            ),
            "accessory_reliability": torch.tensor(
                accessory_visibility,
                dtype=torch.float32,
            ),
            "accessory_valid": torch.tensor(accessory_valid),
            "pose_valid": torch.tensor(pose_valid),
            "mask_valid": torch.tensor(mask_valid),
            "valid": torch.tensor(mask_valid),
        }
