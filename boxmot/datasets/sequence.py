"""Generic sequence loaders driven by declared modality formats and classes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from boxmot.datasets.readers.frames import numeric_frame_paths

if TYPE_CHECKING:
    import torch

    from boxmot.datasets.inputs import DatasetInputs, ModalityInput, SequenceInputs
    from boxmot.structures import CameraModel, Detections, Detections3D, Frame, Tracks


def _validate_fps(fps: float) -> float:
    """Require explicit finite positive timing before opening any inputs."""
    if type(fps) not in {int, float} or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Sensor sequence fps must be a positive finite number.")
    return float(fps)


def _class_ids(classes: Mapping[str, Mapping[str, Any]], *, evaluation: str = "target") -> tuple[int, ...]:
    """Validate normalized configured class IDs without assigning dataset labels."""
    values = tuple(value["id"] for value in classes.values())
    if not values or any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("Dataset classes must have unique nonnegative integer IDs.")
    selected = tuple(value["id"] for value in classes.values() if value.get("evaluation", "target") == evaluation)
    if evaluation == "target" and not selected:
        raise ValueError("Dataset classes must declare at least one target class.")
    return selected


def _instance_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Require explicit PNG encoding rules without assigning dataset conventions."""
    options = dict(options)
    unknown = set(options) - {"class_divisor", "background_id", "ignore_ids"}
    if unknown:
        raise ValueError(f"Unsupported instance-png options: {', '.join(sorted(unknown))}.")
    for name in ("class_divisor", "background_id"):
        if name not in options:
            raise ValueError(f"instance-png options require {name}.")
    return options


def _single_path(modality: ModalityInput, expected_format: str, role: str) -> Path:
    """Validate a single-path modality before selecting its format reader."""
    if modality.format != expected_format:
        raise ValueError(f"{role} requires format {expected_format!r}, got {modality.format!r}.")
    if len(modality.paths) != 1:
        raise ValueError(f"{role} requires exactly one input path.")
    if expected_format not in {"instance-png", "kitti-tracking-labels"} and modality.options:
        raise ValueError(f"{expected_format} does not support reader options: {', '.join(sorted(modality.options))}.")
    return Path(modality.paths[0]).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ImageSample:
    """One RGB frame, optional annotated tracks, and a separate ignore region."""

    frame: Frame
    ground_truth: Tracks | None
    ignore_mask: torch.Tensor | None


class ImageDataset(Sequence[ImageSample]):
    """Load numeric image frames with optional encoded instance annotations.

    Raw roots contain one directory per sequence. ``from_inputs`` reads the
    same data from a resolved dataset manifest. Construction checks filenames;
    RGB pixels and annotations are decoded only when a sample is requested.
    Native frame numbers are retained in timestamps and sample IDs.
    """

    def __init__(
        self,
        image_root: str | Path,
        instances_root: str | Path | None = None,
        *,
        split: str = "train",
        sequence_ids: Sequence[str] | None = None,
        classes: Mapping[str, Mapping[str, Any]],
        fps: float,
        instance_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(split, str) or not split or split != split.strip() or ":" in split:
            raise ValueError("split must be a non-empty string without surrounding whitespace or ':'.")
        self.fps = _validate_fps(fps)
        self.class_ids = _class_ids(classes)
        self.ignore_class_ids = _class_ids(classes, evaluation="ignore")
        self.instance_options = (
            _instance_options(instance_options or {}) if instances_root is not None else dict(instance_options or {})
        )
        self.image_root = Path(image_root).expanduser().resolve()
        self.instances_root = None if instances_root is None else Path(instances_root).expanduser().resolve()
        self.split = split
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image dataset image root does not exist: {self.image_root}")
        if self.instances_root is not None and not self.instances_root.is_dir():
            raise FileNotFoundError(f"Image dataset instances root does not exist: {self.instances_root}")

        available = {
            path.name: path for path in self.image_root.iterdir() if path.is_dir() and not path.name.startswith(".")
        }
        if sequence_ids is None:
            selected = tuple(sorted(available))
        else:
            if isinstance(sequence_ids, (str, bytes)) or not isinstance(sequence_ids, Sequence):
                raise TypeError("sequence_ids must be a sequence of exact directory names, not a string.")
            selected = tuple(sequence_ids)
            if not selected:
                raise ValueError("sequence_ids must contain at least one sequence name.")
            if any(
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or any(character in name for character in ("/", "\\", ":"))
                or name != name.strip()
                for name in selected
            ):
                raise ValueError("sequence_ids must contain non-empty directory names without paths or ':'.")
            if len(set(selected)) != len(selected):
                raise ValueError("sequence_ids must not contain duplicate names.")
            missing = sorted(set(selected) - available.keys())
            if missing:
                raise ValueError(f"Image dataset image root does not contain requested sequences: {missing}")
            selected = tuple(sorted(selected))
        if sequence_ids is None and not selected:
            raise ValueError(f"Image dataset image root contains no sequence directories: {self.image_root}")

        self.sequence_ids = selected
        paths: list[tuple[str, Path, Path | None, dict[str, Any]]] = []
        for sequence_id in selected:
            sequence_root = available[sequence_id]
            if ":" in sequence_id or not sequence_root.resolve().is_relative_to(self.image_root):
                raise ValueError(f"Invalid image sequence directory: {sequence_root}")
            for image_path in numeric_frame_paths(sequence_root):
                annotation_path = None
                if self.instances_root is not None:
                    annotation_path = self.instances_root / sequence_id / image_path.with_suffix(".png").name
                    if not annotation_path.is_file():
                        raise FileNotFoundError(f"Missing paired instance PNG for {image_path}: {annotation_path}")
                paths.append((sequence_id, image_path, annotation_path, self.instance_options))
        self._paths = tuple(paths)

    @classmethod
    def from_inputs(cls, inputs: DatasetInputs) -> ImageDataset:
        """Index images and optional ground truth from a portable dataset manifest."""
        dataset = cls.__new__(cls)
        dataset.image_root = inputs.root
        dataset.instances_root = None
        dataset.split = inputs.split
        dataset.fps = _validate_fps(inputs.fps)
        dataset.class_ids = _class_ids(inputs.classes)
        dataset.ignore_class_ids = _class_ids(inputs.classes, evaluation="ignore")
        dataset.instance_options = {}
        dataset.sequence_ids = tuple(sequence.sequence_id for sequence in inputs.sequences)
        paths: list[tuple[str, Path, Path | None, dict[str, Any]]] = []
        for sequence in inputs.sequences:
            images = sequence.modalities.get("images")
            if images is None:
                raise ValueError(f"Image dataset sequence {sequence.sequence_id!r} requires an images modality.")
            image_root = _single_path(images, "image-directory", "images")
            ground_truth = sequence.modalities.get("ground_truth")
            annotations = None if ground_truth is None else _single_path(ground_truth, "instance-png", "ground_truth")
            options = {} if ground_truth is None else _instance_options(ground_truth.options)
            for image_path in numeric_frame_paths(image_root):
                annotation_path = None if annotations is None else annotations / image_path.with_suffix(".png").name
                if annotation_path is not None and not annotation_path.is_file():
                    raise FileNotFoundError(f"Missing paired instance PNG for {image_path}: {annotation_path}")
                paths.append((sequence.sequence_id, image_path, annotation_path, options))
        dataset._paths = tuple(paths)
        return dataset

    def __len__(self) -> int:
        """Return the number of selected frames without decoding images."""

        return len(self._paths)

    @overload
    def __getitem__(self, index: int) -> ImageSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ImageSample, ...]: ...

    def __getitem__(self, index: int | slice) -> ImageSample | tuple[ImageSample, ...]:
        """Decode one frame or a slice of frames and matching annotations."""

        from boxmot.datasets.readers.images import read_rgb_chw_uint8
        from boxmot.datasets.readers.masks import read_instance_png
        from boxmot.structures import Frame

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"Image dataset indices must be integers or slices, got {type(index).__name__}.")
        sequence_id, image_path, annotation_path, options = self._paths[index]
        frame_index = int(image_path.stem)
        frame = Frame(
            image=read_rgb_chw_uint8(image_path.as_uri(), self.image_root.as_uri()),
            sample_id=f"{self.split}:{sequence_id}:{frame_index}",
            sequence_id=sequence_id,
            frame_index=frame_index,
            timestamp_s=frame_index / self.fps,
            source_uri=image_path.as_uri(),
        )
        ground_truth, ignore_mask = (
            (None, None)
            if annotation_path is None
            else read_instance_png(
                annotation_path, frame, class_ids=self.class_ids, ignore_class_ids=self.ignore_class_ids, **options
            )
        )
        return ImageSample(frame=frame, ground_truth=ground_truth, ignore_mask=ignore_mask)


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Aligned observations and optional camera metadata without decoded pixels."""

    frame_index: int
    image_size: tuple[int, int]
    detections: Detections
    detections_3d: Detections3D
    camera: CameraModel | None
    timestamp_s: float


class MultimodalSequence(Sequence[SensorFrame]):
    """Align configured modalities against an authoritative image timeline.

    Images are required. Predictions, calibration and absolute camera poses are
    optional; frames without a prediction retain empty canonical detections.
    Format readers own decoding and coordinate checks. Dataset classes, split
    and frame rate determine IDs and timing independently of dataset names.
    """

    def __init__(
        self,
        sequence_inputs: SequenceInputs,
        *,
        classes: Mapping[str, Mapping[str, Any]],
        fps: float,
        split: str = "train",
    ) -> None:
        from boxmot.datasets.config import validate_sequence_names
        from boxmot.datasets.readers.frames import image_sequence_size

        self.fps = _validate_fps(fps)
        self.sequence_id = sequence_inputs.sequence_id
        validate_sequence_names([self.sequence_id])
        if not isinstance(split, str) or not split or split != split.strip() or ":" in split:
            raise ValueError("split must be a non-empty string without surrounding whitespace or ':'.")
        self.split = split
        class_ids = _class_ids(classes)
        modalities = sequence_inputs.modalities
        unknown = set(modalities) - {
            "images",
            "ground_truth",
            "ground_truth_3d",
            "ground_truth_objects",
            "detections_2d",
            "detections_3d",
            "calibration",
            "poses",
        }
        if unknown:
            raise ValueError(f"Unsupported sequence modalities: {', '.join(sorted(unknown))}.")
        for role, modality in modalities.items():
            if role == "ground_truth":
                _single_path(modality, "instance-png", role)
                _instance_options(modality.options)
            elif role == "ground_truth_3d":
                # Calibration annotations never enter tracker observations.
                _single_path(modality, "kitti-tracking-labels", role)
            elif role == "ground_truth_objects":
                _single_path(modality, "kitti-object-labels", role)
            elif role != "detections_3d" and modality.options:
                raise ValueError(
                    f"{modality.format} does not support reader options: {', '.join(sorted(modality.options))}."
                )
        if "images" not in modalities:
            raise ValueError("Multimodal sequences require an images modality to define frame alignment.")
        images = _single_path(modalities["images"], "image-directory", "images")
        self._detections = None
        if "detections_2d" in modalities:
            from boxmot.datasets.readers.detections import TrackRcnnSequence

            detections = _single_path(modalities["detections_2d"], "trackrcnn", "detections_2d")
            self._detections = TrackRcnnSequence(
                self.sequence_id, images=images, detections=detections, class_ids=class_ids, split=split
            )
            self.frame_paths = self._detections.frame_paths
            self.image_size = self._detections.image_size
        else:
            self.frame_paths = numeric_frame_paths(images, contiguous=True)
            self.image_size = image_sequence_size(self.frame_paths)
        self._projection = self._poses = self._spatial = None
        self.missing_3d_frames: dict[str, tuple[int, ...]] = {}
        if "calibration" in modalities:
            from boxmot.datasets.readers.calibration import read_kitti_projection

            path = _single_path(modalities["calibration"], "kitti-p2", "calibration")
            self._projection = read_kitti_projection(path)
        if "poses" in modalities:
            from boxmot.datasets.readers.poses import read_camera_to_world_poses

            if self._projection is None:
                raise ValueError("Camera-to-world poses require camera calibration.")
            path = _single_path(modalities["poses"], "camera-to-world-npy", "poses")
            self._poses = read_camera_to_world_poses(path, len(self))
        if "detections_3d" in modalities:
            from boxmot.datasets.readers.boxes3d import KittiDetections3D

            spatial = modalities["detections_3d"]
            if spatial.format != "kitti-detections":
                raise ValueError(f"detections_3d requires format 'kitti-detections', got {spatial.format!r}.")
            if not spatial.paths:
                raise ValueError("detections_3d requires at least one input directory.")
            self._spatial = KittiDetections3D(
                spatial.paths, frame_count=len(self), classes=classes, options=spatial.options
            )
            self.missing_3d_frames = self._spatial.missing_frames

    def __len__(self) -> int:
        """Return the authoritative image frame count without decoding pixels."""
        return len(self.frame_paths)

    @overload
    def __getitem__(self, index: int) -> SensorFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SensorFrame, ...]: ...

    def __getitem__(self, index: int | slice) -> SensorFrame | tuple[SensorFrame, ...]:
        """Decode one frame's masks and sensor predictions without reading RGB."""
        import torch

        from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MaskBatch

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Multimodal sequence indices must be integers or slices.")
        frame_index = int(self.frame_paths[index].stem)
        sample_id = f"{self.split}:{self.sequence_id}:{frame_index}"
        detections = (
            self._detections[index].detections
            if self._detections is not None
            else Detections(
                Boxes(torch.empty((0, 4), dtype=torch.float32)),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.int64),
                sample_id,
                masks=MaskBatch(torch.empty((0, *self.image_size), dtype=torch.bool)),
            )
        )
        detections_3d = (
            self._spatial.read(frame_index, sample_id)
            if self._spatial is not None
            else Detections3D(
                Boxes3D(torch.empty((0, 7), dtype=torch.float32)),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.int64),
                sample_id,
            )
        )
        camera = None
        if self._projection is not None:
            pose = None if self._poses is None else torch.from_numpy(self._poses[frame_index])
            camera = CameraModel(self._projection, self.image_size, pose)
        return SensorFrame(frame_index, self.image_size, detections, detections_3d, camera, frame_index / self.fps)


__all__ = ("ImageDataset", "ImageSample", "MultimodalSequence", "SensorFrame")
