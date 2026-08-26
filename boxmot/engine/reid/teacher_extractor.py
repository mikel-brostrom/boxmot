"""Extract offline human-privileged teacher signals for HP-GRD.

The extractor deliberately sits outside the training and inference graphs.  It
loads a registered ReID teacher checkpoint, aligns semantic masks with each
training image, and writes the tensor-only input consumed by
``boxmot.engine.reid.privileged_cache``::

    uv run python -m boxmot.engine.reid.teacher_extractor \
      --teacher teacher.pt --dataset-index train-samples.json \
      --image-root datasets/Market-1501-v15.09.15 \
      --anatomical-metadata runs/anatomical-metadata \
      --include-leave-part-out --output teacher-signals.pt

``--part-mask-input`` is an alternative to the repository's pose/anatomical
metadata.  It accepts a safe ``torch.save`` dictionary containing
``sample_indices`` and ``part_masks`` plus optional ``part_visibility`` and
``part_confidence`` tensors. Ordered ``part_names`` are propagated when
present; six-part anatomical inputs default to BoxMOT's canonical semantic
order. No teacher, parser, or pose model is downloaded by this command.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from boxmot.engine.reid.privileged_cache import load_dataset_index
from boxmot.reid.core.preprocessing import (
    DEFAULT_PREPROCESS,
    IMAGENET_MEAN_RGB,
    PREPROCESS_REGISTRY,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_PARTS,
    PoseAnatomicalTargetProvider,
)
from boxmot.reid.training.trainer_components.privileged_graph import (
    DatasetSampleProvenance,
    PrivilegedGraphTeacherCache,
    validate_part_names,
)

_IMAGE_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
_IMAGE_NORMALIZATION_STD = (0.229, 0.224, 0.225)
_DESCRIPTOR_KEYS = (
    "norm_concat_bn",
    "global_descriptors",
    "global_descriptor",
    "embeddings",
    "embedding",
    "features",
    "feature",
    "descriptor",
)


@dataclass(frozen=True)
class TeacherExtractionConfig:
    """Filesystem and runtime settings for one offline extraction pass."""

    dataset_index: Path
    image_root: Path
    output: Path
    teacher_checkpoint: Path | None = None
    model_name: str | None = None
    part_names: tuple[str, ...] | None = None
    anatomical_metadata: Path | None = None
    person_mask_dir: Path | None = None
    part_mask_input: Path | None = None
    img_size: tuple[int, int] | None = None
    preprocess: str | None = None
    batch_size: int = 32
    workers: int = 4
    device: str = "auto"
    amp: bool = True
    descriptor_key: str | None = None
    include_leave_part_out: bool = False
    global_confidence_from_parts: bool = False
    fill_value: float = 0.0
    max_intervention_batch: int | None = None
    normalize_descriptors: bool = True
    storage_dtype: torch.dtype = torch.float32
    overwrite: bool = False


@dataclass(frozen=True)
class TeacherExtractionResult:
    """Published tensor bundle and its compact shape summary."""

    output_path: Path
    sample_count: int
    part_count: int
    global_dim: int
    part_dim: int
    leave_part_out_dim: int | None
    model_name: str
    part_names: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable CLI summary."""

        return {
            "tensor_input": str(self.output_path),
            "sample_count": self.sample_count,
            "part_count": self.part_count,
            "global_dim": self.global_dim,
            "part_dim": self.part_dim,
            "leave_part_out_dim": self.leave_part_out_dim,
            "model_name": self.model_name,
            "part_names": list(self.part_names),
        }


class PartMaskSignalStore:
    """Stable-index store for parser/pose masks produced by another tool."""

    def __init__(
        self,
        *,
        sample_indices: torch.Tensor,
        part_masks: torch.Tensor,
        part_visibility: torch.Tensor | None = None,
        part_confidence: torch.Tensor | None = None,
        part_names: Sequence[str] | None = None,
    ) -> None:
        if not torch.is_tensor(sample_indices) or sample_indices.ndim != 1:
            raise ValueError("sample_indices must be a tensor with shape [N]")
        if sample_indices.dtype == torch.bool or sample_indices.dtype.is_floating_point or sample_indices.is_complex():
            raise TypeError("sample_indices must use an integer dtype")
        if not torch.is_tensor(part_masks) or part_masks.ndim != 4:
            raise ValueError("part_masks must be a tensor with shape [N,P,H,W]")
        if part_masks.shape[0] != sample_indices.numel() or part_masks.shape[1] < 1:
            raise ValueError("part_masks must align with sample_indices and contain at least one part")
        if not part_masks.dtype.is_floating_point:
            part_masks = part_masks.float()
        if not bool(torch.isfinite(part_masks).all()):
            raise ValueError("part_masks must contain only finite values")
        if bool(((part_masks < 0) | (part_masks > 1)).any()):
            raise ValueError("part_masks values must be in [0, 1]")

        indices = sample_indices.detach().cpu().long().clone()
        sorted_indices, order = indices.sort()
        if sorted_indices.numel() > 1 and bool((sorted_indices[1:] == sorted_indices[:-1]).any()):
            raise ValueError("sample_indices must be unique")
        masks = part_masks.detach().cpu().float().index_select(0, order).clone()
        strength = masks.flatten(2).amax(dim=2)
        visibility = strength if part_visibility is None else part_visibility
        confidence = strength if part_confidence is None else part_confidence
        visibility = self._validate_scores(visibility, masks.shape[:2], "part_visibility")
        confidence = self._validate_scores(confidence, masks.shape[:2], "part_confidence")
        present = strength > 1e-6

        self.sample_indices = sorted_indices
        self.part_masks = masks
        self.part_visibility = visibility.detach().cpu().float().index_select(0, order) * present
        self.part_confidence = confidence.detach().cpu().float().index_select(0, order) * present
        self.part_names = None if part_names is None else validate_part_names(part_names, int(masks.shape[1]))

    @staticmethod
    def _validate_scores(value: torch.Tensor, shape: torch.Size, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"{name} values must be finite and in [0, 1]")
        return value

    @property
    def part_count(self) -> int:
        """Number of semantic parts in every row."""

        return int(self.part_masks.shape[1])

    def lookup(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masks, visibility, and confidence for one stable index."""

        requested = torch.tensor(int(sample_index), dtype=torch.long)
        position = int(torch.searchsorted(self.sample_indices, requested).item())
        if position >= len(self.sample_indices) or int(self.sample_indices[position]) != int(sample_index):
            raise KeyError(f"Part-mask input has no row for stable sample index {sample_index}")
        return (
            self.part_masks[position].clone(),
            self.part_visibility[position].clone(),
            self.part_confidence[position].clone(),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> PartMaskSignalStore:
        """Load a safe tensor-only parser/pose mask bundle."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("Part-mask input must be a torch.save dictionary")
        allowed = {
            "sample_indices",
            "part_masks",
            "masks",
            "part_visibility",
            "visibility",
            "part_confidence",
            "confidence",
            "part_names",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Part-mask input has unknown fields: {sorted(unknown)}")
        if "sample_indices" not in payload:
            raise KeyError("Part-mask input is missing 'sample_indices'")
        masks = payload.get("part_masks", payload.get("masks"))
        if masks is None:
            raise KeyError("Part-mask input is missing 'part_masks'")
        return cls(
            sample_indices=payload["sample_indices"],
            part_masks=masks,
            part_visibility=payload.get("part_visibility", payload.get("visibility")),
            part_confidence=payload.get("part_confidence", payload.get("confidence")),
            part_names=payload.get("part_names"),
        )


class OfflineTeacherSignalDataset(Dataset):
    """Load clean RGB tensors and spatially aligned training-only part masks."""

    def __init__(
        self,
        samples: Sequence[DatasetSampleProvenance],
        *,
        image_root: str | os.PathLike[str],
        img_size: tuple[int, int],
        preprocess: str = DEFAULT_PREPROCESS,
        mask_store: PartMaskSignalStore | None = None,
        target_provider: Any | None = None,
    ) -> None:
        if (mask_store is None) == (target_provider is None):
            raise ValueError("Exactly one of mask_store or target_provider must be supplied")
        if len(img_size) != 2 or min(int(value) for value in img_size) < 1:
            raise ValueError("img_size must contain two positive integers")
        if preprocess not in PREPROCESS_REGISTRY:
            raise ValueError(f"Unknown preprocess {preprocess!r}; available={sorted(PREPROCESS_REGISTRY)}")
        self.samples = tuple(samples)
        self.image_root = Path(image_root).expanduser().resolve()
        self.img_size = tuple(int(value) for value in img_size)
        self.preprocess = preprocess
        self.mask_store = mask_store
        self.target_provider = target_provider

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, sample: DatasetSampleProvenance) -> Path:
        path = Path(sample.img_path).expanduser()
        candidates = (path,) if path.is_absolute() else (path, self.image_root / path)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        rendered = ", ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(f"Dataset-index image is not a file; checked: {rendered}")

    def __getitem__(self, row_index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[row_index]
        with Image.open(self._image_path(sample)) as source:
            image = source.convert("RGB")
        if self.mask_store is not None:
            masks, visibility, confidence = self.mask_store.lookup(sample.index)
        else:
            target = self.target_provider(row_index, image.size)
            if not isinstance(target, Mapping):
                raise TypeError("Anatomical target provider must return a mapping")
            masks = _required_tensor(target, "masks")
            visibility = _required_tensor(target, "visibility")
            confidence = _required_tensor(target, "reliability")
        image_tensor, masks = align_teacher_image_and_masks(
            image,
            masks,
            img_size=self.img_size,
            preprocess=self.preprocess,
        )
        present = masks.flatten(1).amax(dim=1) > 1e-6
        visibility = visibility.float().clamp(0, 1) * present
        confidence = confidence.float().clamp(0, 1) * present
        if visibility.shape != (masks.shape[0],) or confidence.shape != (masks.shape[0],):
            raise ValueError("Part visibility/confidence must each have shape [P]")
        return {
            "images": image_tensor,
            "sample_indices": torch.tensor(sample.index, dtype=torch.long),
            "part_masks": masks,
            "part_visibility": visibility,
            "part_confidence": confidence,
        }


def align_teacher_image_and_masks(
    image: Image.Image,
    part_masks: torch.Tensor,
    *,
    img_size: tuple[int, int],
    preprocess: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the deployed resize geometry to RGB and semantic masks together."""

    if not torch.is_tensor(part_masks) or part_masks.ndim != 3:
        raise ValueError("part_masks must have shape [P,H,W]")
    target_h, target_w = (int(img_size[0]), int(img_size[1]))
    source_w, source_h = image.size
    masks = part_masks.detach().float().clamp(0, 1)
    if tuple(masks.shape[-2:]) != (source_h, source_w):
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(source_h, source_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if preprocess == "resize":
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    elif preprocess == "resize_pad":
        scale = min(target_w / source_w, target_h / source_h)
        resized_w, resized_h = int(source_w * scale), int(source_h * scale)
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (target_w, target_h), IMAGENET_MEAN_RGB)
        left = (target_w - resized_w) // 2
        top = (target_h - resized_h) // 2
        padded.paste(image, (left, top))
        image = padded
        resized_masks = F.interpolate(
            masks.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        masks = masks.new_zeros((masks.shape[0], target_h, target_w))
        masks[:, top : top + resized_h, left : left + resized_w] = resized_masks
    else:
        raise ValueError(f"Unknown preprocess {preprocess!r}")

    image_tensor = TF.normalize(
        TF.to_tensor(image),
        mean=_IMAGE_NORMALIZATION_MEAN,
        std=_IMAGE_NORMALIZATION_STD,
    )
    return image_tensor, masks.clamp(0, 1)


def resolve_teacher_descriptor(output: object, descriptor_key: str | None = None) -> torch.Tensor:
    """Resolve a two-dimensional descriptor from common ReID output packets."""

    if descriptor_key is not None:
        value = _mapping_path(output, descriptor_key)
        return _as_descriptor(value, descriptor_key)
    if torch.is_tensor(output):
        return _as_descriptor(output, "teacher output")
    if isinstance(output, Mapping):
        for key in _DESCRIPTOR_KEYS:
            if key in output:
                return _as_descriptor(output[key], key)
        raise KeyError(
            "Teacher output mapping has no recognized descriptor key; "
            f"set --descriptor-key (available={sorted(str(key) for key in output)})"
        )
    if isinstance(output, (tuple, list)):
        errors: list[Exception] = []
        for value in output:
            try:
                return resolve_teacher_descriptor(value)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(error)
        if errors:
            raise TypeError("Teacher output sequence contains no descriptor tensor") from errors[-1]
        raise TypeError("Teacher output sequence contains no descriptor tensor")
    raise TypeError(f"Unsupported teacher output type: {type(output).__name__}")


def _mapping_path(output: object, path: str) -> object:
    value = output
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(f"Teacher output has no descriptor path {path!r}")
        value = value[component]
    return value


def _as_descriptor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Teacher descriptor {name!r} must be a tensor")
    if value.ndim == 2:
        descriptor = value
    elif value.ndim > 2:
        descriptor = value.flatten(2).mean(dim=2)
    else:
        raise ValueError(f"Teacher descriptor {name!r} must have shape [B,D] or [B,D,...]")
    if descriptor.shape[1] < 1 or not descriptor.dtype.is_floating_point:
        raise ValueError(f"Teacher descriptor {name!r} must have a positive floating-point feature dimension")
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError(f"Teacher descriptor {name!r} contains non-finite values")
    return descriptor


@torch.no_grad()
def extract_teacher_signal_bundle(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device | str,
    part_names: Sequence[str] | None = None,
    descriptor_key: str | None = None,
    include_leave_part_out: bool = False,
    global_confidence_from_parts: bool = False,
    fill_value: float = 0.0,
    max_intervention_batch: int | None = None,
    normalize_descriptors: bool = True,
    amp: bool = False,
    storage_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Extract global, masked-part, and optional leave-part-out descriptors.

    Part descriptors come from masked-*in* RGB views.  Leave-part-out
    descriptors come from the complementary masked-*out* views, matching the
    semantic intervention performed on the student during HP-GRD training.
    Invalid/missing masks receive zero descriptors and zero reliability.
    """

    device = torch.device(device)
    if storage_dtype not in {torch.float16, torch.float32, torch.float64}:
        raise ValueError("storage_dtype must be float16, float32, or float64")
    if not torch.isfinite(torch.tensor(float(fill_value))):
        raise ValueError("fill_value must be finite")
    if max_intervention_batch is not None and int(max_intervention_batch) < 1:
        raise ValueError("max_intervention_batch must be positive")

    was_training = model.training
    model.to(device).eval()
    collected: dict[str, list[torch.Tensor]] = {
        "sample_indices": [],
        "global_descriptors": [],
        "part_descriptors": [],
        "part_visibility": [],
        "part_confidence": [],
    }
    if include_leave_part_out:
        collected["leave_part_out_descriptors"] = []
    if global_confidence_from_parts:
        collected["global_confidence"] = []
    try:
        for batch in dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError("Teacher dataloader must yield mapping batches")
            images = _required_tensor(batch, "images").to(device=device, non_blocking=True)
            masks = _required_tensor(batch, "part_masks").to(
                device=device,
                dtype=images.dtype,
                non_blocking=True,
            )
            visibility = _required_tensor(batch, "part_visibility").float().clamp(0, 1)
            confidence = _required_tensor(batch, "part_confidence").float().clamp(0, 1)
            indices = _required_tensor(batch, "sample_indices").long()
            if masks.ndim != 4 or masks.shape[0] != images.shape[0]:
                raise ValueError("Batched part_masks must have shape [B,P,H,W]")
            if visibility.shape != masks.shape[:2] or confidence.shape != masks.shape[:2]:
                raise ValueError("Batched part visibility/confidence must have shape [B,P]")
            if indices.shape != (images.shape[0],):
                raise ValueError("Batched sample_indices must have shape [B]")

            global_descriptors = _forward_teacher(
                model,
                images,
                descriptor_key=descriptor_key,
                normalize=normalize_descriptors,
                amp=amp,
                device=device,
            )
            present = masks.flatten(2).amax(dim=2) > 1e-6
            valid = present & (visibility.to(device=device) > 0) & (confidence.to(device=device) > 0)
            part_descriptors = _extract_intervention_descriptors(
                model,
                images,
                masks,
                valid,
                descriptor_dim=global_descriptors.shape[1],
                keep_part=True,
                fill_value=float(fill_value),
                descriptor_key=descriptor_key,
                normalize=normalize_descriptors,
                amp=amp,
                device=device,
                max_forward_batch=max_intervention_batch or images.shape[0],
            )
            collected["sample_indices"].append(indices.detach().cpu())
            collected["global_descriptors"].append(global_descriptors.detach().cpu().to(storage_dtype))
            collected["part_descriptors"].append(part_descriptors.detach().cpu().to(storage_dtype))
            collected["part_visibility"].append((visibility * present.cpu()).cpu())
            collected["part_confidence"].append((confidence * present.cpu()).cpu())
            if include_leave_part_out:
                leave_part_out = _extract_intervention_descriptors(
                    model,
                    images,
                    masks,
                    valid,
                    descriptor_dim=global_descriptors.shape[1],
                    keep_part=False,
                    fill_value=float(fill_value),
                    descriptor_key=descriptor_key,
                    normalize=normalize_descriptors,
                    amp=amp,
                    device=device,
                    max_forward_batch=max_intervention_batch or images.shape[0],
                )
                collected["leave_part_out_descriptors"].append(leave_part_out.detach().cpu().to(storage_dtype))
            if global_confidence_from_parts:
                fused = 0.5 * (visibility + confidence)
                collected["global_confidence"].append(fused.amax(dim=1).cpu())
    finally:
        model.train(was_training)

    if not collected["sample_indices"]:
        raise ValueError("Teacher dataloader produced no batches")
    tensors = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    # The cache constructor is the authoritative downstream contract.  This
    # catches duplicate stable indices, invalid ranges, and shape drift before
    # a potentially expensive extraction result is published.
    resolved_part_names = _resolve_part_names(
        part_names,
        int(tensors["part_descriptors"].shape[1]),
    )
    PrivilegedGraphTeacherCache(part_names=resolved_part_names, **tensors)
    return tensors


def _extract_intervention_descriptors(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    *,
    descriptor_dim: int,
    keep_part: bool,
    fill_value: float,
    descriptor_key: str | None,
    normalize: bool,
    amp: bool,
    device: torch.device,
    max_forward_batch: int,
) -> torch.Tensor:
    batch_size, part_count = masks.shape[:2]
    flat_valid = valid.flatten()
    selected = flat_valid.nonzero(as_tuple=False).flatten()
    output = images.new_zeros((batch_size * part_count, descriptor_dim))
    if selected.numel() == 0:
        return output.reshape(batch_size, part_count, descriptor_dim)

    for chunk in selected.split(int(max_forward_batch)):
        base_rows = torch.div(chunk, part_count, rounding_mode="floor")
        part_rows = torch.remainder(chunk, part_count)
        selected_images = images.index_select(0, base_rows)
        selected_masks = masks[base_rows, part_rows, None].expand(
            -1,
            images.shape[1],
            -1,
            -1,
        )
        if keep_part:
            altered = fill_value + selected_masks * (selected_images - fill_value)
        else:
            altered = selected_images + selected_masks * (fill_value - selected_images)
        descriptors = _forward_teacher(
            model,
            altered,
            descriptor_key=descriptor_key,
            normalize=normalize,
            amp=amp,
            device=device,
        )
        if descriptors.shape[1] != descriptor_dim:
            raise ValueError(
                "Teacher descriptor dimension changed under a masked intervention: "
                f"global={descriptor_dim}, intervention={descriptors.shape[1]}"
            )
        output.index_copy_(0, chunk, descriptors.to(dtype=output.dtype))
    return output.reshape(batch_size, part_count, descriptor_dim)


def _forward_teacher(
    model: nn.Module,
    images: torch.Tensor,
    *,
    descriptor_key: str | None,
    normalize: bool,
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    amp_enabled = bool(amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        descriptor = resolve_teacher_descriptor(model(images), descriptor_key)
    descriptor = descriptor.float()
    if descriptor.shape[0] != images.shape[0]:
        raise ValueError("Teacher descriptor batch dimension does not match its input")
    return F.normalize(descriptor, p=2, dim=1) if normalize else descriptor


def load_registered_teacher(
    checkpoint: str | os.PathLike[str],
    *,
    device: torch.device | str,
    model_name: str | None = None,
    img_size: tuple[int, int] | None = None,
) -> tuple[nn.Module, str, dict[str, Any]]:
    """Build a registered ReID teacher and strictly load its deployed tensors."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint is not a file: {checkpoint}")
    resolved_name = model_name or ReIDModelRegistry.get_model_name(checkpoint)
    if not resolved_name:
        raise ValueError("Unable to infer teacher model; pass --model-name")
    device = torch.device(device)
    model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(checkpoint)
    if img_size is not None:
        model_kwargs["img_size"] = tuple(int(value) for value in img_size)
    model_kwargs = ReIDModelRegistry.deployment_model_kwargs(resolved_name, model_kwargs)
    model = ReIDModelRegistry.build_model(
        name=resolved_name,
        weights=checkpoint,
        num_classes=ReIDModelRegistry.get_nr_classes(checkpoint),
        loss="softmax",
        pretrained=False,
        use_gpu=device.type != "cpu",
        **model_kwargs,
    )
    ReIDModelRegistry.load_deployment_weights(model, checkpoint)
    model.to(device).eval()
    return model, resolved_name, model_kwargs


def _resolve_part_names(
    part_names: Sequence[str] | None,
    part_count: int,
) -> tuple[str, ...]:
    """Resolve unnamed six-part inputs to the canonical anatomical order."""

    if part_names is None:
        if part_count != len(ANATOMICAL_PARTS):
            raise ValueError(
                f"Ordered semantic part names are required for a non-canonical {part_count}-part extractor input"
            )
        part_names = ANATOMICAL_PARTS
    return validate_part_names(part_names, part_count)


def run_teacher_extraction(
    config: TeacherExtractionConfig,
    *,
    model: nn.Module | None = None,
) -> TeacherExtractionResult:
    """Run extraction from files and atomically publish the tensor bundle."""

    if (config.anatomical_metadata is None) == (config.part_mask_input is None):
        raise ValueError("Choose exactly one of anatomical_metadata or part_mask_input")
    if config.batch_size < 1 or config.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    output = Path(config.output)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing teacher-signal bundle: {output}")
    if output.exists() and output.is_dir():
        raise IsADirectoryError(f"Teacher-signal output must be a file: {output}")

    device = _resolve_device(config.device)
    checkpoint_kwargs: dict[str, Any] = {}
    resolved_name = config.model_name or (type(model).__name__ if model is not None else "")
    if model is None:
        if config.teacher_checkpoint is None:
            raise ValueError("teacher_checkpoint is required when model is not supplied")
        model, resolved_name, checkpoint_kwargs = load_registered_teacher(
            config.teacher_checkpoint,
            device=device,
            model_name=config.model_name,
            img_size=config.img_size,
        )
    img_size = config.img_size or tuple(checkpoint_kwargs.get("img_size", (256, 128)))
    if len(img_size) != 2:
        raise ValueError("Resolved teacher img_size must contain height and width")
    preprocess = (
        config.preprocess
        or (
            ReIDModelRegistry.get_checkpoint_preprocess(config.teacher_checkpoint)
            if config.teacher_checkpoint is not None
            else None
        )
        or DEFAULT_PREPROCESS
    )
    samples = load_dataset_index(config.dataset_index)
    mask_store = PartMaskSignalStore.load(config.part_mask_input) if config.part_mask_input is not None else None
    target_provider = None
    if config.anatomical_metadata is not None:
        target_provider = PoseAnatomicalTargetProvider(
            samples,
            image_root=config.image_root,
            metadata_root=config.anatomical_metadata,
            person_mask_dir=config.person_mask_dir,
            compact_nonsemantic=False,
        )
    source_part_names: Sequence[str] | None = ANATOMICAL_PARTS if target_provider is not None else mask_store.part_names
    source_part_count = len(ANATOMICAL_PARTS) if target_provider is not None else mask_store.part_count
    configured_part_names = (
        None if config.part_names is None else validate_part_names(config.part_names, source_part_count)
    )
    if configured_part_names is not None and source_part_names is not None:
        embedded_names = validate_part_names(source_part_names, source_part_count)
        if configured_part_names != embedded_names:
            raise ValueError(
                "Configured ordered part names do not match the mask/anatomical source: "
                f"configured={list(configured_part_names)!r}, "
                f"source={list(embedded_names)!r}"
            )
    resolved_part_names = _resolve_part_names(
        configured_part_names or source_part_names,
        source_part_count,
    )
    dataset = OfflineTeacherSignalDataset(
        samples,
        image_root=config.image_root,
        img_size=tuple(int(value) for value in img_size),
        preprocess=preprocess,
        mask_store=mask_store,
        target_provider=target_provider,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    tensors = extract_teacher_signal_bundle(
        model,
        dataloader,
        device=device,
        part_names=resolved_part_names,
        descriptor_key=config.descriptor_key,
        include_leave_part_out=config.include_leave_part_out,
        global_confidence_from_parts=config.global_confidence_from_parts,
        fill_value=config.fill_value,
        max_intervention_batch=config.max_intervention_batch,
        normalize_descriptors=config.normalize_descriptors,
        amp=config.amp,
        storage_dtype=config.storage_dtype,
    )
    _atomic_save_tensors(
        tensors,
        output,
        part_names=resolved_part_names,
        overwrite=config.overwrite,
    )
    return TeacherExtractionResult(
        output_path=output,
        sample_count=int(tensors["sample_indices"].numel()),
        part_count=int(tensors["part_descriptors"].shape[1]),
        global_dim=int(tensors["global_descriptors"].shape[1]),
        part_dim=int(tensors["part_descriptors"].shape[2]),
        leave_part_out_dim=(
            None if "leave_part_out_descriptors" not in tensors else int(tensors["leave_part_out_descriptors"].shape[2])
        ),
        model_name=str(resolved_name),
        part_names=resolved_part_names,
    )


def _atomic_save_tensors(
    tensors: Mapping[str, torch.Tensor],
    output: Path,
    *,
    part_names: Sequence[str],
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing teacher-signal bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        part_count = int(tensors["part_descriptors"].shape[1])
        names = validate_part_names(part_names, part_count)
        torch.save(
            {"part_names": list(names), "tensors": dict(tensors)},
            temporary_path,
        )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required_tensor(values: Mapping[str, object], key: str) -> torch.Tensor:
    value = values.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a tensor")
    return value


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="Registered ReID teacher checkpoint")
    parser.add_argument("--model-name", help="Teacher registry name when it cannot be inferred")
    parser.add_argument(
        "--part-names",
        nargs="+",
        help="Ordered semantic part names; canonical six-part anatomy is the default",
    )
    parser.add_argument("--dataset-index", type=Path, required=True, help="Stable JSON/JSONL sample index")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for relative image paths")
    mask_source = parser.add_mutually_exclusive_group(required=True)
    mask_source.add_argument("--anatomical-metadata", type=Path, help="Pose/anatomical metadata directory")
    mask_source.add_argument("--part-mask-input", type=Path, help="Tensor bundle with [N,P,H,W] masks")
    parser.add_argument("--person-mask-dir", type=Path, help="Optional external person parser masks")
    parser.add_argument("--output", type=Path, required=True, help="Output consumed by privileged_cache build")
    parser.add_argument("--img-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--preprocess", choices=sorted(PREPROCESS_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--descriptor-key", help="Mapping path for a non-standard teacher output")
    parser.add_argument("--include-leave-part-out", action="store_true")
    parser.add_argument("--global-confidence-from-parts", action="store_true")
    parser.add_argument("--fill-value", type=float, default=0.0, help="Fill in normalized RGB space")
    parser.add_argument("--max-intervention-batch", type=int)
    parser.add_argument("--no-normalize", action="store_true", help="Store raw teacher descriptors")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone registered-teacher extractor."""

    args = _build_parser().parse_args(argv)
    result = run_teacher_extraction(
        TeacherExtractionConfig(
            teacher_checkpoint=args.teacher,
            model_name=args.model_name,
            part_names=None if args.part_names is None else tuple(args.part_names),
            dataset_index=args.dataset_index,
            image_root=args.image_root,
            anatomical_metadata=args.anatomical_metadata,
            person_mask_dir=args.person_mask_dir,
            part_mask_input=args.part_mask_input,
            output=args.output,
            img_size=None if args.img_size is None else tuple(args.img_size),
            preprocess=args.preprocess,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            amp=not args.no_amp,
            descriptor_key=args.descriptor_key,
            include_leave_part_out=args.include_leave_part_out,
            global_confidence_from_parts=args.global_confidence_from_parts,
            fill_value=args.fill_value,
            max_intervention_batch=args.max_intervention_batch,
            normalize_descriptors=not args.no_normalize,
            storage_dtype=getattr(torch, args.storage_dtype),
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
