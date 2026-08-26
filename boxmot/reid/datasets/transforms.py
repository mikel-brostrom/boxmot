"""ReID training transforms (augmentation pipelines).

Augmentations ported from torchreid (deep-person-reid):
- Random2DTranslation: 1.05× upscale → random crop (LMBN-style)
- ColorAugmentation: PCA-based color jitter (Krizhevsky et al.)
- RandomPatch: occlusion simulation with a patch pool (Zhou et al., ICCV 2019)
"""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS, IMAGENET_MEAN_RGB
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_FLIP_PERMUTATION,
    ANATOMICAL_PART_KEYPOINTS,
    COCO_KEYPOINT_FLIP_PERMUTATION,
)
from boxmot.reid.datasets.base import ReIDSample


class ResizePad:
    """Resize preserving aspect ratio with ImageNet-mean padding (PIL version).

    Mirrors ``boxmot.reid.core.preprocessing.resize_pad`` but operates on PIL
    images so it can be used inside a ``torchvision.transforms.Compose`` chain.
    """

    def __init__(self, size: Tuple[int, int]):
        """Args: size as (H, W)."""
        self.target_h, self.target_w = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size  # PIL is (W, H)
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        padded = Image.new("RGB", (self.target_w, self.target_h), IMAGENET_MEAN_RGB)
        pad_left = (self.target_w - new_w) // 2
        pad_top = (self.target_h - new_h) // 2
        padded.paste(img, (pad_left, pad_top))
        return padded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size=({self.target_h}, {self.target_w}))"


def _resize_op(img_size: Tuple[int, int], preprocess: str):
    """Return the PIL resize operation matching the inference preprocess name."""
    if preprocess == "resize_pad":
        return ResizePad(img_size)
    return T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)


def _resize_spatial_masks(
    masks: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize tensors with arbitrary leading mask dimensions."""
    leading_shape = masks.shape[:-2]
    resized = F.interpolate(
        masks.reshape(-1, *masks.shape[-2:]).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.reshape(*leading_shape, *size)


def _resize_anatomical_spatial_targets(
    target: dict[str, torch.Tensor],
    size: tuple[int, int],
) -> None:
    """Resize every dense anatomical target with identical geometry."""
    for key in ("masks", "foreground_mask", "accessory_mask"):
        value = target.get(key)
        if torch.is_tensor(value):
            target[key] = _resize_spatial_masks(value, size)


def _clear_anatomical_spatial_region(
    target: dict[str, torch.Tensor],
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    """Remove supervision hidden by a synthetic occluder."""
    accessory_before = None
    if torch.is_tensor(target.get("accessory_mask")):
        accessory_before = target["accessory_mask"].sum()
    for key in ("masks", "foreground_mask", "accessory_mask"):
        value = target.get(key)
        if torch.is_tensor(value):
            value = value.clone()
            value[..., top : top + height, left : left + width] = 0
            target[key] = value
    if accessory_before is not None:
        retained = (
            target["accessory_mask"].sum()
            / accessory_before.clamp_min(1e-6)
        )
        for key in (
            "accessory_visibility",
            "accessory_reliability",
        ):
            if key in target:
                target[key] = target[key] * retained


def _transform_canonical_grid(
    target: dict[str, torch.Tensor],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    scale: tuple[float, float],
    offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Apply a pixel-center affine transform to canonical sample points."""
    source_width, source_height = source_size
    output_width, output_height = output_size
    scale_x, scale_y = scale
    offset_x, offset_y = offset
    grid = target["canonical_grid"].clone()
    x = (grid[..., 0] + 1.0) * source_width * 0.5 - 0.5
    y = (grid[..., 1] + 1.0) * source_height * 0.5 - 0.5
    x = (x + 0.5) * scale_x - 0.5 + offset_x
    y = (y + 0.5) * scale_y - 0.5 + offset_y
    grid[..., 0] = 2.0 * (x + 0.5) / output_width - 1.0
    grid[..., 1] = 2.0 * (y + 0.5) / output_height - 1.0
    in_bounds = (
        (x >= 0)
        & (x <= output_width - 1)
        & (y >= 0)
        & (y <= output_height - 1)
    )
    target["canonical_grid"] = grid
    for validity_key in (
        "canonical_grid_valid",
        "canonical_grid_pose_valid",
    ):
        if validity_key in target:
            target[validity_key] = (
                target[validity_key] & in_bounds
            )
    pose_keypoints = target.get("pose_keypoints")
    if torch.is_tensor(pose_keypoints):
        pose_keypoints = pose_keypoints.clone()
        pose_x = (
            (pose_keypoints[:, 0] + 1.0)
            * source_width
            * 0.5
            - 0.5
        )
        pose_y = (
            (pose_keypoints[:, 1] + 1.0)
            * source_height
            * 0.5
            - 0.5
        )
        pose_x = (pose_x + 0.5) * scale_x - 0.5 + offset_x
        pose_y = (pose_y + 0.5) * scale_y - 0.5 + offset_y
        pose_in_bounds = (
            (pose_x >= 0)
            & (pose_x <= output_width - 1)
            & (pose_y >= 0)
            & (pose_y <= output_height - 1)
        )
        pose_keypoints[:, 0] = (
            2.0 * (pose_x + 0.5) / output_width - 1.0
        )
        pose_keypoints[:, 1] = (
            2.0 * (pose_y + 0.5) / output_height - 1.0
        )
        pose_keypoints[:, 2] *= pose_in_bounds.to(
            pose_keypoints.dtype
        )
        target["pose_keypoints"] = pose_keypoints


def _invalidate_canonical_grid_region(
    target: dict[str, torch.Tensor],
    *,
    image_size: tuple[int, int],
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    """Invalidate canonical samples covered by a pasted/erased rectangle."""
    image_width, image_height = image_size
    grid = target["canonical_grid"]
    x = (grid[..., 0] + 1.0) * image_width * 0.5 - 0.5
    y = (grid[..., 1] + 1.0) * image_height * 0.5 - 0.5
    covered = (
        (x >= left)
        & (x < left + width)
        & (y >= top)
        & (y < top + height)
    )
    for validity_key in (
        "canonical_grid_valid",
        "canonical_grid_pose_valid",
    ):
        if validity_key in target:
            target[validity_key] = (
                target[validity_key] & ~covered
            )
    pose_keypoints = target.get("pose_keypoints")
    if torch.is_tensor(pose_keypoints):
        pose_x = (
            (pose_keypoints[:, 0] + 1.0) * image_width * 0.5 - 0.5
        )
        pose_y = (
            (pose_keypoints[:, 1] + 1.0) * image_height * 0.5 - 0.5
        )
        pose_covered = (
            (pose_x >= left)
            & (pose_x < left + width)
            & (pose_y >= top)
            & (pose_y < top + height)
        )
        pose_keypoints = pose_keypoints.clone()
        pose_keypoints[:, 2] *= (~pose_covered).to(
            pose_keypoints.dtype
        )
        target["pose_keypoints"] = pose_keypoints


class Random2DTranslation:
    """Randomly translate via scale× upscale → random crop.

    With probability *p* the image is resized to *scale*× the target size and
    then a random crop of the target size is taken.  Otherwise the image is
    simply resized to the target size.

    Reference:
        Zhou et al. "Omni-Scale Feature Learning for Person
        Re-Identification." ICCV 2019.
    """

    def __init__(self, height: int, width: int, p: float = 0.5, scale: float = 1.05):
        if scale < 1.0:
            raise ValueError("Random2DTranslation scale must be >= 1.0")
        self.height = height
        self.width = width
        self.p = p
        self.scale = float(scale)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img.resize((self.width, self.height), Image.BILINEAR)
        new_w = int(round(self.width * self.scale))
        new_h = int(round(self.height * self.scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        x1 = random.randint(0, new_w - self.width)
        y1 = random.randint(0, new_h - self.height)
        return resized.crop((x1, y1, x1 + self.width, y1 + self.height))

    def apply_with_anatomical_target(
        self,
        img: Image.Image,
        target: dict[str, torch.Tensor],
    ) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        """Apply one sampled resize/crop to both RGB and anatomical masks."""
        source_size = img.size
        source_width, source_height = source_size
        masks = target["masks"]
        if random.random() > self.p:
            img = img.resize((self.width, self.height), Image.BILINEAR)
            _resize_anatomical_spatial_targets(
                target,
                (self.height, self.width),
            )
            _transform_canonical_grid(
                target,
                source_size=source_size,
                output_size=(self.width, self.height),
                scale=(
                    self.width / source_width,
                    self.height / source_height,
                ),
            )
            return img, target

        new_w = int(round(self.width * self.scale))
        new_h = int(round(self.height * self.scale))
        x1 = random.randint(0, new_w - self.width)
        y1 = random.randint(0, new_h - self.height)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.crop((x1, y1, x1 + self.width, y1 + self.height))
        resized_masks = _resize_spatial_masks(masks, (new_h, new_w))
        before = resized_masks.flatten(1).sum(dim=1)
        cropped_masks = resized_masks[
            :,
            y1 : y1 + self.height,
            x1 : x1 + self.width,
        ]
        after = cropped_masks.flatten(1).sum(dim=1)
        retained = torch.where(
            before > 1e-6,
            after / before.clamp_min(1e-6),
            torch.zeros_like(before),
        )
        target["masks"] = cropped_masks
        foreground_mask = target.get("foreground_mask")
        if torch.is_tensor(foreground_mask):
            resized_foreground = _resize_spatial_masks(
                foreground_mask,
                (new_h, new_w),
            )
            target["foreground_mask"] = resized_foreground[
                ...,
                y1 : y1 + self.height,
                x1 : x1 + self.width,
            ]
        accessory_mask = target.get("accessory_mask")
        if torch.is_tensor(accessory_mask):
            resized_accessory = _resize_spatial_masks(
                accessory_mask,
                (new_h, new_w),
            )
            before_accessory = resized_accessory.sum()
            target["accessory_mask"] = resized_accessory[
                ...,
                y1 : y1 + self.height,
                x1 : x1 + self.width,
            ]
            retained_accessory = (
                target["accessory_mask"].sum()
                / before_accessory.clamp_min(1e-6)
            )
            target["accessory_visibility"] = (
                target["accessory_visibility"]
                * retained_accessory
            )
            target["accessory_reliability"] = (
                target["accessory_reliability"]
                * retained_accessory
            )
        _transform_canonical_grid(
            target,
            source_size=source_size,
            output_size=(self.width, self.height),
            scale=(
                new_w / source_width,
                new_h / source_height,
            ),
            offset=(-x1, -y1),
        )
        target["visibility"] = target["visibility"] * retained
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(h={self.height}, w={self.width}, p={self.p}, scale={self.scale})"


class ColorAugmentation:
    """PCA-based color augmentation (Krizhevsky et al., NIPS 2012).

    Adds a random linear combination of the ImageNet RGB principal components
    to each pixel, encouraging colour-invariant representations.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.eig_vec = torch.tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        shift = (self.eig_val * alpha) @ self.eig_vec
        return tensor + shift.view(3, 1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomPatch:
    """Occlusion simulation via a patch pool (torchreid / OSNet).

    Extracts random patches from training images and pastes them onto other
    images to simulate partial occlusion.

    Reference:
        Zhou et al. "Learning Generalisable Omni-Scale Representations
        for Person Re-Identification." TPAMI 2021.
    """

    def __init__(
        self,
        prob_happen: float = 0.5,
        pool_capacity: int = 5000,
        min_sample_size: int = 100,
        patch_min_area: float = 0.01,
        patch_max_area: float = 0.5,
        patch_min_ratio: float = 0.1,
        prob_rotate: float = 0.5,
        prob_flip_leftright: float = 0.5,
    ):
        self.prob_happen = prob_happen
        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio
        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright
        self.patchpool: deque = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def set_epoch(self, epoch: int) -> None:
        """Reset mutable augmentation state at an explicit epoch boundary.

        A patch pool that spans epochs cannot be reconstructed by an epoch
        checkpoint without serializing thousands of image crops. Keeping the
        pool epoch-local gives uninterrupted and resumed training identical
        augmentation state while retaining cross-image patch sampling within
        each epoch.
        """
        del epoch
        self.patchpool.clear()

    def _generate_wh(self, W: int, H: int):
        area = W * H
        for _ in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def _transform_patch(self, patch: Image.Image) -> Image.Image:
        if random.random() > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        # Collect a new patch from this image
        w, h = self._generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            self.patchpool.append(img.crop((x1, y1, x1 + w, y1 + h)))

        if len(self.patchpool) < self.min_sample_size:
            return img
        if random.random() > self.prob_happen:
            return img

        patch = random.choice(self.patchpool)
        pW, pH = patch.size
        if pW > W or pH > H:
            return img
        x1 = random.randint(0, W - pW)
        y1 = random.randint(0, H - pH)
        patch = self._transform_patch(patch)
        img = img.copy()
        img.paste(patch, (x1, y1))
        return img

    def apply_with_anatomical_target(
        self,
        img: Image.Image,
        target: dict[str, torch.Tensor],
    ) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        """Paste one patch and mark the covered anatomy as not visible."""
        width, height = img.size
        patch_width, patch_height = self._generate_wh(width, height)
        if patch_width is not None and patch_height is not None:
            source_x = random.randint(0, width - patch_width)
            source_y = random.randint(0, height - patch_height)
            self.patchpool.append(
                img.crop(
                    (
                        source_x,
                        source_y,
                        source_x + patch_width,
                        source_y + patch_height,
                    )
                )
            )
        if (
            len(self.patchpool) < self.min_sample_size
            or random.random() > self.prob_happen
        ):
            return img, target

        patch = random.choice(self.patchpool)
        patch_width, patch_height = patch.size
        if patch_width > width or patch_height > height:
            return img, target
        x0 = random.randint(0, width - patch_width)
        y0 = random.randint(0, height - patch_height)
        patch = self._transform_patch(patch)
        img = img.copy()
        img.paste(patch, (x0, y0))

        masks = target["masks"]
        before = masks.flatten(1).sum(dim=1)
        _clear_anatomical_spatial_region(
            target,
            left=x0,
            top=y0,
            width=patch_width,
            height=patch_height,
        )
        after = target["masks"].flatten(1).sum(dim=1)
        retained = torch.where(
            before > 1e-6,
            after / before.clamp_min(1e-6),
            torch.zeros_like(before),
        )
        _invalidate_canonical_grid_region(
            target,
            image_size=(width, height),
            left=x0,
            top=y0,
            width=patch_width,
            height=patch_height,
        )
        target["visibility"] = target["visibility"] * retained
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.prob_happen}, pool={self.patchpool.maxlen})"


class IdentityPreservingBackgroundMosaic:
    """Replace only an anchor person's background with a four-source mosaic.

    The primary foreground mask belongs only to the labeled anchor sample, so
    its PID remains the sole valid label. A separate union-of-all-people mask
    removes every confidently segmented person from each donor tile before
    compositing.
    """

    def __init__(
        self,
        samples: Sequence[ReIDSample],
        *,
        image_root: str | Path,
        primary_mask_root: str | Path,
        donor_mask_root: str | Path,
        probability: float = 0.3,
        start_epoch: int = 10,
        ramp_end_epoch: int = 30,
        min_foreground_ratio: float = 0.2,
        max_foreground_ratio: float = 0.9,
        feather: float = 1.5,
        dilation: int = 2,
        max_donor_tile_foreground: float = 0.25,
        donor_attempts: int = 24,
        occluder_probability: float = 0.0,
        occluder_min_area: float = 0.05,
        occluder_max_area: float = 0.20,
    ) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("background mosaic probability must be in [0, 1]")
        if start_epoch < 0 or ramp_end_epoch < start_epoch:
            raise ValueError("background mosaic epochs must satisfy 0 <= start <= ramp end")
        if not 0 <= min_foreground_ratio < max_foreground_ratio <= 1:
            raise ValueError("background mosaic foreground ratios must satisfy 0 <= min < max <= 1")
        if feather < 0 or dilation < 0:
            raise ValueError("background mosaic feather and dilation must be non-negative")
        if not 0 <= max_donor_tile_foreground <= 1 or donor_attempts < 1:
            raise ValueError("invalid background mosaic donor settings")
        if not 0 <= occluder_probability <= 1:
            raise ValueError("context occluder probability must be in [0, 1]")
        if not 0 < occluder_min_area <= occluder_max_area <= 1:
            raise ValueError("context occluder area must satisfy 0 < min <= max <= 1")

        self.samples = tuple(samples)
        self.image_root = Path(image_root).expanduser().resolve()
        self.primary_mask_root = Path(primary_mask_root).expanduser().resolve()
        self.donor_mask_root = Path(donor_mask_root).expanduser().resolve()
        self.probability = float(probability)
        self.start_epoch = int(start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.max_foreground_ratio = float(max_foreground_ratio)
        self.feather = float(feather)
        self.dilation = int(dilation)
        self.max_donor_tile_foreground = float(max_donor_tile_foreground)
        self.donor_attempts = int(donor_attempts)
        self.occluder_probability = float(occluder_probability)
        self.occluder_min_area = float(occluder_min_area)
        self.occluder_max_area = float(occluder_max_area)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the one-based training epoch used by the probability ramp."""
        self.epoch = int(epoch)

    def effective_probability(self) -> float:
        """Return the current probability after the configured epoch ramp."""
        if self.epoch <= self.start_epoch:
            return 0.0
        if self.ramp_end_epoch == self.start_epoch or self.epoch >= self.ramp_end_epoch:
            return self.probability
        progress = (self.epoch - self.start_epoch) / (
            self.ramp_end_epoch - self.start_epoch
        )
        return self.probability * progress

    def effective_occluder_probability(self) -> float:
        """Return the independently scheduled realistic-occluder probability."""
        if self.probability > 0:
            scale = self.effective_probability() / self.probability
        else:
            scale = 1.0 if self.epoch > self.start_epoch else 0.0
        return self.occluder_probability * scale

    def _mask_path(
        self,
        sample: ReIDSample,
        mask_root: Path,
    ) -> Path | None:
        image_path = Path(sample.img_path).expanduser().resolve()
        try:
            relative_path = image_path.relative_to(self.image_root)
        except ValueError:
            return None
        return (mask_root / relative_path).with_suffix(".png")

    def _load_mask(
        self,
        sample: ReIDSample,
        size: tuple[int, int],
        *,
        mask_root: Path,
    ) -> np.ndarray | None:
        mask_path = self._mask_path(sample, mask_root)
        if mask_path is None or not mask_path.is_file():
            return None
        with Image.open(mask_path) as mask_image:
            mask_image = mask_image.convert("L")
            if mask_image.size != size:
                mask_image = mask_image.resize(size, Image.Resampling.NEAREST)
            return np.asarray(mask_image, dtype=np.uint8) >= 128

    def _valid_foreground_mask(self, mask: np.ndarray) -> bool:
        foreground_ratio = float(mask.mean())
        if not self.min_foreground_ratio <= foreground_ratio <= self.max_foreground_ratio:
            return False
        height, width = mask.shape
        center = mask[
            height // 4 : max(3 * height // 4, height // 4 + 1),
            width // 4 : max(3 * width // 4, width // 4 + 1),
        ]
        return bool(center.any())

    @staticmethod
    def _quadrants(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
        x_mid = max(width // 2, 1)
        y_mid = max(height // 2, 1)
        return (
            (0, 0, x_mid, y_mid),
            (x_mid, 0, width, y_mid),
            (0, y_mid, x_mid, height),
            (x_mid, y_mid, width, height),
        )

    def _background_tile(
        self,
        sample: ReIDSample,
        target_size: tuple[int, int],
    ) -> Image.Image | None:
        with Image.open(sample.img_path) as donor_handle:
            donor = donor_handle.convert("RGB")
        donor_mask = self._load_mask(
            sample,
            donor.size,
            mask_root=self.donor_mask_root,
        )
        if donor_mask is None or not self._valid_foreground_mask(donor_mask):
            return None

        candidates = []
        for box in self._quadrants(*donor.size):
            x1, y1, x2, y2 = box
            tile_mask = donor_mask[y1:y2, x1:x2]
            candidates.append((float(tile_mask.mean()), random.random(), box, tile_mask))
        foreground_ratio, _, box, tile_mask = min(candidates)
        if foreground_ratio > self.max_donor_tile_foreground:
            return None

        tile = np.asarray(donor.crop(box), dtype=np.uint8).copy()
        background_pixels = tile[~tile_mask]
        if not background_pixels.size:
            return None
        fill_color = np.median(background_pixels, axis=0).astype(np.uint8)
        tile[tile_mask] = fill_color
        return Image.fromarray(tile, mode="RGB").resize(
            target_size,
            Image.Resampling.BILINEAR,
        )

    def _sample_background_tile(
        self,
        *,
        anchor_index: int,
        target_size: tuple[int, int],
    ) -> Image.Image | None:
        if len(self.samples) < 2:
            return None
        anchor_pid = self.samples[anchor_index].pid
        for _ in range(self.donor_attempts):
            donor_index = random.randrange(len(self.samples))
            donor = self.samples[donor_index]
            if donor_index == anchor_index or donor.pid == anchor_pid:
                continue
            tile = self._background_tile(donor, target_size)
            if tile is not None:
                return tile
        return None

    def _build_background(
        self,
        *,
        anchor_index: int,
        size: tuple[int, int],
    ) -> Image.Image | None:
        width, height = size
        background = Image.new("RGB", size, IMAGENET_MEAN_RGB)
        for box in self._quadrants(width, height):
            x1, y1, x2, y2 = box
            tile = self._sample_background_tile(
                anchor_index=anchor_index,
                target_size=(x2 - x1, y2 - y1),
            )
            if tile is None:
                return None
            background.paste(tile, (x1, y1))
        return background

    def __call__(self, image: Image.Image, index: int) -> Image.Image:
        return self.apply_with_status(image, index)[0]

    def _foreground_occluder(
        self,
        *,
        anchor_index: int,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        anchor_pid = self.samples[anchor_index].pid
        for _ in range(self.donor_attempts):
            donor_index = random.randrange(len(self.samples))
            donor = self.samples[donor_index]
            if donor_index == anchor_index or donor.pid == anchor_pid:
                continue
            with Image.open(donor.img_path) as handle:
                donor_image = handle.convert("RGB")
            donor_mask = self._load_mask(
                donor,
                donor_image.size,
                mask_root=self.primary_mask_root,
            )
            if donor_mask is None or not donor_mask.any():
                continue
            ys, xs = np.nonzero(donor_mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            crop = np.asarray(donor_image, dtype=np.uint8)[y0:y1, x0:x1]
            crop_mask = donor_mask[y0:y1, x0:x1]
            target_area = random.uniform(
                self.occluder_min_area,
                self.occluder_max_area,
            ) * image_size[0] * image_size[1]
            scale = math.sqrt(target_area / max(float(crop_mask.sum()), 1.0))
            new_width = max(int(round(crop.shape[1] * scale)), 1)
            new_height = max(int(round(crop.shape[0] * scale)), 1)
            fit_scale = min(
                image_size[0] / new_width,
                image_size[1] / new_height,
                1.0,
            )
            new_width = max(int(round(new_width * fit_scale)), 1)
            new_height = max(int(round(new_height * fit_scale)), 1)
            resized = cv2.resize(
                crop,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            resized_mask = cv2.resize(
                crop_mask.astype(np.uint8),
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            return resized, resized_mask
        return None

    def _apply_foreground_occluder(
        self,
        image: Image.Image,
        foreground_mask: np.ndarray,
        anchor_index: int,
    ) -> Image.Image | None:
        occluder = self._foreground_occluder(
            anchor_index=anchor_index,
            image_size=image.size,
        )
        if occluder is None:
            return None
        patch, patch_mask = occluder
        patch_height, patch_width = patch_mask.shape
        height, width = foreground_mask.shape
        ys, xs = np.nonzero(foreground_mask)
        if not len(xs):
            return None
        side = random.choice(("left", "right", "bottom"))
        if side == "left":
            x0 = 0
            y0 = min(max(int(np.median(ys)) - patch_height // 2, 0), height - patch_height)
        elif side == "right":
            x0 = width - patch_width
            y0 = min(max(int(np.median(ys)) - patch_height // 2, 0), height - patch_height)
        else:
            x0 = min(max(int(np.median(xs)) - patch_width // 2, 0), width - patch_width)
            y0 = height - patch_height
        target_mask = foreground_mask[
            y0 : y0 + patch_height,
            x0 : x0 + patch_width,
        ]
        if not np.any(target_mask & patch_mask):
            return None
        output = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        region = output[y0 : y0 + patch_height, x0 : x0 + patch_width]
        region[patch_mask] = patch[patch_mask]
        return Image.fromarray(output, mode="RGB")

    def apply_with_status(
        self,
        image: Image.Image,
        index: int,
    ) -> tuple[Image.Image, bool]:
        """Apply context replacement and/or a boundary-entering occluder."""
        sample = self.samples[index]
        foreground_mask = self._load_mask(
            sample,
            image.size,
            mask_root=self.primary_mask_root,
        )
        if foreground_mask is None or not self._valid_foreground_mask(foreground_mask):
            return image, False

        changed = False
        if random.random() < self.effective_probability():
            background = self._build_background(anchor_index=index, size=image.size)
            if background is not None:
                alpha = Image.fromarray(foreground_mask.astype(np.uint8) * 255, mode="L")
                if self.dilation:
                    alpha = alpha.filter(ImageFilter.MaxFilter(2 * self.dilation + 1))
                if self.feather:
                    alpha = alpha.filter(ImageFilter.GaussianBlur(self.feather))
                image = Image.composite(image, background, alpha)
                changed = True
        if random.random() < self.effective_occluder_probability():
            occluded = self._apply_foreground_occluder(
                image,
                foreground_mask,
                index,
            )
            if occluded is not None:
                image = occluded
                changed = True
        return image, changed

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.probability}, "
            f"epochs={self.start_epoch}->{self.ramp_end_epoch}, "
            f"primary_masks={self.primary_mask_root}, "
            f"donor_masks={self.donor_mask_root}, "
            f"occluder_p={self.occluder_probability})"
        )


@torch.no_grad()
def cross_camera_same_id_part_mosaic(
    images: torch.Tensor,
    pids: torch.Tensor,
    camera_ids: torch.Tensor,
    *,
    probability: float = 0.35,
    max_regions: int = 2,
    min_replaced_area: float = 0.15,
    max_replaced_area: float = 0.40,
    boundary_jitter: float = 0.05,
    cross_camera_rate: float = 1.0,
    min_unaltered_fraction: float = 0.5,
) -> torch.Tensor:
    """Replace body-aligned regions with independently augmented same-ID crops.

    The input is a normalized PK batch after its independent spatial and
    photometric transforms. Copying normalized pixels is equivalent to copying
    before channel normalization, while keeping donor selection inside the PK
    batch makes the hard identity label and metric-learning relationships valid.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if pids.ndim != 1 or camera_ids.ndim != 1 or len(pids) != len(images):
        raise ValueError("pids and camera_ids must be one-dimensional batch labels")
    if len(camera_ids) != len(images):
        raise ValueError("camera_ids must match the image batch")
    if not 0 <= probability <= 1 or not 0 <= cross_camera_rate <= 1:
        raise ValueError("mosaic probability and cross-camera rate must be in [0, 1]")
    if max_regions not in {1, 2}:
        raise ValueError("max_regions must be 1 or 2")
    if not 0 < min_replaced_area <= max_replaced_area <= 1:
        raise ValueError("replaced area must satisfy 0 < min <= max <= 1")
    if not 0 <= boundary_jitter <= 0.1:
        raise ValueError("boundary_jitter must be in [0, 0.1]")
    if not 0 <= min_unaltered_fraction <= 1:
        raise ValueError("min_unaltered_fraction must be in [0, 1]")

    batch_size, _, height, width = images.shape
    max_augmented = int(math.floor(batch_size * (1.0 - min_unaltered_fraction)))
    if batch_size < 2 or probability == 0 or max_augmented < 1:
        return images

    device = images.device
    candidate_indices = torch.nonzero(
        torch.rand(batch_size, device=device) < probability,
        as_tuple=False,
    ).flatten()
    if candidate_indices.numel() > max_augmented:
        order = torch.randperm(candidate_indices.numel(), device=device)
        candidate_indices = candidate_indices[order[:max_augmented]]
    if candidate_indices.numel() == 0:
        return images

    # Overlapping anatomical bands from the proposal. When two regions are
    # selected, use non-adjacent bands so the requested area is unambiguous.
    base_regions = (
        (0.00, 0.25),
        (0.20, 0.55),
        (0.50, 0.78),
        (0.72, 1.00),
    )
    non_overlapping_pairs = ((0, 2), (0, 3), (1, 3))
    batch_indices = torch.arange(batch_size, device=device)
    output = images.clone()

    for anchor_tensor in candidate_indices:
        anchor_index = int(anchor_tensor.item())
        same_identity = batch_indices[
            (pids == pids[anchor_index]) & (batch_indices != anchor_index)
        ]
        if same_identity.numel() == 0:
            continue
        cross_camera = same_identity[
            camera_ids[same_identity] != camera_ids[anchor_index]
        ]

        region_count = 1
        if max_regions == 2:
            region_count = int(torch.randint(1, 3, (), device=device).item())
        if region_count == 1:
            selected_regions = (
                int(torch.randint(0, len(base_regions), (), device=device).item()),
            )
        else:
            pair_index = int(
                torch.randint(0, len(non_overlapping_pairs), (), device=device).item()
            )
            selected_regions = non_overlapping_pairs[pair_index]

        pixel_regions: list[tuple[int, int]] = []
        for region_index in selected_regions:
            base_start, base_end = base_regions[region_index]
            jitter = (
                torch.empty(2, device=device)
                .uniform_(-boundary_jitter, boundary_jitter)
                .tolist()
            )
            start = max(0.0, min(base_start + jitter[0], 1.0))
            end = max(0.0, min(base_end + jitter[1], 1.0))
            if end <= start:
                continue
            y0 = min(int(round(start * height)), height - 1)
            y1 = min(max(int(round(end * height)), y0 + 1), height)
            pixel_regions.append((y0, y1))
        if not pixel_regions:
            continue

        total_height_fraction = sum(y1 - y0 for y0, y1 in pixel_regions) / height
        if total_height_fraction < min_replaced_area:
            continue
        area_low = max(min_replaced_area, 0.5 * total_height_fraction)
        area_high = min(max_replaced_area, total_height_fraction)
        target_area = float(
            torch.empty((), device=device).uniform_(area_low, area_high).item()
        )
        width_fraction = min(target_area / max(total_height_fraction, 1e-12), 1.0)
        x_size = min(max(int(round(width_fraction * width)), 1), width)

        for y0, y1 in pixel_regions:
            use_cross_camera = (
                cross_camera.numel() > 0
                and torch.rand((), device=device).item() < cross_camera_rate
            )
            donor_pool = cross_camera if use_cross_camera else same_identity
            donor_offset = int(
                torch.randint(0, donor_pool.numel(), (), device=device).item()
            )
            donor_index = int(donor_pool[donor_offset].item())
            x0 = int(torch.randint(0, width - x_size + 1, (), device=device).item())
            x1 = x0 + x_size
            output[anchor_index, :, y0:y1, x0:x1] = images[
                donor_index,
                :,
                y0:y1,
                x0:x1,
            ]
    return output


@torch.no_grad()
def apply_independent_random_erasing(
    images: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Apply the standard ReID Random Erasing policy independently per image."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if not 0 <= probability <= 1:
        raise ValueError("random erasing probability must be in [0, 1]")
    if probability == 0:
        return images

    output = images.clone()
    for index, image in enumerate(images):
        if torch.rand(()).item() >= probability:
            continue
        y0, x0, erase_h, erase_w, value = T.RandomErasing.get_params(
            image,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.33),
            value=[0.0, 0.0, 0.0],
        )
        output[index, :, y0 : y0 + erase_h, x0 : x0 + erase_w] = value.to(
            device=images.device,
            dtype=images.dtype,
        )
    return output


class EpochAwareCompose(T.Compose):
    """Compose transforms and forward deterministic epoch boundaries."""

    def set_epoch(self, epoch: int) -> None:
        """Reset stateful child transforms before an epoch is iterated."""
        for transform in self.transforms:
            setter = getattr(transform, "set_epoch", None)
            if callable(setter):
                setter(epoch)

    @staticmethod
    def _resize_pad_target(
        masks: torch.Tensor,
        source_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        source_w, source_h = source_size
        output_w, output_h = output_size
        scale = min(output_w / source_w, output_h / source_h)
        resized_w = int(source_w * scale)
        resized_h = int(source_h * scale)
        resized = _resize_spatial_masks(
            masks,
            (resized_h, resized_w),
        )
        output = masks.new_zeros(
            (*masks.shape[:-2], output_h, output_w)
        )
        left = (output_w - resized_w) // 2
        top = (output_h - resized_h) // 2
        output[
            ...,
            top : top + resized_h,
            left : left + resized_w,
        ] = resized
        return output

    @staticmethod
    def _apply_random_erasing(
        transform: T.RandomErasing,
        image: torch.Tensor,
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if torch.rand(1).item() >= transform.p:
            return image, target
        if transform.value == "random":
            value = None
        elif isinstance(transform.value, (int, float)):
            value = [float(transform.value)]
        else:
            value = [float(item) for item in transform.value]
        y0, x0, erase_h, erase_w, erase_value = transform.get_params(
            image,
            scale=transform.scale,
            ratio=transform.ratio,
            value=value,
        )
        image = TF.erase(
            image,
            y0,
            x0,
            erase_h,
            erase_w,
            erase_value,
            transform.inplace,
        )
        masks = target["masks"]
        before = masks.flatten(1).sum(dim=1)
        _clear_anatomical_spatial_region(
            target,
            left=x0,
            top=y0,
            width=erase_w,
            height=erase_h,
        )
        after = target["masks"].flatten(1).sum(dim=1)
        retained = torch.where(
            before > 1e-6,
            after / before.clamp_min(1e-6),
            torch.zeros_like(before),
        )
        _invalidate_canonical_grid_region(
            target,
            image_size=(image.shape[-1], image.shape[-2]),
            left=x0,
            top=y0,
            width=erase_w,
            height=erase_h,
        )
        target["visibility"] = target["visibility"] * retained
        return image, target

    def apply_with_anatomical_target(
        self,
        image: Image.Image,
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply spatial transforms consistently to RGB and part targets."""
        target = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for transform in self.transforms:
            if isinstance(transform, ResizePad):
                source_size = image.size
                image = transform(image)
                for key in (
                    "masks",
                    "foreground_mask",
                    "accessory_mask",
                ):
                    value = target.get(key)
                    if torch.is_tensor(value):
                        target[key] = self._resize_pad_target(
                            value,
                            source_size,
                            image.size,
                        )
                source_width, source_height = source_size
                output_width, output_height = image.size
                scale = min(
                    output_width / source_width,
                    output_height / source_height,
                )
                resized_width = int(source_width * scale)
                resized_height = int(source_height * scale)
                left = (output_width - resized_width) // 2
                top = (output_height - resized_height) // 2
                _transform_canonical_grid(
                    target,
                    source_size=source_size,
                    output_size=image.size,
                    scale=(
                        resized_width / source_width,
                        resized_height / source_height,
                    ),
                    offset=(left, top),
                )
            elif isinstance(transform, T.Resize):
                source_size = image.size
                image = transform(image)
                _resize_anatomical_spatial_targets(
                    target,
                    (image.height, image.width),
                )
                _transform_canonical_grid(
                    target,
                    source_size=source_size,
                    output_size=image.size,
                    scale=(
                        image.width / source_size[0],
                        image.height / source_size[1],
                    ),
                )
            elif isinstance(transform, T.RandomHorizontalFlip):
                if torch.rand(1).item() < transform.p:
                    image = TF.hflip(image)
                    target["masks"] = torch.flip(
                        target["masks"],
                        dims=(-1,),
                    )[list(ANATOMICAL_FLIP_PERMUTATION)]
                    if "foreground_mask" in target:
                        target["foreground_mask"] = torch.flip(
                            target["foreground_mask"],
                            dims=(-1,),
                        )
                    if "accessory_mask" in target:
                        target["accessory_mask"] = torch.flip(
                            target["accessory_mask"],
                            dims=(-1,),
                        )
                    target["visibility"] = target["visibility"][
                        list(ANATOMICAL_FLIP_PERMUTATION)
                    ]
                    target["reliability"] = target["reliability"][
                        list(ANATOMICAL_FLIP_PERMUTATION)
                    ]
                    canonical_grid = target["canonical_grid"][
                        list(ANATOMICAL_FLIP_PERMUTATION)
                    ].clone()
                    canonical_grid[..., 0] *= -1
                    target["canonical_grid"] = torch.flip(
                        canonical_grid,
                        dims=(-2,),
                    )
                    target["canonical_grid_valid"] = torch.flip(
                        target["canonical_grid_valid"][
                            list(ANATOMICAL_FLIP_PERMUTATION)
                        ],
                        dims=(-1,),
                    )
                    if "canonical_grid_pose_valid" in target:
                        target["canonical_grid_pose_valid"] = torch.flip(
                            target["canonical_grid_pose_valid"][
                                list(ANATOMICAL_FLIP_PERMUTATION)
                            ],
                            dims=(-1,),
                        )
                    pose_keypoints = target.get("pose_keypoints")
                    if torch.is_tensor(pose_keypoints):
                        pose_keypoints = pose_keypoints[
                            list(COCO_KEYPOINT_FLIP_PERMUTATION)
                        ].clone()
                        pose_keypoints[:, 0] *= -1
                        target["pose_keypoints"] = pose_keypoints
            elif isinstance(transform, Random2DTranslation):
                image, target = transform.apply_with_anatomical_target(
                    image,
                    target,
                )
            elif isinstance(transform, RandomPatch):
                image, target = transform.apply_with_anatomical_target(
                    image,
                    target,
                )
            elif isinstance(transform, T.RandomErasing):
                image, target = self._apply_random_erasing(
                    transform,
                    image,
                    target,
                )
            else:
                image = transform(image)

        masks_present = target["masks"].flatten(1).amax(dim=1) > 1e-6
        target["mask_present"] = masks_present
        target["visibility"] = target["visibility"] * masks_present.to(
            target["visibility"].dtype
        )
        target["reliability"] = target["reliability"] * masks_present.to(
            target["reliability"].dtype
        )
        target["canonical_grid_valid"] = (
            target["canonical_grid_valid"]
            & masks_present[:, None, None]
        )
        accessory_mask = target.get("accessory_mask")
        if torch.is_tensor(accessory_mask):
            accessory_present = accessory_mask.amax() > 1e-6
            if "accessory_valid" in target:
                target["accessory_valid"] = (
                    target["accessory_valid"]
                    & accessory_present
                )
            for key in (
                "accessory_visibility",
                "accessory_reliability",
            ):
                if key in target:
                    target[key] = (
                        target[key]
                        * accessory_present.to(target[key].dtype)
                    )
        pose_keypoints = target.get("pose_keypoints")
        if torch.is_tensor(pose_keypoints):
            target["pose_reliability"] = torch.stack(
                tuple(
                    pose_keypoints[list(indices), 2].mean()
                    for indices in ANATOMICAL_PART_KEYPOINTS
                )
            )
            if "pose_valid" in target:
                target["pose_valid"] = (
                    target["pose_valid"]
                    & (pose_keypoints[:, 2] > 0).any()
                )
        return image, target


def build_train_transforms(
    img_size: Tuple[int, int] = (256, 128),
    *,
    preprocess: str = DEFAULT_PREPROCESS,
    random_erasing: float = 0.5,
    color_jitter: bool = True,
    gaussian_blur: bool = False,
    random_grayscale: float = 0.0,
    random_patch: bool = True,
    random_crop_scale: float = 1.05,
    color_augmentation: bool = True,
) -> EpochAwareCompose:
    """Build the standard ReID training augmentation pipeline.

    Pipeline (torchreid-inspired, with optional aspect-preserving resize):
        [resize_pad] → flip → Random2DTranslation → RandomPatch →
        [ColorJitter] → [GaussianBlur] → [RandomGrayscale] →
        ToTensor → ColorAugmentation → Normalize → [RandomErasing]

    ``Random2DTranslation`` owns the resize for the canonical ``resize``
    preprocess. ``ResizePad`` remains before it when aspect-preserving padded
    geometry is requested.
    """
    h, w = img_size
    ops = []
    if preprocess == "resize_pad":
        ops.append(_resize_op(img_size, preprocess))
    ops.extend([
        T.RandomHorizontalFlip(p=0.5),
        Random2DTranslation(h, w, p=0.5, scale=random_crop_scale),
    ])
    if random_patch:
        ops.append(RandomPatch(prob_happen=0.5))
    if color_jitter:
        # torchreid values: brightness=0.2, contrast=0.15
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0))
    if gaussian_blur:
        ops.append(T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)))
    if random_grayscale > 0:
        ops.append(T.RandomGrayscale(p=random_grayscale))
    ops.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if color_augmentation:
        ops.insert(-1, ColorAugmentation(p=0.5))
    if random_erasing > 0:
        # Zhong et al. "Random Erasing Data Augmentation" §5.3.1:
        # p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.33). Normalization
        # precedes erasing, so zero is the normalized ImageNet-mean fill.
        ops.append(T.RandomErasing(
            p=random_erasing, scale=(0.02, 0.2), ratio=(0.3, 3.33),
            value=0.0,
        ))
    return EpochAwareCompose(ops)


def build_clean_train_transforms(
    img_size: Tuple[int, int] = (256, 128),
    *,
    preprocess: str = DEFAULT_PREPROCESS,
) -> EpochAwareCompose:
    """Build a deterministic train-time teacher view with target alignment.

    ``EpochAwareCompose`` is used instead of the inference compose so dense
    pose/parsing targets receive exactly the same resize or resize-pad
    geometry as the RGB image.
    """
    return EpochAwareCompose(
        [
            _resize_op(img_size, preprocess),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_test_transforms(
    img_size: Tuple[int, int] = (256, 128),
    *,
    preprocess: str = DEFAULT_PREPROCESS,
) -> T.Compose:
    """Build the standard ReID test/val transform pipeline.

    Args:
        img_size: Target (H, W).
        preprocess: Preprocessing method name, must match training.
    """
    return T.Compose([
        _resize_op(img_size, preprocess),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
