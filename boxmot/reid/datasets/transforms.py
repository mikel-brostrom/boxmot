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
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS, IMAGENET_MEAN_RGB


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.prob_happen}, pool={self.patchpool.maxlen})"


IMAGENET_MEAN = [0.485, 0.456, 0.406]


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
) -> T.Compose:
    """Build the standard ReID training augmentation pipeline.

    Pipeline (torchreid-inspired, with aspect-preserving resize):
        resize_pad → flip → Random2DTranslation → RandomPatch →
        [ColorJitter] → [GaussianBlur] → [RandomGrayscale] →
        ToTensor → ColorAugmentation → Normalize → [RandomErasing]
    """
    h, w = img_size
    ops = [
        _resize_op(img_size, preprocess),
        T.RandomHorizontalFlip(p=0.5),
        Random2DTranslation(h, w, p=0.5, scale=random_crop_scale),
    ]
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
        # p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.33), fill=ImageNet mean
        ops.append(T.RandomErasing(
            p=random_erasing, scale=(0.02, 0.2), ratio=(0.3, 3.33),
            value=IMAGENET_MEAN,
        ))
    return T.Compose(ops)


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
