"""PyTorch Dataset wrapper for ReID image samples."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from boxmot.reid.datasets.base import ReIDSample


class ReIDImageDataset(Dataset):
    """Wraps a list of ``ReIDSample`` for PyTorch DataLoader consumption."""

    def __init__(
        self,
        samples: List[ReIDSample],
        transform: Optional[Callable] = None,
        sample_transform: Optional[Callable[[Image.Image, int], Image.Image]] = None,
        *,
        return_clean_view: bool = False,
        clean_transform: Optional[Callable] = None,
        return_clean_anatomical_target: bool = False,
        anatomical_target_provider: Optional[Callable] = None,
        return_sample_index: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.sample_transform = sample_transform
        self.return_clean_view = bool(return_clean_view)
        self.clean_transform = clean_transform
        self.return_clean_anatomical_target = bool(
            return_clean_anatomical_target
        )
        self.anatomical_target_provider = anatomical_target_provider
        self.return_sample_index = bool(return_sample_index)
        self.anatomical_targets_enabled = anatomical_target_provider is not None
        if self.return_clean_anatomical_target and not self.return_clean_view:
            raise ValueError(
                "return_clean_anatomical_target requires return_clean_view"
            )
        if (
            self.return_clean_anatomical_target
            and self.anatomical_target_provider is None
        ):
            raise ValueError(
                "return_clean_anatomical_target requires anatomical targets"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        sample = self.samples[index]
        img = Image.open(sample.img_path).convert("RGB")
        anatomical_target = (
            self.anatomical_target_provider(index, img.size)
            if self.anatomical_targets_enabled
            else None
        )
        source_anatomical_target = anatomical_target
        clean_anatomical_target = None
        clean_img = img.copy() if self.return_clean_view else None
        augmented = False
        if self.sample_transform is not None:
            apply_with_status = getattr(self.sample_transform, "apply_with_status", None)
            if callable(apply_with_status):
                img, augmented = apply_with_status(img, index)
            else:
                transformed = self.sample_transform(img, index)
                augmented = transformed is not img
                img = transformed
        if self.transform is not None:
            if anatomical_target is None:
                img = self.transform(img)
            else:
                apply_with_target = getattr(
                    self.transform,
                    "apply_with_anatomical_target",
                    None,
                )
                if not callable(apply_with_target):
                    raise TypeError(
                        "Anatomical supervision requires a transform with "
                        "apply_with_anatomical_target()"
                    )
                img, anatomical_target = apply_with_target(
                    img,
                    anatomical_target,
                )
        if self.return_clean_view:
            if self.return_clean_anatomical_target:
                clean_transform = self.clean_transform or self.transform
                apply_with_target = getattr(
                    clean_transform,
                    "apply_with_anatomical_target",
                    None,
                )
                if not callable(apply_with_target):
                    raise TypeError(
                        "Clean anatomical supervision requires a transform "
                        "with apply_with_anatomical_target()"
                    )
                clean_img, clean_anatomical_target = apply_with_target(
                    clean_img,
                    source_anatomical_target,
                )
                # The standard image transform may augment the student even
                # when no sample-level mosaic was selected. Every item is a
                # valid paired clean/augmented view in this mode.
                augmented = True
            elif augmented:
                clean_transform = self.clean_transform or self.transform
                if clean_transform is not None:
                    clean_img = clean_transform(clean_img)
            else:
                clean_img = img.clone() if hasattr(img, "clone") else img.copy()
            output = (img, sample.pid, sample.camid, clean_img, augmented)
        else:
            output = (img, sample.pid, sample.camid)
        if anatomical_target is not None:
            compactor = getattr(
                self.anatomical_target_provider,
                "compact_target",
                None,
            )
            if callable(compactor):
                anatomical_target = compactor(anatomical_target)
                if clean_anatomical_target is not None:
                    clean_anatomical_target = compactor(
                        clean_anatomical_target
                    )
            if clean_anatomical_target is not None:
                anatomical_target = dict(anatomical_target)
                anatomical_target["_clean_view"] = clean_anatomical_target
            output = (*output, anatomical_target)
        if self.return_sample_index:
            output = (*output, index)
        return output

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch boundaries to stateful training transforms."""
        for transform in (self.sample_transform, self.transform, self.clean_transform):
            setter = getattr(transform, "set_epoch", None)
            if callable(setter):
                setter(epoch)

    def set_anatomical_targets_enabled(self, enabled: bool) -> None:
        """Enable training-only target generation for the next epoch."""
        self.anatomical_targets_enabled = bool(enabled) and (
            self.anatomical_target_provider is not None
        )
