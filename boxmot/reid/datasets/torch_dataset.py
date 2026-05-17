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
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        sample = self.samples[index]
        img = Image.open(sample.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.pid, sample.camid
