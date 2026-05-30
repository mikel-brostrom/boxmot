"""VeRi-776 vehicle re-identification dataset.

Directory layout::

    VeRi/
        image_train/
            0001_c001_00016450_0.jpg
            ...
        image_test/
            0002_c002_00030600_0.jpg
            ...
        image_query/
            0002_c002_00030600_0.jpg
            ...
        name_train.txt
        name_test.txt
        name_query.txt

Image naming: ``VVVV_cCCC_FFFFFFFF_B.jpg`` where V=vehicle ID, C=cam, F=frame.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import List

from boxmot.reid.datasets.base import BaseReIDDataset, ReIDSample

_PATTERN = re.compile(r"(\d+)_c(\d+)")


class VeRi776(BaseReIDDataset):
    name = "veri776"

    _SUBDIRS = ("VeRi", "veri", "VeRi-776", "veri776")

    def __init__(self, root: str, **kwargs):
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str):
        p = Path(root)
        if (p / "image_train").is_dir():
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "image_train").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Cannot find VeRi-776 dataset under {root}. "
            f"Expected image_train/ directory."
        )

    def _load_split(self, split: str) -> List[ReIDSample]:
        dir_map = {
            "train": "image_train",
            "query": "image_query",
            "gallery": "image_test",
        }
        split_dir = self.root / dir_map[split]
        return _parse_veri_dir(split_dir)


def _parse_veri_dir(directory: Path) -> List[ReIDSample]:
    img_paths = sorted(glob.glob(str(directory / "*.jpg")))
    samples = []
    for path in img_paths:
        fname = os.path.basename(path)
        match = _PATTERN.search(fname)
        if match is None:
            continue
        pid = int(match.group(1))
        camid = int(match.group(2)) - 1
        samples.append(ReIDSample(img_path=path, pid=pid, camid=camid))
    return samples
