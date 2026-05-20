"""Market-1501 dataset.

Directory layout::

    Market-1501-v15.09.15/
        bounding_box_train/
            0002_c1s1_000451_03.jpg
            ...
        bounding_box_test/
            0000_c1s1_000151_01.jpg
            ...
        query/
            0001_c1s1_001051_00.jpg
            ...

Image naming: ``PPPP_cC_sS_NNNN_NN.jpg`` where P=pid, C=cam, S=seq.
PID ``-1`` and ``0000`` are junk/distractors in the test set.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List

from boxmot.reid.datasets.base import BaseReIDDataset, ReIDSample

_PATTERN = re.compile(r"([-\d]+)_c(\d)")


class Market1501(BaseReIDDataset):
    name = "market1501"

    # Common root folder names the dataset ships as
    _SUBDIRS = ("Market-1501-v15.09.15", "Market-1501", "market1501")

    def __init__(self, root: str, **kwargs):
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str):
        """Accept either the dataset subfolder or a parent containing it."""
        p = self._find_root(root)
        if p is None:
            raise FileNotFoundError(
                f"Cannot find Market-1501 dataset under {root}. "
                f"Expected one of {self._SUBDIRS} or the directory itself "
                f"containing bounding_box_train/."
            )
        return p

    def _find_root(self, root: str):
        from pathlib import Path
        p = Path(root)
        if (p / "bounding_box_train").is_dir():
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "bounding_box_train").is_dir():
                return candidate
        return None

    def _load_split(self, split: str) -> List[ReIDSample]:
        dir_map = {
            "train": "bounding_box_train",
            "query": "query",
            "gallery": "bounding_box_test",
        }
        split_dir = self.root / dir_map[split]
        return _parse_market_dir(split_dir, is_train=(split == "train"))


def _parse_market_dir(directory, *, is_train: bool) -> List[ReIDSample]:
    img_paths = sorted(glob.glob(str(directory / "*.jpg")))
    samples = []
    for path in img_paths:
        fname = os.path.basename(path)
        match = _PATTERN.search(fname)
        if match is None:
            continue
        pid = int(match.group(1))
        camid = int(match.group(2)) - 1  # 0-indexed
        if pid == -1:
            continue  # junk
        samples.append(ReIDSample(img_path=path, pid=pid, camid=camid))
    return samples
