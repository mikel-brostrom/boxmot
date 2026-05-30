"""CUHK03 dataset (new protocol, detected).

Directory layout (new protocol)::

    cuhk03-np/
        detected/
            bounding_box_train/
                0001_c1_1.jpg
                ...
            bounding_box_test/
            query/
        labeled/
            bounding_box_train/
            bounding_box_test/
            query/

Image naming follows Market-1501 convention: ``PPPP_cC_N.jpg``.

The "new protocol" (Zhong et al., 2017) reorganises the original .mat
file into the standard bounding_box_train / query / bounding_box_test
splits used by Market-1501 and DukeMTMC-reID.  We default to
``detected/`` (automatic detection crops) which is the harder and more
commonly reported variant.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import List

from boxmot.reid.datasets.base import BaseReIDDataset, ReIDSample

_PATTERN = re.compile(r"([-\d]+)_c(\d)")


class CUHK03(BaseReIDDataset):
    name = "cuhk03"

    _SUBDIRS = ("cuhk03-np", "CUHK03", "cuhk03")

    def __init__(self, root: str, *, variant: str = "detected", **kwargs):
        self._variant = variant
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str):
        p = Path(root)
        # Direct detected/labeled dir with bounding_box_train
        if (p / "bounding_box_train").is_dir():
            return p
        # Check for variant subdir (detected/ or labeled/)
        if (p / self._variant / "bounding_box_train").is_dir():
            return p / self._variant
        # Check standard dataset root names
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / self._variant / "bounding_box_train").is_dir():
                return candidate / self._variant
            if (candidate / "bounding_box_train").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Cannot find CUHK03 dataset under {root}. "
            f"Expected cuhk03-np/{self._variant}/bounding_box_train/ directory. "
            f"Download the new-protocol split from: "
            f"https://github.com/zhunzhong07/person-re-ranking"
        )

    def _load_split(self, split: str) -> List[ReIDSample]:
        dir_map = {
            "train": "bounding_box_train",
            "query": "query",
            "gallery": "bounding_box_test",
        }
        split_dir = self.root / dir_map[split]
        return _parse_cuhk03_dir(split_dir)


def _parse_cuhk03_dir(directory: Path) -> List[ReIDSample]:
    img_paths = sorted(
        glob.glob(str(directory / "*.jpg"))
        + glob.glob(str(directory / "*.png"))
    )
    samples = []
    for path in img_paths:
        fname = os.path.basename(path)
        match = _PATTERN.search(fname)
        if match is None:
            continue
        pid = int(match.group(1))
        camid = int(match.group(2)) - 1
        if pid == -1:
            continue
        samples.append(ReIDSample(img_path=path, pid=pid, camid=camid))
    return samples
