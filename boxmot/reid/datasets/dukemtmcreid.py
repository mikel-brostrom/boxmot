"""DukeMTMC-reID dataset.

Same directory layout as Market-1501::

    DukeMTMC-reID/
        bounding_box_train/
            0001_c2_f0046182.jpg
            ...
        bounding_box_test/
            0002_c1_f0044158.jpg
            ...
        query/
            0005_c2_f0046985.jpg
            ...

Image naming: ``PPPP_cC_fFFFFFF.jpg`` where P=pid, C=cam.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List

from boxmot.reid.datasets.base import BaseReIDDataset, ReIDSample

_PATTERN = re.compile(r"([-\d]+)_c(\d)")


class DukeMTMCreID(BaseReIDDataset):
    name = "dukemtmcreid"

    _SUBDIRS = ("DukeMTMC-reID", "dukemtmc-reid", "dukemtmcreid", "duke")

    def __init__(self, root: str, **kwargs):
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str):
        from pathlib import Path
        p = Path(root)
        # 1. Named subdirectories under root (most specific match)
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "bounding_box_train").is_dir():
                return candidate
        # 2. Named subdirectories under parent (cross-dataset support)
        for sub in self._SUBDIRS:
            candidate = p.parent / sub
            if (candidate / "bounding_box_train").is_dir():
                return candidate
        # 3. Bare root only if its folder name matches a known alias
        #    (prevents silently loading Market/CUHK03 data as Duke)
        if (p / "bounding_box_train").is_dir():
            norm = p.name.lower().replace("-", "").replace("_", "")
            known = {s.lower().replace("-", "").replace("_", "") for s in self._SUBDIRS}
            if norm in known:
                return p
        raise FileNotFoundError(
            f"Cannot find DukeMTMC-reID dataset under {root}. "
            f"Expected one of {self._SUBDIRS} as a subdirectory or the dataset "
            f"root itself containing bounding_box_train/."
        )

    def _load_split(self, split: str) -> List[ReIDSample]:
        dir_map = {
            "train": "bounding_box_train",
            "query": "query",
            "gallery": "bounding_box_test",
        }
        split_dir = self.root / dir_map[split]
        return _parse_duke_dir(split_dir)


def _parse_duke_dir(directory) -> List[ReIDSample]:
    img_paths = sorted(glob.glob(str(directory / "*.jpg")))
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
