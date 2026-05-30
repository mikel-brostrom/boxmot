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
        if (p / "bounding_box_train").is_dir():
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "bounding_box_train").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Cannot find DukeMTMC-reID dataset under {root}. "
            f"Expected bounding_box_train/ directory."
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
