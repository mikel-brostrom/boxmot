"""MSMT17 dataset.

Directory layout::

    MSMT17_V2/ (or MSMT17_V1/)
        train/
            0000/
                0000_000_01_0000000001.jpg
                ...
            ...
        test/
            0000/
                ...
        list_train.txt
        list_val.txt
        list_query.txt
        list_gallery.txt

``list_*.txt`` lines: ``<relative_path> <pid>``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from boxmot.reid.datasets.base import BaseReIDDataset, ReIDSample


class MSMT17(BaseReIDDataset):
    name = "msmt17"

    _SUBDIRS = ("MSMT17_V2", "MSMT17_V1", "MSMT17", "msmt17")

    def __init__(self, root: str, *, merged: bool = False, **kwargs):
        self._merged = merged
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str):
        p = Path(root)
        if (p / "list_train.txt").is_file():
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "list_train.txt").is_file():
                return candidate
        raise FileNotFoundError(
            f"Cannot find MSMT17 dataset under {root}. "
            f"Expected list_train.txt in the dataset root."
        )

    def _load_split(self, split: str) -> List[ReIDSample]:
        list_map = {
            "train": "list_train.txt",
            "query": "list_query.txt",
            "gallery": "list_gallery.txt",
        }
        # V1 layout: images live under train/ and test/ subdirectories,
        # but list files contain paths relative to those subdirectories.
        img_subdir_map = {
            "train": "train",
            "query": "test",
            "gallery": "test",
        }
        if self._merged and split == "train":
            # MSMT17-merged: combine train + query + gallery for training
            samples = []
            for list_name, img_split in [
                ("list_train.txt", "train"),
                ("list_query.txt", "test"),
                ("list_gallery.txt", "test"),
            ]:
                list_file = self.root / list_name
                if list_file.is_file():
                    img_root = self.root / img_split if (self.root / img_split).is_dir() else self.root
                    samples.extend(_parse_msmt17_list(list_file, img_root))
            return samples
        list_file = self.root / list_map[split]
        img_subdir = img_subdir_map[split]
        img_root = self.root / img_subdir if (self.root / img_subdir).is_dir() else self.root
        return _parse_msmt17_list(list_file, img_root)


def _parse_msmt17_list(list_file: Path, root: Path) -> List[ReIDSample]:
    samples = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]
            pid = int(parts[1])
            # Extract camera ID from filename: e.g. 0001_001_01_... -> cam 1
            fname = os.path.basename(rel_path)
            camid = int(fname.split("_")[2]) - 1
            img_path = str(root / rel_path)
            samples.append(ReIDSample(img_path=img_path, pid=pid, camid=camid))
    return samples
