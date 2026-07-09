"""Base class for ReID datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ReIDSample:
    """A single ReID image sample."""
    img_path: str
    pid: int
    camid: int
    source: str = ""


@dataclass
class DatasetSplit:
    """Container for a dataset split (train/query/gallery)."""
    samples: List[ReIDSample] = field(default_factory=list)

    @property
    def num_pids(self) -> int:
        return len({s.pid for s in self.samples})

    @property
    def num_cams(self) -> int:
        return len({s.camid for s in self.samples})

    @property
    def num_imgs(self) -> int:
        return len(self.samples)

    def pid_set(self) -> set[int]:
        return {s.pid for s in self.samples}


class BaseReIDDataset:
    """Base class for person/vehicle re-identification datasets.

    Subclasses implement ``_load_split`` to parse the dataset-specific
    directory layout and return a list of ``ReIDSample``.
    """

    name: str = "base"

    def __init__(self, root: str, *, relabel_train: bool = True):
        self.root = Path(root)
        self._relabel_train = relabel_train

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root}. "
                f"Download the dataset and point --data-dir to its root."
            )

        raw_train = self._with_source(self._load_split("train"), self.name)
        self.query = DatasetSplit(self._with_source(self._load_split("query"), self.name))
        self.gallery = DatasetSplit(self._with_source(self._load_split("gallery"), self.name))

        if relabel_train:
            raw_train = self._relabel(raw_train)
        self.train = DatasetSplit(raw_train)

    def _load_split(self, split: str) -> List[ReIDSample]:
        """Return samples for the given split. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _relabel(samples: List[ReIDSample]) -> List[ReIDSample]:
        """Remap PIDs to a contiguous range starting from 0."""
        pid_map = {pid: idx for idx, pid in enumerate(sorted({s.pid for s in samples}))}
        return [
            ReIDSample(img_path=s.img_path, pid=pid_map[s.pid], camid=s.camid, source=s.source)
            for s in samples
        ]

    @staticmethod
    def _with_source(samples: List[ReIDSample], source: str) -> List[ReIDSample]:
        """Attach a dataset source label used by source-balanced samplers."""
        return [
            ReIDSample(
                img_path=s.img_path,
                pid=s.pid,
                camid=s.camid,
                source=s.source or source,
            )
            for s in samples
        ]

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  root: {self.root}",
            f"  train: {self.train.num_imgs} images, {self.train.num_pids} IDs, {self.train.num_cams} cameras",
            f"  query: {self.query.num_imgs} images, {self.query.num_pids} IDs, {self.query.num_cams} cameras",
            f"  gallery: {self.gallery.num_imgs} images, {self.gallery.num_pids} IDs, {self.gallery.num_cams} cameras",
        ]
        return "\n".join(lines)

    @property
    def num_train_pids(self) -> int:
        return self.train.num_pids


class CombinedReIDDataset:
    """Combine multiple ReID datasets into one for joint training.

    Train splits are concatenated with PIDs remapped to a contiguous
    global range. Query/gallery splits come from the first dataset.
    """

    name: str = "combined"

    def __init__(self, datasets: List[BaseReIDDataset]):
        if not datasets:
            raise ValueError("Must provide at least one dataset")

        # Merge train splits with PID remapping
        all_train: List[ReIDSample] = []
        pid_offset = 0
        cam_offset = 0
        source_names: List[str] = []
        for ds in datasets:
            source_names.append(ds.name)
            for s in ds.train.samples:
                all_train.append(
                    ReIDSample(
                        img_path=s.img_path,
                        pid=s.pid + pid_offset,
                        camid=s.camid + cam_offset,
                        source=s.source or ds.name,
                    )
                )
            pid_offset += ds.train.num_pids
            cam_offset += ds.train.num_cams

        self.train = DatasetSplit(all_train)
        # Use first dataset for default query/gallery evaluation
        self.query = datasets[0].query
        self.gallery = datasets[0].gallery
        self._datasets = datasets
        self._source_names = source_names
        self.name = "+".join(source_names)

    @property
    def num_train_pids(self) -> int:
        return self.train.num_pids

    def summary(self) -> str:
        lines = [
            f"Combined dataset: {self.name}",
            f"  sources: {', '.join(self._source_names)}",
            f"  train: {self.train.num_imgs} images, {self.train.num_pids} IDs, {self.train.num_cams} cameras",
            f"  query: {self.query.num_imgs} images ({self._source_names[0]})",
            f"  gallery: {self.gallery.num_imgs} images ({self._source_names[0]})",
        ]
        return "\n".join(lines)
