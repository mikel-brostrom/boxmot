from __future__ import annotations

"""
MOT Dataset Loader
==================

Provides :class:`MOTDataset` and :class:`MOTSequence` for loading and iterating
over multi-object tracking datasets stored in the **MOT sequence format**.

Expected dataset layout
-----------------------

The MOT sequence format (used by MOTChallenge and supported by annotation tools
such as `CVAT <https://docs.cvat.ai/docs/dataset_management/formats/format-mot/>`_)
organises data as follows::

    <dataset_root>/
    ├── <sequence_1>/
    │   ├── img1/               # frame images (sequentially numbered)
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ...
    │   ├── gt/
    │   │   ├── gt.txt          # ground-truth annotations
    │   │   └── labels.txt      # (optional) class label names
    │   └── seqinfo.ini         # (optional) sequence metadata
    ├── <sequence_2>/
    │   └── ...
    └── ...

Ground-truth file (``gt/gt.txt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each line follows the format::

    frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility[, ...]

* ``frame_id`` – 1-based frame number.
* ``track_id`` – unique identity of the tracked object.
* ``x, y, w, h`` – bounding box (top-left corner + width/height).
* ``not_ignored`` – ``1`` if the annotation should be evaluated, ``0`` otherwise.
* ``class_id`` – integer class label (see ``labels.txt``).
* ``visibility`` – occlusion / visibility ratio in ``[0, 1]``.

Labels file (``gt/labels.txt``) *(optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One class name per line. Mandatory when the dataset uses non-standard labels.

Sequence info (``seqinfo.ini``) *(optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An INI file with at least a ``[Sequence]`` section that can contain::

    [Sequence]
    name=<sequence_name>
    imDir=img1
    frameRate=30
    seqLength=600
    imWidth=1920
    imHeight=1080
    imExt=.jpg

``frameRate`` is used by this module for FPS-based frame downsampling when
a ``target_fps`` is provided.
"""

import configparser
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from boxmot.utils import logger as LOGGER


def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _list_sequence_frames(img_dir: Path) -> list[Path]:
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))


def _sequence_name_from_img_dir(img_dir: Path) -> str:
    return img_dir.parent.name if img_dir.name == "img1" else img_dir.name


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths: list[Path] = []
    seq_info: dict[str, int] = {}
    for seq_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        img_dir = _sequence_img_dir(seq_dir)
        frame_files = _list_sequence_frames(img_dir)
        if not frame_files:
            continue
        seq_paths.append(img_dir)
        seq_info[seq_dir.name] = len(frame_files)
    return seq_paths, seq_info


def read_seq_fps(seq_dir: Path) -> int:
    """Read the original FPS from the ``seqinfo.ini`` of a sequence."""
    cfg_file = seq_dir / "seqinfo.ini"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    return cfg.getint("Sequence", "frameRate")


def compute_fps_mask(frames: np.ndarray, orig_fps: int, target_fps: int) -> np.ndarray:
    """Compute a boolean mask for selecting frames to match ``target_fps``."""
    tgt = min(orig_fps, target_fps)
    step = orig_fps / tgt
    wanted = set(np.arange(1, int(frames.max()) + 1, step).astype(int))
    return np.isin(frames.astype(int), list(wanted))


def _load_text_matrix(path: Path, *, delimiter: str | None = None, comments: str | None = "#") -> np.ndarray:
    """Load a numeric text file into a 2D array."""
    data = np.loadtxt(path, delimiter=delimiter, comments=comments)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return np.asarray(data, dtype=np.float32)


class MOTDataset:
    """Dataset class for MOT-format sequences with optional detection and embedding data."""

    def __init__(
        self,
        mot_root: str,
        det_emb_root: Optional[str] = None,
        model_name: Optional[str] = None,
        reid_name: Optional[str] = None,
        reid_preprocess: Optional[str] = None,
        target_fps: Optional[int] = None,
    ):
        self.root = Path(mot_root)
        self.target_fps = target_fps
        self.seqs: Dict[str, Dict] = {}

        if det_emb_root and model_name and reid_name:
            from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
            preprocess_name = reid_preprocess or DEFAULT_PREPROCESS
            base = Path(det_emb_root) / model_name
            self.dets_dir = base / "dets"
            embs_root = base / "embs"
            self.embs_dir = embs_root / reid_name / preprocess_name
            # Back-compat: if the (suffix-included) directory does not exist
            # but the legacy stem-only directory does, prefer the legacy one
            # so existing on-disk caches keep working.
            #
            # Restricted to ``.pt`` requests because the legacy layout was
            # only ever populated by the PyTorch runtime; reusing it for
            # ``.onnx`` (or other formats) would silently consume PyTorch
            # embeddings as if they belonged to a different model.
            if not self.embs_dir.exists():
                from pathlib import Path as _Path
                _name_path = _Path(reid_name)
                stem = _name_path.stem if _name_path.suffix else str(reid_name)
                if (
                    stem
                    and stem != reid_name
                    and _name_path.suffix.lower() == ".pt"
                ):
                    legacy_dir = embs_root / stem / preprocess_name
                    if legacy_dir.exists():
                        self.embs_dir = legacy_dir
            # Modern back-compat: caches written by ``boxmot.engine.cache``
            # are bucketed under ``reid_cache_key`` (e.g.
            # ``lmbn_n_duke_pt_pytorch_py``). When neither the raw
            # ``<reid_name>`` nor the legacy stem directory is on disk, fall
            # back to the canonical cache key so eval can find embeddings
            # written by a previous generate phase.
            if not self.embs_dir.exists():
                try:
                    from boxmot.data.cache import (legacy_reid_cache_keys,
                                                   reid_cache_key)
                except ImportError:
                    pass
                else:
                    candidates: list[str] = []
                    for backend in ("py", "cpp"):
                        candidates.append(
                            reid_cache_key(reid_name, tracker_backend=backend)
                        )
                        candidates.extend(
                            legacy_reid_cache_keys(reid_name, tracker_backend=backend)
                        )
                    seen: set[str] = set()
                    for key in candidates:
                        if key in seen:
                            continue
                        seen.add(key)
                        candidate_dir = embs_root / key / preprocess_name
                        if candidate_dir.exists():
                            self.embs_dir = candidate_dir
                            break
        else:
            self.dets_dir = self.embs_dir = None

        self._index_sequences()

    def _index_sequences(self) -> None:
        """Index all sequences in the dataset folder."""
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            img_dir = _sequence_img_dir(seq_dir)
            imgs = _list_sequence_frames(img_dir)
            if not imgs:
                continue
            frame_ids = [int(path.stem) for path in imgs]

            if self.dets_dir:
                npy_path = self.dets_dir / f"{name}.npy"
                txt_path = self.dets_dir / f"{name}.txt"
                if npy_path.exists():
                    det_path = npy_path
                elif txt_path.exists():
                    det_path = txt_path
                else:
                    det_path = None
            else:
                det_path = None

            if self.embs_dir:
                npy_path = self.embs_dir / f"{name}.npy"
                txt_path = self.embs_dir / f"{name}.txt"
                if npy_path.exists():
                    emb_path = npy_path
                elif txt_path.exists():
                    emb_path = txt_path
                else:
                    emb_path = None
            else:
                emb_path = None

            self.seqs[name] = {
                "seq_dir": seq_dir,
                "frame_ids": np.array(frame_ids, dtype=int),
                "frame_paths": imgs,
                "det_path": det_path,
                "emb_path": emb_path,
            }

    def sequence_names(self) -> List[str]:
        """Return the list of all available sequence names."""
        return list(self.seqs.keys())

    def get_sequence(
        self,
        name: str,
        show_progress: bool = True,
        progress_queue=None,
    ) -> "MOTSequence":
        """Return a :class:`MOTSequence` iterator for the given sequence."""
        if name not in self.seqs:
            raise KeyError(f"Unknown sequence {name}")
        return MOTSequence(
            name,
            self.seqs[name],
            self.target_fps,
            show_progress=show_progress,
            progress_queue=progress_queue,
        )


class MOTSequence:
    """Single MOT sequence that streams frame data with optional detections and embeddings."""

    def __init__(
        self,
        name: str,
        meta: Dict,
        target_fps: Optional[int],
        show_progress: bool = True,
        progress_queue=None,
    ):
        self.name = name
        self.meta = meta
        self.target_fps = target_fps
        self.show_progress = show_progress
        self.progress_queue = progress_queue
        self.dets: Optional[np.ndarray] = None
        self.embs: Optional[np.ndarray] = None
        self.frame_ids: np.ndarray = meta["frame_ids"]
        self.frame_paths: List[Path] = meta["frame_paths"]
        self._prepare()

    def _prepare(self) -> None:
        """Load detections / embeddings and optionally downsample to ``target_fps``."""
        if self.meta["det_path"] and self.meta["emb_path"]:
            det_path = Path(self.meta["det_path"])
            if det_path.suffix == ".npy":
                self.dets = np.load(det_path, mmap_mode="r")
            else:
                self.dets = _load_text_matrix(det_path, comments="#")

            emb_path = Path(self.meta["emb_path"])
            if emb_path.suffix == ".npy":
                self.embs = np.load(emb_path, mmap_mode="r")
            else:
                self.embs = _load_text_matrix(emb_path, comments="#")

            if self.dets.shape[0] != self.embs.shape[0]:
                raise ValueError(f"Row mismatch in {self.name}")

            if self.target_fps and self.dets.shape[0] > 0:
                seq_info_file = self.meta["seq_dir"] / "seqinfo.ini"
                if not seq_info_file.exists():
                    LOGGER.warning(f"Missing seqinfo.ini in {self.meta['seq_dir']}, skipping FPS downsample")
                else:
                    orig_fps = read_seq_fps(self.meta["seq_dir"])
                    mask = compute_fps_mask(self.dets[:, 0], orig_fps, self.target_fps)

                    self.dets = self.dets[mask]
                    self.embs = self.embs[mask]
                    keep_ids = set(self.dets[:, 0].astype(int))
                    idxs_to_keep = [index for index, fid in enumerate(self.frame_ids) if fid in keep_ids]
                    self.frame_ids = self.frame_ids[idxs_to_keep]
                    self.frame_paths = [self.frame_paths[index] for index in idxs_to_keep]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __iter__(self) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """Yield frame dictionaries one by one."""
        total = len(self.frame_ids)
        progress_queue = self.progress_queue
        for index, (fid, img_path) in enumerate(
            tqdm(
                zip(self.frame_ids, self.frame_paths),
                total=total,
                desc=f"Frames {self.name}",
                disable=not self.show_progress,
            )
        ):
            if progress_queue is not None:
                try:
                    progress_queue.put_nowait((self.name, index + 1, total))
                except Exception:
                    pass

            img = cv2.imread(str(img_path))
            if img is None:
                LOGGER.warning(f"Failed to load {img_path}")
                continue

            if self.dets is not None:
                mask = self.dets[:, 0].astype(int) == fid
                dets_f = self.dets[mask, 1:]
                embs_f = self.embs[mask]
            else:
                dets_f = np.zeros((0, 5))
                embs_f = np.zeros((0, 128))

            yield {
                "frame_id": fid,
                "img": img,
                "dets": dets_f,
                "embs": embs_f,
            }


__all__ = (
    "MOTDataset",
    "MOTSequence",
    "_collect_seq_info",
    "_list_sequence_frames",
    "_sequence_img_dir",
    "_sequence_name_from_img_dir",
    "compute_fps_mask",
    "read_seq_fps",
)