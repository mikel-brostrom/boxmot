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

One class name per line.  Mandatory when the dataset uses non-standard labels.

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
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from boxmot.utils import logger as LOGGER


def read_seq_fps(seq_dir: Path) -> int:
    """Read the original FPS from the ``seqinfo.ini`` of a sequence.

    Args:
        seq_dir: Path to the sequence directory.

    Returns:
        Original frame rate of the sequence.

    Raises:
        FileNotFoundError: If ``seqinfo.ini`` is not found in *seq_dir*.
    """
    cfg_file = seq_dir / "seqinfo.ini"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    return cfg.getint("Sequence", "frameRate")


def compute_fps_mask(frames: np.ndarray, orig_fps: int, target_fps: int) -> np.ndarray:
    """Compute a boolean mask for selecting frames to match *target_fps*.

    Args:
        frames: Array of original frame IDs.
        orig_fps: Original FPS of the sequence (from ``seqinfo.ini``).
        target_fps: Desired FPS.

    Returns:
        Boolean mask indicating which frames to keep.
    """
    tgt = min(orig_fps, target_fps)
    step = orig_fps / tgt
    wanted = set(np.arange(1, int(frames.max()) + 1, step).astype(int))
    return np.isin(frames.astype(int), list(wanted))


class MOTDataset:
    """Dataset class for MOT-format sequences with optional detection and embedding data.

    Scans ``mot_root`` for sequence sub-directories that follow the standard
    MOT layout (see module docstring).  When *det_emb_root*, *model_name* and
    *reid_name* are provided, pre-computed detections and ReID embeddings are
    associated with each sequence.

    Args:
        mot_root: Root path to the MOT dataset (contains sequence folders).
        det_emb_root: Root path for detection and embedding outputs.
        model_name: Name of the detection model used.
        reid_name: Name of the re-identification model used.
        target_fps: FPS to downsample to.  When set, the ``seqinfo.ini``
            inside each sequence is read to determine the original frame rate.
    """

    def __init__(
        self,
        mot_root: str,
        det_emb_root: Optional[str] = None,
        model_name: Optional[str] = None,
        reid_name: Optional[str] = None,
        target_fps: Optional[int] = None
    ):
        self.root = Path(mot_root)
        self.target_fps = target_fps
        self.seqs: Dict[str, Dict] = {}

        if det_emb_root and model_name and reid_name:
            base = Path(det_emb_root) / model_name
            self.dets_dir = base / 'dets'
            self.embs_dir = base / 'embs' / reid_name
        else:
            self.dets_dir = self.embs_dir = None

        self._index_sequences()

    def _index_sequences(self) -> None:
        """Index all sequences in the dataset folder.

        For each sub-directory the loader looks for an ``img1/`` folder first;
        if absent it falls back to using the sequence directory itself as the
        image source.  Image files (``*.jpg``, ``*.png``) are collected and
        sorted by name to determine frame ordering.
        """
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            img_dir = seq_dir / 'img1'
            if not img_dir.exists():
                img_dir = seq_dir
            imgs = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
            if not imgs:
                continue
            frame_ids = [int(p.stem) for p in imgs]

            det_path = self.dets_dir / f'{name}.txt' if self.dets_dir else None
            emb_path = self.embs_dir / f'{name}.txt' if self.embs_dir else None

            self.seqs[name] = {
                'seq_dir': seq_dir,
                'frame_ids': np.array(frame_ids, dtype=int),
                'frame_paths': imgs,
                'det_path': det_path,
                'emb_path': emb_path
            }

    def sequence_names(self) -> List[str]:
        """Return the list of all available sequence names."""
        return list(self.seqs.keys())

    def get_sequence(self, name: str) -> "MOTSequence":
        """Return a :class:`MOTSequence` iterator for the given sequence.

        Args:
            name: Name of the sequence (must match a sub-directory name).

        Raises:
            KeyError: If *name* is not found among indexed sequences.
        """
        if name not in self.seqs:
            raise KeyError(f"Unknown sequence {name}")
        return MOTSequence(name, self.seqs[name], self.target_fps)


class MOTSequence:
    """Single MOT sequence that streams frame data with optional detections and embeddings.

    Iterating over an instance yields dictionaries with the following keys:

    * ``frame_id`` – integer frame number (1-based).
    * ``img`` – BGR image as a NumPy array (loaded via OpenCV).
    * ``dets`` – ``(N, D)`` detection array for the frame (columns after
      ``frame_id`` from the detections file).
    * ``embs`` – ``(N, E)`` ReID embedding array aligned with *dets*.

    When *target_fps* is set and a ``seqinfo.ini`` with ``frameRate`` is
    present, frames (together with detections, embeddings and ground truth)
    are subsampled accordingly.  A temporary ``gt_temp.txt`` is written
    next to the original ``gt.txt`` so that evaluation tools can use the
    filtered annotations.

    Args:
        name: Sequence name.
        meta: Internal metadata dictionary produced by :class:`MOTDataset`.
        target_fps: Desired FPS for downsampling.
    """

    def __init__(self, name: str, meta: Dict, target_fps: Optional[int]):
        self.name = name
        self.meta = meta
        self.target_fps = target_fps
        self.dets: Optional[np.ndarray] = None
        self.embs: Optional[np.ndarray] = None
        self.frame_ids: np.ndarray = meta['frame_ids']
        self.frame_paths: List[Path] = meta['frame_paths']
        self._prepare()

    def _prepare(self) -> None:
        """Load detections / embeddings and optionally downsample to *target_fps*.

        Always ensures gt_temp.txt exists for evaluation: either filtered by FPS
        when downsampling, or a copy of gt.txt when nothing changed.
        """
        updated_gt = False
        gt_dir = self.meta['seq_dir'] / 'gt'

        # 1) Load dets & embs
        if self.meta['det_path'] and self.meta['emb_path']:
            self.dets = np.loadtxt(self.meta['det_path'], comments="#")
            self.embs = np.loadtxt(self.meta['emb_path'], comments="#")
            if self.dets.shape[0] != self.embs.shape[0]:
                raise ValueError(f"Row mismatch in {self.name}")

            # 2) If target_fps is set, build a frame mask using seqinfo.ini
            if self.target_fps:
                seq_info_file = self.meta['seq_dir'] / 'seqinfo.ini'
                if not seq_info_file.exists():
                    LOGGER.warning(f"Missing seqinfo.ini in {self.meta['seq_dir']}, skipping FPS downsample")
                else:
                    orig_fps = read_seq_fps(self.meta['seq_dir'])
                    mask = compute_fps_mask(self.dets[:, 0], orig_fps, self.target_fps)

                    # a) Filter dets / embs / frame_ids / frame_paths
                    self.dets = self.dets[mask]
                    self.embs = self.embs[mask]
                    keep_ids = set(self.dets[:, 0].astype(int))
                    idxs_to_keep = [i for i, fid in enumerate(self.frame_ids) if fid in keep_ids]
                    self.frame_ids = self.frame_ids[idxs_to_keep]
                    self.frame_paths = [self.frame_paths[i] for i in idxs_to_keep]

                    # b) Filter GT and write gt_temp.txt for evaluation
                    orig_gt = np.loadtxt(gt_dir / 'gt.txt', delimiter=',')
                    gt_mask = np.isin(orig_gt[:, 0].astype(int), list(keep_ids))
                    filtered_gt = orig_gt[gt_mask]
                    np.savetxt(
                        gt_dir / 'gt_temp.txt',
                        filtered_gt,
                        delimiter=',',
                        fmt="%d" if filtered_gt.dtype.kind in 'iu' else "%f",
                    )
                    updated_gt = True

        # 3) Ensure gt_temp.txt always exists for the evaluator (copy gt.txt if unchanged)
        if (gt_dir / 'gt.txt').exists() and not updated_gt:
            shutil.copy2(gt_dir / 'gt.txt', gt_dir / 'gt_temp.txt')

    def __iter__(self) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """Yield frame dictionaries one by one.

        Yields:
            A dict with keys ``frame_id``, ``img``, ``dets``, ``embs``.
        """
        for fid, img_p in tqdm(
            zip(self.frame_ids, self.frame_paths),
            total=len(self.frame_ids),
            desc=f"Frames {self.name}",
        ):
            img = cv2.imread(str(img_p))
            if img is None:
                LOGGER.warning(f"Failed to load {img_p}")
                continue

            if self.dets is not None:
                mask = (self.dets[:, 0].astype(int) == fid)
                dets_f = self.dets[mask, 1:]
                embs_f = self.embs[mask]
            else:
                dets_f = np.zeros((0, 5))
                embs_f = np.zeros((0, 128))

            yield {
                'frame_id': fid,
                'img': img,
                'dets': dets_f,
                'embs': embs_f
            }


def process_sequences_lazily(dataset: MOTDataset) -> None:
    """Example usage of lazy-loading and processing sequences from the dataset.

    Args:
        dataset: A :class:`MOTDataset` instance.
    """
    for seq_name in dataset.sequence_names():
        LOGGER.info(f"Processing sequence: {seq_name}")
        for frame_data in dataset.get_sequence(seq_name):
            print(
                f"Seq: {seq_name}, "
                f"Frame: {frame_data['frame_id']}, "
                f"Dets: {frame_data['dets'].shape[0]}"
            )


if __name__ == "__main__":
    dataset = MOTDataset(
        mot_root="./tracking/TrackEval/MOT17-ablation/train",
        det_emb_root="./runs/dets_n_embs",
        model_name="yolox_x_ablation",
        reid_name="lmbn_n_duke",
        target_fps=15
    )

    process_sequences_lazily(dataset)
