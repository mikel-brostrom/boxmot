# ───────────────────────────────────────────────────────── mot17_lazy_dataset.py ──
"""
MOT17DetEmbDataset + MOT17Sequence  (v2, fixed FPS)

* Frames are still pulled lazily through LazyDataLoader
* FPS down-sampling now:
    ─ filters detections/embeddings/GT *and*
    ─ shows the correct number of frames in tqdm / user loops
    ─ never yields “skipped” frames
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional, List, Dict, Generator, Union

import cv2
import numpy as np
from tqdm import tqdm

from boxmot.utils.dataloaders.video import LazyDataLoader
from boxmot.utils import logger as LOGGER        # adjust import if needed


# ───────────────────────────────────── helpers ────────────────────────────────── #
def read_seq_fps(seq_dir: Path) -> int:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    return cfg.getint("Sequence", "frameRate")


def read_seq_len(seq_dir: Path) -> int:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    return cfg.getint("Sequence", "seqLength")


def compute_fps_mask(
    frames: np.ndarray, orig_fps: int, target_fps: int
) -> np.ndarray:
    """
    Return True/False for every element in *frames* so that the resulting
    frame-rate is ≈ target_fps.
    """
    tgt  = min(orig_fps, target_fps)
    step = orig_fps / tgt                        # e.g. 30/15 → 2.0
    wanted = set(np.arange(frames.min(),
                            frames.max() + 1,
                            step).round().astype(int))
    return np.isin(frames.astype(int), list(wanted))


# ───────────────────────────────────── dataset ────────────────────────────────── #
class MOT17DetEmbDataset:
    """
    Manages multiple MOT17 sequences – now with a *fixed* FPS interface.
    """

    def __init__(
        self,
        mot_root: str,
        det_emb_root: Optional[str] = None,
        model_name:   Optional[str] = None,
        reid_name:    Optional[str] = None,
        target_fps:   Optional[int] = None,
    ):
        self.root       = Path(mot_root)
        self.target_fps = target_fps
        self.seqs: Dict[str, Dict] = {}

        if det_emb_root and model_name and reid_name:
            base          = Path(det_emb_root) / model_name
            self.dets_dir = base / "dets"
            self.embs_dir = base / "embs" / reid_name
        else:
            self.dets_dir = self.embs_dir = None

        self._index_sequences()

    # ------------------------------------------------------------------ #
    def _index_sequences(self) -> None:
        """Create an entry per sequence (still no disk scan of img1)."""
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue

            name        = seq_dir.name
            seq_len     = read_seq_len(seq_dir)
            frame_ids   = np.arange(1, seq_len + 1, dtype=int)

            # One LazyDataLoader per sequence (no FPS stride here – handled later)
            orig_fps = read_seq_fps(seq_dir)           # NEW
            stride   = 1                               # default: keep every frame
            if self.target_fps:
                stride = max(1, round(orig_fps / self.target_fps))

            img_loader = LazyDataLoader(
                seq_dir / "img1",
                shuffle=False,
                recursive=False,
                stride=stride,                         # ← pass it in
            )

            det_path = self.dets_dir / f"{name}.txt" if self.dets_dir else None
            emb_path = self.embs_dir / f"{name}.txt" if self.embs_dir else None

            self.seqs[name] = {
                "seq_dir"    : seq_dir,
                "frame_ids"  : frame_ids,
                "loader"     : img_loader,
                "det_path"   : det_path,
                "emb_path"   : emb_path,
            }

    # ------------------------------------------------------------------ #
    def sequence_names(self) -> List[str]:
        return list(self.seqs.keys())

    # ------------------------------------------------------------------ #
    def get_sequence(self, name: str) -> "MOT17Sequence":
        if name not in self.seqs:
            raise KeyError(f"Unknown sequence {name}")
        return MOT17Sequence(name, self.seqs[name], self.target_fps)


# ─────────────────────────────────── sequence ─────────────────────────────────── #
class MOT17Sequence:
    """
    Streams frames, detections, embeddings – with proper FPS sub-sampling.
    """

    def __init__(
        self,
        name: str,
        meta: Dict,
        target_fps: Optional[int],
    ):
        self.name        = name
        self.meta        = meta
        self.target_fps  = target_fps

        self.frame_ids: np.ndarray       = meta["frame_ids"]
        self.loader: LazyDataLoader      = meta["loader"]

        self.dets: Optional[np.ndarray]  = None
        self.embs: Optional[np.ndarray]  = None

        self._prepare()                  # loads dets/embs and builds masks

    # ------------------------------------------------------------------ #
    def _prepare(self) -> None:
        """Load dets/embs → apply FPS mask → write gt_temp.txt once."""
        det_path, emb_path = self.meta["det_path"], self.meta["emb_path"]
        if det_path and emb_path:
            self.dets = np.loadtxt(det_path, comments="#")
            self.embs = np.loadtxt(emb_path, comments="#")
            if self.dets.shape[0] != self.embs.shape[0]:
                raise ValueError(f"Row mismatch in {self.name}")

        # FPS down-sampling --------------------------------------------------
        if self.target_fps:
            orig_fps  = read_seq_fps(self.meta["seq_dir"])
            stride    = max(1, round(orig_fps / self.target_fps))

            keep_mask = (self.frame_ids % stride == 0)     # keep 0-based multiples
            self.frame_ids = self.frame_ids[keep_mask]

            if self.dets is not None:
                rows = (self.dets[:, 0] % stride == 0)
                self.dets = self.dets[rows]
                self.embs = self.embs[rows]

            # GT: write the filtered copy only once
            gt_dir  = self.meta["seq_dir"] / "gt"
            gt      = np.loadtxt(gt_dir / "gt.txt", delimiter=",")
            gt      = gt[gt[:, 0] % stride == 0]
            np.savetxt(gt_dir / "gt_temp.txt", gt, delimiter=",",
                    fmt="%d" if gt.dtype.kind in "iu" else "%f")


    # ------------------------------------------------------------------ #
    def __iter__(self):
        total = len(self.loader)                   # loader already knows the length
        with tqdm(total=total, desc=f"Frames {self.name}") as pbar:
            for batch, fid in zip(self.loader, self.frame_ids):
                img = batch[0]
                if img is None:
                    LOGGER.warning(f"Failed to load frame {fid:06d} in {self.name}")
                    pbar.update(1)
                    continue

                if self.dets is not None:
                    rows   = (self.dets[:, 0] == fid)
                    dets_f = self.dets[rows, 1:]
                    embs_f = self.embs[rows]
                else:
                    dets_f = np.zeros((0, 5))
                    embs_f = np.zeros((0, 128))

                pbar.update(1)
                yield {"frame_id": fid, "img": img, "dets": dets_f, "embs": embs_f}


# ─────────────────────────────────── example ──────────────────────────────────── #
def process_sequences_lazily(dataset: MOT17DetEmbDataset) -> None:
    """Tiny demo loop."""
    for seq_name in dataset.sequence_names():
        LOGGER.info(f"Processing sequence: {seq_name}")
        for frame in dataset.get_sequence(seq_name):
            print(
                f"Seq: {seq_name}, "
                f"Frame: {frame['frame_id']}, "
                f"Dets: {frame['dets'].shape[0]}"
            )


if __name__ == "__main__":
    dataset = MOT17DetEmbDataset(
        mot_root    ="./boxmot/engine/val_utils/data/MOT17-50/train",
        det_emb_root="./runs/dets_n_embs",
        model_name  ="yolox_x_ablation",
        reid_name   ="lmbn_n_duke",
        target_fps  =30,
    )
    process_sequences_lazily(dataset)
