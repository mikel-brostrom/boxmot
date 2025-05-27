import cv2
from pathlib import Path
import configparser
import logging
import concurrent.futures
import numpy as np
import pandas as pd
from typing import Optional

from boxmot.utils import logger as LOGGER


def _read_seq_fps(seq_dir: Path) -> int:
    cfg_file = seq_dir / "seqinfo.ini"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    return cfg.getint("Sequence", "frameRate")


def _compute_fps_mask(frames: np.ndarray, orig_fps: int, target_fps: int) -> np.ndarray:
    # choose the lower of original and target
    tgt = min(orig_fps, target_fps)
    step = orig_fps / tgt
    wanted = set(np.arange(1, int(frames.max()) + 1, step).astype(int))
    return np.isin(frames.astype(int), list(wanted))


def load_detemb_worker(det_path: Path,
                       emb_path: Path,
                       seq_dir: Path,
                       target_fps: Optional[int]):
    """
    Load external detections and embeddings for a given sequence;
    return (seq_name, {"dets": dets, "embs": embs}) or None if missing.
    """
    seq = det_path.stem
    LOGGER.info(f"Loading det+emb for '{seq}'")

    if not emb_path.exists():
        LOGGER.debug(f"No embeddings for '{seq}', skipping")
        return None

    # Load raw dets; first col is frame index
    dets = np.loadtxt(det_path, comments="#", dtype=float)

    # Assert frame count matches images
    img_dir = seq_dir / "img1"
    all_imgs = sorted(img_dir.glob("*.jpg"))
    max_frame = int(dets[:, 0].max())
    assert max_frame == len(all_imgs), (
        f"Frame count mismatch for '{seq}': "
        f"{max_frame} frames in det vs {len(all_imgs)} images"
    )

    # Load embeddings
    embs = np.loadtxt(emb_path, comments="#", dtype=float)
    if dets.shape[0] != embs.shape[0]:
        raise ValueError(f"Row mismatch in {seq}: {dets.shape[0]} dets vs {embs.shape[0]} embs")

    # Optional FPS subsampling
    if target_fps is not None:
        orig_fps = _read_seq_fps(seq_dir)
        mask = _compute_fps_mask(dets[:, 0], orig_fps, target_fps)
        dets = dets[mask]
        embs = embs[mask]
        LOGGER.info(f" Subsampled to {mask.sum()} frames @ {min(orig_fps, target_fps)} FPS")

    return seq, {"dets": dets, "embs": embs}


class MOT17DetEmbDataset:
    """
    Loader that returns each sequence as a list of per-frame dicts:
      - 'img':  cv2 image array (BGR)
      - 'dets': np.ndarray of bounding boxes for that frame (N×4 or N×5)
      - 'embs': np.ndarray of embeddings for that frame (N×D)

    Example streaming usage:
      frames = ds.get_sequence('MOT17-04')
      for frame in frames:
          tracks = tracker.update(frame['dets'], frame['img'], frame['embs'])
    """
    def __init__(self,
                 mot_root: str,
                 split: str = "train",
                 det_emb_root: Optional[str] = None,
                 model_name: str = "yolox_x_ablation",
                 reid_name: str = "lmbn_n_duke",
                 fps: Optional[int]    = None,
                 num_workers: Optional[int] = None):
        self.root       = Path(mot_root)
        self.split      = split
        self.seq_root   = self.root #/ split
        self.target_fps = fps
        self.num_workers = num_workers

        # setup external directories
        if det_emb_root:
            self.dets_dir = Path(det_emb_root) / model_name / "dets"
            self.embs_dir = Path(det_emb_root) / model_name / "embs" / reid_name
        else:
            self.dets_dir = self.embs_dir = None

        self.sequences = {}
        self._build_index()

    def _build_index(self):
        # 1) gather frames & GT
        for seq_dir in sorted(self.seq_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            img_dir = seq_dir / "img1"

            # read GT file
            gt_df = pd.read_csv(
                seq_dir / "gt" / "gt.txt",
                header=None,
                names=["frame","x","y","w","h","id","conf","class","vis"]
            )
            # optional FPS subsample for GT stream
            if self.target_fps:
                orig_fps = _read_seq_fps(seq_dir)
                mask = _compute_fps_mask(gt_df['frame'].to_numpy(), orig_fps, self.target_fps)
                gt_df = gt_df[mask].reset_index(drop=True)
                LOGGER.info(f"[{name}] GT subsampled to {mask.sum()} frames")

            # map frame IDs to image paths
            frame_ids  = sorted(gt_df['frame'].unique())
            frame_paths = [img_dir / f"{fid:06d}.jpg" for fid in frame_ids]

            self.sequences[name] = {
                'frame_ids':   frame_ids,
                'frame_paths': frame_paths,
                'gt':          gt_df
            }

        # 2) if external det+emb given, load & split per-frame
        if self.dets_dir:
            det_files = sorted(self.dets_dir.glob("*.txt"))
            LOGGER.info(f"Loading {len(det_files)} external det files...")

            executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
            futures = {}
            for det_path in det_files:
                seq = det_path.stem
                emb_path = self.embs_dir / f"{seq}.txt"
                seq_dir = self.seq_root / seq
                futures[executor.submit(
                    load_detemb_worker,
                    det_path, emb_path, seq_dir, self.target_fps
                )] = seq

            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                seq, data = res
                if seq not in self.sequences:
                    LOGGER.warning(f"Unknown seq '{seq}' in dets")
                    continue

                # build per-frame list of dicts with cv2 image
                frame_ids   = self.sequences[seq]['frame_ids']
                frame_paths = self.sequences[seq]['frame_paths']
                frames = []
                for fid, img_path in zip(frame_ids, frame_paths):
                    # load image as BGR array
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise FileNotFoundError(f"Failed to load image {img_path}")
                    mask = data['dets'][:,0].astype(int) == fid
                    dets_f = data['dets'][mask, 1:]
                    embs_f = data['embs'][mask]
                    frames.append({'img': img, 'dets': dets_f, 'embs': embs_f})
                self.sequences[seq]['frames'] = frames

            executor.shutdown()
            LOGGER.info("External det+emb loading complete")

    def sequence_names(self) -> list[str]:
        return list(self.sequences.keys())

    def get_sequence(self, name: str) -> list[dict]:
        if name not in self.sequences or 'frames' not in self.sequences[name]:
            raise KeyError(f"Sequence '{name}' not found or has no frames loaded")
        return self.sequences[name]['frames']


if __name__ == "__main__":
    # Example: load MOT17 with external det+emb at 10 FPS
    ds = MOT17DetEmbDataset(
        mot_root     = "./tracking/val_utils/data/MOT17-50",
        split        = "train",
        det_emb_root = "./runs/dets_n_embs",
        model_name   = "yolox_x_ablation",
        reid_name    = "lmbn_n_duke",
        #fps          = 30,
        num_workers  = 4
    )

    seq_name = ds.sequence_names()[0]
    frames = ds.get_sequence(seq_name)
    print(f"Sequence '{seq_name}' has {len(frames)} frames loaded.")
    for idx, frame in enumerate(frames, start=1):
        dets = frame['dets']       # numpy array of bboxes
        img = frame['img']         # cv2 image BGR array
        embs = frame['embs']       # numpy array of embeddings
        print(f"Frame {idx}: {len(dets)} dets, emb shape {embs.shape}, img shape {img.shape}")
