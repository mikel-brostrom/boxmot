import cv2
import numpy as np
import configparser
from pathlib import Path
from typing import Optional, List, Dict, Generator, Union
from boxmot.utils import logger as LOGGER
from tqdm import tqdm


def read_seq_fps(seq_dir: Path) -> int:
    """
    Read the original FPS (frames per second) from the seqinfo.ini file of a sequence.
    Args:
        seq_dir (Path): Path to the sequence directory.
    Returns:
        int: Original frame rate of the sequence.
    Raises:
        FileNotFoundError: If seqinfo.ini is not found.
    """
    cfg_file = seq_dir / "seqinfo.ini"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    return cfg.getint("Sequence", "frameRate")


def compute_fps_mask(frames: np.ndarray, orig_fps: int, target_fps: int) -> np.ndarray:
    """
    Compute a boolean mask for selecting frames to match the target FPS.
    Args:
        frames (np.ndarray): Array of original frame IDs.
        orig_fps (int): Original FPS of the sequence.
        target_fps (int): Desired FPS.
    Returns:
        np.ndarray: Boolean mask indicating which frames to keep.
    """
    tgt = min(orig_fps, target_fps)
    step = orig_fps / tgt
    wanted = set(np.arange(1, int(frames.max()) + 1, step).astype(int))
    return np.isin(frames.astype(int), list(wanted))


class MOT17DetEmbDataset:
    """
    A dataset class to manage MOT17 sequences with optional detection and embedding data.
    """

    def __init__(
        self,
        mot_root: str,
        det_emb_root: Optional[str] = None,
        model_name: Optional[str] = None,
        reid_name: Optional[str] = None,
        target_fps: Optional[int] = None
    ):
        """
        Initialize the MOT17 dataset.
        Args:
            mot_root (str): Root path to MOT dataset.
            det_emb_root (Optional[str], optional): Root path for detection and embedding outputs.
            model_name (Optional[str], optional): Name of detection model used.
            reid_name (Optional[str], optional): Name of re-identification model used.
            target_fps (Optional[int], optional): FPS to downsample to.
        """
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
        """
        Index all sequences in the dataset folder, loading image paths and optional det/emb paths.
        """
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            img_dir = seq_dir / 'img1'
            imgs = sorted(img_dir.glob('*.jpg'))
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
        """
        Get the list of all available sequence names.
        Returns:
            List[str]: Sequence names.
        """
        return list(self.seqs.keys())

    def get_sequence(self, name: str) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """
        Get a generator for frames of a specific sequence.
        Args:
            name (str): Name of the sequence.
        Returns:
            Generator[Dict[str, Union[int, np.ndarray]]]: Frame-wise data generator.
        Raises:
            KeyError: If the sequence name is unknown.
        """
        if name not in self.seqs:
            raise KeyError(f"Unknown sequence {name}")
        return MOT17Sequence(name, self.seqs[name], self.target_fps)


class MOT17Sequence:
    """
    A class to represent a single MOT17 sequence and stream frame data with optional detections and embeddings.
    """

    def __init__(self, name: str, meta: Dict, target_fps: Optional[int]):
        """
        Initialize a MOT17 sequence.
        Args:
            name (str): Sequence name.
            meta (Dict): Sequence metadata.
            target_fps (Optional[int]): Desired FPS for downsampling.
        """
        self.name = name
        self.meta = meta
        self.target_fps = target_fps
        self.dets: Optional[np.ndarray] = None
        self.embs: Optional[np.ndarray] = None
        self.frame_ids: np.ndarray = meta['frame_ids']
        self.frame_paths: List[Path] = meta['frame_paths']
        self._prepare()

    def _prepare(self) -> None:
        # 1) load dets & embs
        if self.meta['det_path'] and self.meta['emb_path']:
            self.dets = np.loadtxt(self.meta['det_path'], comments="#")
            self.embs = np.loadtxt(self.meta['emb_path'], comments="#")
            if self.dets.shape[0] != self.embs.shape[0]:
                raise ValueError(f"Row mismatch in {self.name}")

            # 2) if target_fps, build mask once
            if self.target_fps:
                orig_fps = read_seq_fps(self.meta['seq_dir'])
                mask = compute_fps_mask(self.dets[:,0], orig_fps, self.target_fps)

                # a) filter dets/embs/frame_ids/frame_paths
                self.dets       = self.dets[mask]
                self.embs       = self.embs[mask]
                keep_ids        = set(self.dets[:,0].astype(int))
                idxs_to_keep    = [i for i, fid in enumerate(self.frame_ids) if fid in keep_ids]
                self.frame_ids  = self.frame_ids[idxs_to_keep]
                self.frame_paths= [self.frame_paths[i] for i in idxs_to_keep]

                # b) filter GT once and write out gt_temp.txt
                gt_dir      = self.meta['seq_dir'] / 'gt'
                orig_gt     = np.loadtxt(gt_dir/'gt.txt', delimiter=',')
                gt_mask     = np.isin(orig_gt[:,0].astype(int), list(keep_ids))
                filtered_gt = orig_gt[gt_mask]
                np.savetxt(gt_dir/'gt_temp.txt', filtered_gt, delimiter=',', fmt="%d" if filtered_gt.dtype.kind in 'iu' else "%f")

    def __iter__(self) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """
        Yield frames one by one, with associated detections and embeddings (if available).
        Yields:
            Generator[Dict[str, Union[int, np.ndarray]]]: Frame data dictionary.
        """
        for fid, img_p in tqdm(zip(self.frame_ids, self.frame_paths), total=len(self.frame_ids), desc=f"Frames {self.name}"):
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


def process_sequences_lazily(dataset: MOT17DetEmbDataset) -> None:
    """
    Example usage of lazy-loading and processing sequences from the dataset.
    Args:
        dataset (MOT17DetEmbDataset): MOT17 dataset instance.
    """
    for seq_name in dataset.sequence_names():
        LOGGER.info(f"Processing sequence: {seq_name}")
        for frame_data in dataset.get_sequence(seq_name):
            # Replace this print with your own logic.
            print(f"Seq: {seq_name}, Frame: {frame_data['frame_id']}, Dets: {frame_data['dets'].shape[0]}")


if __name__ == "__main__":
    dataset = MOT17DetEmbDataset(
        mot_root="./tracking/val_utils/data/MOT17-50/train",
        det_emb_root="./runs/dets_n_embs",
        model_name="yolox_x_ablation",
        reid_name="lmbn_n_duke",
        target_fps=15
    )

    process_sequences_lazily(dataset)