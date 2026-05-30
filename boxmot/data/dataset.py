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

from boxmot.data.frame_cache import FrameCache
from boxmot.utils import logger as LOGGER
from boxmot.utils.rich.progress import RichTqdm as tqdm


def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _list_sequence_frames(img_dir: Path) -> list[Path]:
    return sorted(
        p for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.npy"))
        if not p.name.startswith("._")
    )


def _sequence_name_from_img_dir(img_dir: Path) -> str:
    return img_dir.parent.name if img_dir.name == "img1" else img_dir.name


def _infer_sequence_length(seq_dir: Path) -> int:
    """Infer sequence length when frame files are not present locally."""
    seqinfo = seq_dir / "seqinfo.ini"
    if seqinfo.exists():
        try:
            cfg = configparser.ConfigParser()
            cfg.read(seqinfo)
            seq_len = cfg.getint("Sequence", "seqLength")
            if seq_len > 0:
                return seq_len
        except Exception:
            pass

    def _max_frame_from_csv(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            data = np.loadtxt(path, delimiter=",")
            if data.size == 0:
                return 0
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return int(np.max(data[:, 0]))
        except Exception:
            return 0

    gt_len = _max_frame_from_csv(seq_dir / "gt" / "gt.txt")
    det_len = _max_frame_from_csv(seq_dir / "det" / "det.txt")
    return max(gt_len, det_len)


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths: list[Path] = []
    seq_info: dict[str, int] = {}
    for seq_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        img_dir = _sequence_img_dir(seq_dir)
        frame_files = _list_sequence_frames(img_dir)
        if frame_files:
            seq_paths.append(img_dir)
            seq_info[seq_dir.name] = len(frame_files)
            continue

        inferred_len = _infer_sequence_length(seq_dir)
        if inferred_len > 0:
            seq_paths.append(img_dir)
            seq_info[seq_dir.name] = inferred_len
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
        masks_dir: Optional[str] = None,
        seq_pattern: Optional[str] = None,
    ):
        self.root = Path(mot_root)
        self.target_fps = target_fps
        self.seq_pattern = seq_pattern
        self.seqs: Dict[str, Dict] = {}

        if det_emb_root and model_name:
            from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
            preprocess_name = reid_preprocess or DEFAULT_PREPROCESS
            base = Path(det_emb_root) / model_name
            self.dets_dir = base / "dets"
            if reid_name:
                embs_root = base / "embs"
                self.embs_dir = embs_root / reid_name / preprocess_name
            else:
                self.embs_dir = None
        else:
            self.dets_dir = self.embs_dir = None

        self.masks_dir = Path(masks_dir) if masks_dir else None

        # Auto-discover masks from the cache tree if not explicitly provided
        if self.masks_dir is None and det_emb_root and model_name:
            masks_base = Path(det_emb_root) / model_name / "masks"
            if masks_base.is_dir():
                # Use the first subdirectory that contains mask files (.npy or legacy .npz)
                for sub in sorted(masks_base.iterdir()):
                    if sub.is_dir() and (any(sub.glob("*.npy")) or any(sub.glob("*.npz"))):
                        self.masks_dir = sub
                        break

        self._index_sequences()

    def _index_sequences(self) -> None:
        """Index all sequences in the dataset folder."""
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            # Apply sequence pattern filter (e.g. "*-FRCNN")
            if self.seq_pattern:
                from fnmatch import fnmatch
                if not fnmatch(name, self.seq_pattern):
                    continue
            img_dir = _sequence_img_dir(seq_dir)
            imgs = _list_sequence_frames(img_dir)
            if imgs:
                frame_ids = [int(path.stem) for path in imgs]
            else:
                inferred_len = _infer_sequence_length(seq_dir)
                if inferred_len <= 0:
                    continue
                frame_ids = list(range(1, inferred_len + 1))

            if self.dets_dir:
                npy_path = self.dets_dir / f"{name}.npy"
                det_path = npy_path if npy_path.exists() else None
            else:
                det_path = None

            if self.embs_dir:
                npy_path = self.embs_dir / f"{name}.npy"
                emb_path = npy_path if npy_path.exists() else None
            else:
                emb_path = None

            # Mask cache path (.npy, same row order as dets)
            mask_path = None
            if self.masks_dir:
                npy_path = self.masks_dir / f"{name}.npy"
                if npy_path.exists():
                    mask_path = npy_path

            self.seqs[name] = {
                "seq_dir": seq_dir,
                "frame_ids": np.array(frame_ids, dtype=int),
                "frame_paths": imgs,
                "det_path": det_path,
                "emb_path": emb_path,
                "mask_path": mask_path,
            }

    def sequence_names(self) -> List[str]:
        """Return the list of all available sequence names."""
        return list(self.seqs.keys())

    def get_sequence(
        self,
        name: str,
        show_progress: bool = True,
        progress_queue=None,
        skip_image_load: bool = False,
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
            skip_image_load=skip_image_load,
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
        skip_image_load: bool = False,
        n_cache_peers: int = 1,
    ):
        self.name = name
        self.meta = meta
        self.target_fps = target_fps
        self.show_progress = show_progress
        self.progress_queue = progress_queue
        self.skip_image_load = skip_image_load
        self.n_cache_peers = n_cache_peers
        self.dets: Optional[np.ndarray] = None
        self.embs: Optional[np.ndarray] = None
        self.masks_data: Optional[Dict[int, np.ndarray]] = None
        self._masks_flat: Optional[np.ndarray] = None
        self.frame_ids: np.ndarray = meta["frame_ids"]
        self.frame_paths: List[Path] = meta["frame_paths"]
        self._frame_cache: Optional[FrameCache] = None
        if not self.skip_image_load:
            self._frame_cache = FrameCache(self.frame_paths, n_cache_peers=self.n_cache_peers)
        self._prepare()

    def _prepare(self) -> None:
        """Load detections / embeddings and optionally downsample to ``target_fps``."""
        self._det_index: Optional[Dict[int, tuple]] = None
        if self.meta["det_path"]:
            det_path = Path(self.meta["det_path"])
            self.dets = np.load(det_path, mmap_mode="r")

            if self.meta["emb_path"]:
                emb_path = Path(self.meta["emb_path"])
                self.embs = np.load(emb_path, mmap_mode="r")

                if self.dets.shape[0] != self.embs.shape[0]:
                    raise ValueError(f"Row mismatch in {self.name}")

            if self.target_fps and self.dets.shape[0] > 0:
                seq_info_file = self.meta["seq_dir"] / "seqinfo.ini"
                if not seq_info_file.exists():
                    LOGGER.warning(f"Missing seqinfo.ini in {self.meta['seq_dir']}, skipping FPS downsample")
                else:
                    orig_fps = read_seq_fps(self.meta["seq_dir"])
                    fps_mask = compute_fps_mask(self.dets[:, 0], orig_fps, self.target_fps)

                    self.dets = self.dets[fps_mask]
                    if self.embs is not None:
                        self.embs = self.embs[fps_mask]
                    keep_ids = set(self.dets[:, 0].astype(int))
                    idxs_to_keep = [index for index, fid in enumerate(self.frame_ids) if fid in keep_ids]
                    self.frame_ids = self.frame_ids[idxs_to_keep]
                    self.frame_paths = [self.frame_paths[index] for index in idxs_to_keep]
                    self._fps_mask = fps_mask

            # Build frame_id → row-slice index for O(1) per-frame lookup
            if self.dets is not None and self.dets.shape[0] > 0:
                self._det_index = self._build_det_index()

        # Load mask cache
        if self.meta.get("mask_path"):
            mask_path = Path(self.meta["mask_path"])
            # Bit-packed format: (total_dets, H, W_packed) uint8, same row order as dets/embs
            masks_arr = np.load(str(mask_path), mmap_mode="r")
            if hasattr(self, "_fps_mask"):
                masks_arr = masks_arr[self._fps_mask]
            self._masks_flat = masks_arr

    def _build_det_index(self) -> Dict[int, tuple]:
        """Build a mapping from frame_id to (start, end) row indices in self.dets."""
        fids = self.dets[:, 0].astype(int)
        index: Dict[int, tuple] = {}
        n = len(fids)
        if n == 0:
            return index
        # Detections are sorted by frame_id in cache files
        start = 0
        current_fid = fids[0]
        for i in range(1, n):
            if fids[i] != current_fid:
                index[int(current_fid)] = (start, i)
                current_fid = fids[i]
                start = i
        index[int(current_fid)] = (start, n)
        return index

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _stub_image_shape(self) -> tuple[int, int]:
        seqinfo = self.meta["seq_dir"] / "seqinfo.ini"
        if seqinfo.exists():
            try:
                cfg = configparser.ConfigParser()
                cfg.read(seqinfo)
                h = cfg.getint("Sequence", "imHeight")
                w = cfg.getint("Sequence", "imWidth")
                if h > 0 and w > 0:
                    return h, w
            except Exception:
                pass
        return 720, 1280

    def __iter__(self) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """Yield frame dictionaries one by one."""
        total = len(self.frame_ids)
        progress_queue = self.progress_queue
        _img_stub: Optional[np.ndarray] = None

        has_frame_files = len(self.frame_paths) > 0
        if not has_frame_files and total > 0:
            h, w = self._stub_image_shape()
            _img_stub = np.empty((h, w, 3), dtype=np.uint8)
            iter_pairs = ((int(fid), None) for fid in self.frame_ids)
        else:
            iter_pairs = zip(self.frame_ids, self.frame_paths)

        for index, (fid, img_path) in enumerate(
            tqdm(
                iter_pairs,
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

            # After the first frame, reuse a small stub image when caller
            # only needs shape (e.g. tracking replay with cached embeddings).
            if img_path is None and _img_stub is not None:
                img = _img_stub
            elif self.skip_image_load and _img_stub is not None:
                img = _img_stub
            elif self._frame_cache is not None:
                img = self._frame_cache.read_image(img_path)
            elif img_path.suffix == ".npy":
                img = np.load(str(img_path))
                # Convert multi-channel arrays to 3-channel BGR for tracker compatibility
                if img.ndim == 3 and img.shape[2] > 3:
                    img = img[:, :, :3]
                elif img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.imread(str(img_path))
            if img is None:
                LOGGER.warning(f"Failed to load {img_path}")
                continue

            # Create a tiny stub with same shape metadata for subsequent frames
            if self.skip_image_load and _img_stub is None:
                _img_stub = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            if self.dets is not None:
                if self._det_index is not None and fid in self._det_index:
                    start, end = self._det_index[fid]
                    dets_f = np.array(self.dets[start:end, 1:])
                    embs_f = np.array(self.embs[start:end]) if self.embs is not None else np.zeros((end - start, 0))
                    if self._masks_flat is not None:
                        packed = np.array(self._masks_flat[start:end])
                        masks_f = np.unpackbits(packed, axis=-1)
                    else:
                        masks_f = None
                elif self._det_index is not None:
                    # Frame not in index → no detections
                    dets_f = np.zeros((0, self.dets.shape[1] - 1))
                    embs_f = np.zeros((0, self.embs.shape[1] if self.embs is not None else 0))
                    masks_f = None
                else:
                    # Fallback for unsorted data (no index built)
                    mask = self.dets[:, 0].astype(int) == fid
                    dets_f = self.dets[mask, 1:]
                    embs_f = self.embs[mask] if self.embs is not None else np.zeros((dets_f.shape[0], 0))
                    if self._masks_flat is not None:
                        packed = np.array(self._masks_flat[mask])
                        masks_f = np.unpackbits(packed, axis=-1)
                    else:
                        masks_f = None
            else:
                dets_f = np.zeros((0, 5))
                embs_f = np.zeros((0, 128))
                masks_f = None

            # Legacy .npz masks fallback (keyed by frame id)
            if masks_f is None and self.masks_data is not None:
                masks_f = self.masks_data.get(fid)
                # Align masks to detection count
                if masks_f is not None and masks_f.shape[0] != dets_f.shape[0]:
                    masks_f = None

            yield {
                "frame_id": fid,
                "img": img,
                "dets": dets_f,
                "embs": embs_f,
                "masks": masks_f,
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
