from pathlib import Path
import numpy as np
import configparser
import concurrent.futures
from boxmot.utils import logger as LOGGER


def load_sequence_worker(det_path: Path,
                         embs_dir: Path,
                         mot_source: Path | None,
                         target_fps: int | None):
    """
    Worker function for loading and optional FPS-subsampling of a single sequence's detections and embeddings.
    """
    seq = det_path.stem
    LOGGER.info(f"Loading sequence '{seq}' from {det_path}")
    emb_path = embs_dir / f"{seq}.txt"
    if not emb_path.exists():
        LOGGER.debug(f"Skipping sequence '{seq}': no embeddings file found at {emb_path}")
        return None

    dets = np.loadtxt(det_path, comments="#", dtype=float)
    embs = np.loadtxt(emb_path, comments="#", dtype=float)

    if dets.shape[0] != embs.shape[0]:
        raise ValueError(
            f"Row mismatch in {seq}: {dets.shape[0]} dets vs {embs.shape[0]} embs"
        )

    # Optional FPS subsampling
    if target_fps and mot_source:
        seqinfo = mot_source / seq / "seqinfo.ini"
        if not seqinfo.exists():
            raise FileNotFoundError(
                f"Expected seqinfo at {seqinfo} for sequence {seq}, but not found."
            )

        cfg = configparser.ConfigParser()
        cfg.read(seqinfo)
        if "Sequence" not in cfg:
            raise configparser.NoSectionError(
                f"No [Sequence] section in {seqinfo}"
            )

        orig = cfg.getint("Sequence", "frameRate")
        tgt = target_fps if orig >= target_fps else orig
        if orig < target_fps:
            LOGGER.warning(
                f"[{seq}] original FPS={orig} < target={target_fps}, using {orig}"
            )

        step = orig / tgt
        max_frame = int(dets[:, 0].max())
        wanted = set(np.arange(1, max_frame + 1, step).astype(int))
        frames = dets[:, 0].astype(int)
        mask = np.isin(frames, list(wanted))
        dets = dets[mask]
        embs = embs[mask]
        LOGGER.info(
            f"Sequence '{seq}' subsampled from {mask.size} to {dets.shape[0]} frames"
        )

    LOGGER.info(
        f"Sequence '{seq}' loaded: {dets.shape[0]} frames, embedding dim={embs.shape[1] if embs.ndim==2 else embs.size}"
    )
    return seq, {"dets": dets, "embs": embs}


class DetEmbSequenceDataset:
    """
    Per‐sequence loader for dets + embs with optional FPS subsampling, parallelized across sequences.

      root/
        <model_name>/
          dets/  <seq>.txt
          embs/  <reid_name>/ <seq>.txt

    mot_source/
      <seq>/
        seqinfo.ini

    If fps is given, mot_source must point at the parent of each <seq>/seqinfo.ini
    so we can read the original frameRate and drop all other frames.
    """
    def __init__(self,
                 root: str,
                 model_name: str = "yolox_x_ablation",
                 reid_name: str = "lmbn_n_duke",
                 mot_source: str = None,
                 fps: int = None):

        self.root       = Path(root)
        self.dets_dir   = self.root / model_name / "dets"
        self.embs_dir   = self.root / model_name / "embs" / reid_name

        # Optional MOT directory root (contains <seq>/seqinfo.ini)
        self.mot_source = Path(mot_source) if mot_source is not None else None
        self.target_fps = fps

        if self.target_fps is not None and self.mot_source is None:
            raise ValueError(
                "If you set fps, you must also pass mot_source so I can read seqinfo.ini"
            )

        self.sequences: dict[str, dict[str, np.ndarray]] = {}
        self._build_index()

    def _build_index(self):
        """
        Build the dataset index in parallel across all sequences using ProcessPoolExecutor.
        """
        det_paths = sorted(self.dets_dir.glob("*.txt"))
        LOGGER.info(f"Building dataset index: found {len(det_paths)} sequence files in {self.dets_dir}")
        
        # Use a ProcessPool to parallelize loading of each sequence
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    load_sequence_worker,
                    dp,
                    self.embs_dir,
                    self.mot_source,
                    self.target_fps
                ): dp for dp in det_paths
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    seq, data = result
                    self.sequences[seq] = data
        LOGGER.info(f"Finished building index. Successfully loaded {len(self.sequences)} sequences.")

    def sequence_names(self):
        """Return a list of all loaded sequence names."""
        LOGGER.debug("Retrieving sequence names list")
        return list(self.sequences.keys())

    def get_sequence(self, name: str):
        """Retrieve dets and embs arrays for a given sequence name."""
        LOGGER.debug(f"Retrieving data for sequence '{name}'")
        if name not in self.sequences:
            raise KeyError(f"Sequence '{name}' not found")
        return self.sequences[name]

    def summary(self):
        """
        Print a summary of the loaded dataset, including number of sequences,
        total frames, average frames per sequence, and embedding dimension.
        """
        num_seqs = len(self.sequences)
        total_frames = 0
        frame_counts = []
        emb_dim = None

        for seq, data in self.sequences.items():
            dets = data['dets']
            embs = data['embs']
            n = dets.shape[0]
            total_frames += n
            frame_counts.append(n)
            if emb_dim is None and embs.ndim == 2:
                emb_dim = embs.shape[1]

        avg_frames = total_frames / num_seqs if num_seqs > 0 else 0

        LOGGER.info("Dataset Summary:")
        LOGGER.info(f"  Sequences loaded      : {num_seqs}")
        LOGGER.info(f"  Total rows            : {total_frames}")
        LOGGER.info(f"  Average rows per seq  : {int(avg_frames)}")
        if emb_dim is not None:
            LOGGER.info(f"  Embedding dimension   : {emb_dim}")


if __name__ == "__main__":
    ds = DetEmbSequenceDataset(
        root       = "./runs/dets_n_embs",
        model_name = "yolox_x_ablation",
        reid_name  = "lmbn_n_duke",
        mot_source = "./tracking/val_utils/data/MOT17-50/train",
        fps        = 10
    )

    LOGGER.info(f"Available sequences: {ds.sequence_names()}")
    ds.summary()
    data = ds.get_sequence(ds.sequence_names()[0])
    LOGGER.info(f"First sequence shapes: dets={data['dets'].shape}, embs={data['embs'].shape}")