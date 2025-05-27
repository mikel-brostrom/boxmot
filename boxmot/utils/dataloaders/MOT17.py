from pathlib import Path
import pandas as pd

class MOT17Dataset:
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.sequences = {}
        self._build_index()

    def _build_index(self):
        split_dir = self.root / self.split
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir(): continue
            seq_name = seq_dir.name

            # --- collect sorted image paths ---
            img_dir = seq_dir / "img1"
            frame_paths = sorted(img_dir.glob("*.jpg"))

            # --- load GT into DataFrame ---
            gt_file = seq_dir / "gt" / "gt.txt"
            df = pd.read_csv(
                gt_file, header=None,
                names=["frame","id","x","y","w","h","conf","class","vis"]
            )

            # --- map each row to its image path ---
            # note: frames in GT are 1-based
            df["img_path"] = df["frame"].apply(lambda f: frame_paths[f-1])

            # --- store ---
            self.sequences[seq_name] = {
                "frame_paths": frame_paths,
                "gt": df
            }

    def sequence_names(self):
        """List all sequence IDs."""
        return list(self.sequences.keys())

    def get_sequence(self, name: str):
        """
        Get a dict with:
          - frame_paths: [Path, …]
          - gt          : DataFrame with columns [..., img_path]
        """
        return self.sequences[name]


dataset = MOT17Dataset("/Users/mikel.brostrom/boxmot/assets/MOT17-mini")

print(dataset.sequence_names())
# ['MOT17-02-FRCNN', 'MOT17-04-FRCNN']

seq = dataset.get_sequence("MOT17-04-FRCNN")
print(seq["frame_paths"][:4])
# [PosixPath('.../img1/000001.jpg'), ..., PosixPath('.../img1/000004.jpg')]

print(seq["gt"].head())