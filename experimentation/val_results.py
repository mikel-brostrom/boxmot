import sys
import subprocess
from boxmot.utils import EXAMPLES, ROOT, WEIGHTS
from pathlib import Path


mot_seqs_path = ROOT / 'assets' / 'MOT17-mini' / 'train'
gt_folder = ROOT / 'assets' / 'MOT17-mini' / 'train'
seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
d = [seq_path.parent.name for seq_path in seq_paths]
p = subprocess.Popen(
    args=[
        sys.executable, "/home/mikel.brostrom/yolo_tracking/examples/val_utils/scripts/run_mot_challenge.py",
        "--GT_FOLDER", gt_folder,
        "--BENCHMARK", "",
        "--TRACKERS_FOLDER", "/home/mikel.brostrom/yolo_tracking/runs/track/exp",   # project/name
        "--TRACKERS_TO_EVAL", "mot",  # project/name/mot
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", "4",
        "--SKIP_SPLIT_FOL", "True",
        "--SEQ_INFO", *d
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = p.communicate()

print(stdout)