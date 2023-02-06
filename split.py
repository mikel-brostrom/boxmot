import shutil
from pathlib import Path
import pandas as pd

src_fldr=Path("/datadrive/mikel/yolov8_tracking/val_utils/data/MOT17/train")
dst_fldr=Path("/datadrive/mikel/yolov8_tracking/val_utils/data/MOT17-half/train")

if not dst_fldr.is_dir():
    shutil.copytree(src_fldr, dst_fldr)

seq_paths = [f for f in dst_fldr.iterdir() if f.is_dir()]

percent_to_delete = 0.5
for seq_path in seq_paths:
    seq_gt_path = seq_path / 'gt' / 'gt.txt'
    df = pd.read_csv(seq_gt_path, sep=",", header=None)
    nr_seq_imgs = df[0].unique().max()
    #print(nr_seq_imgs)
    split = int(nr_seq_imgs * percent_to_delete)
    print('nr of annotated frames in txt \t\t\t\t\t\t\t\t\t\t\t', split)
    df.drop(df.loc[df[0] > split].index, inplace=True)
    df.to_csv(seq_gt_path, header=None, index=None, sep=',', mode='w')
    jpg_paths = (seq_path / seq_path.name).glob(f'*.jpg')
    for jpg_path in jpg_paths:
        ref = int(jpg_path.with_suffix('').name)
        if ref > split:
            jpg_path.unlink()  # delete file
    jpg_paths = (seq_path / seq_path.name).glob(f'*.jpg')
    print(f'nr of images in {seq_path} after delete\t', sum(1 for _ in jpg_paths))