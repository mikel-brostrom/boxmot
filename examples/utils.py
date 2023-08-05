# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from ultralytics.utils import ops


def write_mot_results(txt_path, results, frame_idx):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)
    mot = torch.cat([
        frame_idx,
        results.boxes.id.unsqueeze(1).to('cpu'),
        ops.xyxy2ltwh(results.boxes.xyxy).to('cpu'),
        results.boxes.conf.unsqueeze(1).to('cpu'),
        results.boxes.cls.unsqueeze(1).to('cpu'),
        dont_care
    ], dim=1)

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    txt_path.touch(exist_ok=True)

    with open(str(txt_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%d')  # save as ints instead of scientific notation
