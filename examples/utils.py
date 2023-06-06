import torch
import numpy as np
from ultralytics.yolo.utils import ops

def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)
    i = torch.full((nr_dets, 1), i)
    mot = torch.cat([
        frame_idx,
        results.boxes.id.unsqueeze(1).to('cpu'),
        ops.xyxy2ltwh(results.boxes.xyxy).to('cpu'),
        results.boxes.conf.unsqueeze(1).to('cpu'),
        results.boxes.cls.unsqueeze(1).to('cpu'),
        dont_care
    ], dim=1)

    with open(str(txt_path) + '.txt', 'ab') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%d')  # save as ints instead of scientific notation