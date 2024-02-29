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


def write_np_mot_results(txt_path, numpy_results, frame_idx):
    # (x, y, x, y, id, conf, cls, ind) --> (id, t, l, w, h, conf, cls, ind)
    tlwh = ops.xyxy2ltwh(numpy_results[:, 0:4])

    mot_result = np.column_stack((
        numpy_results[:, 4],
        tlwh,
        numpy_results[:, 5],
        numpy_results[:, 6],)
    )

    frame_idx_column = np.full((mot_result.shape[0], 1), frame_idx)
    mot = np.hstack([frame_idx_column, mot_result])

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    txt_path.touch(exist_ok=True)

    with open(str(txt_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, mot, fmt='%d')  # save as ints instead of scientific notation