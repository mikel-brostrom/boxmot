import torch
import numpy as np


def python_nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression using numpy
        Args:
            boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(not support relative coordinates),
                shape is (n, 4)
            scores(Tensor): scores, shape is (n, )
            nms_thresh(float): thresh
        Returns:
            indices kept.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    # Use numpy to run nms. Running nms in PyTorch code on CPU is really slow.
    origin_device = boxes.device
    cpu_device = torch.device('cpu')
    boxes = boxes.to(cpu_device).numpy()
    scores = scores.to(cpu_device).numpy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True
    keep = np.nonzero(suppressed == 0)[0]
    keep = torch.from_numpy(keep).to(origin_device)
    return keep
