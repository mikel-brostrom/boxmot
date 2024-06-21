# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywh2tlwh(x):
    """
    Convert bounding box coordinates from (x c, y c, w, h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # xc --> t
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # yc --> l
    y[..., 2] = x[..., 2]                    # width
    y[..., 3] = x[..., 3]                    # height
    return y


def tlwh2xyxy(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


def xyxy2tlwh(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def tlwh2xyah(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h)
    to (center x, center y, aspect ratio, height)`, where the aspect ratio is `width / height`.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] / 2)
    y[..., 1] = x[..., 1] + (x[..., 3] / 2)
    y[..., 2] = x[..., 2] / x[..., 3]
    y[..., 3] = x[..., 3]
    return y


def xyxy2xysr(x):
    """
    Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, s, r) format.

    Args:
        bbox (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        z (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, s, r) format, where
                                          x, y is the center of the box,
                                          s is the scale (area), and
                                          r is the aspect ratio.
    """
    x = x[0:4]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w = y[..., 2] - y[..., 0]  # width
    h = y[..., 3] - y[..., 1]  # height
    y[..., 0] = y[..., 0] + w / 2.0            # x center
    y[..., 1] = y[..., 1] + h / 2.0            # y center
    y[..., 2] = w * h                                  # scale (area)
    y[..., 3] = w / (h + 1e-6)                         # aspect ratio
    y = y.reshape((4, 1))
    return y
