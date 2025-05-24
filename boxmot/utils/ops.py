# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from typing import Tuple, Union

import cv2
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


def letterbox(
    img: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resizes an image to a new shape while maintaining aspect ratio, padding with color if needed.

    Args:
        img (np.ndarray): The original image in BGR format.
        new_shape (Union[int, Tuple[int, int]], optional): Desired size as an integer (e.g., 640) 
            or tuple (width, height). Default is (640, 640).
        color (Tuple[int, int, int], optional): Padding color in BGR format. Default is (114, 114, 114).
        auto (bool, optional): If True, adjusts padding to be a multiple of 32. Default is True.
        scaleFill (bool, optional): If True, stretches the image to fill the new shape. Default is False.
        scaleup (bool, optional): If True, allows scaling up; otherwise, only scales down. Default is True.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
            - Resized and padded image as np.ndarray.
            - Scaling ratio used for width and height as (width_ratio, height_ratio).
            - Padding applied to width and height as (width_padding, height_padding).
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Ensure new_shape is a tuple (width, height)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)  # only scale down

    # Calculate new dimensions and padding
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:  # stretch to fill
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    # Divide padding by 2 for even distribution
    dw /= 2
    dh /= 2

    # Resize image if necessary
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border to the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)
