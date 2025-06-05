from typing import Tuple, Union

import numpy as np
import cv2


def letterbox(
    img: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
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
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img, ratio, (dw, dh)