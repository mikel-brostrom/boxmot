from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np


def resize(crop: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Simple resize to target (H, W). Default preprocessing."""
    return cv2.resize(
        crop,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_pad(crop: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize preserving aspect ratio with padding."""
    target_h, target_w = target_shape
    h, w = crop.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return padded


PREPROCESS_REGISTRY: Dict[str, Callable] = {
    "resize": resize,
    "resize_pad": resize_pad,
}

DEFAULT_PREPROCESS = "resize"


def get_preprocess_fn(name: Optional[str] = None) -> Callable:
    """Get preprocessing function by name. Returns default if name is None."""
    if name is None:
        name = DEFAULT_PREPROCESS
    if name not in PREPROCESS_REGISTRY:
        raise ValueError(
            f"Unknown preprocess '{name}'. "
            f"Available: {list(PREPROCESS_REGISTRY.keys())}"
        )
    return PREPROCESS_REGISTRY[name]
