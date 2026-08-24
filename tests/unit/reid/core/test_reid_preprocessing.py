import numpy as np
from PIL import Image

from boxmot.reid.core.preprocessing import IMAGENET_MEAN_BGR, IMAGENET_MEAN_RGB, resize_pad
from boxmot.reid.datasets.transforms import ResizePad


def test_resize_pad_uses_bgr_imagenet_mean_padding_for_opencv_crops():
    crop_color_bgr = (7, 13, 19)
    crop = np.full((10, 4, 3), crop_color_bgr, dtype=np.uint8)

    padded = resize_pad(crop, (10, 10))
    mean_bgr = np.asarray(IMAGENET_MEAN_BGR, dtype=np.uint8)

    assert padded.shape == (10, 10, 3)
    assert np.all(padded[:, :3] == mean_bgr)
    assert np.all(padded[:, 7:] == mean_bgr)
    assert np.all(padded[:, 3:7] == np.asarray(crop_color_bgr, dtype=np.uint8))


def test_resize_pad_pil_uses_rgb_imagenet_mean_padding_for_training_transforms():
    crop_color_rgb = (19, 13, 7)
    img = Image.new("RGB", (4, 10), crop_color_rgb)

    padded = ResizePad((10, 10))(img)
    arr = np.asarray(padded)
    mean_rgb = np.asarray(IMAGENET_MEAN_RGB, dtype=np.uint8)

    assert arr.shape == (10, 10, 3)
    assert np.all(arr[:, :3] == mean_rgb)
    assert np.all(arr[:, 7:] == mean_rgb)
    assert np.all(arr[:, 3:7] == np.asarray(crop_color_rgb, dtype=np.uint8))
