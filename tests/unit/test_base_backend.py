import numpy as np
import torch
import cv2

from boxmot.reid.backends.base_backend import BaseModelBackend


class DummyBackend(BaseModelBackend):
    def __init__(self):
        self.device = torch.device("cpu")
        self.half = False
        self.input_shape = (16, 8)
        self.mean_array = torch.zeros((1, 3, 1, 1), device=self.device)
        self.std_array = torch.ones((1, 3, 1, 1), device=self.device)
        self.nhwc = False

    def forward(self, im_batch):
        return im_batch

    def load_model(self, w):
        return None


def test_boxes_to_xyxy_keeps_aabb_boxes():
    boxes = np.array([[10, 20, 30, 40, 0.9, 0]], dtype=np.float32)

    xyxy = DummyBackend._boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_array_equal(xyxy[0], np.array([10, 20, 30, 40], dtype=np.float32))


def test_boxes_to_xyxy_converts_obb_detections():
    boxes = np.array([[32, 24, 20, 10, 0.0, 0.9, 0]], dtype=np.float32)

    xyxy = DummyBackend._boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_allclose(xyxy[0], np.array([22, 19, 42, 29], dtype=np.float32), atol=1e-4)


def test_boxes_to_xyxy_converts_obb_track_outputs():
    boxes = np.array([[32, 24, 20, 10, 0.0, 7, 0.9, 0, 5]], dtype=np.float32)

    xyxy = DummyBackend._boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_allclose(xyxy[0], np.array([22, 19, 42, 29], dtype=np.float32), atol=1e-4)


def test_get_crops_accepts_obb_boxes():
    backend = DummyBackend()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[19:29, 22:42] = 255
    boxes = np.array([[32, 24, 20, 10, 0.0]], dtype=np.float32)

    crops = backend.get_crops(boxes, img)

    assert tuple(crops.shape) == (1, 3, 16, 8)
    assert torch.count_nonzero(crops) > 0


def test_get_crops_rectifies_rotated_obb_boxes():
    backend = DummyBackend()
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    rect = ((48.0, 48.0), (40.0, 20.0), 35.0)
    corners = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, corners, (255, 255, 255))
    box = np.array([[48.0, 48.0, 40.0, 20.0, np.deg2rad(35.0)]], dtype=np.float32)

    crops = backend.get_crops(box, img)
    crop = crops[0].permute(1, 2, 0).cpu().numpy()

    assert tuple(crops.shape) == (1, 3, 16, 8)
    assert crop.mean() > 0.2