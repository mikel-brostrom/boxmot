"""Unit tests for the native C++ ReID ABI wrapper.

Skipped automatically when the optional ``reid_capi`` shared library or its
ONNX/OpenCV runtime dependencies are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _model_or_skip() -> Path:
    candidate = ROOT / "models" / "lmbn_n_duke.onnx"
    if not candidate.exists():
        pytest.skip(f"Required ONNX ReID model not found: {candidate}")
    return candidate


def _image_or_skip() -> np.ndarray:
    import cv2

    img_path = ROOT / "assets" / "MOT17-mini" / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg"
    if not img_path.exists():
        pytest.skip(f"Required test image not found: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        pytest.skip(f"Failed to read test image: {img_path}")
    return image


def _load_adapter():
    try:
        from boxmot.native.reid_capi import CppOnnxReID
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Native ReID C ABI unavailable: {exc}")
    return CppOnnxReID


def test_cpp_reid_smoke_aabb():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        boxes = np.array(
            [
                [100, 100, 200, 300],
                [300, 150, 400, 400],
                [50, 80, 150, 250],
            ],
            dtype=np.float32,
        )
        feats = reid.get_features(boxes, image)

        assert feats.dtype == np.float32
        assert feats.ndim == 2
        assert feats.shape[0] == boxes.shape[0]
        assert feats.shape[1] == reid.feature_dim > 0

        # L2 normalised rows
        norms = np.linalg.norm(feats, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
    finally:
        reid.close()


def test_cpp_reid_handles_empty_boxes():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        out = reid.get_features(np.empty((0, 4), dtype=np.float32), image)
        assert isinstance(out, np.ndarray)
        assert out.size == 0
    finally:
        reid.close()


def test_cpp_reid_obb_to_aabb_conversion():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        # OBB rows: (cx, cy, w, h, theta). theta=0 ⇒ AABB equivalent (cx-w/2, ...).
        obb = np.array([[200.0, 250.0, 80.0, 200.0, 0.0]], dtype=np.float32)
        aabb = np.array([[160.0, 150.0, 240.0, 350.0]], dtype=np.float32)

        feats_obb = reid.get_features(obb, image)
        feats_aabb = reid.get_features(aabb, image)

        # When theta=0 the OBB→AABB conversion must match an explicit AABB call.
        assert feats_obb.shape == feats_aabb.shape
        np.testing.assert_allclose(feats_obb, feats_aabb, atol=1e-5)
    finally:
        reid.close()
