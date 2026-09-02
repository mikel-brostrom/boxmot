from __future__ import annotations

import numpy as np
import pytest

from boxmot.engine.tracking.inputs import TrackerInputAdapter


def _dets() -> np.ndarray:
    return np.array([[10, 10, 20, 30, 0.9, 0]], dtype=np.float32)


def _img() -> np.ndarray:
    return np.zeros((32, 48, 3), dtype=np.uint8)


def test_adapter_routes_only_detections_to_motion_only_tracker() -> None:
    calls = []

    class MotionTracker:
        uses_img = False
        uses_embs = False
        supports_masks = False

        @staticmethod
        def update(dets):
            calls.append(dets.copy())
            return np.empty((0, 8), dtype=np.float32)

    adapter = TrackerInputAdapter(MotionTracker())
    adapter.update(
        _dets(),
        img=_img(),
        embs=np.ones((1, 4), dtype=np.float32),
        masks=np.ones((1, 8, 8), dtype=np.uint8),
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], _dets())


def test_adapter_uses_exact_image_requirement_for_precomputed_embeddings() -> None:
    calls = []

    class AppearanceTracker:
        uses_img = True
        uses_embs = True
        supports_masks = False

        @staticmethod
        def requires_image(dets, embs=None, masks=None):
            del masks
            return bool(len(dets) and embs is None)

        @staticmethod
        def update(dets, img=None, embs=None, masks=None):
            calls.append((img, embs, masks))
            return np.empty((0, 8), dtype=np.float32)

    embs = np.ones((1, 4), dtype=np.float32)
    adapter = TrackerInputAdapter(AppearanceTracker())
    adapter.update(_dets(), img=_img(), embs=embs, masks=np.ones((1, 8, 8), dtype=np.uint8))

    assert len(calls) == 1
    routed_img, routed_embs, routed_masks = calls[0]
    assert routed_img is None
    assert routed_embs is embs
    assert routed_masks is None


def test_adapter_passes_image_when_the_active_path_requires_it() -> None:
    calls = []

    class ImageTracker:
        uses_img = True
        uses_embs = False
        supports_masks = False

        @staticmethod
        def requires_image(dets, embs=None, masks=None):
            del dets, embs, masks
            return True

        @staticmethod
        def update(dets, img=None):
            calls.append(img)
            return np.empty((0, 8), dtype=np.float32)

    image = _img()
    TrackerInputAdapter(ImageTracker()).update(_dets(), img=image)

    assert calls[0] is image


def test_adapter_does_not_retry_internal_type_error() -> None:
    class BrokenTracker:
        uses_img = False
        uses_embs = False
        supports_masks = False
        calls = 0

        def update(self, dets):
            del dets
            self.calls += 1
            raise TypeError("tracker bug")

    tracker = BrokenTracker()
    with pytest.raises(TypeError, match="tracker bug"):
        TrackerInputAdapter(tracker).update(_dets(), img=_img())

    assert tracker.calls == 1


def test_adapter_signature_fallback_preserves_external_frame_argument() -> None:
    calls = []

    class ExternalTracker:
        @staticmethod
        def update(dets, frame, embs=None):
            calls.append((dets, frame, embs))
            return np.empty((0, 8), dtype=np.float32)

    image = _img()
    embs = np.ones((1, 4), dtype=np.float32)
    TrackerInputAdapter(ExternalTracker()).update(_dets(), img=image, embs=embs)

    assert len(calls) == 1
    routed_dets, routed_img, routed_embs = calls[0]
    np.testing.assert_array_equal(routed_dets, _dets())
    assert routed_img is image
    assert routed_embs is embs
