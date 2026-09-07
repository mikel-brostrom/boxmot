from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.native.trackers import botsort as native_binding
from boxmot.structures import Boxes, Detections, Tracks
from boxmot.trackers.box.botsort import native as native_module
from boxmot.trackers.box.botsort.tracker import BotSort

from ._helpers import detections_from_rows, empty_native_batch, update_rows


class _FakeLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create(self, cfg):
        self.calls.append(("create", cfg["frame_rate"], cfg["use_embeddings"]))
        return "handle"

    def reset(self, handle):
        self.calls.append(("reset", handle))

    def update(
        self,
        handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        self.calls.append(("update", handle, embeddings, image, geometry.shape[1]))
        return empty_native_batch(geometry.shape[1])

    def destroy(self, handle):
        self.calls.append(("destroy", handle))


@pytest.mark.parametrize(
    ("options", "embeddings", "frame"),
    [
        ({"use_embeddings": True, "use_cmc": False}, True, False),
        ({"use_embeddings": False, "use_cmc": True}, False, True),
        ({"use_embeddings": False, "use_cmc": False}, False, False),
        (
            {"use_embeddings": False, "use_cmc": False, "asso_func": "centroid"},
            False,
            True,
        ),
    ],
)
def test_native_botsort_requirements_follow_resolved_config(options, embeddings, frame):
    tracker = native_module.NativeBotSortTracker(options, library=_FakeLibrary())
    assert tracker.requirements.embeddings is embeddings
    assert tracker.requirements.frame is frame
    assert tracker.requirements.masks is False
    tracker.close()


def test_native_botsort_forwards_supplied_embeddings():
    library = _FakeLibrary()
    tracker = native_module.NativeBotSortTracker(
        {"frame_rate": 15, "use_embeddings": True, "use_cmc": False},
        library=library,
    )
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    detections = Detections(
        geometry=Boxes(torch.tensor([[1, 1, 4, 5], [2, 2, 6, 7]], dtype=torch.float32)),
        scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
        class_ids=torch.tensor([0, 0], dtype=torch.int64),
        sample_id="sample",
        embeddings=embeddings,
    )

    output = tracker.update(detections)
    tracker.reset()
    tracker.close()

    assert isinstance(output, Tracks)
    assert library.calls[0] == ("create", 15, True)
    assert library.calls[1][:2] == ("update", "handle")
    np.testing.assert_array_equal(library.calls[1][2], embeddings.numpy())
    assert library.calls[1][3:] == (None, 4)
    assert library.calls[2:] == [("reset", "handle"), ("destroy", "handle")]


def test_native_botsort_rejects_nonfinite_embeddings_before_native_call():
    library = _FakeLibrary()
    tracker = native_module.NativeBotSortTracker(
        {"use_embeddings": True, "use_cmc": False},
        library=library,
    )
    rows = np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite values"):
        detections_from_rows(rows, embeddings=np.array([[np.nan, 1.0]], dtype=np.float32))
    assert [call[0] for call in library.calls] == ["create"]
    tracker.close()


def test_native_botsort_rejects_model_settings_inside_algorithm_options():
    with pytest.raises(TypeError, match="unexpected option 'with_reid'"):
        native_module.NativeBotSortTracker(
            {"with_reid": False},
            library=_FakeLibrary(),
        )
    with pytest.raises(TypeError, match="unexpected option 'reid_model_path'"):
        native_module.NativeBotSortTracker(
            {"reid_model_path": "model.onnx"},
            library=_FakeLibrary(),
        )


def test_native_botsort_obb_motion_matches_python_without_internal_reid():
    options = {
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.95,
        "proximity_thresh": 0.9,
        "appearance_thresh": 0.25,
        "use_cmc": False,
        "frame_rate": 30,
        "fuse_first_associate": False,
        "use_embeddings": False,
        "second_match_thresh": 0.5,
        "unconfirmed_match_thresh": 0.95,
        "unconfirmed_emb_scale": 2.0,
    }
    python_tracker = BotSort(**options, is_obb=True)
    native_library = native_binding.BotSortLibrary(native_binding.ensure_botsort_cpp_library())
    native_tracker = native_module.NativeBotSortTracker(
        options,
        geometry="obb",
        library=native_library,
    )
    frames = (
        np.array([[40.0, 40.0, 20.0, 10.0, 0.10, 0.95, 0.0]], dtype=np.float32),
        np.array([[43.0, 42.0, 24.0, 12.0, 0.14, 0.96, 0.0]], dtype=np.float32),
        np.array([[47.0, 45.0, 31.0, 15.0, 0.20, 0.94, 0.0]], dtype=np.float32),
        np.array([[52.0, 49.0, 35.0, 18.0, 0.25, 0.93, 0.0]], dtype=np.float32),
    )

    try:
        for detections in frames:
            python_output = update_rows(python_tracker, detections)
            native_output = update_rows(native_tracker, detections)
            assert python_output.shape == native_output.shape == (1, 9)
            np.testing.assert_allclose(
                native_output[:, :5],
                python_output[:, :5],
                rtol=1e-5,
                atol=1e-5,
            )
    finally:
        native_tracker.close()
