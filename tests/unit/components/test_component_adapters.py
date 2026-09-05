from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import boxmot.reid.adapters as reid_adapters
import boxmot.reid.core.crops as crop_module
import boxmot.segmentors.backends.maskrcnn as maskrcnn_adapters
from boxmot.components.timing import timing_event_sink
from boxmot.reid.adapters import RuntimeAppearanceEncoder
from boxmot.reid.protocols import EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.segmentors.backends.maskrcnn import MaskRCNNSegmentor
from boxmot.segmentors.backends.sam import SamSegmentor
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes


def _frame(sample_id: str = "sample") -> Frame:
    image = torch.zeros((3, 6, 8), dtype=torch.uint8)
    image[0] = 10
    image[1] = 20
    image[2] = 30
    return Frame(image, sample_id)


def _aabb_detections(frame: Frame, *, empty: bool = False) -> Detections:
    count = 0 if empty else 1
    return Detections(
        geometry=Boxes(
            torch.empty((0, 4), dtype=torch.float32)
            if empty
            else torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32)
        ),
        scores=torch.full((count,), 0.9, dtype=torch.float32),
        class_ids=torch.zeros((count,), dtype=torch.int64),
        sample_id=frame.sample_id,
    )


def test_reid_adapter_preserves_canonical_cuda_index(monkeypatch) -> None:
    calls = {}

    class Runtime:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.format = SimpleNamespace(id="pytorch")
            self.model = SimpleNamespace(embedding_dim=8, input_shape=(256, 128))

    monkeypatch.setattr(
        reid_adapters,
        "import_module",
        lambda _module: SimpleNamespace(ReID=Runtime),
    )
    spec = ReIDEncoderSpec("pytorch", artifact="model.pt", device="cuda:1")

    encoder = reid_adapters.create_python_reid_encoder(spec)

    assert calls["device"] == torch.device("cuda:1")
    assert encoder.spec is spec


def test_runtime_appearance_encoder_globally_batches_normalizes_and_splits_in_order() -> None:
    calls = []

    class Runtime:
        feature_dim = 3

        def get_features(self, boxes, image):
            calls.append((boxes.copy(), image.copy()))
            return np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)

    first, second, empty = _frame("first"), _frame("second"), _frame("empty")
    oriented = Detections(
        geometry=OrientedBoxes(torch.tensor([[4.0, 3.0, 4.0, 2.0, 0.0]], dtype=torch.float32)),
        scores=torch.tensor([0.8], dtype=torch.float32),
        class_ids=torch.tensor([1], dtype=torch.int64),
        sample_id=first.sample_id,
    )
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx"), Runtime())

    events = []
    with timing_event_sink(events.append):
        results = encoder.encode(
            [first, second, empty],
            [oriented, _aabb_detections(second), _aabb_detections(empty, empty=True)],
        )

    assert encoder.embedding_dim == 3
    assert {event.component for event in events} == {"reid"}
    assert {event.phase for event in events} == {"preprocess", "process", "postprocess"}
    assert len(calls) == 1
    assert calls[0][0].shape == (2, 4)
    assert calls[0][1][0, 0].tolist() == [30, 20, 10]
    assert results[0].shape == (1, 3)
    assert results[1].shape == (1, 3)
    assert results[2].shape == (0, 3)
    torch.testing.assert_close(results[0], torch.tensor([[0.6, 0.8, 0.0]]))
    torch.testing.assert_close(results[1], torch.tensor([[0.0, 0.0, 1.0]]))
    assert all(result.dtype == torch.float32 and result.device.type == "cpu" for result in results)


def test_runtime_appearance_encoder_times_explicit_backend_stages() -> None:
    calls: list[str] = []

    class Runtime:
        feature_dim = 2
        input_shape = (6, 4)
        device = torch.device("cpu")

        def get_features(self, _boxes, _image):
            raise AssertionError("a staged runtime must not use opaque get_features")

        def get_crops(self, boxes, _image):
            calls.append("get_crops")
            return torch.ones((len(boxes), 3, 6, 4), dtype=torch.float32)

        def inference_preprocess(self, crops):
            calls.append("inference_preprocess")
            return crops

        def forward(self, crops):
            calls.append("forward")
            return torch.ones((len(crops), 2), dtype=torch.float32)

        def inference_postprocess(self, features):
            calls.append("inference_postprocess")
            return features.numpy()

    frame = _frame()
    events = []
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("pytorch"), Runtime())

    with timing_event_sink(events.append):
        result = encoder.encode([frame], [_aabb_detections(frame)])[0]

    assert calls == ["get_crops", "inference_preprocess", "forward", "inference_postprocess"]
    assert [event.phase for event in events].count("process") == 1
    assert [event.phase for event in events].count("preprocess") == 2
    assert [event.phase for event in events].count("postprocess") == 3
    torch.testing.assert_close(result, torch.full((1, 2), 2**-0.5))


def test_runtime_appearance_encoder_bounds_global_crop_batches_and_preserves_splits() -> None:
    calls: list[int] = []

    class Runtime:
        feature_dim = 3
        input_shape = (6, 4)

        def get_features(self, boxes, image):
            del image
            offset = sum(calls)
            calls.append(len(boxes))
            values = np.arange(offset + 1, offset + len(boxes) + 1, dtype=np.float32)
            return np.stack((values, np.ones_like(values), np.zeros_like(values)), axis=1)

    def detections(frame: Frame, count: int) -> Detections:
        x1 = torch.arange(count, dtype=torch.float32)
        geometry = torch.stack((x1, torch.zeros_like(x1), x1 + 1.0, torch.ones_like(x1)), dim=1)
        return Detections(
            geometry=Boxes(geometry.contiguous()),
            scores=torch.full((count,), 0.9, dtype=torch.float32),
            class_ids=torch.zeros((count,), dtype=torch.int64),
            sample_id=frame.sample_id,
        )

    first, second = _frame("first"), _frame("second")
    encoder = RuntimeAppearanceEncoder(
        ReIDEncoderSpec("onnx", options=(("batch_size", 2),)),
        Runtime(),
    )

    results = encoder.encode(
        [first, second],
        [detections(first, 3), detections(second, 2)],
    )

    assert calls == [2, 2, 1]
    assert [tuple(result.shape) for result in results] == [(3, 3), (2, 3)]
    assert torch.cat(results)[:, 0].tolist() == sorted(torch.cat(results)[:, 0].tolist())
    torch.testing.assert_close(
        torch.linalg.vector_norm(torch.cat(results), dim=1),
        torch.ones(5),
    )


def test_runtime_appearance_encoder_defaults_to_64_crop_batches() -> None:
    calls: list[int] = []

    class Runtime:
        feature_dim = 2
        input_shape = (6, 4)

        def get_features(self, boxes, image):
            del image
            calls.append(len(boxes))
            return np.ones((len(boxes), 2), dtype=np.float32)

    frame = _frame()
    count = 65
    detections = Detections(
        geometry=Boxes(torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32).repeat(count, 1)),
        scores=torch.full((count,), 0.9, dtype=torch.float32),
        class_ids=torch.zeros((count,), dtype=torch.int64),
        sample_id=frame.sample_id,
    )

    outputs = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx"), Runtime()).encode(
        [frame],
        [detections],
    )

    assert calls == [64, 1]
    assert outputs[0].shape == (65, 2)


@pytest.mark.parametrize("batch_size", (True, 0, -1, 1.5))
def test_runtime_appearance_encoder_rejects_invalid_batch_size(batch_size) -> None:
    class Runtime:
        feature_dim = 3

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        RuntimeAppearanceEncoder(
            ReIDEncoderSpec("onnx", options=(("batch_size", batch_size),)),
            Runtime(),
        )


def test_runtime_appearance_encoder_skips_neural_call_for_all_empty_batches() -> None:
    class Runtime:
        feature_dim = 7

        def get_features(self, boxes, image):
            raise AssertionError("empty batches must not invoke the ReID runtime")

    first, second = _frame("first"), _frame("second")
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx"), Runtime())
    events = []

    with timing_event_sink(events.append):
        results = encoder.encode(
            [first, second],
            [_aabb_detections(first, empty=True), _aabb_detections(second, empty=True)],
        )

    assert [tuple(result.shape) for result in results] == [(0, 7), (0, 7)]
    assert [event.phase for event in events] == ["preprocess"]


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), float("-inf")))
def test_runtime_appearance_encoder_rejects_non_finite_backend_features(invalid: float) -> None:
    class Runtime:
        feature_dim = 2

        def get_features(self, boxes, image):
            del boxes, image
            return np.array([[1.0, invalid]], dtype=np.float32)

    frame = _frame()
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx"), Runtime())

    with pytest.raises(ValueError, match="non-finite embeddings"):
        encoder.encode([frame], [_aabb_detections(frame)])


def test_runtime_appearance_encoder_rejects_zero_norm_backend_features() -> None:
    class Runtime:
        feature_dim = 2

        def get_features(self, boxes, image):
            del boxes, image
            return np.zeros((1, 2), dtype=np.float32)

    frame = _frame()
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx"), Runtime())

    with pytest.raises(ValueError, match="zero-norm embedding"):
        encoder.encode([frame], [_aabb_detections(frame)])


def test_runtime_appearance_encoder_supports_rotated_and_mask_aware_crops() -> None:
    calls = []

    class Runtime:
        feature_dim = 2
        input_shape = (6, 4)

        def get_features(self, boxes, image):
            calls.append((boxes.copy(), image.copy()))
            return np.ones((len(boxes), 2), dtype=np.float32)

    frame = _frame()
    oriented = Detections(
        geometry=OrientedBoxes(torch.tensor([[4.0, 3.0, 4.0, 2.0, 0.0]], dtype=torch.float32)),
        scores=torch.tensor([0.8], dtype=torch.float32),
        class_ids=torch.tensor([1], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    rotated = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx", crop_strategy="rotated"), Runtime())
    assert rotated.requirements == EncoderRequirements(masks=False)
    assert rotated.encode([frame], [oriented])[0].shape == (1, 2)

    masks = torch.zeros((1, frame.height, frame.width), dtype=torch.bool)
    masks[:, 2:4, 2:4] = True
    masked = _aabb_detections(frame).with_masks(MaskBatch(masks))
    mask_aware = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx", crop_strategy="mask_aware"), Runtime())
    assert mask_aware.requirements == EncoderRequirements(masks=True)
    assert mask_aware.encode([frame], [masked])[0].shape == (1, 2)
    mask_mosaic = calls[-1][1]
    assert np.any(mask_mosaic != 0)
    assert np.any(np.all(mask_mosaic == 0, axis=2))


def test_runtime_appearance_encoder_routes_perspective_obb_crops(monkeypatch) -> None:
    captured = {}

    def perspective_crop(box, image, *, max_output_side):
        captured["box"] = box.copy()
        captured["image"] = image.copy()
        captured["max_output_side"] = max_output_side
        return np.full((3, 2, 3), 47, dtype=np.uint8)

    class Runtime:
        feature_dim = 2
        input_shape = (6, 4)

        def get_features(self, boxes, image):
            captured["crop_boxes"] = boxes.copy()
            captured["mosaic"] = image.copy()
            return np.ones((len(boxes), 2), dtype=np.float32)

    monkeypatch.setattr(crop_module, "crop_obb_perspective", perspective_crop)
    frame = _frame()
    geometry = torch.tensor([[4.0, 3.0, 4.0, 2.0, 0.3]], dtype=torch.float32)
    detections = Detections(
        geometry=OrientedBoxes(geometry),
        scores=torch.tensor([0.8], dtype=torch.float32),
        class_ids=torch.tensor([1], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec("onnx", crop_strategy="perspective"), Runtime())

    result = encoder.encode([frame], [detections])

    np.testing.assert_array_equal(captured["box"], geometry[0].numpy())
    assert captured["max_output_side"] == 6
    assert captured["crop_boxes"].tolist() == [[0.0, 0.0, 2.0, 3.0]]
    assert np.all(captured["mosaic"] == 47)
    assert result[0].shape == (1, 2)


def test_sam_segmentor_aligns_and_resizes_prompt_masks_without_calling_for_empty() -> None:
    calls = []

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            mask = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
            return [SimpleNamespace(masks=SimpleNamespace(data=mask))]

    first, second = _frame("first"), _frame("second")
    segmentor = SamSegmentor(SegmentorSpec("sam", artifact="sam.pt"), model=Model())

    results = segmentor.segment([first, second], [_aabb_detections(first), _aabb_detections(second, empty=True)])

    assert len(calls) == 1
    assert calls[0]["bboxes"].tolist() == [[1.0, 1.0, 5.0, 5.0]]
    assert calls[0]["device"] == torch.device("cpu")
    assert results[0].values.shape == (1, 6, 8)
    assert results[1].values.shape == (0, 6, 8)


def test_sam_segmentor_uses_enclosing_aabb_prompts_for_obb_and_preserves_order() -> None:
    calls = []

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            first = torch.zeros((6, 8), dtype=torch.float32)
            first[0, 0] = 1
            second = torch.ones((6, 8), dtype=torch.float32)
            return [SimpleNamespace(masks=SimpleNamespace(data=torch.stack((first, second))))]

    frame = _frame()
    detections = Detections(
        geometry=OrientedBoxes(
            torch.tensor(
                [[4.0, 3.0, 4.0, 2.0, 0.0], [2.0, 2.0, 2.0, 2.0, 0.0]],
                dtype=torch.float32,
            )
        ),
        scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
        class_ids=torch.tensor([0, 1], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    segmentor = SamSegmentor(
        SegmentorSpec("sam", artifact="sam.pt", geometry_mode="obb"),
        model=Model(),
    )

    result = segmentor.segment([frame], [detections])[0]

    np.testing.assert_allclose(calls[0]["bboxes"], [[2.0, 2.0, 6.0, 4.0], [1.0, 1.0, 3.0, 3.0]])
    assert result.values[0].sum() == 1
    assert result.values[1].all()


def test_maskrcnn_segmentor_matches_predictions_by_class_and_iou() -> None:
    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, images):
            assert len(images) == 1
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0], [0.0, 0.0, 2.0, 2.0]]),
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([0, 2]),
                    "masks": torch.stack(
                        (
                            torch.ones((1, 6, 8), dtype=torch.float32),
                            torch.zeros((1, 6, 8), dtype=torch.float32),
                        )
                    ),
                }
            ]

    frame = _frame()
    segmentor = MaskRCNNSegmentor(SegmentorSpec("maskrcnn"), model=Model())

    result = segmentor.segment([frame], [_aabb_detections(frame)])[0]

    assert result.values.shape == (1, 6, 8)
    assert result.values.all()


def test_maskrcnn_segmentor_uses_global_one_to_one_hungarian_assignment(monkeypatch) -> None:
    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, images):
            first_mask = torch.ones((1, 6, 8), dtype=torch.float32)
            second_mask = torch.zeros((1, 6, 8), dtype=torch.float32)
            second_mask[:, 0, 0] = 1
            return [
                {
                    "boxes": torch.zeros((2, 4), dtype=torch.float32),
                    "scores": torch.ones(2, dtype=torch.float32),
                    "labels": torch.zeros(2, dtype=torch.int64),
                    "masks": torch.stack((first_mask, second_mask)),
                }
            ]

    monkeypatch.setattr(
        maskrcnn_adapters,
        "_pairwise_iou",
        lambda left, right: torch.tensor([[0.9, 0.8], [0.85, 0.1]], dtype=torch.float32),
    )
    frame = _frame()
    detections = Detections(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 5.0, 5.0]])),
        scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
        class_ids=torch.tensor([0, 0], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    segmentor = MaskRCNNSegmentor(SegmentorSpec("maskrcnn"), model=Model())

    result = segmentor.segment([frame], [detections])[0]

    assert result.values[0].sum() == 1
    assert result.values[1].all()
    assert segmentor.unmatched_count == 0


def test_maskrcnn_segmentor_maps_classes_and_reports_unmatched(monkeypatch) -> None:
    warnings = []

    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 5.0, 5.0]]),
                    "scores": torch.ones(1),
                    "labels": torch.tensor([10]),
                    "masks": torch.ones((1, 1, 6, 8)),
                }
            ]

    monkeypatch.setattr(maskrcnn_adapters.LOGGER, "warning", lambda *args: warnings.append(args))
    frame = _frame()
    detections = Detections(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0]])),
        scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
        class_ids=torch.tensor([0, 1], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    spec = SegmentorSpec("maskrcnn", options=(("class_mapping", ((0, 10),)),))
    segmentor = MaskRCNNSegmentor(spec, model=Model())

    result = segmentor.segment([frame], [detections])[0]

    assert result.values[0].all()
    assert not result.values[1].any()
    assert segmentor.last_unmatched_count == 1
    assert segmentor.unmatched_count == 1
    assert len(warnings) == 1


def test_maskrcnn_segmentor_never_implicitly_downloads_default_weights() -> None:
    try:
        MaskRCNNSegmentor(SegmentorSpec("maskrcnn"))
    except ValueError as error:
        assert "requires a checkpoint artifact" in str(error)
    else:
        raise AssertionError("Mask R-CNN must require an explicit artifact policy")


def test_maskrcnn_segmentor_rejects_torchvision_download_policy() -> None:
    with pytest.raises(ValueError, match="resolved local checkpoint"):
        MaskRCNNSegmentor(SegmentorSpec("maskrcnn", artifact="torchvision://maskrcnn_resnet50_fpn_coco"))
