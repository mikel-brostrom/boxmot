"""Canonical model adapters for CUDA batch recommendations without real GPUs."""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boxmot.detectors.specs import DetectorSpec
from boxmot.engine.profiling import batching
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Detections, Frame
from boxmot.utils.batching import BatchProfile


@pytest.fixture
def cuda_context(monkeypatch):
    """Record model construction and cache cleanup under the selected device."""

    state = SimpleNamespace(active=False, devices=[], cleanups=0, synchronizations=[])
    monkeypatch.setattr(batching, "resolve_device", lambda value: torch.device(value))

    @contextmanager
    def device_context(device):
        assert not state.active
        state.active = True
        state.devices.append(str(device))
        try:
            yield
        finally:
            state.active = False

    def empty_cache():
        assert state.active
        state.cleanups += 1

    def synchronize(device):
        assert state.active
        state.synchronizations.append(str(device))

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return state


def test_detector_profiles_canonical_batches_at_configured_resolution(monkeypatch, cuda_context) -> None:
    calls = []
    profile = BatchProfile(batch_size=3, measurements=())
    spec = DetectorSpec(backend="ultralytics", artifact="detector.pt", device="2", options=(("image_size", (12, 18)),))

    class Detector:
        def predict(self, frames):
            assert cuda_context.active
            assert all(isinstance(frame, Frame) and frame.image_size == (12, 18) for frame in frames)
            assert len({frame.sample_id for frame in frames}) == len(frames)
            calls.append(len(frames))
            return []

    def create_detector(selected):
        assert cuda_context.active
        assert selected == replace(spec, device="cuda:2")
        return Detector()

    def profile_batches(run_batch, device, **options):
        assert device == torch.device("cuda:2")
        assert options == {"max_batch_size": 3, "memory_fraction": 0.7, "warmup": 2, "repeats": 4}
        run_batch(1)
        run_batch(3)
        return profile

    monkeypatch.setattr(batching, "create_detector", create_detector)
    monkeypatch.setattr(batching, "profile_batch_size", profile_batches)

    result = batching.profile_detector(spec, max_batch_size=3, memory_fraction=0.7, warmup=2, repeats=4)

    assert result is profile
    assert calls == [1, 1, 3]
    assert cuda_context.devices == ["cuda:2"]
    assert cuda_context.cleanups == 1
    assert not cuda_context.active


def test_detector_initializes_lazy_cuda_weights_before_memory_baseline(monkeypatch, cuda_context) -> None:
    state = SimpleNamespace(resident_bytes=0, initialized=False)

    class LazyDetector:
        def predict(self, frames):
            assert len(frames) == 1
            assert torch.is_inference_mode_enabled()
            state.resident_bytes = 7 * 1024**3
            state.initialized = True
            return []

    def profile_batches(run_batch, device, **options):
        # On an 8 GiB GPU this model's 0.5 GiB activation batch fits 80% of
        # the free 1 GiB; including the lazy 7 GiB weight allocation would
        # incorrectly reject it against an 80% budget of the entire GPU.
        assert state.initialized
        assert state.resident_bytes == 7 * 1024**3
        assert cuda_context.synchronizations == ["cuda:2"]
        assert not torch.is_inference_mode_enabled()
        return BatchProfile(batch_size=1, measurements=())

    monkeypatch.setattr(batching, "create_detector", lambda spec: LazyDetector())
    monkeypatch.setattr(batching, "profile_batch_size", profile_batches)
    spec = DetectorSpec(backend="ultralytics", artifact="detector.pt", device="cuda:2")

    assert batching.profile_detector(spec, max_batch_size=1).batch_size == 1


def test_reid_candidates_reach_model_without_default_crop_chunking(monkeypatch, cuda_context) -> None:
    from boxmot.reid.adapters import RuntimeAppearanceEncoder

    batches = []
    spec = ReIDEncoderSpec(
        backend="pytorch",
        artifact="reid.pt",
        device="cuda:2",
        precision="fp16",
        options=(("batch_size", 2), ("image_size", (8, 4))),
    )

    class Runtime:
        device = "cpu"
        input_shape = (8, 4)
        embedding_dim = 4

        def get_crops(self, boxes, image):
            return torch.ones((len(boxes), 3, 8, 4))

        def inference_preprocess(self, payload):
            return payload

        def forward(self, payload):
            assert cuda_context.active
            batches.append(len(payload))
            return torch.ones((len(payload), 4))

        def inference_postprocess(self, features):
            return features

    def create_encoder(selected):
        assert cuda_context.active
        assert selected.precision == "fp16"
        assert selected.option_values() == {"batch_size": 80, "image_size": (8, 4)}
        return RuntimeAppearanceEncoder(selected, Runtime())

    def profile_batches(run_batch, device, **options):
        assert device == torch.device("cuda:2")
        assert options["max_batch_size"] == 80
        for count in (1, 80):
            outputs = run_batch(count)
            assert len(outputs) == 1
            assert outputs[0].shape == (count, 4)
        return BatchProfile(batch_size=80, measurements=())

    monkeypatch.setattr(batching, "create_reid_encoder", create_encoder)
    monkeypatch.setattr(batching, "profile_batch_size", profile_batches)

    result = batching.profile_reid(spec, max_batch_size=80)

    assert result.batch_size == 80
    assert batches == [1, 80]
    assert spec.option_values()["batch_size"] == 2
    assert cuda_context.cleanups == 1


@pytest.mark.parametrize("backend", ("pytorch", "torchscript"))
def test_reid_generates_aligned_crop_detections(monkeypatch, cuda_context, backend) -> None:
    class Encoder:
        def encode(self, frames, detections):
            assert len(frames) == len(detections) == 1
            frame, batch = frames[0], detections[0]
            assert isinstance(frame, Frame) and isinstance(batch, Detections)
            assert frame.sample_id == batch.sample_id
            assert frame.image_size == (256, 128)
            assert batch.geometry.values.tolist() == [[0, 0, 128, 256]] * 3
            return [torch.ones((3, 4))]

    def profile_batches(run_batch, device, **options):
        run_batch(3)
        return BatchProfile(batch_size=3, measurements=())

    monkeypatch.setattr(batching, "create_reid_encoder", lambda spec: Encoder())
    monkeypatch.setattr(batching, "profile_batch_size", profile_batches)
    spec = ReIDEncoderSpec(backend=backend, artifact="reid.pt", device="cuda:2")

    assert batching.profile_reid(spec, max_batch_size=3).batch_size == 3


@pytest.mark.parametrize("component", ("detector", "reid"))
def test_profile_releases_cyclic_models_before_emptying_cache(monkeypatch, cuda_context, component) -> None:
    model_refs = []

    class Model:
        def __init__(self):
            self.cycle = self
            model_refs.append(weakref.ref(self))

        def predict(self, frames):
            return []

    def cleanup():
        assert cuda_context.active
        assert all(reference() is None for reference in model_refs)
        cuda_context.cleanups += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", cleanup)
    monkeypatch.setattr(batching, "create_detector", lambda spec: Model())
    monkeypatch.setattr(batching, "create_reid_encoder", lambda spec: Model())
    monkeypatch.setattr(
        batching, "profile_batch_size", lambda *args, **kwargs: BatchProfile(batch_size=1, measurements=())
    )
    spec = (
        DetectorSpec(backend="ultralytics", artifact="detector.pt", device="cuda:2")
        if component == "detector"
        else ReIDEncoderSpec(backend="pytorch", artifact="reid.pt", device="cuda:2")
    )

    getattr(batching, f"profile_{component}")(spec, max_batch_size=1)

    assert cuda_context.cleanups == 1


@pytest.mark.parametrize("component", ("detector", "reid"))
def test_profile_cleans_up_and_propagates_inference_errors(monkeypatch, cuda_context, component) -> None:
    failure = RuntimeError("invalid model shape")

    def failed_profile(*args, **kwargs):
        raise failure

    monkeypatch.setattr(batching, "create_detector", lambda spec: SimpleNamespace(predict=lambda frames: []))
    monkeypatch.setattr(batching, "create_reid_encoder", lambda spec: object())
    monkeypatch.setattr(batching, "profile_batch_size", failed_profile)
    spec = (
        DetectorSpec(backend="yolox", artifact="detector.pt", device="cuda:2")
        if component == "detector"
        else ReIDEncoderSpec(backend="pytorch", artifact="reid.pt", device="cuda:2")
    )

    with pytest.raises(RuntimeError, match="invalid model shape") as error:
        getattr(batching, f"profile_{component}")(spec)

    assert error.value is failure
    assert cuda_context.cleanups == 1
    assert not cuda_context.active


@pytest.mark.parametrize("component", ("detector", "reid"))
@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_profile_rejects_non_cuda_before_loading_models(monkeypatch, component, device) -> None:
    def forbidden_factory(spec):
        pytest.fail("Invalid devices must not load a model.")

    monkeypatch.setattr(batching, "create_detector", forbidden_factory)
    monkeypatch.setattr(batching, "create_reid_encoder", forbidden_factory)
    spec = (
        DetectorSpec(backend="ultralytics", artifact="detector.pt", device=device)
        if component == "detector"
        else ReIDEncoderSpec(backend="pytorch", artifact="reid.pt", device=device)
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        getattr(batching, f"profile_{component}")(spec)


@pytest.mark.parametrize("component", ("detector", "reid"))
@pytest.mark.parametrize(
    "options", ({"max_batch_size": 0}, {"memory_fraction": float("nan")}, {"warmup": -1}, {"repeats": 0})
)
def test_profile_validates_search_controls_before_loading_models(monkeypatch, component, options) -> None:
    def forbidden_factory(spec):
        pytest.fail("Invalid search controls must not load a model.")

    monkeypatch.setattr(batching, "create_detector", forbidden_factory)
    monkeypatch.setattr(batching, "create_reid_encoder", forbidden_factory)
    spec = (
        DetectorSpec(backend="ultralytics", artifact="detector.pt", device="cuda:0")
        if component == "detector"
        else ReIDEncoderSpec(backend="pytorch", artifact="reid.pt", device="cuda:0")
    )

    with pytest.raises(ValueError):
        getattr(batching, f"profile_{component}")(spec, **options)


@pytest.mark.parametrize(
    "backend,artifact", (("ultralytics", "detector.onnx"), ("yolox", "detector.engine"), ("custom", "model.pt"))
)
def test_detector_rejects_non_torch_artifacts(backend, artifact) -> None:
    spec = DetectorSpec(backend=backend, artifact=artifact, device="cuda:0")
    with pytest.raises(ValueError, match="exported or custom backends are not supported"):
        batching.profile_detector(spec)


@pytest.mark.parametrize("backend", ("onnx", "tensorrt", "openvino", "coreml", "native"))
def test_reid_rejects_backends_with_untracked_memory_or_fixed_batches(backend) -> None:
    spec = ReIDEncoderSpec(backend=backend, artifact="reid.onnx", device="cuda:0")
    with pytest.raises(ValueError, match="only PyTorch and TorchScript"):
        batching.profile_reid(spec)


def test_detector_rejects_precision_ignored_by_current_backend() -> None:
    spec = DetectorSpec(backend="ultralytics", artifact="detector.pt", device="cuda:0", precision="fp16")
    with pytest.raises(ValueError, match="requires precision='fp32'"):
        batching.profile_detector(spec)


def test_rtdetr_requires_local_model_directory(tmp_path: Path, monkeypatch, cuda_context) -> None:
    monkeypatch.setattr(batching, "create_detector", lambda spec: SimpleNamespace(predict=lambda frames: []))
    monkeypatch.setattr(
        batching, "profile_batch_size", lambda *args, **kwargs: BatchProfile(batch_size=1, measurements=())
    )
    spec = DetectorSpec(backend="rtdetr", artifact=str(tmp_path), device="cuda:2")
    assert batching.profile_detector(spec, max_batch_size=1).batch_size == 1
    with pytest.raises(ValueError, match="local RT-DETR"):
        batching.profile_detector(replace(spec, artifact=str(tmp_path / "missing")))


def test_synthetic_images_are_deterministic_without_changing_global_rng() -> None:
    state = torch.random.get_rng_state().clone()
    first = batching._synthetic_frame((8, 4))
    second = batching._synthetic_frame((8, 4))
    assert torch.equal(first.image, second.image)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("value,expected", ((None, (640, 640)), (24, (24, 24)), ((24,), (24, 24)), ((8, 4), (8, 4))))
def test_input_dimensions_follow_detector_options(value, expected) -> None:
    assert batching._image_size(value, (640, 640)) == expected


@pytest.mark.parametrize("value", (0, -1, True, (4, 0), (True, 4), "640", (1, 2, 3)))
def test_invalid_input_dimensions_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="image_size"):
        batching._image_size(value, (640, 640))
