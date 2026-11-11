"""GPU batch recommendations through the canonical perception interfaces."""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

import torch

from boxmot.detectors.factory import create_detector
from boxmot.detectors.specs import DetectorSpec
from boxmot.reid.factory import create_reid_encoder
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame
from boxmot.utils.batching import BatchProfile, profile_batch_size, validate_batch_options
from boxmot.utils.devices import normalize_device, resolve_device


def _cuda_device(value: str) -> torch.device:
    """Reject non-CUDA profiling before constructing or loading any model."""

    normalized = normalize_device(value)
    if not normalized.startswith("cuda:"):
        raise ValueError("GPU batch profiling requires a CUDA device, such as device='cuda:0'.")
    return resolve_device(normalized)


def _image_size(value: object, default: tuple[int, int]) -> tuple[int, int]:
    """Normalize the model's configured size into a positive height/width pair."""

    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)
    elif isinstance(value, tuple) and len(value) == 1:
        value = (value[0], value[0])
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError("image_size must be a positive integer or a positive (height, width) pair.")
    return value


def _synthetic_frame(image_size: tuple[int, int]) -> Frame:
    """Create deterministic RGB pixels without changing the caller's RNG state."""

    generator = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (3, *image_size), dtype=torch.uint8, generator=generator)
    return Frame(image, sample_id="batch-profile:0")


def profile_detector(
    spec: DetectorSpec,
    *,
    max_batch_size: int = 256,
    memory_fraction: float = 0.8,
    warmup: int = 1,
    repeats: int = 3,
) -> BatchProfile:
    """Recommend a CUDA detector frame batch using synthetic canonical inference.

    Supports Ultralytics and YOLOX PyTorch checkpoints and RT-DETR model
    directories. Results reflect preprocessing, model inference, and
    postprocessing at the selected model's input resolution. Synthetic images
    cannot reproduce every scene's detection or mask workload.
    """

    if not isinstance(spec, DetectorSpec):
        raise TypeError("spec must be a DetectorSpec.")
    validate_batch_options(
        max_batch_size=max_batch_size, memory_fraction=memory_fraction, warmup=warmup, repeats=repeats
    )
    artifact = Path(spec.artifact) if spec.artifact is not None else None
    if (
        spec.backend not in {"ultralytics", "yolox", "rtdetr"}
        or artifact is None
        or (spec.backend in {"ultralytics", "yolox"} and artifact.suffix.lower() != ".pt")
        or (spec.backend == "rtdetr" and not artifact.is_dir())
    ):
        raise ValueError(
            "Detector batch profiling supports Ultralytics/YOLOX .pt checkpoints and local RT-DETR "
            "model directories; exported or custom backends are not supported."
        )
    if spec.precision != "fp32":
        raise ValueError("Detector batch profiling currently requires precision='fp32'.")
    image_size = _image_size(spec.option_values().get("image_size"), (640, 640))
    device = _cuda_device(spec.device)
    sample = _synthetic_frame(image_size)
    frames = tuple(Frame(sample.image, sample_id=f"batch-profile:{index}") for index in range(max_batch_size))
    detector = None
    with torch.cuda.device(device):
        try:
            detector = create_detector(replace(spec, device=str(device)))
            # Some detector factories defer GPU placement and predictor setup
            # until their first call. Establish model residency before the
            # profiler snapshots its baseline allocation and available VRAM.
            with torch.inference_mode():
                detector.predict(frames[:1])
            torch.cuda.synchronize(device)

            def run_batch(batch_size: int) -> list[Detections]:
                """Run a complete frame batch without changing model options."""

                assert detector is not None
                return detector.predict(frames[:batch_size])

            return profile_batch_size(
                run_batch,
                device,
                max_batch_size=max_batch_size,
                memory_fraction=memory_fraction,
                warmup=warmup,
                repeats=repeats,
            )
        finally:
            # Models can contain reference cycles; collect them before returning
            # cached blocks so a subsequent ReID profile sees the released VRAM.
            detector = None
            gc.collect()
            torch.cuda.empty_cache()


def profile_reid(
    spec: ReIDEncoderSpec,
    *,
    max_batch_size: int = 256,
    memory_fraction: float = 0.8,
    warmup: int = 1,
    repeats: int = 3,
) -> BatchProfile:
    """Recommend a CUDA ReID crop batch using the canonical appearance encoder.

    Supports PyTorch and TorchScript encoders. The temporary encoder's crop cap
    is raised to ``max_batch_size`` so each candidate reaches the model in one
    inference call. The supplied immutable specification remains unchanged.
    """

    if not isinstance(spec, ReIDEncoderSpec):
        raise TypeError("spec must be a ReIDEncoderSpec.")
    validate_batch_options(
        max_batch_size=max_batch_size, memory_fraction=memory_fraction, warmup=warmup, repeats=repeats
    )
    if spec.backend not in {"pytorch", "torchscript"}:
        raise ValueError(
            "ReID batch profiling supports only PyTorch and TorchScript encoders; "
            "exported or custom backends are not supported."
        )
    if spec.precision not in {"fp32", "fp16"}:
        raise ValueError("ReID batch profiling requires precision='fp32' or precision='fp16'.")
    options = spec.option_values()
    image_size = _image_size(options.get("image_size"), (256, 128))
    options["batch_size"] = max_batch_size
    device = _cuda_device(spec.device)
    profile_spec = replace(spec, device=str(device), options=tuple(sorted(options.items())))
    frame = _synthetic_frame(image_size)
    box = torch.tensor([[0.0, 0.0, float(frame.width), float(frame.height)]], dtype=torch.float32)
    encoder = None
    with torch.cuda.device(device):
        try:
            encoder = create_reid_encoder(profile_spec)

            def run_batch(batch_size: int) -> list[torch.Tensor]:
                """Encode the requested number of actual detection crops."""

                assert encoder is not None
                detections = Detections(
                    geometry=Boxes(box.repeat(batch_size, 1)),
                    scores=torch.ones(batch_size, dtype=torch.float32),
                    class_ids=torch.zeros(batch_size, dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
                return encoder.encode((frame,), (detections,))

            return profile_batch_size(
                run_batch,
                device,
                max_batch_size=max_batch_size,
                memory_fraction=memory_fraction,
                warmup=warmup,
                repeats=repeats,
            )
        finally:
            encoder = None
            gc.collect()
            torch.cuda.empty_cache()


__all__ = ("profile_detector", "profile_reid")
