"""Stage crop limits and authored ReID caps must reach constructed runtimes."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.engine.materialization import InlineExecutor, SourceSample, StagePlan
from boxmot.engine.materialization.stages import embed
from boxmot.reid.adapters import RuntimeAppearanceEncoder
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame


class RecordingRuntime:
    """Record real adapter inference batches and emit distinguishable rows."""

    embedding_dim = 2
    input_shape = (2, 2)

    def __init__(self) -> None:
        self.batches: list[int] = []

    def get_features(self, boxes: np.ndarray, image: np.ndarray) -> torch.Tensor:
        offset = sum(self.batches)
        self.batches.append(len(boxes))
        return torch.stack(
            (torch.ones(len(boxes)), torch.arange(offset + 1, offset + len(boxes) + 1, dtype=torch.float32)),
            dim=1,
        )


class SerializedInlineExecutor(InlineExecutor):
    """Exercise process task serialization with a locally observable runtime."""

    def map(self, function, items):
        return super().map(function, [pickle.loads(pickle.dumps(item)) for item in items])


@pytest.fixture
def inputs() -> tuple[list[Frame], list[Detections]]:
    """Include empty frames and a frame spanning multiple inference batches."""
    frames = [Frame(torch.zeros((3, 2, 2), dtype=torch.uint8), sample_id=str(i)) for i in range(3)]
    detections = [
        Detections(
            geometry=Boxes(torch.tensor([[0, 0, 2, 2]], dtype=torch.float32).repeat(count, 1)),
            scores=torch.ones(count, dtype=torch.float32),
            class_ids=torch.zeros(count, dtype=torch.int64),
            sample_id=frame.sample_id,
        )
        for frame, count in zip(frames, (90, 0, 185), strict=True)
    ]
    return frames, detections


@pytest.fixture
def runtimes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[ReIDEncoderSpec, RecordingRuntime]]:
    created: list[tuple[ReIDEncoderSpec, RecordingRuntime]] = []

    def factory(spec: ReIDEncoderSpec) -> RuntimeAppearanceEncoder:
        runtime = RecordingRuntime()
        created.append((spec, runtime))
        return RuntimeAppearanceEncoder(spec, runtime)

    monkeypatch.setattr(embed, "_WORKER_ENCODERS", {})
    monkeypatch.setattr(embed, "create_reid_encoder", factory)
    return created


def _stage(
    encoder: ReIDEncoderSpec | RuntimeAppearanceEncoder, *, override_batch_size: bool = False
) -> embed.EmbedStage:
    sample = SourceSample(
        sample_id="0",
        split="train",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(2, 2),
        source_uri="file:///tmp/image.jpg",
        source_sha256="a" * 64,
    )
    return embed.EmbedStage(encoder, [sample], encoder_fingerprint="b" * 64, override_batch_size=override_batch_size)


@pytest.mark.parametrize("executor_kind", ["inline", "process"])
@pytest.mark.parametrize("authored_batch", [None, 64, 256])
@pytest.mark.parametrize("override_batch_size", [False, True])
def test_stage_batch_reaches_runtime_and_preserves_row_order(
    inputs, runtimes, executor_kind, authored_batch, override_batch_size
) -> None:
    """Only automatic sizing replaces an authored encoder inference cap."""
    options = {"embedding_dim": 2, "image_size": (2, 2)}
    if authored_batch is not None:
        options["batch_size"] = authored_batch
    original = ReIDEncoderSpec(backend="pytorch", options=tuple(sorted(options.items())), preprocessing="resize")
    context = SimpleNamespace(
        stage_plan=StagePlan.create(
            "embed", batch_size=128, workers=2 if executor_kind == "process" else 1, executor=executor_kind
        ),
        executor=SerializedInlineExecutor() if executor_kind == "process" else InlineExecutor(),
    )
    frames, detections = inputs

    stage = _stage(original, override_batch_size=override_batch_size)
    [(shard, returned_detections, dimension, outputs)] = stage._infer(context, [(7, frames, detections)])

    assert shard == 7
    assert returned_detections is detections
    assert dimension == 2
    assert len(runtimes) == 1
    spec, runtime = runtimes[0]
    expected_batch = 128 if override_batch_size or authored_batch is None else authored_batch
    assert spec.option_values() == {**options, "batch_size": expected_batch}
    assert spec.preprocessing == "resize"
    assert original.option_values() == options
    assert runtime.batches == ([64, 64, 64, 64, 19] if expected_batch == 64 else [128, 128, 19])
    assert [tuple(output.shape) for output in outputs] == [(90, 2), (0, 2), (185, 2)]
    expected = torch.stack((torch.ones(275), torch.arange(1, 276, dtype=torch.float32)), dim=1)
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    torch.testing.assert_close(torch.cat(outputs), expected)


def test_worker_cache_distinguishes_requested_inference_batches(inputs, runtimes) -> None:
    """A resumed stage with a new batch size needs a differently configured encoder."""
    spec = ReIDEncoderSpec(backend="pytorch")
    frames, detections = inputs

    for batch_size in (128, 32, 128):
        embed._encode_from_spec((spec, 0, frames, detections, batch_size, False))

    assert [item.option_values()["batch_size"] for item, _ in runtimes] == [128, 32]
    assert runtimes[0][1].batches == [128, 128, 19] * 2
    assert runtimes[1][1].batches == [32] * 8 + [19]

    _stage(spec).release()
    assert embed._WORKER_ENCODERS == {}


def test_worker_cache_distinguishes_automatic_override(inputs, runtimes) -> None:
    """An automatically sized run cannot reuse a cached authored batch cap."""
    spec = ReIDEncoderSpec(backend="pytorch", options=(("batch_size", 64),))
    frames, detections = inputs

    for override_batch_size in (False, True, False):
        embed._encode_from_spec((spec, 0, frames, detections, 128, override_batch_size))

    assert [item.option_values()["batch_size"] for item, _ in runtimes] == [64, 128]
    assert runtimes[0][1].batches == [64, 64, 64, 64, 19] * 2
    assert runtimes[1][1].batches == [128, 128, 19]

    _stage(spec).release()
    assert embed._WORKER_ENCODERS == {}


@pytest.mark.parametrize("override_batch_size", [False, True])
def test_custom_encoder_retains_its_own_inference_limit(inputs, runtimes, override_batch_size) -> None:
    """Supplied encoder instances keep the configuration owned by their caller."""
    runtime = RecordingRuntime()
    encoder = RuntimeAppearanceEncoder(ReIDEncoderSpec(backend="pytorch"), runtime)
    frames, detections = inputs
    context = SimpleNamespace(stage_plan=StagePlan.create("embed", batch_size=128), executor=InlineExecutor())

    _stage(encoder, override_batch_size=override_batch_size)._infer(context, [(0, frames, detections)])

    assert runtimes == []
    assert runtime.batches == [64, 64, 64, 64, 19]
