"""Saved appearance reuse preserves source, crop and encoder identities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from boxmot.engine.eval.saved_input_cache import CachedAppearanceEncoder
from boxmot.reid.protocols import EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch


class _Encoder:
    embedding_dim = 2

    def __init__(self, *, masks: bool = False) -> None:
        self.requirements = EncoderRequirements(masks=masks)
        self.calls = 0

    def encode(self, frames, detections):
        self.calls += 1
        return [detection.geometry.values[:, :2].contiguous().clone() for detection in detections]


def _sample(tmp_path: Path) -> tuple[Frame, Detections]:
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), (12, 24, 36)).save(image)
    frame = Frame(torch.full((3, 6, 8), 12, dtype=torch.uint8), "val:drive:0", source_uri=image.as_uri())
    detections = Detections(
        Boxes(torch.tensor([[1, 1, 4, 5], [2, 2, 6, 5]], dtype=torch.float32)),
        torch.tensor([0.8, 0.9], dtype=torch.float32),
        torch.tensor([2**54, 2], dtype=torch.int64),
        frame.sample_id,
    )
    return frame, detections


def _cached(tmp_path: Path, encoder: _Encoder, **spec_kwargs) -> CachedAppearanceEncoder:
    return CachedAppearanceEncoder(encoder, ReIDEncoderSpec(backend="fixture", **spec_kwargs), tmp_path / "cache")


def test_appearance_cache_reuses_ordered_embeddings_without_shared_memory(tmp_path: Path) -> None:
    frame, detections = _sample(tmp_path)
    encoder = _Encoder()
    cached = _cached(tmp_path, encoder)
    first = cached.encode([frame], [detections])[0]
    expected = first.clone()
    first.zero_()
    torch.testing.assert_close(cached.encode([frame], [detections])[0], expected)
    assert encoder.calls == 1
    reversed_rows = detections.select(torch.tensor([1, 0], dtype=torch.int64))
    torch.testing.assert_close(cached.encode([frame], [reversed_rows])[0], expected.flip(0))
    assert encoder.calls == 2
    assert cached.embedding_dim == 2
    assert cached.requirements == encoder.requirements


@pytest.mark.parametrize("change", ["source", "pixels", "boxes", "scores", "class_ids", "sample"])
def test_appearance_cache_invalidates_changed_inputs(tmp_path: Path, change: str) -> None:
    frame, detections = _sample(tmp_path)
    encoder = _Encoder()
    cached = _cached(tmp_path, encoder)
    cached.encode([frame], [detections])
    if change == "source":
        Image.new("RGB", (8, 6), (10, 20, 30)).save(tmp_path / "frame.png")
    elif change == "pixels":
        frame.image[0, 0, 0] += 1
    elif change == "boxes":
        detections.geometry.values[0, 0] += 0.1
    elif change == "scores":
        detections.scores[0] -= 0.1
    elif change == "class_ids":
        # Adjacent large int64 IDs must not collide through float serialization.
        detections.class_ids[0] += 1
    else:
        frame = replace(frame, sample_id="val:drive:1")
        detections = replace(detections, sample_id=frame.sample_id)
    cached.encode([frame], [detections])
    assert encoder.calls == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_sha256": "b" * 64},
        {"precision": "fp16"},
        {"device": "mps"},
        {"preprocessing": "custom"},
        {"options": (("batch_size", 4),)},
    ],
)
def test_appearance_cache_invalidates_full_encoder_spec(tmp_path: Path, changes: dict) -> None:
    frame, detections = _sample(tmp_path)
    encoder = _Encoder()
    spec = ReIDEncoderSpec(backend="fixture", artifact="weights.pt", artifact_sha256="a" * 64)
    CachedAppearanceEncoder(encoder, spec, tmp_path / "cache").encode([frame], [detections])
    CachedAppearanceEncoder(encoder, replace(spec, **changes), tmp_path / "cache").encode([frame], [detections])
    assert encoder.calls == 2


def test_appearance_cache_tracks_required_masks_and_recovers_corrupt_payloads(tmp_path: Path) -> None:
    frame, detections = _sample(tmp_path)
    encoder = _Encoder(masks=True)
    cached = _cached(tmp_path, encoder)
    with pytest.raises(ValueError, match="requires detection masks"):
        cached.encode([frame], [detections])
    detections = detections.with_masks(MaskBatch(torch.ones((2, 6, 8), dtype=torch.bool)))
    cached.encode([frame], [detections])
    detections.masks.values[0, 0, 0] = False
    cached.encode([frame], [detections])
    assert encoder.calls == 2
    for payload in (tmp_path / "cache").glob("*/values.npy"):
        payload.write_bytes(b"corrupt")
    cached.encode([frame], [detections])
    assert encoder.calls == 3


def test_appearance_cache_validates_alignment_provenance_and_encoder_outputs(tmp_path: Path, monkeypatch) -> None:
    frame, detections = _sample(tmp_path)
    encoder = _Encoder()
    cached = _cached(tmp_path, encoder)
    with pytest.raises(ValueError, match="artifact SHA-256"):
        _cached(tmp_path, encoder, artifact="weights.pt")
    with pytest.raises(ValueError, match="aligned"):
        cached.encode([frame], [])
    with pytest.raises(ValueError, match="same sample_id"):
        cached.encode([replace(frame, sample_id="other")], [detections])
    with pytest.raises(ValueError, match="local image source_uri"):
        cached.encode([replace(frame, source_uri=None)], [detections])
    with pytest.raises(ValueError, match="nonlocal scheme"):
        cached.encode([replace(frame, source_uri="https://example.org/frame.png")], [detections])
    monkeypatch.setattr(encoder, "encode", lambda *_: [torch.zeros((2, 3), dtype=torch.float32)])
    with pytest.raises(ValueError, match="2 columns"):
        cached.encode([frame], [detections])
    assert not list((tmp_path / "cache").glob("*/_SUCCESS"))
    empty = detections.select(torch.tensor([], dtype=torch.int64))
    assert cached.encode([replace(frame, source_uri=None)], [empty])[0].shape == (0, 2)
