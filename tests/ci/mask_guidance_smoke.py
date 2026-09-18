"""Exercise real EdgeTAM propagation, shared segmentation, and sequence reset.

Run explicitly with a downloaded checkpoint; ordinary unit tests need no model:
``python -m tests.ci.mask_guidance_smoke --checkpoint /path/to/edgetam.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from boxmot import ByteTrack
from boxmot.components.artifacts import sha256_artifact
from boxmot.segmentors import SegmentorSpec, create_segmentor
from boxmot.segmentors.propagation.model import effective_precision
from boxmot.segmentors.propagation.weights import resolve_edgetam_checkpoint
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from boxmot.trackers import MaskGuidanceConfig
from boxmot.utils.devices import resolve_device


def _arriving_frame(index: int) -> np.ndarray:
    """Create this frame on demand without storing or decoding a source video."""
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[8:80, 8:40] = (40, 200, 240)
    image[16:72, 72:112] = (200, 100, 40)
    image[0, 0] = (index % 256, 0, 0)
    return image


def _synchronize(device: torch.device) -> None:
    """Include asynchronous accelerator work in measured update latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _assert_bounded_history(propagator) -> None:
    """Check all retained fields against the official forward attention windows."""
    state = propagator._state
    assert len(state["images"]) <= 2
    assert len(state["cached_features"]) <= 1
    assert len(propagator._objects) <= propagator.max_objects
    assert set(propagator._masks) == set(propagator._objects)
    for mask in propagator._masks.values():
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool and mask.device == propagator._mean.device
        assert tuple(mask.shape) == propagator.frame_shape
    for outputs in propagator._objects.values():
        conditioned = outputs["cond_frame_outputs"]
        recent = outputs["non_cond_frame_outputs"]
        assert len(conditioned) == 1
        assert len(recent) <= 15
        assert sum("maskmem_features" in output for output in recent.values()) <= 6
        for output in (*conditioned.values(), *recent.values()):
            assert set(output) <= {"maskmem_features", "maskmem_pos_enc", "obj_ptr"}
        assert all("maskmem_features" in output for output in conditioned.values())


def _state_signature(value: Any) -> Any:
    """Record temporal storage identity and contents without keeping tensor copies."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().reshape(-1).view(torch.uint8).cpu().numpy()
        return id(value), tuple(value.shape), str(value.dtype), hashlib.sha256(raw).digest()
    if isinstance(value, np.ndarray):
        return id(value), value.shape, str(value.dtype), hashlib.sha256(value).digest()
    if isinstance(value, dict):
        return id(value), tuple((key, _state_signature(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return id(value), tuple(_state_signature(item) for item in value)
    return value


def _assert_shared_segmentation(propagator, checkpoint: Path, frame: np.ndarray, rows: np.ndarray) -> None:
    """Use one loaded official model without altering its established video state."""
    attributes = ("_state", "_objects", "_masks", "_last_observed", "last_frame_index", "frame_shape")
    before = tuple(_state_signature(getattr(propagator, name)) for name in attributes)
    canonical_frame = Frame(
        torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1).contiguous(), "shared-segmentation"
    )
    detections = Detections(
        Boxes(torch.from_numpy(rows[:, :4].copy())),
        scores=torch.from_numpy(rows[:, 4].copy()),
        class_ids=torch.from_numpy(rows[:, 5].astype(np.int64)),
        sample_id=canonical_frame.sample_id,
    )
    spec = SegmentorSpec(
        backend="edgetam",
        artifact=str(checkpoint),
        artifact_sha256=sha256_artifact(checkpoint),
        device=str(propagator.device),
        precision=effective_precision(propagator.device),
        geometry_mode="aabb",
    )
    segmentor = create_segmentor(spec, model=propagator.predictor)
    assert segmentor.model is propagator.predictor
    masks = segmentor.segment([canonical_frame], [detections])[0]
    assert isinstance(masks, MaskBatch)
    assert masks.values.shape == (len(rows), *frame.shape[:2])
    assert masks.values.dtype == torch.bool and masks.values.device.type == "cpu"
    assert masks.values.is_contiguous()
    assert all(mask.any() for mask in masks.values)
    for index, row in enumerate(rows):
        x1, y1, x2, y2 = row[:4].astype(int)
        inside = masks.values[index, y1:y2, x1:x2].sum()
        assert inside > masks.values[index].sum() / 2, "Mask is not aligned with its detection box"
    assert segmentor._predictor._features is None
    assert segmentor._predictor._orig_hw is None
    assert not segmentor._predictor._is_image_set
    after = tuple(_state_signature(getattr(propagator, name)) for name in attributes)
    assert after == before, "Standalone segmentation changed temporal propagation state"


def main() -> None:
    """Process arriving frames twice and report measured tracker/EdgeTAM speed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=20, help="Frames per pass; at least four (default: 20).")
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("--frames must be at least four to exercise delayed identity confirmation")
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    checkpoint = resolve_edgetam_checkpoint(args.checkpoint)
    detections = np.array([[8, 8, 40, 80, 0.95, 0], [72, 16, 112, 72, 0.95, 0]], dtype=np.float32)
    tracker = ByteTrack(mask_guidance=MaskGuidanceConfig(checkpoint=checkpoint, device=str(device)))
    runs: list[list[np.ndarray]] = []
    mask_runs: list[list[dict[int, np.ndarray]]] = []
    latencies: list[float] = []
    for pass_index in range(2):
        outputs = []
        temporal_masks = []
        for index in range(args.frames):
            frame = _arriving_frame(index)
            _synchronize(device)
            started = perf_counter()
            output = tracker.update(detections[:1] if index == 0 else detections, frame)
            _synchronize(device)
            if index > 0:
                # The first update initializes the model; report processing
                # latency separately from that one-time startup expense.
                latencies.append(perf_counter() - started)
            assert output.dtype == np.float64 and output.flags.c_contiguous
            assert output.shape == (1 if index < 2 else 2, 8)
            outputs.append(output.copy())
            guidance = tracker._mask_guidance
            assert guidance is not None
            _assert_bounded_history(guidance._propagator)
            if pass_index == 0 and index == 2:
                # Interleave image inference after temporal memory exists, then
                # continue streaming. The second pass is an uninterrupted oracle.
                _assert_shared_segmentation(guidance._propagator, checkpoint, frame, detections)
            # Exercise the public lazy rendering view outside tracker timing.
            public_masks = guidance.masks
            temporal_masks.append({track_id: mask.copy() for track_id, mask in public_masks.items()})
            if index >= 3:
                assert set(public_masks) == {0, 1}
                assert all(mask.shape == frame.shape[:2] and mask.dtype == bool for mask in public_masks.values())
                assert all(not mask.flags.writeable for mask in public_masks.values())
                assert all(mask.any() for mask in public_masks.values())
        runs.append(outputs)
        mask_runs.append(temporal_masks)
        tracker.reset()
        assert guidance._masks == {}
    for first, second in zip(*runs):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(*mask_runs):
        assert first.keys() == second.keys()
        for track_id in first:
            np.testing.assert_array_equal(first[track_id], second[track_id])
    elapsed = sum(latencies)
    print(
        "Streaming EdgeTAM smoke passed: propagation, shared standalone segmentation, "
        "new identities, bounded history, and exact reset parity."
    )
    print(
        f"Tracking + EdgeTAM on {device}: {len(latencies) / elapsed:.2f} FPS, "
        f"{1000 * elapsed / len(latencies):.1f} ms/frame across {len(latencies)} timed updates "
        "(model initialization, standalone segmentation, rendering mask transfers, and detector inference excluded)."
    )


if __name__ == "__main__":
    main()
