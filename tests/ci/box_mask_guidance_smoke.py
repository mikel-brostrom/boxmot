"""Run real EdgeTAM propagation through every supported Python box tracker.

Invoke explicitly with an installed mask-guidance dependency group and weights:
``python -m tests.ci.box_mask_guidance_smoke --checkpoint models/edgetam.pt --device cpu``.
The nine trackers reuse one explicitly constructed official model, with fresh
temporal state for each tracker and two identical passes separated by reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from boxmot.components.artifacts import sha256_artifact
from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
from boxmot.segmentors.propagation.model import EDGETAM_REVISION, build_edgetam_predictor, effective_precision
from boxmot.segmentors.propagation.weights import resolve_edgetam_checkpoint
from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig, create_tracker
from boxmot.trackers.common.protocols import Tracker
from boxmot.utils.devices import resolve_device
from tests.ci.mask_guidance_smoke import _arriving_frame, _assert_bounded_history, _synchronize

TRACKERS = (
    "boosttrack", "botsort", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "sfsort", "strongsort"
)
FRAME_COUNT = 7
EMPTY_FRAME = 4
BOXES = np.array([[8, 8, 40, 80, 0.95, 0], [72, 16, 112, 72, 0.95, 0]], dtype=np.float32)


def _create_tracker(name: str, guidance: MaskGuidance) -> Tracker:
    """Use stable confirmation and supplied appearance features without downloads."""
    options: dict[str, Any] = {"asso_func": "iou", "min_hits": 1, "max_age": 5}
    if name in {"boosttrack", "botsort", "occluboost"}:
        options.update(use_embeddings=True, use_cmc=False)
    if name in {"botsort", "bytetrack"}:
        options["track_buffer"] = 5
    if name == "deepocsort":
        options.update(use_embeddings=True, cmc_off=True)
    if name == "hybridsort":
        options.update(use_embeddings=True, cmc_method=None)
    if name == "strongsort":
        options["n_init"] = 1
    if name == "sfsort":
        options.update(frame_width=128, frame_height=96, central_timeout=5, marginal_timeout=5)
    return create_tracker(name, mask_guidance=guidance, **options)


def _inputs(tracker: Tracker, index: int, *, sequence: str) -> tuple[Detections, Frame]:
    """Deliver distinct appearance features and correctly ordered source pixels."""
    rows = BOXES[:0] if index == EMPTY_FRAME else BOXES[:1] if index == 0 else BOXES
    sample_id = f"{sequence}/{index}"
    pixels = _arriving_frame(index)
    frame = Frame(
        torch.from_numpy(pixels[..., ::-1].copy()).permute(2, 0, 1).contiguous(),
        sample_id=sample_id,
        sequence_id=sequence,
        frame_index=index,
    )
    detections = Detections(
        Boxes(torch.from_numpy(rows[:, :4].copy())),
        scores=torch.from_numpy(rows[:, 4].copy()),
        class_ids=torch.from_numpy(rows[:, 5].astype(np.int64)),
        embeddings=torch.eye(2, 4)[: len(rows)].contiguous() if tracker.requirements.embeddings else None,
        sample_id=sample_id,
    )
    return detections, frame


def _assert_reset(propagator: EdgeTAMMaskPropagator) -> None:
    """Ensure reused model weights carry no tracker-specific temporal state."""
    assert propagator._state is None
    assert propagator._objects == {}
    assert propagator._masks == {}
    assert propagator._last_observed == {}
    assert propagator.last_frame_index == -1
    assert propagator.frame_shape is None


def _run_tracker(name: str, checkpoint: Path, device: torch.device, predictor: Any) -> dict[str, Any]:
    """Check canonical outputs, mask bounds, empty frames, recovery, and reset parity."""
    propagator = EdgeTAMMaskPropagator(checkpoint, device=device, predictor=predictor, max_objects=2)
    _assert_reset(propagator)
    assert propagator.predictor is predictor
    guidance = MaskGuidance(MaskGuidanceConfig(checkpoint, device=str(device), max_objects=2), propagator=propagator)
    tracker = _create_tracker(name, guidance)
    assert tracker.requirements.frame
    output_passes = []
    mask_passes = []
    latencies = []
    for run in range(2):
        outputs = []
        mask_digests = []
        confirmed_ids = None
        for index in range(FRAME_COUNT):
            detections, frame = _inputs(tracker, index, sequence=f"{name}-{run}")
            _synchronize(device)
            started = perf_counter()
            result = tracker.update(detections, frame)
            _synchronize(device)
            latencies.append(perf_counter() - started)
            assert isinstance(result, Tracks)
            result.validate()
            assert isinstance(result.geometry, Boxes)
            assert result.masks is None, "Guidance masks must remain auxiliary to box tracking output"
            assert result.sample_id == detections.sample_id
            assert propagator.last_frame_index == index
            _assert_bounded_history(propagator)
            outputs.append(result.to_aabb_rows().numpy().copy())
            assert len(guidance._masks) <= 2
            mask_digests.append({key: hashlib.sha256(mask).hexdigest() for key, mask in guidance._masks.items()})
            for mask in guidance._masks.values():
                assert mask.shape == (96, 128) and mask.dtype == np.bool_
                assert mask.any(), f"{name}: expected visible synthetic object mask at frame {index}"
                assert not mask.flags.writeable
            if index == 0:
                assert guidance._masks == {}, "New tracker or reset inherited previous temporal masks"
            if index == EMPTY_FRAME - 1:
                confirmed_ids = set(result.track_ids.tolist())
                assert len(confirmed_ids) == 2
                assert set(guidance._masks) == confirmed_ids
            if index == EMPTY_FRAME:
                assert len(result) == 0
                assert set(guidance._masks) == confirmed_ids, "Empty frames must advance existing masks"
            if index == FRAME_COUNT - 1:
                assert set(result.track_ids.tolist()) == confirmed_ids, "Clear recovery changed confirmed identities"
                assert set(guidance._masks) == confirmed_ids
        output_passes.append(outputs)
        mask_passes.append(mask_digests)
        tracker.reset()
        _assert_reset(propagator)
        assert guidance._masks == {}
    for first, second in zip(*output_passes, strict=True):
        np.testing.assert_array_equal(first, second)
    assert mask_passes[0] == mask_passes[1], f"{name}: reset changed propagation for identical pixels and prompts"
    result = {"tracker": name, "updates": len(latencies), "mean_update_ms": 1000 * float(np.mean(latencies))}
    print(f"{name}: passed {len(latencies)} updates, {result['mean_update_ms']:.1f} ms/update", flush=True)
    return result


def main() -> None:
    """Load one model and report the nine independent tracker smoke checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON validation report.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    checkpoint = resolve_edgetam_checkpoint(args.checkpoint)
    predictor = build_edgetam_predictor(checkpoint, device)
    results = [_run_tracker(name, checkpoint, device, predictor) for name in TRACKERS]
    summary = {
        "device": str(device),
        "precision": effective_precision(device),
        "edgetam_revision": EDGETAM_REVISION,
        "checkpoint_sha256": sha256_artifact(checkpoint),
        "max_objects": 2,
        "model_loads": 1,
        "frames_per_pass": FRAME_COUNT,
        "passes_per_tracker": 2,
        "latency_scope": "Tracker plus propagation only; excludes model load, detector, and ReID inference.",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("All nine Python box trackers passed real EdgeTAM propagation, recovery, bounds, and exact reset parity.")


if __name__ == "__main__":
    main()
