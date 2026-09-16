"""Compare EdgeTAM precision and object batches on a bounded moving fixture.

Run ``python -m tests.ci.mask_guidance_batch_benchmark --checkpoint models/edgetam.pt
--device mps --output runs/edgetam-benchmark.json`` with the optional real model.
This measures propagation, not MOT17 tracking accuracy or a full tuning trial.
"""

from __future__ import annotations

import argparse
import gc
import json
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
from boxmot.segmentors.propagation.model import build_edgetam_predictor, inference_context
from tests.ci.mask_guidance_parity import _object_boxes, _parity_frame
from tests.ci.mask_guidance_smoke import _assert_bounded_history, _synchronize


def _scene(index: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Tile eight separated moving objects with their previous-frame prompts."""
    image = np.empty((192, 256, 3), dtype=np.uint8)
    boxes = {}
    for track_id in range(8):
        x, y = (track_id % 4) * 64, (track_id // 4) * 96
        phase = index + track_id * 3
        image[y : y + 96, x : x + 64] = np.roll(_parity_frame(phase)[:, :64], track_id % 3, axis=2)
        boxes[track_id] = _object_boxes(max(0, index - 1) + track_id * 3)[0] + [x, y, x, y]
    return image, boxes


def _run(checkpoint: Path, device: torch.device, frames: int, precision: str, batch: int) -> tuple:
    """Measure synchronized propagation after spatial and pointer memory warmup."""
    model = build_edgetam_predictor(checkpoint, device, precision=precision)
    propagator = EdgeTAMMaskPropagator(checkpoint, device=device, predictor=model, batch_size=batch)
    timings, masks = [], []
    context = partial(inference_context, precision=precision)
    with patch("boxmot.segmentors.propagation.edgetam.inference_context", context):
        for index in range(frames):
            image, boxes = _scene(index)
            _synchronize(device)
            start = perf_counter()
            current = propagator.propagate(index, image, boxes if index else {}, {})
            _synchronize(device)
            timings.append(perf_counter() - start)
            _assert_bounded_history(propagator)
            if index:
                assert set(current) == set(boxes)
                # Diagnostic comparison reads full masks only after timed inference.
                masks.append(torch.stack([current[track_id] for track_id in sorted(current)]).detach().cpu().numpy())
    memory_dtypes = {
        str(memory["maskmem_features"].dtype)
        for histories in propagator._objects.values()
        for history in histories.values()
        for memory in history.values()
        if memory.get("maskmem_features") is not None
    }
    result = {
        "precision": precision,
        "batch_size": batch,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "memory_dtypes": sorted(memory_dtypes),
        "frame_seconds": timings,
        "steady_ms_per_frame": float(np.mean(timings[16:]) * 1000),
        "steady_fps": float(1 / np.mean(timings[16:])),
        "diagnostic_mask_transfers_timed": False,
    }
    propagator.reset()
    del propagator, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result, np.stack(masks)


def _compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    """Report binary mask drift, including small objects rather than only background."""
    intersection = (reference & actual).sum(axis=(-2, -1))
    union = (reference | actual).sum(axis=(-2, -1))
    iou = np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union != 0)
    changed = np.count_nonzero(reference != actual)
    return {
        "changed_pixels": int(changed),
        "changed_pixel_fraction": float(changed / reference.size),
        "mean_mask_iou": float(iou.mean()),
        "minimum_mask_iou": float(iou.min()),
        "empty_reference_masks": int(np.count_nonzero(~reference.any(axis=(-2, -1)))),
        "empty_actual_masks": int(np.count_nonzero(~actual.any(axis=(-2, -1)))),
    }


def main() -> None:
    """Run each configuration sequentially and write reproducible measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 20:
        parser.error("Use at least 20 frames to measure after spatial and pointer memory warmup.")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    records, masks = {}, {}
    for precision, batch in (("fp32", 1), ("fp16", 1), ("fp16", 4)):
        name = f"{precision}_batch{batch}"
        records[name], masks[name] = _run(args.checkpoint, device, args.frames, precision, batch)
        print(name, json.dumps(records[name]), flush=True)
    results = {
        "torch": torch.__version__,
        "device": str(device),
        "frames": args.frames,
        "objects": 8,
        "steady_frame_indices": [16, args.frames - 1],
        "measurements": records,
        "fp16_vs_fp32": _compare(masks["fp32_batch1"], masks["fp16_batch1"]),
        "batch4_vs_singleton_fp16": _compare(masks["fp16_batch1"], masks["fp16_batch4"]),
        "combined_speedup": records["fp32_batch1"]["steady_ms_per_frame"]
        / records["fp16_batch4"]["steady_ms_per_frame"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
