"""Compare full-mask downloads with resident-tensor association statistics.

Run ``python -m tests.ci.mask_guidance_transfer_benchmark --device mps
--dense --output runs/mask-transfer-benchmark.json``. No model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from boxmot.trackers.common.association.masks import apply_mask_guidance
from tests.ci.mask_guidance_smoke import _synchronize


def _fixture(device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """Place 64 separate rectangles on a 1080p frame before any timed work."""
    masks = np.zeros((64, 1080, 1920), dtype=np.uint8)
    boxes = []
    for index in range(64):
        x, y = (index % 8) * 240 + 40, (index // 8) * 135 + 12
        masks[index, y : y + 96, x : x + 64] = 1
        boxes.append([x, y, x + 100, y + 100])
    return torch.from_numpy(masks).to(device=device, dtype=torch.bool), np.asarray(boxes, dtype=float)


def _measure(operation, device: torch.device) -> tuple[dict, np.ndarray]:
    """Warm twice, then time five synchronized calls and their CPU transfers."""
    for _ in range(2):
        operation()
    timings, transfers = [], []
    original_cpu = torch.Tensor.cpu
    for _ in range(5):
        calls = []

        def cpu(tensor, *args, **kwargs):
            calls.append(
                {
                    "bytes": tensor.numel() * tensor.element_size(),
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                }
            )
            return original_cpu(tensor, *args, **kwargs)

        _synchronize(device)
        with patch.object(torch.Tensor, "cpu", cpu):
            start = perf_counter()
            result = operation()
            _synchronize(device)
            timings.append((perf_counter() - start) * 1000)
        transfers.append(calls)
    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "samples_ms": timings,
        "transfer_calls": [len(calls) for calls in transfers],
        "transfer_bytes": [sum(call["bytes"] for call in calls) for calls in transfers],
        "transfers": transfers[0],
    }, result


def main() -> None:
    """Measure sparse and clear stages, optionally adding a dense 16-object case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    resident, boxes = _fixture(device)
    records = {}
    for name, count in [("sparse", 64), ("clear", 64)] + ([("dense", 16)] if args.dense else []):
        costs = np.full((count, count), 0.4 if name == "dense" else 0.95)
        np.fill_diagonal(costs, 0.4)
        if name == "sparse":
            costs[np.arange(count), (np.arange(count) + 1) % count] = 0.4
        tensors = list(resident[:count].unbind())

        def downloaded():
            masks = [mask for batch in resident[:count].split(4) for mask in batch.cpu().numpy()]
            return apply_mask_guidance(costs, boxes[:count], masks, threshold=0.5)

        def statistics():
            return apply_mask_guidance(costs, boxes[:count], tensors, threshold=0.5)

        old, expected = _measure(downloaded, device)
        new, actual = _measure(statistics, device)
        np.testing.assert_array_equal(actual, expected)
        records[name] = {
            "objects": count,
            "download": old,
            "statistics": new,
            "costs_equal": True,
            "speedup": old["mean_ms"] / new["mean_ms"],
        }
        print(name, json.dumps(records[name]), flush=True)
    output = {
        "torch": torch.__version__,
        "device": str(device),
        "image_size": [1080, 1920],
        "warmup_calls": 2,
        "timed_calls": 5,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
