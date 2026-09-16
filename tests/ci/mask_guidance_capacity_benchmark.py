"""Measure 64-object EdgeTAM capacity in fresh MPS processes at fixed FP16.

Run ``python -m tests.ci.mask_guidance_capacity_benchmark --checkpoint
models/edgetam.pt --output runs/edgetam-capacity/single.json``. Repeat with
``--workers 4 --batches 4 8`` to measure concurrent sequence workers.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import subprocess
import threading
import traceback
from pathlib import Path
from time import perf_counter, time
from typing import Any
from uuid import uuid4

import numpy as np
import psutil
import torch
from PIL import Image

from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
from boxmot.segmentors.propagation.model import build_edgetam_predictor
from tests.ci.mask_guidance_parity import _object_boxes, _parity_frame
from tests.ci.mask_guidance_smoke import _assert_bounded_history

_GIB = 1024**3


def _scene(index: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Tile 64 disjoint textured moving objects and scale to a 1080p frame."""
    image = np.empty((768, 512, 3), dtype=np.uint8)
    boxes = {}
    scale = np.array([1920 / 512, 1080 / 768] * 2)
    for track_id in range(64):
        x, y = (track_id % 8) * 64, (track_id // 8) * 96
        phase = index + track_id * 3
        image[y : y + 96, x : x + 64] = np.roll(_parity_frame(phase)[:, :64], track_id % 3, axis=2)
        boxes[track_id] = (_object_boxes(max(0, index - 1) + track_id * 3)[0] + [x, y, x, y]) * scale
    return np.asarray(Image.fromarray(image).resize((1920, 1080))), boxes


def _system_memory() -> dict[str, int]:
    """Record memory and raw psutil counters, which can include macOS page I/O."""
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "available_bytes": memory.available,
        "swap_used_bytes": swap.used,
        "swap_in_bytes": swap.sin,
        "swap_out_bytes": swap.sout,
    }


def _vm_swap_counters() -> dict[str, int]:
    """Read actual macOS swapping; psutil 5.9.5 sin/sout expose page I/O instead."""
    output = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True, text=True, timeout=3).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))
    return {
        "vm_swap_in_bytes": int(re.search(r"Swapins:\s+(\d+)", output).group(1)) * page_size,
        "vm_swap_out_bytes": int(re.search(r"Swapouts:\s+(\d+)", output).group(1)) * page_size,
    }


class _MemorySampler:
    """Sample live and driver allocations; MPS has no public peak-memory API."""

    def __init__(self) -> None:
        self.samples: list[list[float | int]] = []
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        """Append timestamp, driver bytes, live bytes and process RSS."""
        self.samples.append(
            [
                time(),
                torch.mps.driver_allocated_memory(),
                torch.mps.current_allocated_memory(),
                self.process.memory_info().rss,
            ]
        )

    def _run(self) -> None:
        while not self.stop_event.wait(0.02):
            self.sample()

    def start(self) -> None:
        self.sample()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self.sample()


def _worker(config: dict[str, Any], worker: int, barrier: Any) -> None:
    """Build one independent sequence model and time warmed recurrent inference."""
    output = Path(config["directory"]) / f"worker-{worker}.json"
    record: dict[str, Any] = {"worker": worker, "batch_size": config["batch"], "status": "failed"}
    sampler = None
    try:
        torch.set_num_threads(2)
        if not torch.backends.mps.is_available():
            raise RuntimeError("This benchmark requires host MPS access.")
        recommended = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(config["memory_limit_bytes"] / recommended)
        record["memory_limit_bytes"] = config["memory_limit_bytes"]
        record["system_before"] = _system_memory()
        sampler = _MemorySampler()
        sampler.start()
        start = perf_counter()
        model = build_edgetam_predictor(Path(config["checkpoint"]), torch.device("mps"), precision="fp16")
        propagator = EdgeTAMMaskPropagator(
            config["checkpoint"], device="mps", predictor=model, max_objects=64, batch_size=config["batch"]
        )
        torch.mps.synchronize()
        record["model_load_seconds"] = perf_counter() - start
        record["parameter_dtypes"] = sorted({str(parameter.dtype) for parameter in model.parameters()})
        timings, boundaries = [], []
        for index in range(config["warmup"] + config["frames"]):
            frame, boxes = _scene(index)
            if index == config["warmup"]:
                print(f"batch {config['batch']} worker {worker}: warm, waiting for peers", flush=True)
                barrier.wait(timeout=600)
            torch.mps.synchronize()
            wall_start, start = time(), perf_counter()
            current = propagator.propagate(index, frame, boxes if index else {}, {})
            torch.mps.synchronize()
            timings.append(perf_counter() - start)
            boundaries.append([wall_start, time()])
            sampler.sample()
            _assert_bounded_history(propagator)
            if index:
                assert len(propagator._objects) == 64 and set(current) == set(boxes)
                assert all(mask.dtype == torch.bool and mask.device.type == "mps" for mask in current.values())
            if index in (1, 8, 16, config["warmup"] + config["frames"] - 1):
                print(f"batch {config['batch']} worker {worker}: frame {index}, {timings[-1]:.3f} s", flush=True)
        sampler.stop()
        record["samples"] = sampler.samples
        sampler = None
        steady = np.array(timings[config["warmup"] :]) * 1000
        record.update(
            status="ok",
            frame_seconds=timings,
            steady_mean_ms=float(steady.mean()),
            steady_median_ms=float(np.median(steady)),
            steady_p90_ms=float(np.percentile(steady, 90)),
            timed_wall_start=boundaries[config["warmup"]][0],
            timed_wall_end=boundaries[-1][1],
            sampled_peak_driver_bytes=max(row[1] for row in record["samples"]),
            sampled_peak_live_bytes=max(row[2] for row in record["samples"]),
            sampled_peak_rss_bytes=max(row[3] for row in record["samples"]),
            final_driver_bytes=torch.mps.driver_allocated_memory(),
            final_live_bytes=torch.mps.current_allocated_memory(),
            resident_objects=len(propagator._objects),
            memory_dtypes=sorted(
                {
                    str(memory["maskmem_features"].dtype)
                    for histories in propagator._objects.values()
                    for history in histories.values()
                    for memory in history.values()
                    if memory.get("maskmem_features") is not None
                }
            ),
            system_after=_system_memory(),
        )
        # Diagnostic downloads and compression happen after timing and memory sampling.
        packed = np.stack([np.packbits(current[key].detach().cpu().numpy()) for key in sorted(current)])
        record["empty_final_masks"] = int(np.count_nonzero(~packed.any(axis=1)))
        np.savez_compressed(output.with_suffix(".npz"), masks=packed)
    except Exception as error:
        barrier.abort()
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
        if sampler is not None:
            sampler.stop()
            record["samples"] = sampler.samples
        print(f"batch {config['batch']} worker {worker}: {record['error']}", flush=True)
    finally:
        output.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Run candidate batches sequentially, with fresh processes for every model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=19)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--memory-limit-gib", type=float)
    args = parser.parse_args()
    if args.warmup < 19 or args.frames < 3 or args.workers < 1 or min(args.batches) < 1:
        parser.error("Require warmup >=19, frames >=3, positive workers and batches.")
    if args.memory_limit_gib is not None and (not np.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0):
        parser.error("The per-worker memory limit must be finite and positive.")
    if not torch.backends.mps.is_available():
        parser.error("This benchmark requires host MPS access.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory_limit = (
        int(args.memory_limit_gib * _GIB)
        if args.memory_limit_gib
        else int(min(psutil.virtual_memory().available * 0.70, torch.mps.recommended_max_memory() * 0.8) / args.workers)
    )
    results: dict[str, Any] = {
        "torch": torch.__version__,
        "psutil": psutil.__version__,
        "precision": "fp16",
        "device": "mps",
        "objects": 64,
        "image_size": [1080, 1920],
        "workers": args.workers,
        "warmup_frames": args.warmup,
        "timed_frames": args.frames,
        "per_worker_memory_limit_bytes": memory_limit,
        "recommended_gpu_bytes": torch.mps.recommended_max_memory(),
        "system_before": _system_memory(),
        "measurements": [],
        "memory_sampling_interval_ms": 20,
        "swap_counter_note": (
            "swap_in/out_bytes are raw psutil values; vm_swap_in/out_bytes are macOS Swapins/Swapouts."
        ),
    }
    context = mp.get_context("spawn")
    reference = None
    run_id = uuid4().hex[:8]
    for batch in args.batches:
        directory = args.output.parent / f"{args.output.stem}-{run_id}-batch-{batch}"
        directory.mkdir(exist_ok=True)
        barrier = context.Barrier(args.workers)
        config = {
            "checkpoint": str(args.checkpoint.resolve()),
            "directory": str(directory),
            "batch": batch,
            "warmup": args.warmup,
            "frames": args.frames,
            "memory_limit_bytes": memory_limit,
        }
        processes = [context.Process(target=_worker, args=(config, worker, barrier)) for worker in range(args.workers)]
        before = {**_system_memory(), **_vm_swap_counters()}
        for process in processes:
            process.start()
        system_samples = [before]
        low_memory_since = None
        abort_reason = None
        last_vm_sample = time()
        vm_counters = {key: value for key, value in before.items() if key.startswith("vm_")}
        while any(process.is_alive() for process in processes):
            memory = _system_memory()
            if time() - last_vm_sample >= 1:
                vm_counters = _vm_swap_counters()
                last_vm_sample = time()
            system_samples.append(memory)
            if memory["available_bytes"] < 2 * _GIB:
                low_memory_since = time() if low_memory_since is None else low_memory_since
                if time() - low_memory_since > 2:
                    abort_reason = "Available system memory remained below 2 GiB for two seconds."
            else:
                low_memory_since = None
            if vm_counters["vm_swap_out_bytes"] - before["vm_swap_out_bytes"] > _GIB:
                abort_reason = "System swap-outs increased by more than 1 GiB during this candidate."
            if memory["swap_used_bytes"] - before["swap_used_bytes"] > _GIB:
                abort_reason = "System swap usage increased by more than 1 GiB during this candidate."
            if abort_reason:
                barrier.abort()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
            elif any(process.exitcode not in (None, 0) for process in processes):
                barrier.abort()
            for process in processes:
                process.join(timeout=0.05)
        records = []
        for worker, process in enumerate(processes):
            path = directory / f"worker-{worker}.json"
            records.append(
                json.loads(path.read_text())
                if path.exists() and process.exitcode == 0
                else {"worker": worker, "status": "failed", "error": f"exit code {process.exitcode} without results"}
            )
        result: dict[str, Any] = {
            "batch_size": batch,
            "artifact_directory": str(directory),
            "system_before": before,
            "system_after": {**_system_memory(), **_vm_swap_counters()},
            "minimum_system_available_bytes": min(sample["available_bytes"] for sample in system_samples),
            "abort_reason": abort_reason,
            "workers": [{key: value for key, value in record.items() if key != "samples"} for record in records],
        }
        if all(record["status"] == "ok" for record in records):
            elapsed = max(record["timed_wall_end"] for record in records) - min(
                record["timed_wall_start"] for record in records
            )
            result["aggregate_fps"] = args.workers * args.frames / elapsed
            result["sum_per_worker_peak_driver_bytes"] = sum(record["sampled_peak_driver_bytes"] for record in records)
            packed = np.load(directory / "worker-0.npz")["masks"]
            if reference is None:
                reference = packed
            result["final_changed_pixels_vs_first_success"] = int(np.unpackbits(packed ^ reference).sum())
            result["final_changed_pixels_between_workers"] = [
                int(np.unpackbits(np.load(directory / f"worker-{worker}.npz")["masks"] ^ packed).sum())
                for worker in range(args.workers)
            ]
        results["measurements"].append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        summary = {
            "batch_size": batch,
            "aggregate_fps": result.get("aggregate_fps"),
            "abort_reason": abort_reason,
            "workers": [
                {
                    key: record[key]
                    for key in ("worker", "status", "steady_median_ms", "sampled_peak_driver_bytes", "error")
                    if key in record
                }
                for record in records
            ],
        }
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
