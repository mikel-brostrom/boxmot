"""Measure CUDA inference throughput to select a practical batch size."""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from boxmot.utils.devices import normalize_device, resolve_device


@dataclass(frozen=True)
class BatchMeasurement:
    """One feasible batch's median latency and incremental peak tensor memory."""

    batch_size: int
    latency_s: float
    items_per_second: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class BatchProfile:
    """Recommended batch size and the feasible measurements used to select it."""

    batch_size: int
    measurements: tuple[BatchMeasurement, ...]


def validate_batch_options(*, max_batch_size: int, memory_fraction: float, warmup: int, repeats: int) -> None:
    """Validate profiling settings before allocating models or CUDA inputs."""

    for name, value, minimum in (
        ("max_batch_size", max_batch_size, 1),
        ("warmup", warmup, 0),
        ("repeats", repeats, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if (
        isinstance(memory_fraction, bool)
        or not isinstance(memory_fraction, (float, int))
        or not math.isfinite(memory_fraction)
        or not 0 < memory_fraction <= 1
    ):
        raise ValueError("memory_fraction must be greater than 0 and at most 1.")


def _available_memory(device: torch.device, resident_bytes: int) -> int:
    """Account for reusable allocator memory and changes in other GPU users."""

    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Reserved memory includes both live tensors and reusable cached blocks.
    # Subtract only the tensors present before profiling, so retained model
    # workspaces count toward each candidate's incremental memory budget.
    return max(0, free_bytes + torch.cuda.memory_reserved(device) - resident_bytes)


def _measure_batch(
    run_batch: Callable[[int], object],
    batch_size: int,
    device: torch.device,
    resident_bytes: int,
    warmup: int,
    repeats: int,
) -> BatchMeasurement:
    """Measure complete, synchronized calls without retaining their outputs."""

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        run_batch(batch_size)
    torch.cuda.synchronize(device)

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        run_batch(batch_size)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    latency = statistics.median(durations)
    if latency <= 0:
        raise RuntimeError("CUDA batch profiling measured a non-positive inference latency.")
    return BatchMeasurement(
        batch_size=batch_size,
        latency_s=latency,
        items_per_second=batch_size / latency,
        peak_memory_bytes=max(0, torch.cuda.max_memory_allocated(device) - resident_bytes),
    )


def profile_batch_size(
    run_batch: Callable[[int], object],
    device: str | torch.device,
    *,
    max_batch_size: int = 256,
    memory_fraction: float = 0.8,
    warmup: int = 1,
    repeats: int = 3,
) -> BatchProfile:
    """Benchmark a loaded model's CUDA inference callback on increasing batches.

    ``run_batch`` must execute one actual batch of the requested size and must
    not retain its outputs. Load models before calling this function. Latency
    includes everything in the callback, including preprocessing and transfers.

    Candidates are powers of two plus ``max_batch_size``. Stop at the first CUDA
    out-of-memory error or candidate exceeding ``memory_fraction`` of available
    memory. Existing tensors are excluded from the incremental peak, while
    allocator caches and changes in free GPU memory are accounted for. Select
    the smallest feasible candidate within 5% of the best measured throughput.
    The result applies to the profiled inputs, precision, and current GPU load.
    """

    validate_batch_options(
        max_batch_size=max_batch_size, memory_fraction=memory_fraction, warmup=warmup, repeats=repeats
    )
    normalized = normalize_device(device)
    if not normalized.startswith("cuda:"):
        raise ValueError("Automatic batch profiling requires a CUDA device.")
    selected_device = resolve_device(normalized)
    measurements: list[BatchMeasurement] = []
    stop_reason = "memory headroom"

    with torch.cuda.device(selected_device), torch.inference_mode():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(selected_device)
        resident_bytes = torch.cuda.memory_allocated(selected_device)
        initial_available = _available_memory(selected_device, resident_bytes)
        batch_size = 1
        try:
            while True:
                available = min(initial_available, _available_memory(selected_device, resident_bytes))
                if available == 0:
                    break
                out_of_memory = False
                try:
                    measurement = _measure_batch(
                        run_batch, batch_size, selected_device, resident_bytes, warmup, repeats
                    )
                except torch.cuda.OutOfMemoryError:
                    out_of_memory = True
                # Leave the exception handler before collecting so the failed
                # callback's traceback cannot keep its CUDA tensors alive.
                if out_of_memory:
                    stop_reason = "CUDA out of memory"
                    break
                available = min(available, _available_memory(selected_device, resident_bytes))
                if measurement.peak_memory_bytes > available * memory_fraction:
                    break
                measurements.append(measurement)
                if batch_size == max_batch_size:
                    break
                batch_size = min(batch_size * 2, max_batch_size)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if not measurements:
        raise RuntimeError(
            f"Batch size 1 could not fit on {selected_device} within the profiling memory budget "
            f"({memory_fraction:.0%} of available memory; stopped by {stop_reason}). "
            "Reduce the input size or free GPU memory before profiling."
        )
    threshold = max(item.items_per_second for item in measurements) * 0.95
    selected = next(item for item in measurements if item.items_per_second >= threshold)
    return BatchProfile(batch_size=selected.batch_size, measurements=tuple(measurements))


__all__ = ("BatchMeasurement", "BatchProfile", "profile_batch_size", "validate_batch_options")
