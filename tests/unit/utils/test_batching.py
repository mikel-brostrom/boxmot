"""CUDA batch profiling regressions using a deterministic allocator and clock."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from boxmot.utils import batching


class FakeCuda:
    """Simulate live, cached, and externally occupied memory on one device."""

    def __init__(self) -> None:
        self.device = torch.device("cuda:1")
        self.active_device: torch.device | None = None
        self.resident = 1_000
        self.capacity = 10_000
        self.cache = 0
        self.pinned_cache = 0
        self.peak = self.resident
        self.clock = 0.0
        self.sync_delay = 0.0
        self.calls: list[int] = []
        self.empty_calls = 0
        self.sync_calls = 0
        self.output_ref: weakref.ReferenceType | None = None
        self.failure_ref: weakref.ReferenceType | None = None

    @contextmanager
    def context(self, device: torch.device) -> Iterator[None]:
        """Check all allocator operations happen on the requested device."""

        assert device == self.device
        self.active_device = device
        try:
            yield
        finally:
            self.active_device = None

    def empty_cache(self) -> None:
        """Release cached memory after callback outputs and errors are freed."""

        assert self.active_device == self.device
        assert self.failure_ref is None or self.failure_ref() is None
        assert self.output_ref is None or self.output_ref() is None
        self.cache = self.pinned_cache
        self.empty_calls += 1

    def synchronize(self, device: torch.device) -> None:
        """Advance the fake clock for work completed during synchronization."""

        assert device == self.device == self.active_device
        self.clock += self.sync_delay
        self.sync_calls += 1

    def reset_peak(self, device: torch.device) -> None:
        """Reset the candidate's allocation peak to its model's resident size."""

        assert device == self.device
        self.peak = self.resident

    def run(self, batch_size: int, *, duration: float, memory: int = 100) -> None:
        """Simulate a full inference invocation and its allocator activity."""

        assert self.active_device == self.device
        assert torch.is_inference_mode_enabled()
        self.calls.append(batch_size)
        self.clock += duration
        self.cache = max(self.cache, memory)
        self.peak = max(self.peak, self.resident + memory)


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    """Install CUDA mocks without initializing accelerator hardware."""

    fake = FakeCuda()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", fake.context)
    monkeypatch.setattr(torch.cuda, "synchronize", fake.synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake.empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake.reset_peak)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: fake.resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: fake.resident + fake.cache)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: fake.peak)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (fake.capacity - fake.cache, 1_000_000))
    monkeypatch.setattr(batching.time, "perf_counter", lambda: fake.clock)
    return fake


def test_selects_smallest_batch_with_near_best_throughput(cuda: FakeCuda) -> None:
    """A throughput plateau should prefer a smaller, lower-memory batch."""

    throughput = {1: 10.0, 2: 20.0, 4: 30.0, 8: 31.0, 10: 29.0}

    def run(batch_size: int) -> None:
        cuda.run(batch_size, duration=batch_size / throughput[batch_size], memory=batch_size * 100)

    result = batching.profile_batch_size(run, "1", max_batch_size=10, warmup=2, repeats=3)

    assert result.batch_size == 4
    assert [item.batch_size for item in result.measurements] == [1, 2, 4, 8, 10]
    assert [item.items_per_second for item in result.measurements] == pytest.approx(list(throughput.values()))
    assert result.measurements[-1].peak_memory_bytes == 1_000
    assert cuda.calls == [batch for batch in throughput for _ in range(5)]
    assert cuda.empty_calls == 2
    assert cuda.active_device is None


def test_latency_uses_median_and_includes_gpu_synchronization(cuda: FakeCuda) -> None:
    """The wall clock must include outstanding GPU work but resist outliers."""

    cuda.sync_delay = 0.01
    durations = iter([10.0, 0.1, 3.0, 0.2])

    def run(batch_size: int) -> None:
        cuda.run(batch_size, duration=next(durations))

    result = batching.profile_batch_size(run, cuda.device, max_batch_size=1, warmup=1, repeats=3)

    assert result.measurements[0].latency_s == pytest.approx(0.21)
    assert result.measurements[0].items_per_second == pytest.approx(1 / 0.21)
    assert cuda.sync_calls == 5


def test_memory_budget_counts_available_cache_but_excludes_resident_model(cuda: FakeCuda) -> None:
    """Model weights and cached blocks must not make a feasible batch fail."""

    cuda.resident = 5_000
    cuda.capacity = 500
    cuda.pinned_cache = 100

    def run(batch_size: int) -> None:
        cuda.run(batch_size, duration=0.1, memory=batch_size * 200)

    result = batching.profile_batch_size(run, cuda.device, max_batch_size=4, warmup=0, repeats=1)

    assert result.batch_size == 2
    assert [item.peak_memory_bytes for item in result.measurements] == [200, 400]
    assert cuda.calls == [1, 2, 4]


def test_memory_budget_accounts_for_other_gpu_users(cuda: FakeCuda) -> None:
    """Memory claimed elsewhere during profiling tightens the live budget."""

    def run(batch_size: int) -> None:
        cuda.run(batch_size, duration=0.1, memory=batch_size * 100)
        if batch_size == 2:
            cuda.capacity = 200

    result = batching.profile_batch_size(run, cuda.device, max_batch_size=8, warmup=0, repeats=1)

    assert result.batch_size == 1
    assert len(result.measurements) == 1
    assert cuda.calls == [1, 2]


def test_oom_recovers_after_releasing_failed_callback_traceback(cuda: FakeCuda) -> None:
    """The exception's frames must not keep temporary tensors live at cleanup."""

    class TemporaryTensor:
        pass

    def run(batch_size: int) -> object:
        cuda.run(batch_size, duration=0.1)
        temporary = TemporaryTensor()
        if batch_size == 4:
            cuda.failure_ref = weakref.ref(temporary)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        cuda.output_ref = weakref.ref(temporary)
        return temporary

    result = batching.profile_batch_size(run, cuda.device, max_batch_size=8, warmup=0, repeats=1)

    assert result.batch_size == 2
    assert [item.batch_size for item in result.measurements] == [1, 2]
    assert cuda.calls == [1, 2, 4]
    assert cuda.failure_ref() is None
    assert cuda.empty_calls == 2


@pytest.mark.parametrize("oom", [False, True])
def test_no_feasible_batch_raises_actionable_error(cuda: FakeCuda, oom: bool) -> None:
    """Do not return an unmeasured fallback when even batch one cannot fit."""

    def run(batch_size: int) -> None:
        if oom:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        cuda.run(batch_size, duration=0.1, memory=9_000)

    with pytest.raises(RuntimeError, match="Batch size 1 could not fit on cuda:1") as error:
        batching.profile_batch_size(run, cuda.device, max_batch_size=4, warmup=0, repeats=1)

    assert ("CUDA out of memory" if oom else "memory headroom") in str(error.value)
    assert cuda.active_device is None


def test_non_oom_errors_are_propagated(cuda: FakeCuda) -> None:
    """Unsupported shapes and backend defects must not be treated as OOM."""

    error = RuntimeError("unsupported batch dimension")

    def run(batch_size: int) -> None:
        raise error

    with pytest.raises(RuntimeError) as caught:
        batching.profile_batch_size(run, cuda.device)

    assert caught.value is error
    assert cuda.empty_calls == 2
    assert cuda.active_device is None


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_devices_fail_without_running_callback(device: str) -> None:
    def run(batch_size: int) -> None:
        raise AssertionError("Inference must not start on an unsupported device")

    with pytest.raises(ValueError, match="requires a CUDA device"):
        batching.profile_batch_size(run, device)


def test_unavailable_cuda_fails_without_running_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def run(batch_size: int) -> None:
        raise AssertionError("Inference must not start on an unavailable device")

    with pytest.raises(RuntimeError, match="cuda:0 is unavailable"):
        batching.profile_batch_size(run, "cuda:0")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_batch_size", 0),
        ("max_batch_size", 1.5),
        ("max_batch_size", True),
        ("warmup", -1),
        ("warmup", 1.5),
        ("warmup", True),
        ("repeats", 0),
        ("repeats", 1.5),
        ("repeats", True),
        ("memory_fraction", 0),
        ("memory_fraction", -0.1),
        ("memory_fraction", 1.1),
        ("memory_fraction", float("nan")),
        ("memory_fraction", float("inf")),
        ("memory_fraction", True),
        ("memory_fraction", "0.8"),
    ],
)
def test_invalid_options_fail_before_cuda_probing(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    def unexpected_probe() -> None:
        raise AssertionError("Validation must precede device initialization")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    with pytest.raises(ValueError, match=name):
        batching.profile_batch_size(lambda size: None, "cuda:0", **{name: value})
