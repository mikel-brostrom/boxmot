"""Accelerator lifecycle checks for materialization stages."""

from __future__ import annotations

from boxmot.engine.materialization.stages import _runtime


def test_release_accelerator_memory_collects_and_purges_selected_backend(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_runtime.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_runtime.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(_runtime.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_runtime.torch.mps, "empty_cache", lambda: calls.append("mps"))

    _runtime.release_accelerator_memory("mps")

    assert calls == ["gc", "mps"]

    calls.clear()
    _runtime.release_accelerator_memory("cuda:0")

    assert calls == ["gc", "cuda"]


def test_release_accelerator_memory_does_not_initialize_unavailable_backends(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_runtime.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(_runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_runtime.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(_runtime.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(_runtime.torch.mps, "empty_cache", lambda: calls.append("mps"))

    _runtime.release_accelerator_memory("mps")

    assert calls == ["gc"]


def test_release_accelerator_memory_ignores_cpu(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_runtime.gc, "collect", lambda: calls.append("gc"))

    _runtime.release_accelerator_memory("cpu")

    assert calls == []
