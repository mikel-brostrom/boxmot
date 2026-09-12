"""Device selection regressions without requiring accelerator hardware."""

from __future__ import annotations

import os

import pytest
import torch

from boxmot.utils.devices import normalize_device, resolve_device


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu", "cpu"),
        (" MPS ", "mps"),
        ("cuda", "cuda:0"),
        ("0", "cuda:0"),
        ("10", "cuda:10"),
        (" CUDA:01 ", "cuda:1"),
        (torch.device("cuda"), "cuda:0"),
        (torch.device("cuda:1"), "cuda:1"),
        (torch.device("cpu"), "cpu"),
        (torch.device("mps"), "mps"),
    ],
)
def test_normalization_does_not_probe_hardware(
    monkeypatch: pytest.MonkeyPatch, value: str | torch.device, expected: str
) -> None:
    def unexpected_probe() -> None:
        raise AssertionError("Syntax normalization must not query accelerators")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, "is_built", unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)
    assert normalize_device(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", " ", "auto", "none", None, 0, -1, True, "-1", "cuda:-1", "cuda:", "cuda:1.0",
     "0,1", "cuda:0,1", [0, 1], (0, 1), "[0]", "cudanone:0", "cuda: 0", "xpu:0"],
)
def test_invalid_or_multiple_devices_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="expected a single device"):
        resolve_device(value)


@pytest.mark.parametrize("mask", [None, "2,4", "GPU-existing-mask", "-1"])
def test_selection_preserves_visibility_and_current_device(
    monkeypatch: pytest.MonkeyPatch, mask: str | None
) -> None:
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    environment = dict(os.environ)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def unexpected_switch(*args: object) -> None:
        raise AssertionError("Selection must not switch the process's CUDA device")

    monkeypatch.setattr(torch.cuda, "set_device", unexpected_switch)
    monkeypatch.setattr(torch.cuda, "device", unexpected_switch)
    for value, expected in [("1", "cuda:1"), ("cpu", "cpu"), ("mps", "mps"), ("cuda", "cuda:0")]:
        assert resolve_device(value) == torch.device(expected)
        assert os.environ == environment
    with pytest.raises(RuntimeError, match="cuda:9 is unavailable"):
        resolve_device("9")
    assert os.environ == environment


def test_cpu_selection_does_not_initialize_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe() -> None:
        raise AssertionError("Selecting CPU must not probe accelerators")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_validation_checks_logical_indices_instead_of_digit_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 11)
    assert resolve_device("10") == torch.device("cuda:10")
    with pytest.raises(RuntimeError, match=r"reports 11 CUDA device\(s\)"):
        resolve_device("11")

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match=r"reports 1 CUDA device\(s\)"):
        resolve_device(torch.device("cuda:9"))


def test_cuda_unavailability_is_not_silently_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"reports 0 CUDA device\(s\)"):
        resolve_device("cuda")


@pytest.mark.parametrize(("built", "available"), [(False, False), (False, True), (True, False)])
def test_mps_requires_both_build_and_runtime_support(
    monkeypatch: pytest.MonkeyPatch, built: bool, available: bool
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: built)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="mps is unavailable"):
        resolve_device(torch.device("mps"))
