"""Keep PyTorch guidance usable when optional TFLite modules are unavailable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from boxmot.resources import paths
from boxmot.segmentors import propagation


@pytest.fixture
def factory_without_tflite(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the factory afresh with both optional TFLite modules blocked."""
    monkeypatch.setitem(sys.modules, "boxmot.segmentors.exporters.edgetam.bundle", None)
    monkeypatch.setitem(sys.modules, "boxmot.segmentors.propagation.tflite", None)
    monkeypatch.delitem(sys.modules, "boxmot.segmentors.propagation.factory", raising=False)
    monkeypatch.setattr(propagation, "factory", None, raising=False)
    return importlib.import_module("boxmot.segmentors.propagation.factory")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
def test_pytorch_factory_preserves_device_and_guidance_options_without_tflite(
    factory_without_tflite: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Selecting a checkpoint must construct the PyTorch backend on its requested device."""
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"checkpoint")
    expected = SimpleNamespace()
    calls: list[tuple[Path, str, int, float]] = []

    def construct(
        selected: Path, *, device: str, max_objects: int, prompt_overlap: float
    ) -> SimpleNamespace:
        """Capture backend configuration without loading a model or accelerator."""
        calls.append((selected, device, max_objects, prompt_overlap))
        return expected

    monkeypatch.setitem(
        sys.modules,
        "boxmot.segmentors.propagation.edgetam",
        SimpleNamespace(EdgeTAMMaskPropagator=construct),
    )

    assert factory_without_tflite.mask_propagation_device(checkpoint, device) == device
    actual = factory_without_tflite.create_mask_propagator(
        checkpoint, device=device, max_objects=96, prompt_overlap=0.2
    )

    assert actual is expected
    assert calls == [(checkpoint, device, 96, 0.2)]


def test_unresolved_checkpoint_keeps_device_without_loading_tflite(
    factory_without_tflite: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuning may choose a device before official weights have been downloaded."""
    monkeypatch.chdir(tmp_path)
    model_directory = tmp_path / "models"
    monkeypatch.setattr(paths, "_default_weights_directory", lambda: model_directory)

    assert factory_without_tflite.mask_propagation_device("edgetam.pt", "mps") == "mps"
    assert not model_directory.exists()


@pytest.mark.parametrize("authored", ["bundle", "models/bundle", "absolute"])
def test_bundle_directory_selects_cpu_without_importing_exporter(
    factory_without_tflite: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authored: str,
) -> None:
    """Bundle discovery follows the same model-directory lookup as checkpoints."""
    from boxmot.segmentors.propagation.weights import is_edgetam_tflite_bundle

    monkeypatch.chdir(tmp_path)
    model_directory = tmp_path / "models"
    monkeypatch.setattr(paths, "_default_weights_directory", lambda: model_directory)
    bundle = model_directory / "bundle"
    bundle.mkdir(parents=True)
    checkpoint = bundle if authored == "absolute" else Path(authored)

    assert is_edgetam_tflite_bundle(checkpoint)
    assert factory_without_tflite.mask_propagation_device(checkpoint, "mps") == "cpu"
