"""Export setup respects explicit weights and read-only installations."""

from pathlib import Path
from types import SimpleNamespace

from boxmot.reid.exporters import workflow


def test_explicit_checkpoint_preparation_creates_no_default_directory(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "custom.pt"
    weights.touch()
    args = SimpleNamespace(weights=weights, verbose=True)
    prepared = (object(), object())
    monkeypatch.setattr(workflow, "prepare_export_model", lambda _args: prepared)

    def refuse_directory_creation(*args, **kwargs):
        raise PermissionError("The installed package and unused model cache are read-only")

    monkeypatch.setattr(Path, "mkdir", refuse_directory_creation)

    assert workflow.prepare_export(args) == prepared
    assert args.weights == weights
