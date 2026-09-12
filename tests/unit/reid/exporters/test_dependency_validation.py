"""Exporters validate dependencies before beginning conversion."""

from __future__ import annotations

import pytest

import boxmot.reid.exporters.backends.base as exporter_base
from boxmot.reid.exporters.backends.base import BaseExporter
from boxmot.utils.dependencies import MissingDependencyError


def test_missing_export_extra_stops_before_conversion(monkeypatch, tmp_path) -> None:
    """A library export must propagate actionable errors without installation."""
    calls = []

    class ExampleExporter(BaseExporter):
        extra = "onnx"

        def export(self):
            pytest.fail("Conversion must not start with missing dependencies")

    def missing_extra(extra: str, *, purpose: str) -> None:
        calls.append((extra, purpose))
        raise MissingDependencyError("Run python -m boxmot.engine.cli install --extra onnx")

    monkeypatch.setattr(exporter_base, "require_extra", missing_extra)
    exporter = ExampleExporter(model=object(), im=object(), file=tmp_path / "model.pt", verbose=False)

    with pytest.raises(MissingDependencyError, match="install --extra onnx"):
        exporter.export()

    assert calls == [("onnx", "ExampleExporter export")]
    assert not hasattr(exporter, "checker")
