"""The EdgeTAM export command preserves dynamic-validation and failure contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


@pytest.fixture
def export_calls(monkeypatch):
    """Record domain calls without importing optional conversion runtimes."""

    calls = []
    module = ModuleType("boxmot.segmentors.exporters.edgetam.export")

    def export(checkpoint: Path, output: Path, **kwargs) -> Path:
        calls.append((checkpoint, output, kwargs))
        return output

    module.export_edgetam = export
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_edgetam_export_uses_automatic_validation_by_default(export_calls, tmp_path) -> None:
    result = CliRunner().invoke(boxmot, ["export-edgetam", "--output", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert export_calls == [
        (Path("edgetam.pt"), tmp_path, {"max_objects": 96, "validate_objects": None, "converter_python": None})
    ]
    assert f"Exported EdgeTAM TFLite bundle: {tmp_path}" in result.output


def test_edgetam_export_passes_explicit_object_counts(export_calls, tmp_path) -> None:
    checkpoint = tmp_path / "weights.pt"
    output = tmp_path / "bundle"
    result = CliRunner().invoke(
        boxmot,
        [
            "export-edgetam",
            "--weights",
            str(checkpoint),
            "--output",
            str(output),
            "--max-objects",
            "16",
            "--validate-objects",
            "1,2,16,2",
            "--converter-python",
            sys.executable,
        ],
    )

    assert result.exit_code == 0, result.output
    assert export_calls == [
        (
            checkpoint,
            output,
            {"max_objects": 16, "validate_objects": (1, 2, 16), "converter_python": Path(sys.executable)},
        )
    ]


@pytest.mark.parametrize(
    "options",
    [
        ["--max-objects", "0"],
        ["--validate-objects", "0"],
        ["--validate-objects", "-1"],
        ["--validate-objects", ""],
        ["--validate-objects", "none"],
        ["--validate-objects", "one,two"],
        ["--max-objects", "2", "--validate-objects", "1,3"],
    ],
)
def test_edgetam_export_rejects_invalid_object_counts_before_conversion(export_calls, tmp_path, options) -> None:
    result = CliRunner().invoke(boxmot, ["export-edgetam", "--output", str(tmp_path), *options])

    assert result.exit_code == 2, result.output
    assert export_calls == []


def test_edgetam_export_does_not_report_success_after_validation_failure(monkeypatch, tmp_path) -> None:
    module = ModuleType("boxmot.segmentors.exporters.edgetam.export")

    def export(*_args, **_kwargs):
        raise RuntimeError("Object input has a static shape signature")

    module.export_edgetam = export
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = CliRunner().invoke(boxmot, ["export-edgetam", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error: Object input has a static shape signature" in result.output
    assert "Exported EdgeTAM" not in result.output
