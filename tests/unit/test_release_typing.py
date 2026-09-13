"""Verify the installed-release guard catches missing autocomplete metadata."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import boxmot
from tests.ci.release_contract import check_typing_metadata


def test_release_typing_metadata() -> None:
    check_typing_metadata()


def test_release_rejects_missing_typing_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boxmot, "__file__", str(tmp_path / "__init__.py"))

    with pytest.raises(AssertionError, match="Missing packaged typing marker"):
        check_typing_metadata()


@pytest.mark.parametrize(
    "module_name,alias_name",
    (
        ("boxmot.detectors._model_names", "DetectorName"),
        ("boxmot.reid._model_names", "ReIDName"),
        ("boxmot.trackers.common._model_names", "TrackerName"),
    ),
)
def test_release_rejects_untyped_factory_names(
    module_name: str, alias_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.import_module(module_name), alias_name, str)

    with pytest.raises(AssertionError, match=rf"{alias_name} must include Literal"):
        check_typing_metadata()
