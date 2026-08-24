"""Tracker package-boundary tests."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def test_tracker_package_exports_canonical_occluboost(monkeypatch: pytest.MonkeyPatch) -> None:
    trackers_module = importlib.import_module("boxmot.trackers")
    sentinel = object()
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return SimpleNamespace(OccluBoost=sentinel)

    monkeypatch.delitem(trackers_module.__dict__, "OccluBoost", raising=False)
    monkeypatch.setattr(trackers_module, "import_module", fake_import_module)

    assert trackers_module.__all__ == ("OccluBoost",)
    assert trackers_module.OccluBoost is sentinel
    assert imported_modules == ["boxmot.trackers.bbox.occluboost"]
    assert importlib.util.find_spec("boxmot.trackers.occluboost") is None
