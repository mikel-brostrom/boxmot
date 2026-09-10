"""Tracker package-boundary tests."""

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.trackers.common.registry import TRACKER_DEFINITIONS
from boxmot.trackers.common.specs import TrackerFamily

_TRACKER_EXPORTS = tuple(
    (tracker_name, entry.class_path.rsplit(".", 1)[1], entry.class_path.rsplit(".", 1)[0])
    for tracker_name, entry in _TRACKER_MANIFEST.items()
)
_TRACKER_NAMES = tuple(class_name for _, class_name, _ in _TRACKER_EXPORTS)
_BOX_TRACKER_NAMES = tuple(
    name for name, definition in TRACKER_DEFINITIONS.items() if definition.capabilities.family is TrackerFamily.BOX
)
_ALGORITHM_TRACK_MODELS = (
    "boosttrack",
    "botsort",
    "bytetrack",
    "deepocsort",
    "hybridsort",
    "ocsort",
    "strongsort",
)
_TRACKERS_ROOT = Path(__file__).resolve().parents[3] / "boxmot" / "trackers"
_SHARED_SUPPORT_MODULES = ("base", "config", "factory", "manifest", "protocols", "registry", "specs")


def _absolute_imports(path: Path) -> set[str]:
    """Return the absolute imports declared by ``path``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def test_implementation_packages_do_not_reexport_tracker_classes() -> None:
    implementation_packages = tuple(
        importlib.import_module(entry.class_path.rsplit(".", 2)[0]) for entry in _TRACKER_MANIFEST.values()
    )

    for package in implementation_packages:
        assert not hasattr(package, "__all__")
        for class_name in _TRACKER_NAMES:
            assert not hasattr(package, class_name)


@pytest.mark.parametrize(("class_name", "module_name"), ((item[1], item[2]) for item in _TRACKER_EXPORTS))
def test_package_root_lazily_loads_and_caches_canonical_class(
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
    module_name: str,
) -> None:
    boxmot_module = importlib.import_module("boxmot")
    sentinel = object()
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return SimpleNamespace(**{class_name: sentinel})

    monkeypatch.delitem(boxmot_module.__dict__, class_name, raising=False)
    monkeypatch.setattr(boxmot_module, "import_module", fake_import_module)

    try:
        assert getattr(boxmot_module, class_name) is sentinel
        assert getattr(boxmot_module, class_name) is sentinel
        assert imported_modules == [module_name]
    finally:
        boxmot_module.__dict__.pop(class_name, None)


def test_package_root_exports_the_registered_tracker_classes() -> None:
    boxmot_module = importlib.import_module("boxmot")
    registry_module = importlib.import_module("boxmot.trackers.common.registry")

    for registry_name, class_name, _ in _TRACKER_EXPORTS:
        tracker_class = registry_module.get_tracker_class(registry_name)
        assert getattr(boxmot_module, class_name) is tracker_class
        assert tracker_class.__name__ == class_name


def test_package_root_exports_the_canonical_tracker_factory() -> None:
    boxmot_module = importlib.import_module("boxmot")
    factory_module = importlib.import_module("boxmot.trackers.common.factory")

    assert boxmot_module.create_tracker is factory_module.create_tracker


def test_tracker_imports_stay_lazy_in_a_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import sys
from importlib import import_module

import boxmot
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST

assert {name for name in sys.modules if name.startswith("boxmot.trackers")} == {
    "boxmot.trackers",
    "boxmot.trackers.common",
    "boxmot.trackers.common.manifest",
}

for entry in _TRACKER_MANIFEST.values():
    import_module(entry.class_path.rsplit(".", 2)[0])

import_module("boxmot.trackers.common")
import_module("boxmot.trackers.common.box")

assert not any(
    name.startswith("boxmot.trackers.") and name.endswith(".tracker")
    for name in sys.modules
)
assert not any(name in sys.modules for name in ("cv2", "numpy", "torch", "yaml"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_legacy_branded_tracker_aliases_are_not_public() -> None:
    boxmot_module = importlib.import_module("boxmot")

    for legacy_name in ("BoTSORT", "BYTETracker", "DeepOCSort", "HybridSORT", "OCSORT", "StrongSORT"):
        assert not hasattr(boxmot_module, legacy_name)


def test_occluboost_reports_its_canonical_class_name() -> None:
    from boxmot import OccluBoost

    tracker = OccluBoost(use_embeddings=False, use_cmc=False)

    assert tracker.name == "OccluBoost"


def test_manifest_uses_direct_algorithm_implementation_paths() -> None:
    for tracker_name, class_name, module_name in _TRACKER_EXPORTS:
        assert module_name == f"boxmot.trackers.{tracker_name}.tracker"
        assert _TRACKER_MANIFEST[tracker_name].class_path == f"{module_name}.{class_name}"
        native_path = _TRACKER_MANIFEST[tracker_name].native_class_path
        if native_path is not None:
            assert native_path.rsplit(".", 1)[0] == f"boxmot.trackers.{tracker_name}.native"


def test_algorithm_track_models_live_with_their_owning_tracker() -> None:
    """Keep algorithm-specific track state out of the shared implementation tree."""

    violations: list[str] = []
    for tracker_name in _ALGORITHM_TRACK_MODELS:
        owner_module = _TRACKERS_ROOT / tracker_name / "track.py"
        legacy_module = _TRACKERS_ROOT / "common" / "track_models" / f"{tracker_name}.py"
        tracker_module = _TRACKERS_ROOT / tracker_name / "tracker.py"
        expected_import = f"boxmot.trackers.{tracker_name}.track"

        if not owner_module.is_file():
            violations.append(f"missing {owner_module.relative_to(_TRACKERS_ROOT.parent.parent)}")
        if legacy_module.exists():
            relative = legacy_module.relative_to(_TRACKERS_ROOT.parent.parent)
            violations.append(f"algorithm model remains shared: {relative}")
        if expected_import not in _absolute_imports(tracker_module):
            relative = tracker_module.relative_to(_TRACKERS_ROOT.parent.parent)
            violations.append(f"{relative} does not import {expected_import}")

    forbidden_imports = {
        f"boxmot.trackers.common.track_models.{tracker_name}" for tracker_name in _ALGORITHM_TRACK_MODELS
    }
    for source_path in _TRACKERS_ROOT.rglob("*.py"):
        for imported in _absolute_imports(source_path):
            if imported in forbidden_imports:
                violations.append(
                    f"{source_path.relative_to(_TRACKERS_ROOT.parent.parent)} imports removed module {imported}"
                )

    assert violations == []
    assert not (_TRACKERS_ROOT / "common" / "track_models").exists()


@pytest.mark.parametrize("package_name", ("bbox", "box", "hybrid", "mask", "multimodal"))
def test_removed_or_moved_tracker_packages_stay_absent(package_name: str) -> None:
    assert importlib.util.find_spec(f"boxmot.trackers.{package_name}") is None
    assert not (_TRACKERS_ROOT / package_name).exists()


@pytest.mark.parametrize("module_name", _SHARED_SUPPORT_MODULES)
def test_shared_support_has_one_canonical_module_under_common(module_name: str) -> None:
    """Moving support code must not leave a second importable class identity."""

    assert importlib.util.find_spec(f"boxmot.trackers.common.{module_name}") is not None
    assert importlib.util.find_spec(f"boxmot.trackers.{module_name}") is None
    assert not (_TRACKERS_ROOT / f"{module_name}.py").exists()


def test_public_tracker_contracts_resolve_to_canonical_common_classes() -> None:
    """Factory callers and direct tracker users must share contract identities."""

    from boxmot import create_tracker, trackers
    from boxmot.trackers.common import protocols, specs
    from boxmot.trackers.common.factory import create_tracker as common_create_tracker

    assert trackers.TrackerSpec is specs.TrackerSpec
    assert create_tracker is trackers.create_tracker is common_create_tracker
    for name in ("TrackerCapabilities", "TrackerFamily"):
        assert getattr(trackers, name) is getattr(specs, name)
    for name in ("Tracker", "TrackerRequirements", "ReIDConfigurableTracker"):
        assert getattr(trackers, name) is getattr(protocols, name)


def test_all_trackers_inherit_the_canonical_common_base() -> None:
    """Both box and multimodal trackers retain one validated update boundary."""

    for _, class_name, module_name in _TRACKER_EXPORTS:
        tracker_class = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(tracker_class, BaseTracker)


def test_box_tracker_implementations_share_the_box_base() -> None:
    from boxmot.trackers.common.box.base import BoxTracker

    for tracker_name in _BOX_TRACKER_NAMES:
        tracker_class = importlib.import_module(f"boxmot.trackers.{tracker_name}.tracker").__dict__[
            _TRACKER_MANIFEST[tracker_name].class_path.rsplit(".", 1)[1]
        ]
        assert issubclass(tracker_class, BoxTracker)


def test_removed_common_tracks_exports_stay_absent() -> None:
    common_module = importlib.import_module("boxmot.trackers.common")

    assert "tracks" not in common_module.__all__
    assert "BoxTrack" not in common_module.__all__
    assert "SortBoxTrack" not in common_module.__all__
    assert not hasattr(common_module, "tracks")
    assert not hasattr(common_module, "BoxTrack")
    assert not hasattr(common_module, "SortBoxTrack")
