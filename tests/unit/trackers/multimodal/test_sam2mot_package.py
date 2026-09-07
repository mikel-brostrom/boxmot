"""Sam2Mot package ownership and dependency-boundary tests."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

from boxmot.structures import GeometryKind
from boxmot.trackers.registry import get_tracker_definition
from boxmot.trackers.specs import TrackerCapabilities, TrackerFamily


def test_sam2mot_uses_the_multimodal_implementation_path() -> None:
    """The public class must resolve to the representation-first package."""

    boxmot = importlib.import_module("boxmot")
    implementation = importlib.import_module("boxmot.trackers.multimodal.sam2mot.tracker")

    assert boxmot.Sam2Mot is implementation.Sam2Mot
    assert implementation.Sam2Mot.__module__ == "boxmot.trackers.multimodal.sam2mot.tracker"


def test_direct_sam2mot_class_declares_static_multimodal_capabilities() -> None:
    """Direct construction exposes the same immutable facts as the registry."""

    implementation = importlib.import_module("boxmot.trackers.multimodal.sam2mot.tracker")

    expected = TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB, GeometryKind.OBB}),
        requires_masks=True,
        accepts_masks=True,
        requires_frame=True,
        accepts_frame=True,
    )
    assert implementation.Sam2Mot.capabilities == expected
    assert get_tracker_definition("sam2mot").capabilities == expected


def test_removed_sam2mot_and_hybrid_packages_stay_absent() -> None:
    """Do not retain aliases for either superseded implementation path."""

    assert importlib.util.find_spec("boxmot.trackers.sam2mot") is None
    assert importlib.util.find_spec("boxmot.trackers.hybrid") is None


def test_multimodal_packages_do_not_reexport_or_add_a_symmetry_base() -> None:
    """The taxonomy is internal and has no base class without shared invariants."""

    package = importlib.import_module("boxmot.trackers.multimodal")
    implementation_package = importlib.import_module("boxmot.trackers.multimodal.sam2mot")
    tracker_root = Path(package.__file__).resolve().parent

    assert not hasattr(package, "Sam2Mot")
    assert not hasattr(implementation_package, "Sam2Mot")
    assert not (tracker_root / "base.py").exists()


def test_sam2mot_does_not_own_generic_segmentation_inference() -> None:
    """Reusable SAM/SAM2 model inference remains in ``boxmot.segmentors``."""

    implementation = importlib.import_module("boxmot.trackers.multimodal.sam2mot.tracker")
    source_path = Path(implementation.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert not any(name == "boxmot.segmentors" or name.startswith("boxmot.segmentors.") for name in imports)
    assert "ultralytics" not in imports
    assert importlib.util.find_spec("boxmot.segmentors.backends.sam") is not None
