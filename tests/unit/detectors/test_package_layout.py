"""Detector package layout and lazy-import contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from boxmot.detectors.factory import _DETECTOR_FACTORIES
from tests._paths import REPO_ROOT


def test_public_detector_exports_do_not_load_implementation_modules() -> None:
    probe = (
        "import json, sys; "
        "import boxmot.detectors as detectors; "
        "print(json.dumps({'exports': sorted(detectors.__all__), "
        "'implementations': sorted(name for name in sys.modules "
        "if name.startswith('boxmot.detectors.backends.'))}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload == {
        "exports": ["Detector", "DetectorCapabilities", "DetectorSpec", "create_detector"],
        "implementations": [],
    }


def test_detector_implementations_have_only_canonical_backend_paths() -> None:
    for module_name in ("adapters", "base", "ultralytics", "yolox", "rtdetr"):
        assert importlib.util.find_spec(f"boxmot.detectors.{module_name}") is None
    for module_name in ("base", "ultralytics", "yolox", "rtdetr"):
        assert importlib.util.find_spec(f"boxmot.detectors.backends.{module_name}") is not None

    assert set(_DETECTOR_FACTORIES.entries.values()) == {
        "boxmot.detectors.backends.ultralytics:UltralyticsDetector",
        "boxmot.detectors.backends.yolox:YoloXDetector",
        "boxmot.detectors.backends.rtdetr:RTDetrDetector",
    }
