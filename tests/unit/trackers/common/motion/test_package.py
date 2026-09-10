"""Import-boundary coverage for shared tracker motion code."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests._paths import REPO_ROOT


def _run_import_probe(script: str, *args: str) -> None:
    """Check imports in a fresh interpreter, independent of pytest collection."""

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_motion_namespaces_do_not_load_implementations() -> None:
    _run_import_probe(
        """
        import sys
        from importlib import import_module

        namespaces = {
            "boxmot.trackers.common.motion",
            "boxmot.trackers.common.motion.cmc",
            "boxmot.trackers.common.motion.kalman_filters",
        }
        for namespace in namespaces:
            import_module(namespace)

        assert {name for name in sys.modules if name.startswith("boxmot.trackers.common.motion")} == namespaces
        assert not any(name in sys.modules for name in ("numpy", "scipy", "cv2", "torch", "filterpy"))
        """
    )


def test_kalman_noise_configuration_does_not_load_filters() -> None:
    _run_import_probe(
        """
        import sys
        from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig

        assert KalmanNoiseConfig().is_default
        assert {name for name in sys.modules if name.startswith("boxmot.trackers.common.motion")} == {
            "boxmot.trackers.common.motion",
            "boxmot.trackers.common.motion.kalman_filters",
            "boxmot.trackers.common.motion.kalman_filters.noise",
        }
        assert not any(name in sys.modules for name in ("scipy", "cv2", "torch", "filterpy"))
        """
    )


def test_cmc_registry_validation_and_disabled_factory_keep_estimators_lazy() -> None:
    _run_import_probe(
        """
        import sys
        from boxmot.trackers.common import create_cmc as shared_create_cmc
        from boxmot.trackers.common.motion.cmc.registry import available_cmc_methods, create_cmc, get_cmc_method

        assert shared_create_cmc is create_cmc
        assert available_cmc_methods() == ("ecc", "orb", "sift", "sof")
        assert create_cmc(None) is None
        assert get_cmc_method(None) is None
        assert create_cmc("ecc", enabled=False) is None
        assert create_cmc("invalid", enabled=False) is None
        try:
            create_cmc("invalid")
        except ValueError as error:
            assert "Unknown cmc_method" in str(error)
        else:
            raise AssertionError("An unknown enabled estimator must be rejected")

        assert {name for name in sys.modules if name.startswith("boxmot.trackers.common.motion")} == {
            "boxmot.trackers.common.motion",
            "boxmot.trackers.common.motion.cmc",
            "boxmot.trackers.common.motion.cmc.registry",
        }
        assert not any(name in sys.modules for name in ("numpy", "scipy", "cv2", "torch", "filterpy"))
        """
    )


@pytest.mark.parametrize("method", ("ecc", "orb", "sift", "sof"))
def test_cmc_factory_loads_only_the_selected_estimator(method: str) -> None:
    _run_import_probe(
        """
        import sys
        from boxmot.trackers.common.motion.cmc.registry import create_cmc, get_cmc_method

        method = sys.argv[1]
        estimator = create_cmc(f" {method.upper()} ", scale=0.5)
        assert type(estimator) is get_cmc_method(method)
        assert estimator.scale == 0.5
        prefix = "boxmot.trackers.common.motion.cmc"
        assert type(estimator).__module__ == f"{prefix}.{method}"
        assert {
            name for name in ("ecc", "orb", "sift", "sof") if f"{prefix}.{name}" in sys.modules
        } == {method}
        """,
        method,
    )


def test_shared_motion_exports_resolve_to_the_canonical_implementations() -> None:
    from boxmot.trackers import common
    from boxmot.trackers.common.motion import models
    from boxmot.trackers.common.motion.cmc import integration, registry

    for name in ("MotionModelAdapter", "MotionModelKind", "create_motion_model"):
        assert getattr(common, name) is getattr(models, name)
    for name in ("apply_cmc_to_tracks", "cmc_detection_boxes", "reset_cmc"):
        assert getattr(common, name) is getattr(integration, name)
    assert common.create_cmc is registry.create_cmc
