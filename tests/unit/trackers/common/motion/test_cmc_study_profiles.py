"""Historical study controls must stay isolated from production estimators."""

from __future__ import annotations

import sys

import pytest

from boxmot.trackers.common.motion.cmc.base import BaseCMC
from tests.performance.trackers.motion.cmc_study_profiles import _load_historical_classes


def _sources(base_import: str = "from boxmot.trackers.common.motion.cmc.base import BaseCMC") -> dict[str, bytes]:
    """Supply distinguishable historical kernels without a Git dependency."""
    return {
        "base": b"class BaseCMC:\n    marker = 'frozen-base'\n",
        "ecc": f"{base_import}\nclass ECC(BaseCMC):\n    kernel = 'frozen-ecc'\n".encode(),
        "sof": f"{base_import}\nclass SOF(BaseCMC):\n    kernel = 'frozen-sof'\n".encode(),
    }


def test_historical_kernels_use_frozen_base_without_replacing_production() -> None:
    """An estimator-only restore would silently benchmark new preprocessing."""
    sources = _sources()
    original_sources = sources.copy()
    production_module = sys.modules[BaseCMC.__module__]

    classes = _load_historical_classes(sources)

    assert classes["ecc"].__bases__ == classes["sof"].__bases__
    assert classes["ecc"].__bases__[0] is not BaseCMC
    assert classes["ecc"]().marker == "frozen-base"
    assert classes["ecc"]().kernel == "frozen-ecc"
    assert classes["sof"]().kernel == "frozen-sof"
    assert sources == original_sources
    assert sys.modules[BaseCMC.__module__] is production_module
    assert production_module.BaseCMC is BaseCMC


@pytest.mark.parametrize(
    "base_import",
    (
        "from boxmot.trackers.common.motion.cmc.base import BaseCMC as DifferentBase",
        "from boxmot.trackers.common.motion.cmc.base import BaseCMC, AnotherClass",
        "from another.package import BaseCMC",
    ),
)
def test_historical_loader_rejects_unreviewed_import_shapes(base_import: str) -> None:
    """A changed historical dependency must fail explicitly instead of drifting."""
    with pytest.raises(ValueError, match="Unexpected BaseCMC import"):
        _load_historical_classes(_sources(base_import))
