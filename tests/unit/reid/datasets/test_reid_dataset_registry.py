"""Tests for the lazy ReID dataset registry."""

from __future__ import annotations

import subprocess
import sys

import pytest

from boxmot.reid.datasets.registry import (
    DATASET_REGISTRY,
    get_dataset_class,
    get_dataset_spec,
    registered_dataset_names,
)
from tests._paths import REPO_ROOT


def test_dataset_registry_import_keeps_benchmarks_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import boxmot.reid.datasets; "
            "print('boxmot.reid.datasets.market1501' in sys.modules); "
            "print('boxmot.reid.datasets.msmt17' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines() == ["False", "False"]


def test_dataset_registry_exposes_canonical_names():
    assert registered_dataset_names() == tuple(DATASET_REGISTRY)
    assert {
        "market1501",
        "mot171501",
        "cuhk03",
        "duke",
        "msmt17",
        "msmt17_merged",
        "veri",
    }.issubset(DATASET_REGISTRY)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("mot17_1501", "mot171501"),
        ("dukemtmcreid", "duke"),
        ("MSMT17-MERGED", "msmt17_merged"),
        ("veri776", "veri"),
        ("cuhk03_np", "cuhk03"),
    ],
)
def test_dataset_aliases_resolve_without_importing_implementations(alias, canonical):
    assert get_dataset_spec(alias).name == canonical


def test_dataset_class_is_imported_on_demand():
    dataset_class = get_dataset_class("market1501")

    assert dataset_class.__name__ == "Market1501"
    assert dataset_class.__module__ == "boxmot.reid.datasets.market1501"


def test_unknown_dataset_reports_canonical_choices():
    with pytest.raises(ValueError, match="Unknown dataset 'missing'"):
        get_dataset_spec("missing")
