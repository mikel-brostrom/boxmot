"""ReID dataset registry and dataloader construction."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from boxmot.reid.datasets.base import BaseReIDDataset, CombinedReIDDataset
from boxmot.reid.datasets.cuhk03 import CUHK03
from boxmot.reid.datasets.dukemtmcreid import DukeMTMCreID
from boxmot.reid.datasets.market1501 import Market1501
from boxmot.reid.datasets.msmt17 import MSMT17
from boxmot.reid.datasets.veri776 import VeRi776

DATASET_REGISTRY: Dict[str, Type[BaseReIDDataset]] = {
    "market1501": Market1501,
    "cuhk03": CUHK03,
    "duke": DukeMTMCreID,
    "dukemtmcreid": DukeMTMCreID,
    "msmt17": MSMT17,
    "msmt17_merged": MSMT17,
    "veri": VeRi776,
    "veri776": VeRi776,
}


def build_dataset(name: str, root: str, **kwargs: Any) -> BaseReIDDataset:
    """Instantiate a ReID dataset by name."""
    key = name.lower().replace("-", "").replace("_", "")
    # Handle MSMT17-merged variant
    if key in ("msmt17merged",):
        return MSMT17(root=root, merged=True, **kwargs)
    # Normalize common aliases
    if key in ("dukemtmcreid", "dukemtmc", "duke"):
        key = "duke"
    if key in ("veri776", "veri"):
        key = "veri"
    if key in ("cuhk03", "cuhk03np"):
        key = "cuhk03"
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY.keys())} (also: msmt17_merged)"
        )
    return DATASET_REGISTRY[key](root=root, **kwargs)


def build_combined_dataset(names: List[str], root: str, **kwargs: Any) -> CombinedReIDDataset:
    """Load multiple datasets and combine their train splits with PID remapping.

    Query/gallery splits are kept per-dataset for separate evaluation.
    """
    datasets = [build_dataset(n.strip(), root, **kwargs) for n in names]
    return CombinedReIDDataset(datasets)


__all__ = (
    "DATASET_REGISTRY",
    "BaseReIDDataset",
    "CombinedReIDDataset",
    "CUHK03",
    "DukeMTMCreID",
    "Market1501",
    "MSMT17",
    "VeRi776",
    "build_dataset",
    "build_combined_dataset",
)
