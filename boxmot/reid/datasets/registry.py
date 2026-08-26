"""Lazy registry and construction helpers for ReID datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from boxmot.reid.datasets.base import BaseReIDDataset, CombinedReIDDataset


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Import target and constructor defaults for one ReID dataset."""

    name: str
    module: str
    class_name: str
    aliases: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> type[BaseReIDDataset]:
        """Import and return the configured dataset class."""
        return getattr(import_module(self.module), self.class_name)

    def build(self, root: str, **kwargs: Any) -> BaseReIDDataset:
        """Instantiate this dataset with registered defaults."""
        constructor_kwargs = {**self.kwargs, **kwargs}
        return self.resolve()(root=root, **constructor_kwargs)


DATASET_SPECS = (
    DatasetSpec("market1501", "boxmot.reid.datasets.market1501", "Market1501"),
    DatasetSpec(
        "mot171501",
        "boxmot.reid.datasets.market1501",
        "MOT17Market1501",
        aliases=("mot17_1501", "mot17market1501"),
    ),
    DatasetSpec("cuhk03", "boxmot.reid.datasets.cuhk03", "CUHK03", aliases=("cuhk03np",)),
    DatasetSpec(
        "duke",
        "boxmot.reid.datasets.dukemtmcreid",
        "DukeMTMCreID",
        aliases=("dukemtmcreid", "dukemtmc"),
    ),
    DatasetSpec("msmt17", "boxmot.reid.datasets.msmt17", "MSMT17"),
    DatasetSpec(
        "msmt17_merged",
        "boxmot.reid.datasets.msmt17",
        "MSMT17",
        kwargs={"merged": True},
    ),
    DatasetSpec("veri", "boxmot.reid.datasets.veri776", "VeRi776", aliases=("veri776",)),
)


def _normalize_dataset_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


_DATASET_SPECS_BY_ALIAS = {
    _normalize_dataset_name(alias): spec
    for spec in DATASET_SPECS
    for alias in (spec.name, *spec.aliases)
}
DATASET_REGISTRY = {spec.name: spec for spec in DATASET_SPECS}


def registered_dataset_names() -> tuple[str, ...]:
    """Return canonical dataset names accepted by configuration surfaces."""
    return tuple(DATASET_REGISTRY)


def get_dataset_spec(name: str) -> DatasetSpec:
    """Resolve a dataset name or alias without importing its implementation."""
    normalized = _normalize_dataset_name(name)
    try:
        return _DATASET_SPECS_BY_ALIAS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASET_REGISTRY)}"
        ) from exc


def get_dataset_class(name: str) -> type[BaseReIDDataset]:
    """Resolve the implementation class for a dataset name or alias."""
    return get_dataset_spec(name).resolve()


def build_dataset(name: str, root: str, **kwargs: Any) -> BaseReIDDataset:
    """Instantiate a registered ReID dataset by name or alias."""
    return get_dataset_spec(name).build(root, **kwargs)


def build_combined_dataset(
    names: list[str] | tuple[str, ...],
    root: str,
    **kwargs: Any,
) -> CombinedReIDDataset:
    """Combine registered training datasets with global PID remapping."""
    datasets = [build_dataset(name.strip(), root, **kwargs) for name in names]
    return CombinedReIDDataset(datasets)


__all__ = (
    "DATASET_REGISTRY",
    "DATASET_SPECS",
    "DatasetSpec",
    "build_combined_dataset",
    "build_dataset",
    "get_dataset_class",
    "get_dataset_spec",
    "registered_dataset_names",
)
