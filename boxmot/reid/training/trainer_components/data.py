"""Dataset construction, samplers, and dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from boxmot.reid.datasets import CombinedReIDDataset, build_combined_dataset, build_dataset
from boxmot.reid.datasets.sampler import PKSampler, SourceBalancedPKSampler
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import (
    build_test_transforms,
)
from boxmot.reid.training.augmentations import (
    augmentation_config_from_options,
    build_training_augmentation_bundle,
    pav_requires_clean_view,
)
from boxmot.reid.training.trainer_components.helpers import (
    _seed_data_worker,
)
from boxmot.reid.training.trainer_components.types import (
    DatasetBundle,
    LoaderBundle,
)
from boxmot.utils import logger as LOGGER


class _DataMixin:
    @property
    def train_batch_size(self) -> int:
        """Effective PK-sampled training batch size."""
        if self.source_balance_groups:
            return sum(group.batch_size for group in self.source_balance_groups)
        return self.p * self.k

    @staticmethod
    def _normalize_data_spec(spec: dict[str, Any]) -> dict[str, Any]:
        name = str(spec.get("name") or spec.get("dataset") or "").strip()
        root = str(spec.get("root") or spec.get("path") or "").strip()
        if not name:
            raise ValueError("data_specs entries must define a dataset name")
        if not root:
            raise ValueError(f"data_specs entry for '{name}' must define a root/path")

        normalized = {"name": name, "root": str(Path(root).expanduser().resolve())}
        for key in ("config", "train", "val", "query", "gallery"):
            if spec.get(key) is not None:
                normalized[key] = str(spec[key])
        return normalized

    @staticmethod
    def _dataset_lookup_key(name: str) -> str:
        key = str(name).lower().replace("-", "").replace("_", "")
        if key in {"dukemtmcreid", "dukemtmc", "duke"}:
            return "duke"
        if key in {"mot171501", "mot17market1501"}:
            return "mot171501"
        if key in {"veri776", "veri"}:
            return "veri"
        if key in {"cuhk03", "cuhk03np"}:
            return "cuhk03"
        if key == "msmt17merged":
            return "msmt17merged"
        return key

    def _data_root_for_name(self, name: str) -> str:
        return self._data_roots_by_name.get(self._dataset_lookup_key(name), self.data_dir)

    def _build_dataset_bundle(self) -> DatasetBundle:
        """Load the configured dataset and identify the primary validation split."""
        if self.data_specs:
            dataset_names = [spec["name"] for spec in self.data_specs]
            if len(self.data_specs) > 1:
                LOGGER.info(f"Loading combined dataset from data specs: {dataset_names}")
                datasets = [build_dataset(spec["name"], spec["root"]) for spec in self.data_specs]
                dataset = CombinedReIDDataset(datasets)
            else:
                spec = self.data_specs[0]
                LOGGER.info(f"Loading dataset '{spec['name']}' from {spec['root']}")
                dataset = build_dataset(spec["name"], spec["root"])
            default_eval_name = dataset_names[0].lower()
        else:
            dataset_names = [name.strip() for name in self.dataset_name.split(",") if name.strip()]
            if len(dataset_names) > 1:
                LOGGER.info(f"Loading combined dataset from: {dataset_names}")
                dataset = build_combined_dataset(dataset_names, self.data_dir)
                default_eval_name = dataset_names[0].lower()
            else:
                LOGGER.info(f"Loading dataset '{self.dataset_name}' from {self.data_dir}")
                dataset = build_dataset(self.dataset_name, self.data_dir)
                default_eval_name = self.dataset_name.lower()
        LOGGER.info(dataset.summary())
        self._train_sample_count = len(dataset.train.samples)
        self._train_samples = tuple(dataset.train.samples)
        return DatasetBundle(
            dataset=dataset,
            num_classes=dataset.num_train_pids,
            default_eval_name=default_eval_name,
        )

    def _build_loader_bundle(self, data: DatasetBundle) -> LoaderBundle:
        """Build train, primary validation, and optional cross-domain loaders."""
        train_loader = self._build_train_loader(data.dataset)
        query_loader, gallery_loader = self._build_test_loaders(data.dataset)
        cross_domain: Dict[str, Tuple[DataLoader, DataLoader]] = {}
        for eval_dataset_name in self.eval_datasets:
            if eval_dataset_name.strip().lower() == data.default_eval_name:
                continue
            try:
                eval_root = self._data_root_for_name(eval_dataset_name)
                eval_dataset = build_dataset(eval_dataset_name, eval_root)
                query, gallery = self._build_test_loaders(eval_dataset)
                cross_domain[eval_dataset_name] = (query, gallery)
                LOGGER.info(
                    f"Cross-domain eval: loaded '{eval_dataset_name}' "
                    f"from {eval_root} ({eval_dataset.query.num_imgs}q / {eval_dataset.gallery.num_imgs}g)"
                )
            except Exception as exc:
                LOGGER.warning(f"Skipping cross-domain eval dataset '{eval_dataset_name}': {exc}")
        return LoaderBundle(
            train=train_loader,
            query=query_loader,
            gallery=gallery_loader,
            cross_domain=cross_domain,
        )

    def _pav_requires_clean_view(self, batch_size: int) -> bool:
        """Return whether PAV needs clean tensors for consistency or reversion."""
        return pav_requires_clean_view(
            augmentation_config_from_options(self),
            batch_size=batch_size,
        )

    def _build_train_loader(self, dataset) -> DataLoader:
        if self.source_balance_groups:
            sampler = SourceBalancedPKSampler(
                dataset.train.samples,
                self.source_balance_groups,
                seed=self.seed,
            )
            batch_size = sampler.batch_size
        else:
            sampler = PKSampler(
                dataset.train.samples,
                p=self.p,
                k=self.k,
                seed=self.seed,
                steps_per_epoch=self.pk_steps_per_epoch,
                camera_aware=self.camera_aware_sampler,
            )
            batch_size = self.p * self.k
        augmentations = build_training_augmentation_bundle(
            self,
            dataset,
            batch_size=batch_size,
        )
        torch_ds = ReIDImageDataset(
            dataset.train.samples,
            transform=augmentations.image_transform,
            sample_transform=augmentations.sample_transform,
            return_clean_view=augmentations.return_clean_view,
            clean_transform=augmentations.clean_transform,
            return_clean_anatomical_target=(augmentations.return_clean_anatomical_target),
            anatomical_target_provider=(augmentations.anatomical_target_provider),
            return_sample_index=(
                self.global_ap_loss_weight > 0
                or any(
                    weight > 0
                    for weight in (
                        self.hpgrd_global_weight,
                        self.hpgrd_part_weight,
                        self.hpgrd_background_weight,
                        self.hpgrd_part_drop_weight,
                    )
                )
            ),
        )
        return DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=_seed_data_worker,
            generator=self._train_generator,
        )

    def _build_test_loaders(self, dataset) -> Tuple[DataLoader, DataLoader]:
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)
        query_ds = ReIDImageDataset(dataset.query.samples, transform=transform)
        gallery_ds = ReIDImageDataset(dataset.gallery.samples, transform=transform)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=False,
            persistent_workers=False,
        )
        return DataLoader(query_ds, **loader_kwargs), DataLoader(gallery_ds, **loader_kwargs)
