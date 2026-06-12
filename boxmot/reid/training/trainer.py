"""ReID model trainer with training loop, validation, and checkpointing."""

from __future__ import annotations

import copy
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import build_combined_dataset, build_dataset
from boxmot.reid.datasets.sampler import PKSampler
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import build_test_transforms, build_train_transforms
from boxmot.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
)
from boxmot.reid.training.losses import METRIC_LOSS_REGISTRY, CenterLoss, CrossEntropyLabelSmooth
from boxmot.utils import logger as LOGGER


@dataclass
class TrainMetrics:
    """Metrics collected during a training epoch."""
    epoch: int
    loss: float
    id_loss: float
    triplet_loss: float
    center_loss: float
    lr: float
    elapsed_s: float


@dataclass
class ValMetrics:
    """Metrics from validation (CMC + mAP)."""
    epoch: int
    mAP: float
    rank1: float
    rank5: float
    rank10: float
    dataset: str = ""


@dataclass
class TrainResult:
    """Final result from a ReID training run."""
    best_epoch: int
    best_mAP: float
    best_rank1: float
    weights_path: Path
    history: List[TrainMetrics] = field(default_factory=list)
    val_history: List[ValMetrics] = field(default_factory=list)


class ReIDTrainer:
    """Orchestrates ReID model training.

    Supports softmax (cross-entropy with label smoothing) and triplet loss
    with optional center loss, matching the existing backbone forward()
    contracts.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        data_dir: str,
        *,
        loss_type: str = "triplet",
        preprocess: str = "resize",
        img_size: Tuple[int, int] = (256, 128),
        batch_size: int = 64,
        lr: float = 3.5e-4,
        weight_decay: float = 5e-4,
        epochs: int = 120,
        warmup_epochs: int = 10,
        eval_interval: int = 10,
        p: int = 16,
        k: int = 4,
        margin: float = 0.3,
        label_smooth: float = 0.1,
        center_loss_weight: float = 5e-4,
        eta_min: float = 1e-7,
        pretrained: bool = True,
        device: str = "cpu",
        project: str = "runs/reid_train",
        name: str = "exp",
        num_workers: int = 4,
        seed: int = 42,
        eval_datasets: Optional[List[str]] = None,
        ema_decay: Optional[float] = None,
        gaussian_blur: bool = False,
        random_grayscale: float = 0.0,
        color_jitter: bool = False,
        random_erasing: float = 0.5,
        resume: Optional[str] = None,
        metric_feature: str = "auto",
        inference_feature: str = "concat_bn",
        feat_dim: int = 512,
        neck_dim: int = 512,
        branch_aware_metric: bool = False,
        branch_metric_part_weight: float = 0.5,
        head_pool: str = "avg",
        head_parts: tuple[int, ...] = (1, 2),
        head_warmup_epochs: int = 0,
        head_warmup_lr_mult: float = 2.0,
        explicit_hparams: Iterable[str] | None = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.loss_type = loss_type
        self.preprocess = preprocess
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.eval_interval = eval_interval
        self.p = p
        self.k = k
        self.margin = margin
        self.label_smooth = label_smooth
        self.center_loss_weight = center_loss_weight
        self.eta_min = eta_min
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.project = Path(project)
        self.name = name
        self.num_workers = num_workers
        self.seed = seed
        self.eval_datasets = eval_datasets or []
        self.ema_decay = ema_decay
        self.gaussian_blur = gaussian_blur
        self.random_grayscale = random_grayscale
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing
        self.resume = resume
        self.metric_feature = metric_feature
        self.inference_feature = inference_feature
        self.feat_dim = feat_dim
        self.neck_dim = neck_dim
        self.branch_aware_metric = branch_aware_metric
        self.branch_metric_part_weight = branch_metric_part_weight
        self.head_pool = head_pool
        self.head_parts = self._normalize_head_parts(head_parts)
        self.head_warmup_epochs = head_warmup_epochs
        self.head_warmup_lr_mult = head_warmup_lr_mult
        self.explicit_hparams = set(explicit_hparams or ())

    def _empty_cache(self):
        """Release unused memory held by the MPS/CUDA allocator."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def _normalize_head_parts(head_parts) -> tuple[int, ...]:
        """Normalize CSL-TinyViT head part granularities from CLI/API inputs."""
        if isinstance(head_parts, str):
            parts = [part for part in head_parts.replace(";", ",").split(",") if part.strip()]
            return tuple(int(part) for part in parts)
        if isinstance(head_parts, int):
            return (int(head_parts),)
        return tuple(int(part) for part in head_parts)

    def run(self) -> TrainResult:
        """Execute the full training pipeline."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # 1. Build dataset (supports comma-separated names for joint training)
        ds_names = [n.strip() for n in self.dataset_name.split(",") if n.strip()]
        if len(ds_names) > 1:
            LOGGER.info(f"Loading combined dataset from: {ds_names}")
            dataset = build_combined_dataset(ds_names, self.data_dir)
        else:
            LOGGER.info(f"Loading dataset '{self.dataset_name}' from {self.data_dir}")
            dataset = build_dataset(self.dataset_name, self.data_dir)
        LOGGER.info(dataset.summary())
        num_classes = dataset.num_train_pids

        # 2. Build model (before dataloaders so img_size can be synced)
        LOGGER.info(f"Building model '{self.model_name}' with {num_classes} classes, loss='{self.loss_type}'")
        model = self._build_model(num_classes)
        model = model.to(self.device)

        # Sync img_size with model if the model declares its own (e.g. vit_tiny uses 384×128)
        if hasattr(model, "img_size") and model.img_size != self.img_size:
            LOGGER.info(f"Syncing img_size with model architecture: {self.img_size} → {model.img_size}")
            self.img_size = model.img_size

        # 3. Build dataloaders
        train_loader = self._build_train_loader(dataset)
        query_loader, gallery_loader = self._build_test_loaders(dataset)

        # 2b. Build cross-domain eval loaders
        # For combined datasets, the first component provides default query/gallery;
        # for single datasets, it's the training dataset itself.
        if "," in self.dataset_name:
            default_eval_name = self.dataset_name.split(",")[0].strip().lower()
        else:
            default_eval_name = self.dataset_name.lower()

        xdomain_loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}
        for eval_ds_name in self.eval_datasets:
            if eval_ds_name.strip().lower() == default_eval_name:
                continue  # already covered by default query/gallery
            try:
                xds = build_dataset(eval_ds_name, self.data_dir)
                q_loader, g_loader = self._build_test_loaders(xds)
                xdomain_loaders[eval_ds_name] = (q_loader, g_loader)
                LOGGER.info(
                    f"Cross-domain eval: loaded '{eval_ds_name}' "
                    f"({xds.query.num_imgs}q / {xds.gallery.num_imgs}g)"
                )
            except Exception as e:
                LOGGER.warning(f"Skipping cross-domain eval dataset '{eval_ds_name}': {e}")

        # 3b. Build EMA (momentum) model if requested
        ema_model: Optional[nn.Module] = None
        if self.ema_decay:
            ema_model = copy.deepcopy(model)
            for p in ema_model.parameters():
                p.requires_grad_(False)
            LOGGER.info(f"EMA model enabled (decay={self.ema_decay})")
        # For validation: use EMA model when available, else the trained model
        val_model = ema_model if ema_model is not None else model

        # 4. Build losses
        is_vit = self._is_vit(model)
        label_smooth = self.label_smooth
        if is_vit and label_smooth > 0:
            # Mild smoothing (0.05) works well with soft-margin triplet + center loss
            label_smooth = 0.05
            LOGGER.info(
                f"ViT detected: reducing label smoothing to {label_smooth} "
                f"(was {self.label_smooth})"
            )
        criterion_id = CrossEntropyLabelSmooth(num_classes, epsilon=label_smooth)
        # Build metric loss (triplet, ms, or none for softmax-only)
        soft_margin = is_vit
        criterion_metric = None
        if self.loss_type in METRIC_LOSS_REGISTRY:
            MetricLossClass = METRIC_LOSS_REGISTRY[self.loss_type]
            if self.loss_type == "triplet":
                criterion_metric = MetricLossClass(margin=self.margin, soft_margin=soft_margin)
            else:
                criterion_metric = MetricLossClass()
            LOGGER.info(f"Metric loss: {MetricLossClass.__name__}")

        # MS loss handles positive cluster tightness internally via its
        # positive pair weighting — center loss is redundant and can interfere.
        if self.loss_type == "ms" and self.center_loss_weight > 0:
            LOGGER.info("MS loss active: disabling center loss (redundant)")
            self.center_loss_weight = 0

        # Determine metric feature dim for center loss by a dummy forward pass.
        metric_dim = self._probe_feat_dim(model)
        criterion_center = CenterLoss(num_classes, metric_dim)
        criterion_center = criterion_center.to(self.device)

        # 5. Build optimizer — ViTs need AdamW + proper param groups
        if is_vit:
            self._apply_vit_training_defaults()
            # Keep EMA disabled by default for CSL-TinyViT. The selected
            # recipes use explicit augmentation instead of EMA smoothing.
            if self.ema_decay is None:
                self.ema_decay = 0
                LOGGER.info("ViT detected: leaving EMA disabled by default")
            # NOTE: augmentation auto-scaling removed — augmentations should be
            # controlled explicitly via CLI flags.  The default recipe already
            # enables color_jitter, gaussian_blur, random_grayscale, and
            # random_erasing; auto-overriding breaks ablation experiments that
            # intentionally disable them.
            param_groups = self._build_vit_param_groups(model)
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
            LOGGER.info(
                f"ViT training: AdamW (lr={self.lr:.1e}, wd={self.weight_decay}), "
                f"layer-decay LR, grad clip=1.0, DropPath enabled"
            )
        else:
            params = [{"params": model.parameters()}]
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.5)

        # 6. LR scheduler — must be created BEFORE warmup LR reduction
        # so CosineAnnealingLR captures the full base LR as its _initial_lr.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - self.warmup_epochs, eta_min=self.eta_min
        )

        # Store base LR per param group for warmup (preserves layer-decay scaling)
        # Then reduce LR so epoch 1 starts at base_lr/warmup_epochs (not 100%).
        for pg in optimizer.param_groups:
            pg["_base_lr"] = pg["lr"]
            if self.warmup_epochs > 0:
                pg["lr"] = pg["lr"] / self.warmup_epochs

        # 6b. Resume from checkpoint if requested
        start_epoch = 1
        best_mAP = 0.0
        best_rank1 = 0.0
        best_epoch = 0
        if self.resume:
            resume_path = Path(self.resume)
            if resume_path.is_dir():
                # Prefer last.pt (full state), fall back to best.pt (weights only)
                if (resume_path / "last.pt").exists():
                    resume_path = resume_path / "last.pt"
                elif (resume_path / "best.pt").exists():
                    resume_path = resume_path / "best.pt"
                    LOGGER.warning("last.pt not found, falling back to best.pt (optimizer state will be reset)")
                else:
                    raise FileNotFoundError(f"No checkpoint found in: {resume_path}")
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "optimizer_center" in ckpt:
                optimizer_center.load_state_dict(ckpt["optimizer_center"])
            # Re-create scheduler for the (possibly changed) T_max.
            # PyTorch's last_epoch param has off-by-one issues with the
            # incremental get_lr() formula, so we set last_epoch and LR
            # via closed-form cosine directly.
            resumed_epoch = ckpt.get("epoch", 0)
            cosine_epoch = max(resumed_epoch - self.warmup_epochs, 0)
            new_T_max = self.epochs - self.warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=new_T_max, eta_min=self.eta_min
            )
            scheduler.last_epoch = cosine_epoch
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = self.eta_min + (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * cosine_epoch / new_T_max)
                ) / 2
            if ema_model is not None and "ema_state_dict" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_mAP = ckpt.get("best_mAP", ckpt.get("mAP", 0.0))
            best_rank1 = ckpt.get("rank1", 0.0)
            best_epoch = ckpt.get("epoch", 0)
            LOGGER.info(
                f"Resumed from {resume_path} (epoch {ckpt['epoch']}, "
                f"mAP={best_mAP:.2%}, R1={best_rank1:.2%})"
            )

        # 7. Output directory
        if self.resume:
            save_dir = Path(self.resume) if Path(self.resume).is_dir() else Path(self.resume).parent
        else:
            save_dir = self._make_save_dir()
        LOGGER.info(f"Saving results to {save_dir}")

        # 7b. Persist hyperparameters (effective values after auto-scaling)
        hparams = {
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "data_dir": str(self.data_dir),
            "img_size": list(self.img_size),
            "preprocess": self.preprocess,
            "loss_type": self.loss_type,
            "pretrained": self.pretrained,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW" if is_vit else "Adam",
            "scheduler": "CosineAnnealingLR",
            "eta_min": self.eta_min,
            "warmup_epochs": self.warmup_epochs,
            "label_smooth": label_smooth,
            "margin": self.margin,
            "center_loss_weight": self.center_loss_weight,
            "metric_feature": self._effective_metric_feature(),
            "inference_feature": self.inference_feature,
            "feat_dim": self.feat_dim,
            "neck_dim": self.neck_dim,
            "head_pool": self.head_pool,
            "head_parts": list(self.head_parts),
            "branch_aware_metric": self.branch_aware_metric,
            "branch_metric_part_weight": self.branch_metric_part_weight,
            "head_warmup_epochs": self.head_warmup_epochs,
            "head_warmup_lr_mult": self.head_warmup_lr_mult,
            "p": self.p,
            "k": self.k,
            "seed": self.seed,
            "device": str(self.device),
            "num_workers": self.num_workers,
            "color_jitter": self.color_jitter,
            "gaussian_blur": self.gaussian_blur,
            "random_grayscale": self.random_grayscale,
            "random_erasing": self.random_erasing,
            "ema_decay": self.ema_decay,
            "soft_margin_triplet": soft_margin,
            "flip_tta": is_vit,
            "eval_interval": self.eval_interval,
            "eval_datasets": self.eval_datasets,
            "is_vit": is_vit,
            "grad_clip": 1.0 if is_vit else 0.0,
            "num_classes": num_classes,
            "metric_dim": metric_dim,
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        if is_vit:
            hparams["layer_decay"] = 0.95
            hparams["drop_path_rate"] = 0.1
        (save_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))
        LOGGER.info(f"Saved hyperparameters to {save_dir / 'hparams.json'}")

        # 8. Training loop
        best_weights = save_dir / "best.pt"
        history: List[TrainMetrics] = []
        val_history: List[ValMetrics] = []

        # Restore prior metrics when resuming so the full history is preserved
        if self.resume:
            prev_metrics = save_dir / "metrics.json"
            if prev_metrics.exists():
                try:
                    prev = json.loads(prev_metrics.read_text())
                    for t in prev.get("train", []):
                        if t["epoch"] < start_epoch:
                            history.append(TrainMetrics(
                                epoch=t["epoch"], loss=t["loss"],
                                id_loss=t["id_loss"], triplet_loss=t["triplet_loss"],
                                center_loss=t["center_loss"], lr=t["lr"], elapsed_s=0.0,
                            ))
                    for v in prev.get("val", []):
                        if v["epoch"] < start_epoch:
                            # Support grouped format: {epoch, ds1: {...}, ds2: {...}}
                            if "mAP" in v:
                                # Old flat format
                                val_history.append(ValMetrics(
                                    epoch=v["epoch"], mAP=v["mAP"],
                                    rank1=v["rank1"], rank5=v.get("rank5", 0.0),
                                    rank10=v.get("rank10", 0.0),
                                    dataset=v.get("dataset", ""),
                                ))
                            else:
                                # New grouped format
                                for key, metrics in v.items():
                                    if key == "epoch":
                                        continue
                                    val_history.append(ValMetrics(
                                        epoch=v["epoch"], mAP=metrics["mAP"],
                                        rank1=metrics["rank1"],
                                        rank5=metrics.get("rank5", 0.0),
                                        rank10=metrics.get("rank10", 0.0),
                                        dataset=key,
                                    ))
                    LOGGER.info(
                        f"Restored {len(history)} train and {len(val_history)} "
                        f"val entries from prior metrics.json"
                    )
                except Exception as e:
                    LOGGER.warning(f"Could not restore prior metrics: {e}")

        from tqdm import tqdm

        epoch_bar = tqdm(range(start_epoch, self.epochs + 1), desc="Training", unit="epoch",
                         initial=start_epoch - 1, total=self.epochs)
        for epoch in epoch_bar:
            metrics = self._train_epoch(
                epoch, model, train_loader,
                criterion_id, criterion_metric, criterion_center,
                optimizer, optimizer_center, scheduler,
                ema_model=ema_model,
                grad_clip=1.0 if is_vit else 0.0,
            )
            history.append(metrics)
            self._empty_cache()
            epoch_bar.set_postfix(
                loss=f"{metrics.loss:.4f}",
                id=f"{metrics.id_loss:.4f}",
                tri=f"{metrics.triplet_loss:.4f}",
                lr=f"{metrics.lr:.6f}",
            )

            # Validation
            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                # Calibrate EMA model's BN stats before evaluation
                if ema_model is not None:
                    self._calibrate_bn(val_model, train_loader)
                val = self._validate(epoch, val_model, query_loader, gallery_loader)
                # For combined datasets, default query/gallery comes from the first component
                val.dataset = default_eval_name if "," in self.dataset_name else self.dataset_name
                val_history.append(val)
                epoch_bar.set_postfix(
                    loss=f"{metrics.loss:.4f}",
                    mAP=f"{val.mAP:.2%}",
                    R1=f"{val.rank1:.2%}",
                    lr=f"{metrics.lr:.6f}",
                )
                if val.mAP > best_mAP:
                    best_mAP = val.mAP
                    best_rank1 = val.rank1
                    best_epoch = epoch
                    self._save_checkpoint(val_model, best_weights, epoch, val)
                    tqdm.write(f"  ✓ New best model (mAP={val.mAP:.2%}, R1={val.rank1:.2%}) -> {best_weights}")

                # Cross-domain evaluation
                for xds_name, (xq_loader, xg_loader) in xdomain_loaders.items():
                    xval = self._validate(epoch, val_model, xq_loader, xg_loader)
                    xval.dataset = xds_name
                    val_history.append(xval)
                    tqdm.write(
                        f"  → {xds_name}: mAP={xval.mAP:.2%}  R1={xval.rank1:.2%}  R5={xval.rank5:.2%}"
                    )

                # Restore train mode so EMA buffer counts stay aligned
                val_model.train()

                # Free MPS/CUDA memory accumulated during evaluation
                self._empty_cache()

            # Save last.pt every 10 epochs (full state for resume)
            if epoch % 10 == 0 or epoch == self.epochs:
                last_weights = save_dir / "last.pt"
                self._save_checkpoint(
                    val_model, last_weights, epoch,
                    val_history[-1] if val_history else None,
                    optimizer=optimizer, optimizer_center=optimizer_center,
                    ema_model=ema_model, best_mAP=best_mAP,
                )
                # Persist metrics so far (survives interruptions)
                self._save_metrics(save_dir, history, val_history, best_epoch, best_mAP, best_rank1)

        # Final metrics save
        self._save_metrics(save_dir, history, val_history, best_epoch, best_mAP, best_rank1)

        LOGGER.info(
            f"Training complete. Best epoch={best_epoch}  "
            f"mAP={best_mAP:.2%}  R1={best_rank1:.2%}"
        )
        return TrainResult(
            best_epoch=best_epoch,
            best_mAP=best_mAP,
            best_rank1=best_rank1,
            weights_path=best_weights,
            history=history,
            val_history=val_history,
        )

    def _build_model(self, num_classes: int) -> nn.Module:
        model = ReIDModelRegistry.build_model(
            name=self.model_name,
            weights=Path(f"{self.model_name}_{self.dataset_name}.pt"),
            num_classes=num_classes,
            loss=self.loss_type,
            pretrained=self.pretrained,
            use_gpu=self.device.type != "cpu",
            inference_feature=self.inference_feature,
            feat_dim=self.feat_dim,
            neck_dim=self.neck_dim,
            head_pool=self.head_pool,
            head_parts=self.head_parts,
            branch_metric=self.branch_aware_metric,
        )
        if hasattr(model, "head") and hasattr(model.head, "metric_feature"):
            model.head.metric_feature = self._effective_metric_feature()
        if hasattr(model, "head") and hasattr(model.head, "set_pooling"):
            model.head.set_pooling(self.head_pool)
        if hasattr(model, "head") and hasattr(model.head, "set_branch_metric"):
            model.head.set_branch_metric(self.branch_aware_metric)
        return model

    def _build_train_loader(self, dataset) -> DataLoader:
        transform = build_train_transforms(
            self.img_size,
            preprocess=self.preprocess,
            color_jitter=self.color_jitter,
            gaussian_blur=self.gaussian_blur,
            random_grayscale=self.random_grayscale,
            random_erasing=self.random_erasing,
        )
        torch_ds = ReIDImageDataset(dataset.train.samples, transform=transform)
        sampler = PKSampler(dataset.train.samples, p=self.p, k=self.k)
        # On MPS (unified memory) persistent workers compete with GPU for RAM;
        # keep them only on CUDA where host→device overlap justifies the cost.
        use_persistent = self.num_workers > 0 and self.device.type == "cuda"
        return DataLoader(
            torch_ds,
            batch_size=self.p * self.k,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=use_persistent,
        )

    def _build_test_loaders(self, dataset) -> Tuple[DataLoader, DataLoader]:
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)
        query_ds = ReIDImageDataset(dataset.query.samples, transform=transform)
        gallery_ds = ReIDImageDataset(dataset.gallery.samples, transform=transform)
        # Fewer workers on MPS to reduce unified-memory pressure
        nw = min(self.num_workers, 2) if self.device.type == "mps" else self.num_workers
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=nw,
            pin_memory=self.device.type == "cuda",
            shuffle=False,
            persistent_workers=False,
        )
        return DataLoader(query_ds, **loader_kwargs), DataLoader(gallery_ds, **loader_kwargs)

    def _probe_feat_dim(self, model: nn.Module) -> int:
        """Run a dummy forward to determine the training embedding dimension."""
        was_training = model.training
        model.train()
        dummy = torch.randn(2, 3, *self.img_size, device=self.device)
        with torch.no_grad():
            out = model(dummy)
        if not was_training:
            model.eval()
        if isinstance(out, tuple):
            if isinstance(out[1], dict):
                return out[1]["global"].shape[1]
            return out[1].shape[1]  # triplet: (logits, features)
        if isinstance(out, list):
            return out[0].shape[1]  # multi-branch softmax: list of logits
        return out.shape[1]

    # ------------------------------------------------------------------
    # ViT-specific training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_vit(model: nn.Module) -> bool:
        """Check if the model is a ViT variant (has transformer blocks + patch embed)."""
        return hasattr(model, "blocks") and hasattr(model, "patch_embed")

    def _effective_metric_feature(self) -> str:
        """Resolve the metric feature mode for multi-branch models."""
        if self.metric_feature != "auto":
            return self.metric_feature
        return "concat_bn" if self.loss_type == "ms" else "raw_mean"

    def _apply_vit_training_defaults(self) -> None:
        """Apply ViT training conveniences unless the caller set values explicitly."""
        # AdamW uses decoupled weight decay: effective WD = lr x wd.
        # The default wd=5e-4 (calibrated for Adam L2-reg) gives negligible
        # regularization with AdamW, so use the ViT recipe default unless the
        # caller intentionally passed a lower value for an ablation.
        if "weight_decay" not in self.explicit_hparams and self.weight_decay < 0.01:
            self.weight_decay = 0.1

        if "warmup_epochs" not in self.explicit_hparams and self.warmup_epochs <= 10:
            self.warmup_epochs = 20

        # ViTs with AdamW need ~2x higher LR than CNNs with Adam. Preserve
        # explicit lower LRs so LR sweeps test the requested value.
        if "lr" not in self.explicit_hparams and self.lr <= 3.5e-4:
            self.lr = 7e-4

        # ViTs need stronger center loss to tighten positive clusters. Preserve
        # explicit zero so loss ablations can remove center loss.
        if (
            "center_loss_weight" not in self.explicit_hparams
            and self.loss_type != "ms"
            and self.center_loss_weight <= 5e-4
        ):
            self.center_loss_weight = 5e-3

    def _build_vit_param_groups(self, model: nn.Module) -> list:
        """Build parameter groups with layer-decay LR and no-WD filtering.

        Layer-decay assigns geometrically decreasing LR to earlier blocks:
            lr_scale = layer_decay ** (depth - layer_idx)
        Patch embed and pos_embed get the lowest LR; the classifier head
        gets the base LR.

        No weight decay is applied to: bias, LayerNorm, InstanceNorm,
        cls_token, pos_embed, and AIN gate parameters.
        """
        # Mild layer decay so all layers adapt to the ReID domain.
        # 0.75 (ViT-B canonical) is too aggressive for small ViTs — it
        # effectively freezes early layers.  0.95 gives ~2x LR range.
        depth = getattr(model, "depth", len(model.blocks))
        layer_decay = 0.95

        # Identify parameters that should not have weight decay.
        # "bn" covers BatchNorm gamma in hybrid CNN-Transformer models
        # (e.g. CSL-TinyViT's Conv2d_BN and BNNeck3 modules).
        no_wd_keywords = {
            "bias", "cls_token", "pos_embed",
            "norm", "ln", "bn", "in_norm", "gate",
        }

        def _no_wd(name: str) -> bool:
            return any(kw in name for kw in no_wd_keywords)

        def _get_layer_id(name: str) -> int:
            """Map parameter name to layer index (0=patch_embed, 1..depth=blocks, depth+1=head)."""
            if name.startswith("patch_embed") or name.startswith("cls_token") or name.startswith("pos_embed"):
                return 0
            # "blocks." is the standard ViT naming; "layers." is used by
            # CSL-TinyViT (self.layers registered first, self.blocks alias).
            if name.startswith("blocks.") or name.startswith("layers."):
                block_idx = int(name.split(".")[1])
                return block_idx + 1
            # head / classifier / neck / omni_scale layers
            return depth + 1

        param_groups: dict[str, dict] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            layer_id = _get_layer_id(name)
            lr_scale = layer_decay ** (depth + 1 - layer_id)
            wd = 0.0 if _no_wd(name) else self.weight_decay

            group_key = f"layer_{layer_id}_wd_{wd}"
            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr": self.lr * lr_scale,
                    "weight_decay": wd,
                    "is_head": layer_id == depth + 1,
                }
            param_groups[group_key]["params"].append(param)

        LOGGER.info(
            f"ViT param groups: {len(param_groups)} groups, "
            f"layer_decay={layer_decay}, depth={depth}"
        )
        return list(param_groups.values())

    def _head_warmup_active(self, epoch: int) -> bool:
        """Return whether this epoch should train only neck/head parameters."""
        return self.head_warmup_epochs > 0 and epoch <= self.head_warmup_epochs

    @staticmethod
    def _is_head_or_neck_param(name: str) -> bool:
        return name.startswith("head.") or name.startswith("neck.")

    def _set_head_warmup_trainability(self, model: nn.Module, enabled: bool) -> None:
        """Freeze/unfreeze backbone parameters for head-only warmup."""
        for name, param in model.named_parameters():
            param.requires_grad_(not enabled or self._is_head_or_neck_param(name))

    def _apply_head_warmup_lrs(self, optimizer) -> None:
        """Use zero LR for backbone groups and a boosted LR for head groups."""
        for group in optimizer.param_groups:
            base_lr = group.get("_base_lr", group.get("lr", self.lr))
            group["lr"] = base_lr * self.head_warmup_lr_mult if group.get("is_head", False) else 0.0

    def _metric_loss_for_features(self, criterion_metric, features, pids: torch.Tensor) -> torch.Tensor:
        """Compute metric loss for a tensor feature or branch feature dict."""
        if isinstance(features, dict):
            if not self.branch_aware_metric:
                key = self._effective_metric_feature()
                selected = features.get(key, features["raw_mean"])
                return criterion_metric(F.normalize(selected, p=2, dim=1), pids)

            weighted_losses = []
            total_weight = 0.0
            branch_weights = [("global", 1.0)] + [
                (key, self.branch_metric_part_weight)
                for key in self._sorted_part_feature_keys(features)
            ]
            for key, weight in branch_weights:
                if key in features and weight > 0:
                    branch_features = F.normalize(features[key], p=2, dim=1)
                    weighted_losses.append(criterion_metric(branch_features, pids) * weight)
                    total_weight += weight
            if weighted_losses and total_weight > 0:
                return sum(weighted_losses) / total_weight
            return torch.zeros((), device=self.device, requires_grad=True)

        return criterion_metric(F.normalize(features, p=2, dim=1), pids)

    @staticmethod
    def _sorted_part_feature_keys(features: dict) -> list[str]:
        """Return part feature keys sorted by numeric suffix: part0, part1, ..."""
        def part_index(key: str) -> int:
            try:
                return int(key[4:])
            except ValueError:
                return 10**9

        return sorted(
            (key for key in features if key.startswith("part") and key[4:].isdigit()),
            key=part_index,
        )

    @staticmethod
    def _center_features(features):
        """Use the global raw branch for center loss when branch metrics are enabled."""
        if isinstance(features, dict):
            return features.get("global", features.get("raw_mean"))
        return features

    def _train_epoch(
        self, epoch, model, loader, criterion_id, criterion_metric, criterion_center,
        optimizer, optimizer_center, scheduler, *, ema_model=None, grad_clip: float = 0.0,
    ) -> TrainMetrics:
        from tqdm import tqdm

        requested_head_warmup = self._head_warmup_active(epoch)
        head_warmup_supported = any(group.get("is_head", False) for group in optimizer.param_groups)
        head_warmup_active = requested_head_warmup and head_warmup_supported
        if requested_head_warmup and not head_warmup_supported and epoch == 1:
            LOGGER.warning("Head warmup requested, but optimizer has no separate head parameter group; ignoring")
        self._set_head_warmup_trainability(model, head_warmup_active)
        if head_warmup_active:
            self._apply_head_warmup_lrs(optimizer)
            if epoch == 1:
                LOGGER.info(
                    f"Head warmup enabled for {self.head_warmup_epochs} epochs "
                    f"(head_lr_mult={self.head_warmup_lr_mult:g})"
                )

        model.train()
        total_loss = 0.0
        total_id = 0.0
        total_tri = 0.0
        total_cen = 0.0
        n_batches = 0
        t0 = time.monotonic()

        # AMP: mixed precision on CUDA for ~2x throughput, skip on CPU/MPS
        use_amp = self.device.type == "cuda"
        if not hasattr(self, "_scaler"):
            self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scaler = self._scaler

        batch_bar = tqdm(
            loader, desc=f"  Epoch {epoch}/{self.epochs}",
            leave=False, unit="batch",
        )
        for imgs, pids, _ in batch_bar:
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(imgs)

                if isinstance(output, tuple):
                    logits, features = output
                else:
                    logits = output
                    features = None

                # ID loss — supports multi-branch (part-based) logits
                if isinstance(logits, list):
                    loss_id = sum(criterion_id(lg, pids) for lg in logits) / len(logits)
                else:
                    loss_id = criterion_id(logits, pids)
                loss = loss_id

                # Triplet loss — L2-normalize features so Euclidean distance in
                # triplet loss aligns with cosine distance used at evaluation.
                loss_tri = torch.tensor(0.0, device=self.device)
                if criterion_metric is not None and features is not None:
                    loss_tri = self._metric_loss_for_features(criterion_metric, features, pids)
                    loss = loss + loss_tri

                # Center loss — only on embeddings, never on logits
                center_features = self._center_features(features)
                loss_cen = torch.tensor(0.0, device=self.device)
                if center_features is not None and self.center_loss_weight > 0 and not head_warmup_active:
                    loss_cen = criterion_center(center_features, pids) * self.center_loss_weight
                    loss = loss + loss_cen

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping (ViT training stability)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)

            # Center loss has its own optimizer with special LR
            if center_features is not None and self.center_loss_weight > 0 and not head_warmup_active:
                scaler.unscale_(optimizer_center)
                for param in criterion_center.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / self.center_loss_weight)
                scaler.step(optimizer_center)

            scaler.update()

            # EMA update (parameters + buffers)
            # Float buffers (BN running_mean/var) are EMA'd so their
            # statistics match the EMA model's feature distribution.
            # Integer buffers (num_batches_tracked, index tensors) are copied.
            if ema_model is not None:
                decay = self.ema_decay
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)
                for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
                    if ema_b.is_floating_point():
                        ema_b.data.mul_(decay).add_(model_b.data, alpha=1.0 - decay)
                    else:
                        ema_b.data.copy_(model_b.data)

            total_loss += loss.item()
            total_id += loss_id.item()
            total_tri += loss_tri.item()
            total_cen += loss_cen.item()
            n_batches += 1
            batch_bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        # Scheduler step
        if epoch > self.warmup_epochs:
            scheduler.step()
        elif self.warmup_epochs > 0:
            # Linear warmup — respect per-group base LR (layer-decay)
            warmup_factor = epoch / self.warmup_epochs
            for pg in optimizer.param_groups:
                base_lr = pg.get("_base_lr", self.lr)
                pg["lr"] = base_lr * warmup_factor

        elapsed = time.monotonic() - t0
        return TrainMetrics(
            epoch=epoch,
            loss=total_loss / max(n_batches, 1),
            id_loss=total_id / max(n_batches, 1),
            triplet_loss=total_tri / max(n_batches, 1),
            center_loss=total_cen / max(n_batches, 1),
            lr=optimizer.param_groups[0]["lr"],
            elapsed_s=elapsed,
        )

    @torch.no_grad()
    def _calibrate_bn(self, model, data_loader, num_batches: int = 50):
        """Run forward passes to calibrate BN running stats for the EMA model."""
        model.train()
        with torch.no_grad():
            for i, (imgs, _, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                imgs = imgs.to(self.device)
                model(imgs)
        model.eval()
        self._empty_cache()

    def _validate(self, epoch, model, query_loader, gallery_loader) -> ValMetrics:
        use_flip = self._is_vit(model)
        q_feats, q_pids, q_camids = extract_features(model, query_loader, self.device, desc="Query", flip_tta=use_flip)
        g_feats, g_pids, g_camids = extract_features(
            model, gallery_loader, self.device, desc="Gallery", flip_tta=use_flip
        )
        distmat = compute_distance_matrix(q_feats, g_feats)
        del q_feats, g_feats
        cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)
        del distmat, q_pids, g_pids, q_camids, g_camids
        return ValMetrics(
            epoch=epoch,
            mAP=mAP,
            rank1=float(cmc[0]) if len(cmc) > 0 else 0.0,
            rank5=float(cmc[4]) if len(cmc) > 4 else 0.0,
            rank10=float(cmc[9]) if len(cmc) > 9 else 0.0,
        )

    @staticmethod
    def _get_num_classes(model) -> int:
        """Infer num_classes from model's classifier layer."""
        if hasattr(model, "classifier"):
            return model.classifier.out_features
        # Multi-branch models (e.g. CSLTinyViT with BNNeck3 head)
        for name, module in model.named_modules():
            if name.endswith("classifier") and hasattr(module, "out_features"):
                return module.out_features
        return -1

    def _save_checkpoint(
        self, model, path: Path, epoch: int, val: Optional[ValMetrics],
        optimizer=None, optimizer_center=None,
        ema_model=None, best_mAP: float = 0.0,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "num_classes": self._get_num_classes(model),
            "preprocess": self.preprocess,
            "best_mAP": best_mAP,
            "inference_feature": self.inference_feature,
            "feat_dim": self.feat_dim,
            "neck_dim": self.neck_dim,
            "head_pool": self.head_pool,
            "head_parts": list(self.head_parts),
            "branch_aware_metric": self.branch_aware_metric,
            "branch_metric_part_weight": self.branch_metric_part_weight,
            "head_warmup_epochs": self.head_warmup_epochs,
            "head_warmup_lr_mult": self.head_warmup_lr_mult,
        }
        if val is not None:
            state["mAP"] = val.mAP
            state["rank1"] = val.rank1
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if optimizer_center is not None:
            state["optimizer_center"] = optimizer_center.state_dict()
        if ema_model is not None:
            state["ema_state_dict"] = ema_model.state_dict()
        torch.save(state, path)

    def _save_metrics(
        self, save_dir: Path,
        history: List[TrainMetrics],
        val_history: List[ValMetrics],
        best_epoch: int, best_mAP: float, best_rank1: float,
    ):
        """Persist full training & validation history to metrics.json."""
        # Group val entries by epoch
        from collections import OrderedDict
        val_by_epoch: OrderedDict[int, dict] = OrderedDict()
        for v in val_history:
            if v.epoch not in val_by_epoch:
                val_by_epoch[v.epoch] = {"epoch": v.epoch}
            val_by_epoch[v.epoch][v.dataset] = {
                "mAP": round(v.mAP, 4), "rank1": round(v.rank1, 4),
                "rank5": round(v.rank5, 4), "rank10": round(v.rank10, 4),
            }

        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_mAP": round(best_mAP, 4),
            "best_rank1": round(best_rank1, 4),
            "train": [
                {
                    "epoch": m.epoch, "loss": round(m.loss, 5),
                    "id_loss": round(m.id_loss, 5),
                    "triplet_loss": round(m.triplet_loss, 5),
                    "center_loss": round(m.center_loss, 5),
                    "lr": round(m.lr, 8),
                }
                for m in history
            ],
            "val": list(val_by_epoch.values()),
        }
        path = save_dir / "metrics.json"
        path.write_text(json.dumps(data, indent=2))
        LOGGER.info(f"Saved training metrics to {path}")

    def _make_save_dir(self) -> Path:
        base = self.project / self.name
        if base.exists():
            idx = 1
            while (self.project / f"{self.name}_{idx}").exists():
                idx += 1
            base = self.project / f"{self.name}_{idx}"
        base.mkdir(parents=True, exist_ok=True)
        return base
