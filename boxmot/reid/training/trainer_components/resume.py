"""Checkpoint resume and optimizer-state restoration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from boxmot.reid.backbones.anatomical_registry import (
    V8_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression import (
    MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION,
)
from boxmot.reid.training.config import load_train_hparams
from boxmot.reid.training.losses import (
    CenterLoss,
)
from boxmot.reid.training.provenance import (
    anatomical_metadata_provenance,
    checkpoint_pretrained_provenance,
    restore_model_pretrained_provenance,
)
from boxmot.reid.training.resume import (
    OPTIMIZER_CONTRACT_VERSION,
    build_resume_contract,
    contract_differences,
)
from boxmot.reid.training.trainer_components.types import (
    LoaderBundle,
    LossBundle,
    ModelBundle,
    OptimizationBundle,
    ResumeState,
)
from boxmot.utils import logger as LOGGER


class _ResumeMixin:
    def _resolve_resume_path(self) -> Optional[Path]:
        """Resolve a resume directory to its resumable checkpoint."""
        if not self.resume:
            return None
        resume_path = Path(self.resume)
        if resume_path.is_dir():
            if (resume_path / "last.pt").exists():
                return resume_path / "last.pt"
            if (resume_path / "best.pt").exists():
                raise ValueError(
                    f"Cannot resume {resume_path}: only best.pt exists, and best.pt is an "
                    "inference-only checkpoint. Use it as pretrained weights or restore last.pt."
                )
            raise FileNotFoundError(f"No checkpoint found in: {resume_path}")
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return resume_path

    def _restore_if_needed(
        self,
        model: ModelBundle,
        loaders: LoaderBundle,
        losses: LossBundle,
        optimization: OptimizationBundle,
    ) -> ResumeState:
        """Restore live/EMA model state and compatible optimizer state."""
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return ResumeState()

        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        resumable = checkpoint.get("resumable", "optimizer" in checkpoint)
        if not resumable:
            raise ValueError(
                f"Cannot resume {resume_path}: checkpoint_type={checkpoint.get('checkpoint_type', 'unknown')!r} "
                "does not contain a resumable training state"
            )
        if checkpoint.get("checkpoint_precision") == "float16":
            LOGGER.warning(
                f"Resuming legacy lossy FP16 training state from {resume_path}. "
                "The run remains seeded, but its first resumed step cannot be bit-exact. "
                "New last.pt checkpoints preserve native training precision."
            )
        self._assert_resume_compatible(checkpoint, resume_path)
        saved_num_classes = checkpoint.get("num_classes")
        current_num_classes = self._get_num_classes(model.model)
        if saved_num_classes is not None and int(saved_num_classes) != current_num_classes:
            raise ValueError(
                f"Cannot resume {resume_path}: checkpoint has {int(saved_num_classes)} classes, "
                f"but the current dataset/model has {current_num_classes}"
            )
        required_state = {"optimizer", "optimizer_center", "rng_state"}
        if self.center_loss_weight > 0:
            required_state.add("center_loss_state_dict")
        if self.classifier_loss != "ce":
            required_state.add("classifier_loss_state_dict")
        if self._effective_ema_decay():
            required_state.add("ema_state_dict")
        if checkpoint.get("resume_contract") is not None:
            required_state.add("scheduler")
        if self.deterministic and self.device.type == "cuda":
            required_state.add("grad_scaler")
        if self._global_ap is not None:
            required_state.add("training_state")
        missing_state = sorted(required_state - checkpoint.keys())
        if missing_state:
            raise ValueError(f"Cannot resume {resume_path}: missing training state: {', '.join(missing_state)}")
        pretrained_provenance = checkpoint_pretrained_provenance(checkpoint)
        model.model.load_state_dict(checkpoint["state_dict"], strict=resumable)
        restore_model_pretrained_provenance(model.model, pretrained_provenance)
        self._restore_center_loss_state(
            checkpoint,
            losses.criterion_center,
            model.model,
            loaders.train,
            resume_path,
        )
        self._restore_classifier_loss_state(
            checkpoint,
            losses.criterion_id,
            resume_path,
        )
        if "optimizer" in checkpoint:
            optimization.optimizer.load_state_dict(checkpoint["optimizer"])
        if "optimizer_center" in checkpoint:
            optimization.optimizer_center.load_state_dict(checkpoint["optimizer_center"])
        self._restore_training_auxiliary_state(checkpoint.get("training_state"))
        resumed_epoch = int(checkpoint.get("epoch", 0))
        self._assert_resume_metrics_consistent(resume_path, resumed_epoch)
        optimization.scheduler = self._build_resume_scheduler(
            optimization.optimizer,
            resumed_epoch,
            resume_path,
            checkpoint,
        )
        if model.ema_model is not None:
            model.ema_model.load_state_dict(
                checkpoint.get("ema_state_dict", checkpoint["state_dict"]),
                strict=resumable,
            )
            restore_model_pretrained_provenance(model.ema_model, pretrained_provenance)
        scaler_state = checkpoint.get("grad_scaler")
        if scaler_state is not None:
            if not hasattr(self, "_scaler"):
                self._scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=self.device.type == "cuda",
                )
            self._scaler.load_state_dict(scaler_state)
        self._restore_rng_state(checkpoint.get("rng_state"))
        best_mAP = float(checkpoint.get("best_mAP") or checkpoint.get("mAP", 0.0))
        best_rank1 = float(checkpoint.get("best_rank1", checkpoint.get("rank1", 0.0)))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        if not best_epoch:
            best_epoch, best_rank1 = self._legacy_best_state(
                resume_path,
                fallback_epoch=resumed_epoch,
                fallback_rank1=best_rank1,
            )
        LOGGER.info(f"Resumed from {resume_path} (epoch {resumed_epoch}, mAP={best_mAP:.2%}, R1={best_rank1:.2%})")
        return ResumeState(
            start_epoch=resumed_epoch + 1,
            best_mAP=best_mAP,
            best_rank1=best_rank1,
            best_epoch=best_epoch,
        )

    def _resolved_resume_values(self) -> dict[str, Any]:
        """Return the effective semantic values used for compatibility checks."""
        values = dict(vars(self))
        values["hpgrd_manifest_sha256"] = self._hpgrd_manifest_sha256
        values["retrieval_dataset_sha256"] = self._retrieval_dataset_sha256
        values["optimizer_contract_version"] = OPTIMIZER_CONTRACT_VERSION
        recipe = self._resolve_training_recipe_for_model_name()
        if recipe is not None:
            values.update(
                training_recipe=recipe.name,
                optimizer=recipe.optimizer_name,
                layer_decay=recipe.layer_decay(self),
                grad_clip=recipe.grad_clip,
                flip_tta=self.flip_tta if self.flip_tta is not None else recipe.default_flip_tta,
            )
        values["metric_feature"] = self._effective_metric_feature()
        return values

    def _anatomical_metadata_provenance(self) -> dict[str, Any] | None:
        """Return the immutable annotation snapshot bound to this run.

        Anatomical assets can contain thousands of masks, so their content is
        hashed once per configured root pair and reused by later epoch
        checkpoints. A new trainer (including every resume invocation) always
        computes a fresh snapshot before compatibility is checked.
        """
        if not self.anatomical_auxiliary:
            return None
        cache_key = (
            self.anatomical_metadata_dir,
            self.anatomical_person_mask_dir,
        )
        cached = getattr(
            self,
            "_anatomical_metadata_provenance_cache",
            None,
        )
        if cached is not None and cached[0] == cache_key:
            return dict(cached[1])
        provenance = anatomical_metadata_provenance(*cache_key)
        self._anatomical_metadata_provenance_cache = (
            cache_key,
            provenance,
        )
        return dict(provenance)

    def _resume_contract(self) -> dict[str, Any]:
        """Return the canonical contract for exact ablation/resume matching."""
        contract = build_resume_contract(self._resolved_resume_values())
        anatomical_provenance = self._anatomical_metadata_provenance()
        if anatomical_provenance is not None:
            contract["data"]["anatomical_metadata_sha256"] = (
                anatomical_provenance["sha256"]
            )
        loss_contract = contract["loss"]
        hpgrd_enabled = self._hpgrd_enabled()
        if self.global_ap_loss_weight <= 0:
            global_ap_only_keys = (
                "global_ap_loss_weight",
                "global_ap_temperature",
                "global_ap_topk",
                "global_ap_memory_size",
                "global_ap_momentum",
                "global_ap_max_age",
            )
            for key in global_ap_only_keys:
                loss_contract.pop(key)
            if not hpgrd_enabled:
                for key in tuple(loss_contract):
                    if key.startswith("global_ap_"):
                        loss_contract.pop(key)
        if not hpgrd_enabled:
            for key in tuple(loss_contract):
                if key.startswith("hpgrd_"):
                    loss_contract.pop(key)
        if self.global_ap_loss_weight <= 0 and not hpgrd_enabled:
            loss_contract.pop("retrieval_dataset_sha256")
        if self.multilevel_suppression:
            contract["model"]["multilevel_suppression_version"] = (
                MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION
            )
        return contract

    @staticmethod
    def _legacy_best_state(
        resume_path: Path,
        *,
        fallback_epoch: int,
        fallback_rank1: float,
    ) -> tuple[int, float]:
        """Recover best-model state from metrics written by older trainers."""
        metrics_path = resume_path.parent / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                return (
                    int(metrics.get("best_epoch", fallback_epoch)),
                    float(metrics.get("best_rank1", fallback_rank1)),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return fallback_epoch, fallback_rank1

    def _assert_resume_compatible(self, checkpoint: dict[str, Any], resume_path: Path) -> None:
        """Reject checkpoints whose learned-state contract differs from this run."""
        requested = self._resume_contract()
        saved = checkpoint.get("resume_contract")
        legacy = saved is None
        if legacy:
            legacy_hparams = load_train_hparams(resume_path)
            if not legacy_hparams:
                raise ValueError(
                    f"Cannot verify resume compatibility for legacy checkpoint {resume_path}: hparams.json is missing"
                )
            saved = build_resume_contract(legacy_hparams, partial=True)

        differences = contract_differences(saved, requested, compare_common_only=legacy)
        if legacy:
            differences.append(
                "optimization.optimizer_contract_version: "
                f"saved='<missing>', requested={OPTIMIZER_CONTRACT_VERSION} "
                "(optimizer grouping/warmup semantics changed; use the model weights as pretrained instead)"
            )
        if legacy and self.anatomical_auxiliary:
            legacy_target_type = legacy_hparams.get("anatomical_target_type")
            if legacy_target_type is None:
                if legacy_hparams.get("anatomical_teacher_momentum") is not None:
                    legacy_target_type = V8_ANATOMICAL_TARGET_TYPE
            if legacy_target_type != self.anatomical_target_type:
                differences.append(
                    "model.anatomical_target_type: "
                    f"saved={legacy_target_type or '<missing>'!r}, "
                    f"requested={self.anatomical_target_type!r}"
                )
        previous_epochs = self._resume_target_epochs(resume_path, checkpoint)
        if previous_epochs is not None and self.epochs < previous_epochs:
            differences.append(
                f"target_epochs: saved={previous_epochs!r}, requested={self.epochs!r} "
                "(resume may extend, but may not shorten a run)"
            )
        resumed_epoch = int(checkpoint.get("epoch", 0))
        if resumed_epoch >= self.epochs:
            differences.append(
                f"checkpoint_epoch: saved={resumed_epoch!r}, requested target={self.epochs!r} (run is already complete)"
            )
        if differences:
            details = "\n  - ".join(differences[:20])
            if len(differences) > 20:
                details += f"\n  - ... and {len(differences) - 20} more"
            raise ValueError(
                f"Refusing incompatible resume from {resume_path}:\n  - {details}\n"
                "Use matching arguments or start in a clean project/name directory."
            )
        if legacy:
            LOGGER.warning(
                f"Resume contract missing from {resume_path}; accepted after matching all "
                "available legacy hparams. The next checkpoint will store an exact fingerprint."
            )

    @staticmethod
    def _assert_resume_metrics_consistent(resume_path: Path, checkpoint_epoch: int) -> None:
        """Reject checkpoint progress that cannot be represented by saved history."""
        metrics_path = resume_path.parent / "metrics.json"
        if not metrics_path.exists():
            if checkpoint_epoch > 0:
                raise ValueError(
                    f"Cannot resume {resume_path}: metrics.json is missing for checkpoint epoch {checkpoint_epoch}"
                )
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            train_history = metrics.get("train") or []
            metrics_epoch = max(
                (int(item.get("epoch", 0)) for item in train_history),
                default=0,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot resume {resume_path}: invalid metrics.json: {exc}") from exc
        if metrics_epoch < checkpoint_epoch:
            raise ValueError(
                f"Cannot resume {resume_path}: checkpoint is at epoch {checkpoint_epoch}, but "
                f"metrics.json ends at epoch {metrics_epoch}. The missing history cannot be "
                "reconstructed exactly."
            )
        if metrics_epoch > checkpoint_epoch:
            LOGGER.warning(
                f"metrics.json is ahead of last.pt ({metrics_epoch} > {checkpoint_epoch}); "
                "discarding later metric entries and replaying those epochs deterministically"
            )

    def _resume_target_epochs(self, resume_path: Path, ckpt: dict) -> Optional[int]:
        """Return the epoch target saved by the run being resumed."""
        if ckpt.get("epochs") is not None:
            return int(ckpt["epochs"])

        run_dir = resume_path if resume_path.is_dir() else resume_path.parent
        for filename in ("hparams.json", "metrics.json"):
            path = run_dir / filename
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text())
                epochs = raw.get("epochs")
                if epochs is None and isinstance(raw.get("optimization"), dict):
                    epochs = raw["optimization"].get("epochs")
            except Exception:
                continue
            if epochs is not None:
                return int(epochs)
        return None

    def _build_resume_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        resumed_epoch: int,
        resume_path: Path,
        ckpt: dict,
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        """Build a resume scheduler without increasing LR when extending a run."""
        previous_epochs = self._resume_target_epochs(resume_path, ckpt)
        extending_run = (
            previous_epochs is not None
            and self.epochs > previous_epochs
            and resumed_epoch >= self.warmup_epochs
            and "optimizer" in ckpt
        )
        if extending_run:
            remaining_epochs = max(self.epochs - resumed_epoch, 1)
            for group in optimizer.param_groups:
                current_lr = float(group["lr"])
                group["initial_lr"] = current_lr
                group["_base_lr"] = current_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=self.eta_min,
            )
            LOGGER.info(
                f"Extending cosine LR from epoch {resumed_epoch}/{previous_epochs} "
                f"to {self.epochs}: continuing from checkpoint LR over "
                f"{remaining_epochs} epochs"
            )
            return scheduler

        if "scheduler" in ckpt:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(self.epochs - self.warmup_epochs, 1),
                eta_min=self.eta_min,
            )
            scheduler.load_state_dict(ckpt["scheduler"])
            LOGGER.info(f"Restored exact scheduler state from epoch {resumed_epoch}")
            return scheduler

        if self.warmup_epochs > 0 and resumed_epoch < self.warmup_epochs:
            warmup_factor = max(resumed_epoch, 1) / self.warmup_epochs
            for group in optimizer.param_groups:
                base_lr = float(group.get("_base_lr", group.get("initial_lr", group["lr"])))
                group["_base_lr"] = base_lr
                group["initial_lr"] = base_lr

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(self.epochs - self.warmup_epochs, 1),
                eta_min=self.eta_min,
            )
            scheduler.last_epoch = 0
            for group in optimizer.param_groups:
                group["lr"] = float(group["_base_lr"]) * warmup_factor
            LOGGER.info(
                f"Resuming linear LR warmup at epoch {resumed_epoch}/{self.warmup_epochs}: "
                f"warmup_factor={warmup_factor:.3f}"
            )
            return scheduler

        # Normal resume within the active cosine schedule. PyTorch's
        # last_epoch param has off-by-one issues with the incremental get_lr()
        # formula, so set last_epoch and LR via the closed-form cosine.
        cosine_epoch = max(resumed_epoch - self.warmup_epochs, 0)
        new_T_max = max(self.epochs - self.warmup_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=new_T_max,
            eta_min=self.eta_min,
        )
        scheduler.last_epoch = cosine_epoch
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = (
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_epoch / new_T_max)) / 2
            )
        return scheduler

    def _restore_classifier_loss_state(self, ckpt: dict, criterion_id: nn.Module, resume_path: Path) -> None:
        """Restore train-only margin classifier weights when resuming."""
        if self.classifier_loss == "ce":
            return

        state = ckpt.get("classifier_loss_state_dict")
        if state is None:
            LOGGER.warning(
                f"{resume_path} has no classifier_loss_state_dict; "
                f"initializing {self.classifier_loss} classifier from scratch"
            )
            return

        try:
            criterion_id.load_state_dict(state)
            LOGGER.info(f"Restored {self.classifier_loss} classifier state from checkpoint")
        except RuntimeError as exc:
            LOGGER.warning(f"Could not restore {self.classifier_loss} classifier state from {resume_path}: {exc}")

    def _restore_center_loss_state(
        self,
        ckpt: dict,
        criterion_center: CenterLoss,
        model: nn.Module,
        train_loader: DataLoader,
        resume_path: Path,
    ) -> None:
        """Restore center-loss centers, or initialize them for older checkpoints."""
        if self.center_loss_weight <= 0:
            return

        center_state = ckpt.get("center_loss_state_dict")
        if center_state is not None:
            try:
                criterion_center.load_state_dict(center_state)
                LOGGER.info("Restored center loss state from checkpoint")
                return
            except RuntimeError as exc:
                LOGGER.warning(f"Could not restore center loss state from {resume_path}: {exc}")

        if "optimizer_center" in ckpt:
            LOGGER.warning(
                f"{resume_path} has optimizer_center but no center_loss_state_dict; "
                "initializing center-loss centers from resumed model features"
            )
        self._initialize_center_loss_from_features(model, criterion_center, train_loader)

    def _initialize_center_loss_from_features(
        self,
        model: nn.Module,
        criterion_center: CenterLoss,
        train_loader: DataLoader,
    ) -> None:
        """Initialize missing center-loss centers from per-class feature means."""
        was_training = model.training
        model.eval()

        centers = torch.zeros_like(criterion_center.centers.data)
        counts = torch.zeros(criterion_center.num_classes, device=self.device)

        try:
            with torch.no_grad():
                for batch in train_loader:
                    imgs, pids = batch[:2]
                    imgs = imgs.to(self.device)
                    pids = pids.to(self.device)
                    output = model(imgs)
                    _, features = self._split_model_output(output)
                    center_features = self._center_features(features)
                    if center_features is None:
                        continue

                    center_features = center_features.detach()
                    if center_features.shape[1] != criterion_center.feat_dim:
                        raise RuntimeError(
                            "Center feature dimension does not match center-loss checkpoint state: "
                            f"{center_features.shape[1]} != {criterion_center.feat_dim}"
                        )

                    valid = (pids >= 0) & (pids < criterion_center.num_classes)
                    if not valid.any():
                        continue

                    valid_pids = pids[valid].long()
                    centers.index_add_(0, valid_pids, center_features[valid])
                    counts.index_add_(0, valid_pids, torch.ones_like(valid_pids, dtype=counts.dtype))
        finally:
            if was_training:
                model.train()

        seen = counts > 0
        if not seen.any():
            LOGGER.warning("Could not initialize center-loss centers: no valid class features found")
            return

        centers[seen] = centers[seen] / counts[seen].unsqueeze(1)
        criterion_center.centers.data[seen] = centers[seen]
        LOGGER.info(
            f"Initialized center-loss centers from resumed model features "
            f"({int(seen.sum().item())}/{criterion_center.num_classes} classes)"
        )
