"""Parameter grouping, trainability schedules, and EMA policy."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from boxmot.utils import logger as LOGGER


class _OptimizationMixin:
    def _apply_vit_training_defaults(self) -> None:
        """Apply transformer training conveniences unless the caller set values explicitly."""
        # AdamW uses decoupled weight decay: effective WD = lr x wd.
        # The default wd=5e-4 (calibrated for Adam L2-reg) gives negligible
        # regularization with AdamW, so use the transformer recipe default unless the
        # caller intentionally passed a lower value for an ablation.
        if "weight_decay" not in self.explicit_hparams and self.weight_decay < 0.01:
            self.weight_decay = 0.1

        if "warmup_epochs" not in self.explicit_hparams and self.warmup_epochs <= 10:
            self.warmup_epochs = 20

        # Transformer-style ReID backbones with AdamW need ~2x higher LR than CNNs with Adam. Preserve
        # explicit lower LRs so LR sweeps test the requested value.
        if "lr" not in self.explicit_hparams and self.lr <= 3.5e-4:
            self.lr = 7e-4

        # Transformer-style ReID backbones need stronger center loss to tighten positive clusters. Preserve
        # explicit zero so loss ablations can remove center loss.
        if (
            "center_loss_weight" not in self.explicit_hparams
            and self.loss_type != "ms"
            and self.center_loss_weight <= 5e-4
        ):
            self.center_loss_weight = 5e-3

    def _vit_layer_id_for_param(self, name: str, depth: int) -> int:
        """Map a parameter to its LR-decay unit.

        Flat ViTs expose individual transformer blocks, while hierarchical
        backbones such as CSL-TinyViT expose stages through ``layers``. The
        configured ``layer_decay`` therefore means per-block decay for the
        former and explicitly per-stage decay for the latter.
        """
        if name.startswith(("patch_embed", "cls_token", "pos_embed")):
            return 0
        # "blocks." is the standard ViT naming; "layers." is used by
        # CSL-TinyViT (self.layers registered first, self.blocks alias).
        if name.startswith(("blocks.", "layers.")):
            return int(name.split(".")[1]) + 1
        return depth + 1

    def _vit_lr_scale_for_param(self, name: str, depth: int) -> float:
        """Return the LR scale for a transformer parameter under the active LR profile."""
        if name.startswith("head.mcpt."):
            return self.mcpt_lr_multiplier
        if self._is_reid_adaptation_param(name):
            return 1.0

        layer_id = self._vit_layer_id_for_param(name, depth)
        if self.vit_lr_profile == "reid_lrd":
            if name.startswith("patch_embed") or name.startswith(("blocks.0.", "layers.0.")):
                return 0.05
            if name.startswith(("blocks.1.", "layers.1.")):
                return 0.10
            if name.startswith(("blocks.2.", "layers.2.")):
                return 0.25
            if name.startswith(("blocks.3.", "layers.3.")):
                return 0.50
            return 1.0

        stage_decay = self.layer_decay
        return stage_decay ** (depth + 1 - layer_id)

    @staticmethod
    def _is_vit_no_weight_decay_param(
        name: str,
        param: nn.Parameter,
        owner: nn.Module | None,
        explicit_no_wd: set[str],
    ) -> bool:
        """Return whether a transformer parameter must bypass weight decay.

        Shape and owning-module checks avoid name collisions such as
        ``bn_global.classifier.weight`` and ``reduction`` layers whose path
        happens to contain ``bn`` or ``ln``. Those matrix/kernel weights must
        still be regularized. Scalar/vector gates, normalization affine
        parameters, real biases, and positional/token seeds remain exempt.
        """
        if name in explicit_no_wd or param.ndim <= 1:
            return True
        if isinstance(
            owner,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            ),
        ):
            return True

        leaf_name = name.rsplit(".", 1)[-1]
        if (
            leaf_name == "bias"
            or leaf_name.endswith("_bias")
            or leaf_name.startswith("attention_bias")
        ):
            return True
        return leaf_name in {
            "cls_token",
            "pos_embed",
            "register_seed",
            "identity_register_seed",
            "body_slot_seed",
            "body_slot_roles",
            "row_logits",
            "dataset_row_logits",
        }

    def _build_vit_param_groups(self, model: nn.Module) -> list:
        """Build parameter groups with layer-decay LR and no-WD filtering.

        Layer-decay assigns geometrically decreasing LR to earlier units. A
        unit is a block for flat ViTs and a stage for hierarchical TinyViTs:
            lr_scale = layer_decay ** (depth - unit_idx)
        Patch embed and pos_embed get the lowest LR; the classifier head
        gets the base LR.

        No weight decay is applied to bias, normalization, token/register,
        logit, and scalar residual-gate parameters.
        """
        depth = getattr(model, "depth", len(model.blocks))

        explicit_no_wd: set[str] = set()
        no_weight_decay = getattr(model, "no_weight_decay", None)
        if callable(no_weight_decay):
            explicit_no_wd.update(str(name) for name in no_weight_decay())
        parameter_owners = {
            id(parameter): module
            for module in model.modules()
            for parameter in module.parameters(recurse=False)
        }

        param_groups: dict[str, dict] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lr_scale = self._vit_lr_scale_for_param(name, depth)
            no_weight_decay = self._is_vit_no_weight_decay_param(
                name,
                param,
                parameter_owners.get(id(param)),
                explicit_no_wd,
            )
            wd = 0.0 if no_weight_decay else self.weight_decay

            is_reid_adaptation = self._is_reid_adaptation_param(name)
            group_key = f"lr_{lr_scale:.6g}_wd_{wd}"
            if self.gradual_unfreeze:
                role = "reid" if is_reid_adaptation else "backbone"
                group_key = f"{role}_{group_key}"
            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr": self.lr * lr_scale,
                    "weight_decay": wd,
                    "is_head": is_reid_adaptation,
                    "is_backbone": not is_reid_adaptation,
                    "lr_scale": lr_scale,
                    "no_weight_decay": no_weight_decay,
                }
            else:
                param_groups[group_key]["is_head"] |= is_reid_adaptation
                param_groups[group_key]["is_backbone"] |= not is_reid_adaptation
            param_groups[group_key]["params"].append(param)

        LOGGER.info(
            f"Transformer param groups: {len(param_groups)} groups, lr_profile={self.vit_lr_profile}, depth={depth}"
        )
        return list(param_groups.values())

    def _build_cnn_param_groups(self, model: nn.Module) -> list[dict]:
        """Build Adam parameter groups for CNN models.

        Keep the historical single-group behavior unless a staged freeze/warmup
        schedule needs backbone and ReID-specific parameters to have separate LR
        state.
        """
        if not (self.gradual_unfreeze or self.head_warmup_epochs > 0 or self.backbone_freeze_epochs > 0):
            return [{"params": model.parameters()}]

        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if self._is_reid_adaptation_param(name):
                head_params.append(param)
            else:
                backbone_params.append(param)

        parameter_groups = []
        if backbone_params:
            parameter_groups.append(
                {
                    "params": backbone_params,
                    "is_head": False,
                    "is_backbone": True,
                }
            )
        if head_params:
            parameter_groups.append(
                {
                    "params": head_params,
                    "is_head": True,
                    "is_backbone": False,
                }
            )
        return parameter_groups

    def _build_mobilenetv4_param_groups(self, model: nn.Module) -> list[dict]:
        """Build AdamW groups for MobileNetV4 with no decay on norm/bias params."""
        parameter_owners = {
            id(parameter): module
            for module in model.modules()
            for parameter in module.parameters(recurse=False)
        }
        grouped: dict[tuple[bool, bool, float], dict] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_head = self._is_reid_adaptation_param(name)
            no_decay = self._is_cnn_no_weight_decay_param(
                name,
                param,
                parameter_owners.get(id(param)),
            )
            lr_scale = (
                self.mcpt_lr_multiplier
                if name.startswith("head.mcpt.")
                else 1.0
            )
            key = (is_head, no_decay, lr_scale)
            if key not in grouped:
                grouped[key] = {
                    "params": [],
                    "lr": (
                        self.lr
                        if is_head
                        else self.lr * self.backbone_lr_mult
                    )
                    * lr_scale,
                    "weight_decay": 0.0 if no_decay else self.weight_decay,
                    "is_head": is_head,
                    "is_backbone": not is_head,
                    "lr_scale": lr_scale,
                    "no_weight_decay": no_decay,
                }
            grouped[key]["params"].append(param)

        return list(grouped.values())

    @staticmethod
    def _is_cnn_no_weight_decay_param(
        name: str,
        param: nn.Parameter,
        owner: nn.Module | None = None,
    ) -> bool:
        """Exempt only true bias, scalar/vector, and normalization parameters."""
        return (
            name.endswith(".bias")
            or param.ndim <= 1
            or isinstance(
                owner,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                    nn.GroupNorm,
                    nn.LayerNorm,
                ),
            )
        )

    def _head_warmup_active(self, epoch: int) -> bool:
        """Return whether this epoch should train only neck/head parameters.

        If backbone freeze is configured, the head-only phase starts after the
        freeze warm-start instead of being silently overlapped and skipped.
        """
        if self.head_warmup_epochs <= 0:
            return False
        start_epoch = self.backbone_freeze_epochs if self.backbone_freeze_epochs > 0 else 0
        return start_epoch < epoch <= start_epoch + self.head_warmup_epochs

    def _head_warmup_start_epoch(self) -> int:
        """Return the first epoch of the effective head-only warmup phase."""
        return self.backbone_freeze_epochs + 1 if self.backbone_freeze_epochs > 0 else 1

    def _backbone_freeze_active(self, epoch: int) -> bool:
        """Return whether this epoch should keep pretrained backbone stages frozen."""
        return self.backbone_freeze_epochs > 0 and epoch <= self.backbone_freeze_epochs

    def _gradual_unfreeze_phase(self, epoch: int) -> str | None:
        """Return the active staged-unfreeze phase for this epoch."""
        if not self.gradual_unfreeze:
            return None
        if epoch <= self.gradual_unfreeze_head_epochs:
            return "head"
        if epoch <= self.gradual_unfreeze_stage_epochs:
            return "stage"
        return "full"

    def _gradual_backbone_lr_active(self, epoch: int) -> bool:
        """Return whether trainable backbone groups should use the temporary LR drop."""
        return (
            self.gradual_unfreeze
            and self.gradual_unfreeze_backbone_lr_epochs > 0
            and epoch > self.gradual_unfreeze_head_epochs
            and epoch <= self.gradual_unfreeze_stage_epochs + self.gradual_unfreeze_backbone_lr_epochs
        )

    def _effective_id_loss_weight(self, epoch: int) -> float:
        """Return ID loss weight after applying any temporary early CE boost."""
        if self.early_id_loss_epochs > 0 and epoch <= self.early_id_loss_epochs:
            return self.early_id_loss_weight if self.early_id_loss_weight > 0 else self.id_loss_weight
        return self.id_loss_weight

    def _effective_center_loss_weight(self, epoch: int) -> float:
        """Return center loss weight after applying the optional epoch ramp."""
        if self.center_loss_weight <= 0 or self.center_loss_ramp_end_epoch <= 0:
            return self.center_loss_weight
        if epoch <= self.center_loss_ramp_start_epoch:
            return 0.0
        if epoch <= self.center_loss_ramp_end_epoch:
            span = self.center_loss_ramp_end_epoch - self.center_loss_ramp_start_epoch
            return self.center_loss_weight * ((epoch - self.center_loss_ramp_start_epoch) / span)
        return self.center_loss_weight

    def _effective_csmm_loss_weight(self, epoch: int) -> float:
        """Return the CSMM weight after the delayed linear ramp."""
        if self.csmm_loss_weight <= 0 or epoch <= self.csmm_start_epoch:
            return 0.0
        if epoch < self.csmm_ramp_end_epoch:
            span = self.csmm_ramp_end_epoch - self.csmm_start_epoch
            return self.csmm_loss_weight * ((epoch - self.csmm_start_epoch) / span)
        return self.csmm_loss_weight

    def _effective_treeboost_loss_weight(self, epoch: int) -> float:
        """Return the TreeBoost-AP weight after its delayed linear ramp."""
        if self.treeboost_loss_weight <= 0 or epoch <= self.treeboost_start_epoch:
            return 0.0
        if epoch < self.treeboost_ramp_end_epoch:
            span = self.treeboost_ramp_end_epoch - self.treeboost_start_epoch
            return self.treeboost_loss_weight * ((epoch - self.treeboost_start_epoch) / span)
        return self.treeboost_loss_weight

    def _effective_late_interaction_scale(self, epoch: int) -> float:
        """Return the shared matcher/distillation ramp multiplier."""
        if not self.hierarchical_late_interaction or epoch <= self.late_interaction_start_epoch:
            return 0.0
        if epoch < self.late_interaction_ramp_end_epoch:
            span = self.late_interaction_ramp_end_epoch - self.late_interaction_start_epoch
            return (epoch - self.late_interaction_start_epoch) / span
        return 1.0

    @staticmethod
    def _is_head_or_neck_param(name: str) -> bool:
        return name in {
            "fusion_logits",
            "identity_register_seed",
            "body_slot_seed",
            "body_slot_roles",
        } or name.startswith(
            (
                "head.",
                "neck.",
                "spatial_neck.",
                "fpn_projections.",
                "output_norms.",
                "post_fusion_mixer_module.",
                "identity_register_modules.",
                "body_slot_modules.",
            )
        )

    @staticmethod
    def _is_reid_adaptation_param(name: str) -> bool:
        return (
            name == "fusion_logits"
            or name == "identity_register_seed"
            or name in {"body_slot_seed", "body_slot_roles"}
            or name.startswith(
                (
                    "head.",
                    "neck.",
                    "spatial_neck.",
                    "feature_fusion_module.",
                    "identity_register_modules.",
                    "body_slot_modules.",
                    "fpn_projections.",
                    "output_norms.",
                    "post_fusion_mixer_module.",
                )
            )
            or ".reid_adapters." in name
        )

    @staticmethod
    def _last_vit_stage_index(model: nn.Module) -> int | None:
        layers = getattr(model, "layers", None)
        if layers is not None:
            return len(layers) - 1
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            return len(blocks) - 1
        return None

    @staticmethod
    def _is_last_vit_stage_param(name: str, stage_index: int | None) -> bool:
        if stage_index is None or stage_index < 0:
            return False
        return name.startswith((f"layers.{stage_index}.", f"blocks.{stage_index}."))

    @staticmethod
    def _last_backbone_stage_prefixes(model: nn.Module) -> tuple[str, ...]:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return ()
        for attr_name in ("stages", "blocks", "features"):
            container = getattr(backbone, attr_name, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                return (f"backbone.{attr_name}.{len(container) - 1}.",)
        return ()

    def _is_last_stage_param(self, model: nn.Module, name: str, stage_index: int | None) -> bool:
        return self._is_last_vit_stage_param(name, stage_index) or name.startswith(
            self._last_backbone_stage_prefixes(model)
        )

    @staticmethod
    def _is_attention_param(name: str) -> bool:
        return ".attn." in name

    @staticmethod
    def _initial_parameter_trainability(model: nn.Module) -> dict[str, bool]:
        """Return the model's permanent trainability contract.

        Freeze schedules are temporary. They must not re-enable parameters
        that a module intentionally constructed as frozen, such as BNNeck
        biases or EMA-teacher weights. Cache the contract before the first
        schedule transition so later phases can restore only parameters that
        belong to the optimizer's trainable model.
        """
        attribute = "_boxmot_initial_parameter_trainability"
        trainability = getattr(model, attribute, None)
        named_parameters = tuple(model.named_parameters())
        parameter_names = {name for name, _ in named_parameters}
        if trainability is None:
            trainability = {name: bool(parameter.requires_grad) for name, parameter in named_parameters}
            setattr(model, attribute, trainability)
        elif set(trainability) != parameter_names:
            raise RuntimeError("Model parameters changed after the training freeze contract was established")
        return trainability

    def _set_scheduled_trainability(
        self,
        model: nn.Module,
        schedule_allows: Callable[[str], bool],
    ) -> None:
        """Apply a temporary schedule without overriding permanent freezes."""
        initial = self._initial_parameter_trainability(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(initial[name] and bool(schedule_allows(name)))

    def _set_head_warmup_trainability(self, model: nn.Module, enabled: bool) -> None:
        """Freeze/unfreeze backbone parameters for head-only warmup."""
        self._set_scheduled_trainability(
            model,
            lambda name: not enabled or self._is_head_or_neck_param(name),
        )

    def _set_backbone_freeze_trainability(self, model: nn.Module, enabled: bool) -> None:
        """Freeze pretrained backbone parameters while keeping ReID-specific modules trainable."""
        adapt_spatial_norm = enabled and getattr(model, "timm_head_mode", None) == "spatial_adapt_norm"
        self._set_scheduled_trainability(
            model,
            lambda name: (
                not enabled
                or self._is_reid_adaptation_param(name)
                or (adapt_spatial_norm and name.startswith("backbone.norm_head."))
            ),
        )
        if not enabled:
            return
        for module_name in ("patch_embed", "layers"):
            module = getattr(model, module_name, None)
            if module is not None:
                module.eval()
        if getattr(model, "layers", None) is None:
            blocks = getattr(model, "blocks", None)
            if blocks is not None:
                blocks.eval()
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            backbone.eval()
            if adapt_spatial_norm:
                norm_head = getattr(backbone, "norm_head", None)
                if norm_head is not None:
                    norm_head.train()

    def _set_gradual_unfreeze_trainability(self, model: nn.Module, phase: str) -> None:
        """Apply staged CSL-TinyViT unfreeze trainability for the active phase."""
        last_stage_index = self._last_vit_stage_index(model)

        def schedule_allows(name: str) -> bool:
            if phase == "head":
                return self._is_reid_adaptation_param(name)
            if phase == "stage":
                return self._is_reid_adaptation_param(name) or (
                    self._is_last_stage_param(model, name, last_stage_index) and not self._is_attention_param(name)
                )
            return True

        self._set_scheduled_trainability(model, schedule_allows)

        if phase == "full":
            return

        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            patch_embed.eval()
        layers = getattr(model, "layers", None)
        if layers is not None:
            if phase == "head":
                layers.eval()
                return
            for index, layer in enumerate(layers):
                layer.train(index == last_stage_index)
            return
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            if phase == "head":
                blocks.eval()
                return
            for index, block in enumerate(blocks):
                block.train(index == last_stage_index)
            return

        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return
        if phase == "head":
            backbone.eval()
            return
        for attr_name in ("stages", "blocks", "features"):
            container = getattr(backbone, attr_name, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                backbone.eval()
                last_index = len(container) - 1
                for index, stage in enumerate(container):
                    stage.train(index == last_index)
                return

    def _apply_head_warmup_lrs(self, optimizer) -> list[float]:
        """Use zero LR for backbone groups and a boosted LR for head groups."""
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            scheduled_lr = group.get("lr", self.lr)
            group["lr"] = scheduled_lr * self.head_warmup_lr_mult if group.get("is_head", False) else 0.0
        return original_lrs

    def _apply_epoch_warmup_lrs(self, optimizer, epoch: int) -> bool:
        """Set the linear-warmup LR used by the current epoch.

        Applying this at epoch start makes epochs ``1..warmup_epochs`` use
        factors ``1/N..N/N``. Checkpoints consequently contain the LR that was
        actually used for their epoch instead of the following epoch's LR.
        """
        if self.warmup_epochs <= 0 or epoch > self.warmup_epochs:
            return False
        if epoch < 1:
            raise ValueError(f"Training epochs are one-indexed, got {epoch}")
        warmup_factor = epoch / self.warmup_epochs
        for group in optimizer.param_groups:
            base_lr = group.get("_base_lr", group.get("initial_lr", self.lr))
            group["lr"] = base_lr * warmup_factor
        return True

    @staticmethod
    def _group_has_trainable_params(group: dict) -> bool:
        """Return whether an optimizer group currently owns any trainable parameter."""
        return any(param.requires_grad for param in group.get("params", ()))

    def _apply_gradual_backbone_lrs(self, optimizer, epoch: int | None = None) -> list[float]:
        """Zero frozen groups and temporarily reduce trainable backbone LR groups."""
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        backbone_lr_active = True if epoch is None else self._gradual_backbone_lr_active(epoch)
        for group in optimizer.param_groups:
            if not self._group_has_trainable_params(group):
                group["lr"] = 0.0
            elif backbone_lr_active and group.get("is_backbone", False):
                group["lr"] *= self.gradual_unfreeze_backbone_lr_mult
        return original_lrs

    def _optimizer_lr_summary(self, optimizer) -> tuple[float, float, float]:
        """Return max active LR plus active backbone/head LR summaries."""
        active_lrs = []
        backbone_lrs = []
        head_lrs = []
        for group in optimizer.param_groups:
            if not self._group_has_trainable_params(group):
                continue
            lr = float(group["lr"])
            if lr <= 0:
                continue
            active_lrs.append(lr)
            if group.get("is_backbone", False):
                backbone_lrs.append(lr)
            if group.get("is_head", False):
                head_lrs.append(lr)
        fallback = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        return (
            max(active_lrs, default=fallback),
            max(backbone_lrs, default=0.0),
            max(head_lrs, default=0.0),
        )

    @staticmethod
    def _trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
        """Return trainable and total parameter counts."""
        total = 0
        trainable = 0
        for param in model.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        return trainable, total

    @staticmethod
    @torch.no_grad()
    def _update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
        """Update EMA parameters and persistent buffers by state-dict key.

        Some backbones create non-persistent inference caches dynamically. Such
        buffers can differ between the live and EMA module trees and must not
        shift positional parameter/buffer pairing. They are intentionally
        absent from ``state_dict`` and are regenerated by the owning module.
        """
        ema_state = ema_model.state_dict(keep_vars=True)
        model_state = model.state_dict(keep_vars=True)
        if ema_state.keys() != model_state.keys():
            missing = sorted(model_state.keys() - ema_state.keys())
            unexpected = sorted(ema_state.keys() - model_state.keys())
            raise RuntimeError(
                f"EMA state structure differs from the live model: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.shape != model_value.shape:
                raise RuntimeError(
                    f"EMA state shape mismatch for {name}: {tuple(ema_value.shape)} != {tuple(model_value.shape)}"
                )
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    @staticmethod
    def _ema_decay_for_update(target_decay: float, update: int) -> float:
        """Ramp EMA decay so initialization cannot dominate early validation.

        A fixed high decay retains randomly initialized ReID layers for many
        epochs on small datasets. Capping it with an update-aware schedule
        makes the EMA follow the live model closely at startup, then smoothly
        approaches the configured long-term decay.
        """
        if update < 1:
            raise ValueError("EMA update must be positive")
        warmup_decay = (1.0 + update) / (10.0 + update)
        return min(float(target_decay), warmup_decay)

    def _part_relation_enabled(self) -> bool:
        """Return whether the training-only part EMA teacher is required."""
        return self.part_relation_weight > 0 or self.part_to_global_weight > 0

    def _effective_ema_decay(self) -> float:
        """Resolve validation EMA or the training-only part-teacher EMA."""
        if self.ema_decay:
            return float(self.ema_decay)
        if self._part_relation_enabled():
            return self.part_relation_teacher_momentum
        return 0.0
