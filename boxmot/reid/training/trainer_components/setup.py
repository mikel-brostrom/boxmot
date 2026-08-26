"""Model, criterion, and optimizer-bundle construction."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from boxmot.reid.backbones import get_backbone_spec
from boxmot.reid.backbones.families.csl_tinyvit.pretrained import (
    load_pretrained_tinyvit_checkpoint,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.ablation import resolve_csl_tinyvit_ablation
from boxmot.reid.training.losses import (
    METRIC_LOSS_REGISTRY,
    AdaSPLoss,
    ArcFaceLoss,
    CenterLoss,
    CosFaceLoss,
    CrossEntropyLabelSmooth,
    CrossScaleMajorityMarginLoss,
    TreeBoostAPLoss,
)
from boxmot.reid.training.model_options import build_reid_model_kwargs
from boxmot.reid.training.recipes import (
    TrainingRecipe,
    build_training_recipe,
    default_recipe_for_family,
)
from boxmot.reid.training.trainer_components.types import (
    LossBundle,
    ModelBundle,
    OptimizationBundle,
)
from boxmot.utils import logger as LOGGER


class _SetupMixin:
    def _build_model_bundle(self, num_classes: int) -> ModelBundle:
        """Build live and optional EMA models with finalized training-family defaults."""
        LOGGER.info(f"Building model '{self.model_name}' with {num_classes} classes, loss='{self.loss_type}'")
        pre_build_recipe = self._resolve_training_recipe_for_model_name()
        if pre_build_recipe is not None:
            pre_build_recipe.apply_pre_build_defaults(self)
            self._validate_config()
        configured_pretrained = self.pretrained
        configured_pretrained_weights = getattr(self, "pretrained_weights", None)
        use_local_pretraining = bool(configured_pretrained_weights) and not self.resume
        if self.resume and (configured_pretrained or configured_pretrained_weights):
            LOGGER.info("Resume requested: skipping redundant pretrained-weight initialization")
            self.pretrained = False
        elif use_local_pretraining:
            LOGGER.info(
                "Local human-pretrained TinyViT backbone requested: "
                "skipping the model-zoo initialization"
            )
            self.pretrained = False
        try:
            model = self._build_model(num_classes)
            if use_local_pretraining:
                if not self.model_name.startswith("csl_tinyvit"):
                    raise ValueError(
                        "pretrained_weights currently supports CSL-TinyViT exact-backbone "
                        f"exports, got model={self.model_name!r}"
                    )
                load_pretrained_tinyvit_checkpoint(
                    model,
                    configured_pretrained_weights,
                )
            model = model.to(self.device)
        finally:
            self.pretrained = configured_pretrained
        if hasattr(model, "img_size") and model.img_size != self.img_size:
            LOGGER.info(f"Syncing img_size with model architecture: {self.img_size} → {model.img_size}")
            self.img_size = model.img_size

        recipe = self._resolve_training_recipe(model)
        recipe.apply_defaults(self)
        self._validate_config()

        ema_model: Optional[nn.Module] = None
        effective_ema_decay = self._effective_ema_decay()
        if effective_ema_decay:
            ema_model = copy.deepcopy(model)
            for parameter in ema_model.parameters():
                parameter.requires_grad_(False)
            purpose = (
                "part-relation teacher"
                if not self.ema_decay and self._part_relation_enabled()
                else "validation and training"
            )
            LOGGER.info(
                f"EMA model enabled (target_decay={effective_ema_decay}, "
                "startup_schedule=update_warmup, "
                f"purpose={purpose})"
            )
        return ModelBundle(
            model=model,
            ema_model=ema_model,
            val_model=(ema_model if ema_model is not None and bool(self.ema_decay) else model),
            is_transformer=recipe.family == "transformer",
            training_family=recipe.family,
            recipe=recipe,
        )

    def _build_loss_bundle(self, model: ModelBundle, num_classes: int) -> LossBundle:
        """Resolve and construct ID, metric, and center-loss modules."""
        recipe = self._recipe_for_bundle(model)
        label_smooth = recipe.resolve_label_smooth(self, self.label_smooth)

        soft_margin = self._use_soft_margin_triplet(recipe.default_triplet_soft_margin)
        criterion_metric = None
        metric_objective_active = self.metric_loss_weight > 0 or (
            self.compact_deployment_head and self.compact_metric_loss_weight > 0
        )
        if self.loss_type in METRIC_LOSS_REGISTRY and metric_objective_active:
            metric_loss_class = METRIC_LOSS_REGISTRY[self.loss_type]
            model_metric_kwargs = getattr(model.model, "metric_loss_kwargs", {})
            if not isinstance(model_metric_kwargs, dict):
                raise TypeError("model.metric_loss_kwargs must be a dictionary")
            metric_loss_kwargs = model_metric_kwargs.get(self.loss_type, {})
            if not isinstance(metric_loss_kwargs, dict):
                raise TypeError(
                    f"model.metric_loss_kwargs[{self.loss_type!r}] must be a dictionary"
                )
            criterion_metric = (
                metric_loss_class(margin=self.margin, soft_margin=soft_margin)
                if self.loss_type == "triplet"
                else metric_loss_class(**metric_loss_kwargs)
            )
            details = f" with {metric_loss_kwargs}" if metric_loss_kwargs else ""
            LOGGER.info(f"Metric loss: {metric_loss_class.__name__}{details}")
        criterion_csmm = None
        if self.csmm_loss_weight > 0:
            criterion_csmm = CrossScaleMajorityMarginLoss(
                margin=self.csmm_margin,
                temperature=self.csmm_temperature,
                topk_negatives=self.csmm_topk_negatives,
            )
            LOGGER.info(
                "CSMM auxiliary loss: "
                f"weight={self.csmm_loss_weight:g}, margin={self.csmm_margin:g}, "
                f"temperature={self.csmm_temperature:g}, topk={self.csmm_topk_negatives}, "
                f"ramp={self.csmm_start_epoch}-{self.csmm_ramp_end_epoch}"
            )
        criterion_treeboost = None
        if self.treeboost_loss_weight > 0:
            criterion_treeboost = TreeBoostAPLoss(
                coarse_coefficient=self.treeboost_coarse_coefficient,
                fine_coefficient=self.treeboost_fine_coefficient,
                node_coefficient=self.treeboost_node_coefficient,
                regression_coefficient=self.treeboost_regression_coefficient,
                difficulty_floor=self.treeboost_difficulty_floor,
                regression_tolerance=self.treeboost_regression_tolerance,
                temperature=self.treeboost_temperature,
            )
            LOGGER.info(
                "TreeBoost-AP auxiliary loss: "
                f"weight={self.treeboost_loss_weight:g}, "
                f"node={self.treeboost_node_coefficient:g}, "
                f"regression={self.treeboost_regression_coefficient:g}, "
                f"temperature={self.treeboost_temperature:g}, "
                f"ramp={self.treeboost_start_epoch}-{self.treeboost_ramp_end_epoch}"
            )
        criterion_adasp = None
        if self.adasp_loss_weight > 0:
            criterion_adasp = AdaSPLoss(
                temperature=self.adasp_temperature,
            )
            LOGGER.info(
                "AdaSP loss: "
                f"weight={self.adasp_loss_weight:g}, "
                f"scale={self.adasp_scale:g}, "
                f"temperature={self.adasp_temperature:g}"
            )

        if (
            self.loss_type == "ms"
            and self.center_loss_weight > 0
            and not bool(getattr(model.model, "allow_center_with_ms", False))
        ):
            LOGGER.info("MS loss active: disabling center loss (redundant)")
            self.center_loss_weight = 0

        metric_dim = self._probe_feat_dim(model.model)
        self._initialize_retrieval_training_components(
            self._deployment_feature_dim(model.model, metric_dim)
        )
        classifier_dim = self._probe_classifier_feat_dim(model.model) if self.classifier_loss != "ce" else metric_dim
        criterion_id = self._build_classifier_loss(
            num_classes,
            classifier_dim,
            label_smooth,
        ).to(self.device)
        criterion_center = CenterLoss(num_classes, metric_dim).to(self.device)
        return LossBundle(
            criterion_id=criterion_id,
            criterion_metric=criterion_metric,
            criterion_csmm=criterion_csmm,
            criterion_treeboost=criterion_treeboost,
            criterion_adasp=criterion_adasp,
            criterion_center=criterion_center,
            label_smooth=label_smooth,
            soft_margin=soft_margin,
            metric_dim=metric_dim,
            classifier_dim=classifier_dim,
        )

    def _build_optimization_bundle(
        self,
        model: ModelBundle,
        losses: LossBundle,
    ) -> OptimizationBundle:
        """Build model/center optimizers and the cosine scheduler."""
        classifier_parameters = list(losses.criterion_id.parameters()) if self.classifier_loss != "ce" else []
        recipe = self._recipe_for_bundle(model)
        # Establish the permanent trainability contract at the same boundary
        # used to decide optimizer membership. Epoch schedules may only narrow
        # this set temporarily; they must never introduce new trainable model
        # parameters after optimizer construction.
        self._initial_parameter_trainability(model.model)
        parameter_groups = recipe.build_param_groups(self, model.model)
        if classifier_parameters:
            parameter_groups.append(recipe.classifier_param_group(self, classifier_parameters))
        optimizer = recipe.build_optimizer(self, parameter_groups)
        if recipe.family == "transformer":
            LOGGER.info(
                f"Transformer training: {recipe.optimizer_name} (lr={self.lr:.1e}, wd={self.weight_decay}), "
                f"lr_profile={self.vit_lr_profile}, grad clip={recipe.grad_clip:.1f}, DropPath enabled"
            )
        else:
            LOGGER.info(
                f"{recipe.family.upper()} training: {recipe.optimizer_name} "
                f"(lr={self.lr:.1e}, wd={self.weight_decay}), "
                f"grad clip={recipe.grad_clip:.1f}"
            )

        optimizer_center = torch.optim.SGD(losses.criterion_center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.eta_min,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["_base_lr"] = parameter_group["lr"]
            if self.warmup_epochs > 0:
                parameter_group["lr"] /= self.warmup_epochs
        return OptimizationBundle(
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            scheduler=scheduler,
            grad_clip=recipe.grad_clip,
        )

    def _effective_model_kwargs(
        self,
        model: nn.Module | None = None,
    ) -> dict[str, object]:
        """Return the exact constructor contract used for build or replay."""
        model_kwargs = build_reid_model_kwargs(self)
        if self.timm_model_name:
            model_kwargs["timm_model_name"] = self.timm_model_name
        if self.model_name.startswith("mobilenetv4_"):
            model_kwargs.update(
                timm_head_mode=getattr(
                    model,
                    "timm_head_mode",
                    self.timm_head_mode,
                ),
                mobilenetv4_last_stride=getattr(
                    model,
                    "mobilenetv4_last_stride",
                    self.mobilenetv4_last_stride,
                ),
                mobilenetv4_neck_mode=getattr(
                    model,
                    "mobilenetv4_neck_mode",
                    self.mobilenetv4_neck_mode,
                ),
            )
            resolved_timm_model = getattr(model, "timm_model_name", None)
            if resolved_timm_model:
                model_kwargs["timm_model_name"] = resolved_timm_model
        return model_kwargs

    def _build_model(self, num_classes: int) -> nn.Module:
        model_kwargs = self._effective_model_kwargs()
        model = ReIDModelRegistry.build_model(
            name=self.model_name,
            weights=Path(f"{self.model_name}_{self.dataset_name}.pt"),
            num_classes=num_classes,
            loss=self._model_loss_type(),
            pretrained=self.pretrained,
            use_gpu=self.device.type != "cpu",
            **model_kwargs,
        )
        if hasattr(model, "head") and hasattr(model.head, "metric_feature"):
            model.head.metric_feature = self._effective_metric_feature()
        if hasattr(model, "head") and hasattr(model.head, "set_pooling"):
            model.head.set_pooling(self.head_pool)
        if hasattr(model, "head") and hasattr(model.head, "set_branch_metric"):
            model.head.set_branch_metric(self.branch_aware_metric)
        self._log_hierarchical_reid_config(model)
        return model

    @staticmethod
    def _max_drop_path(model: nn.Module) -> float:
        max_drop = 0.0
        for module in model.modules():
            drop_prob = getattr(module, "drop_prob", None)
            if drop_prob is not None:
                max_drop = max(max_drop, float(drop_prob))
        return max_drop

    def _log_hierarchical_reid_config(self, model: nn.Module) -> None:
        """Log active component and architecture settings after construction."""
        if not self.model_name.startswith(("csl_tinyvit", "mobilenetv4")):
            return
        family_label = (
            "CSL-TinyViT"
            if self.model_name.startswith("csl_tinyvit")
            else "MobileNetV4"
        )
        ablation_plan = resolve_csl_tinyvit_ablation(self)
        LOGGER.info(
            f"{family_label} ablation components: "
            + ", ".join(ablation_plan.active_names)
        )
        head = getattr(model, "head", None)
        block_windows = []
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "blocks"):
                layer_windows = [
                    getattr(block, "window_size", None) for block in layer.blocks if hasattr(block, "window_size")
                ]
                if layer_windows:
                    block_windows.append(layer_windows)
        LOGGER.info(
            f"{family_label} active config: "
            f"max_drop_path={self._max_drop_path(model):.3f}, "
            f"metric_feature={getattr(head, 'metric_feature', None)}, "
            f"inference_feature={getattr(head, 'inference_feature', None)}, "
            f"head_type={getattr(model, 'head_type', None)}, "
            "multiscale_channel_alpha="
            f"{getattr(head, 'multiscale_channel_alpha', None)}, "
            f"head_pool={getattr(head, 'head_pool', None)}, "
            f"part_pooling={getattr(head, 'part_pooling', None)}, "
            f"num_part_tokens={getattr(head, 'num_part_tokens', None)}, "
            f"decouple_patterns={getattr(head, 'decouple_patterns', None)}, "
            f"stripe_visibility={getattr(head, 'stripe_visibility', None)}, "
            f"drop_global_aux={getattr(head, 'drop_global_aux_enabled', None)}, "
            f"drop_global_aux_ratio={getattr(head, 'drop_global_aux_ratio', None)}, "
            f"scale_balanced_branches={getattr(head, 'scale_balanced_branches', None)}, "
            f"hierarchical_branch_attention={getattr(head, 'hierarchical_branch_attention_enabled', None)}, "
            f"branch_set_attention={getattr(head, 'branch_set_attention_enabled', None)}, "
            f"multiscale_query_decoder={getattr(head, 'multiscale_query_decoder_enabled', None)}, "
            f"hierarchical_late_interaction={getattr(head, 'hierarchical_late_interaction_enabled', None)}, "
            f"mcpt_mode={getattr(model, 'mcpt_mode', None)}, "
            f"mcpt_hidden_dim={getattr(model, 'mcpt_hidden_dim', None)}, "
            f"mcpt_max_displacement={getattr(model, 'mcpt_max_displacement', None)}, "
            f"feature_fusion={getattr(model, 'feature_fusion', None)}, "
            f"pyramid_resize_mode={getattr(model, 'pyramid_resize_mode', None)}, "
            f"spatial_conv_mode={getattr(model, 'spatial_conv_mode', None)}, "
            f"post_fusion_mixer={getattr(model, 'post_fusion_mixer', None)}, "
            f"post_fusion_mixer_kernel={getattr(model, 'post_fusion_mixer_kernel', None)}, "
            f"post_fusion_mixer_gamma_init={getattr(model, 'post_fusion_mixer_gamma_init', None)}, "
            f"reid_adapter_stages={getattr(model, 'reid_adapter_stages', None)}, "
            f"reid_adapter_reduction={getattr(model, 'reid_adapter_reduction', None)}, "
            f"reid_adapter_suppression_tau={getattr(model, 'reid_adapter_suppression_tau', None)}, "
            f"attention_window_layout={getattr(model, 'attention_window_layout', None)}, "
            f"attention_bias={getattr(model, 'attention_bias', None)}, "
            "interpolate_pretrained_attention_bias="
            f"{getattr(model, 'interpolate_pretrained_attention_bias', None)}, "
            f"attention_mask={getattr(model, 'attention_mask', None)}, "
            f"attention_shift={getattr(model, 'attention_shift', None)}, "
            f"stage3_global={getattr(model, 'stage3_global', None)}, "
            f"stage3_downsample={getattr(model, 'stage3_downsample', None)}, "
            f"stage2_width_merge_after={getattr(model, 'stage2_width_merge_after', None)}, "
            f"stage2_mlp_ratio={getattr(model, 'stage2_mlp_ratio', None)}, "
            f"stage3_mlp_ratio={getattr(model, 'stage3_mlp_ratio', None)}, "
            f"stage2_depth={getattr(model, 'stage2_depth', None)}, "
            f"stage3_depth={getattr(model, 'stage3_depth', None)}, "
            f"width_first_hierarchy={getattr(model, 'width_first_hierarchy', None)}, "
            "identity_registers="
            f"{getattr(model, 'identity_registers_enabled', None)}, "
            "identity_register_count="
            f"{getattr(model, 'identity_register_count', None)}, "
            "identity_register_dim="
            f"{getattr(model, 'identity_register_dim', None)}, "
            "identity_register_gate_init="
            f"{getattr(model, 'identity_register_gate_init', None)}, "
            f"native_branch_widths={getattr(model, 'native_branch_widths', None)}, "
            f"fine_map_dim={getattr(model, 'fine_map_dim', None)}, "
            f"compact_deployment_head={getattr(model, 'compact_deployment_head', None)}, "
            f"windows={block_windows}"
        )
        match_count = getattr(model, "pretrained_match_count", None)
        total_count = getattr(model, "pretrained_total_count", None)
        if match_count is not None and total_count is not None:
            LOGGER.info(
                f"{family_label} pretrained tensor match count: {match_count}/{total_count} "
                f"from {getattr(model, 'pretrained_url', None)}"
            )

    @staticmethod
    def _declared_feature_dim(
        model: nn.Module,
        attribute_names: tuple[str, ...],
    ) -> int | None:
        """Resolve an explicit training descriptor dimension from model metadata."""
        owners = (model, getattr(model, "head", None))
        for owner in owners:
            if owner is None:
                continue
            for attribute_name in attribute_names:
                value = getattr(owner, attribute_name, None)
                if callable(value):
                    value = value()
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"{owner.__class__.__name__}.{attribute_name} must be a positive integer, "
                        f"got {value!r}"
                    )
                return value
        return None

    def _safe_training_probe(self, model: nn.Module):
        """Run a legacy shape probe without changing model state or RNG streams.

        Canonical backbones expose descriptor dimensions directly. This
        fallback exists for third-party/legacy models whose output contract
        cannot be inferred statically. Training forwards update BatchNorm
        buffers and may clear non-persistent inference caches, so every module
        mode and buffer slot is restored exactly in ``finally``.
        """
        rng_state = self._capture_rng_state()
        module_modes = [(module, module.training) for module in model.modules()]
        module_buffers = []
        for module in model.modules():
            snapshots = {
                name: (buffer, None if buffer is None else buffer.detach().clone())
                for name, buffer in module._buffers.items()
            }
            module_buffers.append(
                (module, snapshots, set(module._non_persistent_buffers_set))
            )

        try:
            model.train()
            dummy = torch.randn(2, 3, *self.img_size, device=self.device)
            with torch.no_grad():
                return model(dummy)
        finally:
            with torch.no_grad():
                for module, snapshots, non_persistent in module_buffers:
                    for name in tuple(module._buffers):
                        if name not in snapshots:
                            del module._buffers[name]
                    for name, (original, snapshot) in snapshots.items():
                        if original is None:
                            module._buffers[name] = None
                        else:
                            original.copy_(snapshot)
                            module._buffers[name] = original
                    module._non_persistent_buffers_set = non_persistent
            for module, training in module_modes:
                module.training = training
            self._restore_rng_state(rng_state)

    def _probe_feat_dim(self, model: nn.Module) -> int:
        """Return the center-loss embedding dimension without mutating the model."""
        declared_dim = self._declared_feature_dim(
            model,
            ("center_dim", "metric_dim"),
        )
        if declared_dim is not None:
            return declared_dim
        LOGGER.warning(
            f"{model.__class__.__name__} has no explicit center/metric dimension; "
            "using a state-preserving training probe"
        )
        out = self._safe_training_probe(model)
        _, features = self._split_model_output(out)
        center_features = self._center_features(features)
        if isinstance(center_features, torch.Tensor):
            return center_features.shape[1]
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0].shape[1]  # multi-branch softmax: list of logits
        return out.shape[1]

    def _probe_classifier_feat_dim(self, model: nn.Module) -> int:
        """Return the margin-classifier descriptor dimension safely."""
        declared_dim = self._declared_feature_dim(
            model,
            ("classifier_dim", "metric_dim"),
        )
        if declared_dim is not None:
            return declared_dim
        LOGGER.warning(
            f"{model.__class__.__name__} has no explicit classifier dimension; "
            "using a state-preserving training probe"
        )
        out = self._safe_training_probe(model)
        _, features = self._split_model_output(out)
        classifier_features = self._classification_features(features)
        if classifier_features is None:
            raise RuntimeError(f"classifier_loss={self.classifier_loss} requires embedding features")
        return classifier_features.shape[1]

    @staticmethod
    def _split_model_output(output):
        """Unpack training output into (logits, features) across backbone contracts."""
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        if isinstance(output, list) and len(output) == 2:
            return output[0], output[1]
        return output, None

    # ------------------------------------------------------------------
    # Training-family helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_vit(model: nn.Module) -> bool:
        """Check if the model is a transformer-like variant."""
        return hasattr(model, "blocks") and hasattr(model, "patch_embed")

    def _backbone_spec(self):
        """Return the registered backbone spec for this trainer, when available."""
        try:
            return get_backbone_spec(self.model_name)
        except KeyError:
            return None

    @classmethod
    def _detect_training_family(cls, model: nn.Module) -> str:
        """Return the canonical training family for a model instance."""
        explicit_family = getattr(model, "training_family", None)
        if explicit_family is not None:
            return cls._normalize_training_family(explicit_family)
        return "transformer" if cls._is_vit(model) else "cnn"

    @staticmethod
    def _normalize_training_family(family: str) -> str:
        family = str(family).lower()
        if family in {"vit", "transformer"}:
            return "transformer"
        if family in {"cnn", "convnet"}:
            return "cnn"
        if family in {"hybrid", "legacy"}:
            return family
        raise ValueError(f"Unsupported training_family={family!r}")

    @classmethod
    def _training_recipe_for_family(cls, family: str) -> TrainingRecipe:
        """Build the recipe object for a canonical training family."""
        return default_recipe_for_family(cls._normalize_training_family(family))

    def _resolve_training_recipe_for_model_name(self) -> TrainingRecipe | None:
        """Resolve recipes that need defaults before model construction."""
        spec = self._backbone_spec()
        return build_training_recipe(spec=spec) if spec is not None else None

    def _resolve_training_recipe(self, model: nn.Module) -> TrainingRecipe:
        """Resolve the model's family-specific training recipe."""
        explicit_recipe = getattr(model, "training_recipe", None)
        if explicit_recipe is not None:
            return build_training_recipe(str(explicit_recipe))
        spec = self._backbone_spec()
        if spec is not None:
            return build_training_recipe(spec=spec)
        return self._training_recipe_for_family(self._detect_training_family(model))

    def _recipe_for_bundle(self, model: ModelBundle) -> TrainingRecipe:
        """Resolve the recipe stored on, or implied by, a model bundle."""
        if model.recipe is not None:
            return model.recipe
        family = "transformer" if model.is_transformer else model.training_family
        return self._training_recipe_for_family(family)

    def _model_loss_type(self) -> str:
        """Choose the backbone output contract needed by the configured losses."""
        if self.loss_type == "softmax" and self.classifier_loss == "ce":
            return "softmax"
        if self.loss_type == "ms":
            return "ms"
        return "triplet"

    def _use_soft_margin_triplet(self, default_soft_margin: bool) -> bool:
        """Resolve hard-margin vs softplus batch-hard triplet behavior."""
        if self.triplet_soft_margin is not None:
            return bool(self.triplet_soft_margin)
        return default_soft_margin

    def _build_classifier_loss(self, num_classes: int, feat_dim: int, label_smooth: float) -> nn.Module:
        """Build the ID-classification criterion."""
        if self.classifier_loss == "ce":
            return CrossEntropyLabelSmooth(num_classes, epsilon=label_smooth)
        if self.classifier_loss == "arcface":
            return ArcFaceLoss(
                feat_dim=feat_dim,
                num_classes=num_classes,
                scale=self.arcface_scale,
                margin=self.arcface_margin,
            )
        if self.classifier_loss == "cosface":
            return CosFaceLoss(
                feat_dim=feat_dim,
                num_classes=num_classes,
                scale=self.cosface_scale,
                margin=self.cosface_margin,
            )
        raise ValueError(f"Unsupported classifier_loss: {self.classifier_loss}")

    def _effective_metric_feature(self) -> str:
        """Resolve the metric feature mode for multi-branch models."""
        if self.metric_feature != "auto":
            return self.metric_feature
        return "concat_bn" if self.loss_type == "ms" else "raw_mean"

    def _aux_ce_weight_for_epoch(self, epoch: int) -> float:
        """Return the active auxiliary classifier CE weight for this epoch."""
        if self.aux_ce_drop_epoch > 0 and epoch > self.aux_ce_drop_epoch:
            return 0.0
        return self.aux_ce_weight
