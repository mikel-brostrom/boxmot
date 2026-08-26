"""Top-level fit orchestration and the per-epoch training loop."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from boxmot.reid.backbones.anatomical_registry import (
    EMA_ANATOMICAL_TARGET_TYPES,
)
from boxmot.reid.datasets.transforms import (
    apply_independent_random_erasing,
    cross_camera_same_id_part_mosaic,
)
from boxmot.reid.training.trainer_components.types import (
    DatasetBundle,
    LoaderBundle,
    LossBundle,
    ModelBundle,
    OptimizationBundle,
    ResumeState,
    TrainMetrics,
    TrainResult,
    _TrainingTimeEstimator,
)
from boxmot.utils import logger as LOGGER


class _TrainingLoopMixin:
    def _fit(
        self,
        *,
        save_dir: Path,
        data: DatasetBundle,
        models: ModelBundle,
        loaders: LoaderBundle,
        losses: LossBundle,
        optimization: OptimizationBundle,
        state: ResumeState,
        run_started_at: float,
    ) -> TrainResult:
        """Run epoch orchestration after all setup and restore work is complete."""
        from tqdm import tqdm

        best_weights = save_dir / "best.pt"
        history, val_history = self._restore_history(save_dir, state.start_epoch)
        latest_primary_val = next(
            (metrics for metrics in reversed(val_history) if metrics.dataset == data.default_eval_name),
            None,
        )
        epoch_durations_s: list[float] = []
        forward_durations_s: list[float] = []
        eval_durations_s: list[float] = []
        fallback_epoch_s, fallback_eval_s = self._restore_timing_averages(save_dir)
        time_estimator = _TrainingTimeEstimator(
            total_epochs=self.epochs,
            eval_interval=self.eval_interval,
            fallback_epoch_s=fallback_epoch_s,
            fallback_eval_s=fallback_eval_s,
        )
        epoch_bar = tqdm(
            range(state.start_epoch, self.epochs + 1),
            desc="Training",
            unit="epoch",
            initial=state.start_epoch - 1,
            total=self.epochs,
            # tqdm's default ETA folds validation into one apparent epoch and
            # spikes after every evaluation. Show only the custom wall ETA.
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
        for epoch in epoch_bar:
            try:
                metrics = self._train_epoch(
                    epoch,
                    models.model,
                    loaders.train,
                    losses.criterion_id,
                    losses.criterion_metric,
                    losses.criterion_center,
                    optimization.optimizer,
                    optimization.optimizer_center,
                    optimization.scheduler,
                    criterion_csmm=losses.criterion_csmm,
                    criterion_treeboost=losses.criterion_treeboost,
                    criterion_adasp=losses.criterion_adasp,
                    ema_model=models.ema_model,
                    grad_clip=optimization.grad_clip,
                )
            except RuntimeError as exc:
                if not self._handle_oom(
                    exc,
                    optimization.optimizer,
                    optimization.optimizer_center,
                ):
                    raise
                raise RuntimeError(
                    f"{self.device.type.upper()} out of memory during training. "
                    "Cached memory was cleared; reduce --batch-size or --p-ids and resume from last.pt."
                ) from exc
            history.append(metrics)
            epoch_durations_s.append(metrics.elapsed_s)
            forward_durations_s.append(metrics.forward_elapsed_s)
            time_estimator.add_epoch(
                metrics.elapsed_s,
                phase=self._training_phase_for_eta(epoch),
            )
            self._clear_memory(threshold=self.MEMORY_CLEAR_THRESHOLD)

            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                evaluation_started_at = time.monotonic()
                self._set_mcpt_epoch(models.val_model, epoch)
                if models.ema_model is not None and models.val_model is models.ema_model:
                    self._calibrate_bn(models.val_model, loaders.train)
                if self.mcpt_mode != "none":
                    head = self._model_head(models.val_model)
                    capture = getattr(
                        head,
                        "enable_mcpt_visualization_capture",
                        None,
                    )
                    if not callable(capture):
                        raise RuntimeError("Enabled MCPT model is missing visualization capture")
                    capture(100)
                try:
                    val = self._validate(
                        epoch,
                        models.val_model,
                        loaders.query,
                        loaders.gallery,
                    )
                except RuntimeError as exc:
                    if not self._handle_oom(exc):
                        raise
                    raise RuntimeError(
                        f"{self.device.type.upper()} out of memory during validation. "
                        "Cached memory was cleared; reduce --batch-size and resume from last.pt."
                    ) from exc
                val.dataset = data.default_eval_name
                if self.mcpt_mode != "none":
                    head = self._model_head(models.val_model)
                    pop_capture = getattr(
                        head,
                        "pop_mcpt_visualization_capture",
                        None,
                    )
                    captured = pop_capture() if callable(pop_capture) else None
                    self._save_mcpt_energy_maps(save_dir, epoch, captured)
                if self.mcpt_mode != "none" and self.mcpt_disabled_eval:
                    self._set_mcpt_force_disabled(models.val_model, True)
                    try:
                        disabled_val = self._validate(
                            epoch,
                            models.val_model,
                            loaders.query,
                            loaders.gallery,
                        )
                    finally:
                        self._set_mcpt_force_disabled(models.val_model, False)
                    val.mcpt_disabled_mAP = disabled_val.mAP
                    val.mcpt_disabled_rank1 = disabled_val.rank1
                    tqdm.write(
                        "  → MCPT disabled control: "
                        f"mAP={disabled_val.mAP:.2%}  R1={disabled_val.rank1:.2%}  "
                        f"ΔmAP={val.mAP - disabled_val.mAP:+.2%}"
                    )
                val_history.append(val)
                latest_primary_val = val
                if val.mAP > state.best_mAP:
                    state.best_mAP = val.mAP
                    state.best_rank1 = val.rank1
                    state.best_epoch = epoch
                    self.checkpoint_manager.save_best(
                        best_weights,
                        model=models.val_model,
                        epoch=epoch,
                        val=val,
                        criterion_center=losses.criterion_center,
                        criterion_classifier=losses.criterion_id,
                        best_mAP=state.best_mAP,
                        best_epoch=state.best_epoch,
                        best_rank1=state.best_rank1,
                    )
                    tqdm.write(f"  ✓ New best model (mAP={val.mAP:.2%}, R1={val.rank1:.2%}) -> {best_weights}")

                for dataset_name, (query_loader, gallery_loader) in loaders.cross_domain.items():
                    try:
                        cross_domain_val = self._validate(
                            epoch,
                            models.val_model,
                            query_loader,
                            gallery_loader,
                        )
                    except RuntimeError as exc:
                        if not self._handle_oom(exc):
                            raise
                        raise RuntimeError(
                            f"{self.device.type.upper()} out of memory during cross-domain validation. "
                            "Cached memory was cleared; reduce --batch-size and resume from last.pt."
                        ) from exc
                    cross_domain_val.dataset = dataset_name
                    val_history.append(cross_domain_val)
                    tqdm.write(
                        f"  → {dataset_name}: mAP={cross_domain_val.mAP:.2%}  "
                        f"R1={cross_domain_val.rank1:.2%}  R5={cross_domain_val.rank5:.2%}"
                    )
                models.val_model.train()
                self._clear_memory(threshold=self.MEMORY_CLEAR_THRESHOLD)
                evaluation_elapsed_s = time.monotonic() - evaluation_started_at
                eval_durations_s.append(evaluation_elapsed_s)
                time_estimator.add_evaluation(evaluation_elapsed_s)

            if epoch % 10 == 0 or epoch == self.epochs:
                self._save_metrics(
                    save_dir,
                    history,
                    val_history,
                    state.best_epoch,
                    state.best_mAP,
                    state.best_rank1,
                    average_epoch_time_s=self._average_duration(epoch_durations_s),
                    average_forward_time_s=self._average_duration(forward_durations_s),
                    average_eval_time_s=self._average_duration(eval_durations_s),
                    total_end_to_end_time_s=time.monotonic() - run_started_at,
                )
                self.checkpoint_manager.save_last(
                    save_dir / "last.pt",
                    model=models.model,
                    epoch=epoch,
                    val=latest_primary_val,
                    optimizer=optimization.optimizer,
                    optimizer_center=optimization.optimizer_center,
                    criterion_center=losses.criterion_center,
                    criterion_classifier=losses.criterion_id,
                    ema_model=models.ema_model,
                    best_mAP=state.best_mAP,
                    scheduler=optimization.scheduler,
                    grad_scaler=getattr(self, "_scaler", None),
                    training_state=self._training_auxiliary_state(),
                    best_epoch=state.best_epoch,
                    best_rank1=state.best_rank1,
                )

            remaining_s = time_estimator.estimate_remaining_s(epoch)
            postfix = {
                "loss": f"{metrics.loss:.4f}",
                "id": f"{metrics.id_loss:.4f}",
                "tri": f"{metrics.triplet_loss:.4f}",
                "lr": f"{metrics.lr:.6f}",
                "epoch_s": f"{time_estimator.epoch_duration_s:.1f}",
                "wall_eta": self._format_eta(remaining_s),
            }
            if time_estimator.evaluation_duration_s > 0:
                postfix["eval_s"] = f"{time_estimator.evaluation_duration_s:.1f}"
            if latest_primary_val is not None and latest_primary_val.epoch == epoch:
                postfix.update(
                    mAP=f"{latest_primary_val.mAP:.2%}",
                    R1=f"{latest_primary_val.rank1:.2%}",
                )
            epoch_bar.set_postfix(postfix)

        self._save_metrics(
            save_dir,
            history,
            val_history,
            state.best_epoch,
            state.best_mAP,
            state.best_rank1,
            average_epoch_time_s=self._average_duration(epoch_durations_s),
            average_forward_time_s=self._average_duration(forward_durations_s),
            average_eval_time_s=self._average_duration(eval_durations_s),
            total_end_to_end_time_s=time.monotonic() - run_started_at,
        )
        self._save_training_plots(save_dir, history, val_history)
        LOGGER.info(
            f"Training complete. Best epoch={state.best_epoch}  mAP={state.best_mAP:.2%}  R1={state.best_rank1:.2%}"
        )
        return TrainResult(
            best_epoch=state.best_epoch,
            best_mAP=state.best_mAP,
            best_rank1=state.best_rank1,
            weights_path=best_weights,
            history=history,
            val_history=val_history,
        )

    def _train_epoch(
        self,
        epoch,
        model,
        loader,
        criterion_id,
        criterion_metric,
        criterion_center,
        optimizer,
        optimizer_center,
        scheduler,
        *,
        criterion_csmm=None,
        criterion_treeboost=None,
        criterion_adasp=None,
        ema_model=None,
        grad_clip: float = 0.0,
    ) -> TrainMetrics:
        from tqdm import tqdm

        self._seed_training_epoch(epoch, loader)
        self._apply_epoch_warmup_lrs(optimizer, epoch)
        privileged_retrieval_scale = self._retrieval_auxiliary_schedule_scale(epoch)
        anatomical_training_active = self._anatomical_training_active(epoch)
        model.train()
        self._set_anatomical_runtime_active(
            model,
            anatomical_training_active,
        )
        multilevel_suppression_progress = self._set_multilevel_suppression_progress(model, epoch)
        self._set_mcpt_epoch(model, epoch)
        if self.anatomical_auxiliary and self.anatomical_target_type == "privileged_mask_pose_attention":
            head = self._model_head(model)
            set_gate_active = getattr(
                head,
                "set_anatomical_attention_gate_active",
                None,
            )
            if not callable(set_gate_active):
                raise RuntimeError("Privileged mask-pose attention is missing its gate schedule hook")
            set_gate_active(epoch > self.backbone_freeze_epochs)
        backbone_freeze_active = self._backbone_freeze_active(epoch)
        gradual_unfreeze_phase = self._gradual_unfreeze_phase(epoch)
        gradual_backbone_original_lrs = None
        head_warmup_original_lrs = None
        head_warmup_active = False
        if gradual_unfreeze_phase:
            self._set_gradual_unfreeze_trainability(model, gradual_unfreeze_phase)
            if epoch == 1:
                LOGGER.info(
                    "Gradual unfreeze enabled: "
                    f"ReID modules through epoch {self.gradual_unfreeze_head_epochs}, "
                    f"last stage through epoch {self.gradual_unfreeze_stage_epochs}, "
                    "then full backbone"
                )
                if self.gradual_unfreeze_backbone_lr_epochs > 0:
                    LOGGER.info(
                        "Gradual unfreeze backbone LR drop active from epoch "
                        f"{self.gradual_unfreeze_head_epochs + 1} through epoch "
                        f"{self.gradual_unfreeze_stage_epochs + self.gradual_unfreeze_backbone_lr_epochs} "
                        f"(backbone_lr_mult={self.gradual_unfreeze_backbone_lr_mult:g})"
                    )
        elif backbone_freeze_active:
            self._set_backbone_freeze_trainability(model, True)
            if epoch == 1:
                LOGGER.info(
                    f"Backbone freeze warm-start enabled for {self.backbone_freeze_epochs} epochs; "
                    "training neck, feature fusion, adapters, and head"
                )
        else:
            requested_head_warmup = self._head_warmup_active(epoch)
            head_warmup_supported = any(group.get("is_head", False) for group in optimizer.param_groups)
            head_warmup_active = requested_head_warmup and head_warmup_supported
            if requested_head_warmup and not head_warmup_supported and epoch == 1:
                LOGGER.warning("Head warmup requested, but optimizer has no separate head parameter group; ignoring")
            self._set_head_warmup_trainability(model, head_warmup_active)

        if gradual_unfreeze_phase:
            gradual_backbone_original_lrs = self._apply_gradual_backbone_lrs(optimizer, epoch)
            if epoch in {
                1,
                self.gradual_unfreeze_head_epochs + 1,
                self.gradual_unfreeze_stage_epochs + 1,
            }:
                trainable, total = self._trainable_parameter_summary(model)
                active_lr, backbone_lr, head_lr = self._optimizer_lr_summary(optimizer)
                LOGGER.info(
                    f"Gradual unfreeze phase '{gradual_unfreeze_phase}': "
                    f"trainable_params={trainable}/{total}, "
                    f"active_lr={active_lr:.2e}, backbone_lr={backbone_lr:.2e}, head_lr={head_lr:.2e}"
                )
        elif head_warmup_active:
            head_warmup_original_lrs = self._apply_head_warmup_lrs(optimizer)
            if epoch == self._head_warmup_start_epoch():
                LOGGER.info(
                    f"Head warmup enabled for {self.head_warmup_epochs} epochs "
                    f"(head_lr_mult={self.head_warmup_lr_mult:g})"
                )
        epoch_lr, epoch_backbone_lr, epoch_head_lr = self._optimizer_lr_summary(optimizer)

        id_loss_weight = self._effective_id_loss_weight(epoch)
        center_loss_weight = self._effective_center_loss_weight(epoch)
        csmm_loss_weight = self._effective_csmm_loss_weight(epoch)
        treeboost_loss_weight = self._effective_treeboost_loss_weight(epoch)
        late_interaction_scale = self._effective_late_interaction_scale(epoch)

        running_losses = torch.zeros(50, device=self.device)
        running_multilevel_suppression_loss = torch.tensor(
            0.0,
            device=self.device,
        )
        running_multilevel_suppression_diagnostics = torch.zeros(
            len(self._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS),
            device=self.device,
        )
        running_privileged_losses = torch.zeros(7, device=self.device)
        n_batches = 0
        forward_elapsed_s = 0.0
        forward_events: list[tuple[Any, Any]] = []
        t0 = time.monotonic()

        # AMP: mixed precision on CUDA for ~2x throughput, skip on CPU/MPS
        use_amp = self.device.type == "cuda"
        if not hasattr(self, "_scaler"):
            self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scaler = self._scaler

        batch_bar = tqdm(
            loader,
            desc=f"  Epoch {epoch}/{self.epochs}",
            leave=False,
            unit="batch",
        )
        for batch in batch_bar:
            sample_indices = None
            retrieval_indices_required = self._global_ap is not None or self._hpgrd_enabled()
            if retrieval_indices_required:
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    raise RuntimeError("GlobalAP/HP-GRD requires a stable sample index in every training batch")
                *batch_values, sample_indices = batch
                batch = tuple(batch_values)
            anatomical_targets = None
            clean_anatomical_targets = None
            if len(batch) == 6:
                (
                    imgs,
                    pids,
                    camera_ids,
                    clean_imgs,
                    augmented,
                    anatomical_targets,
                ) = batch
                clean_imgs = clean_imgs.to(self.device)
                augmented = augmented.to(self.device, dtype=torch.bool)
            elif len(batch) == 5:
                imgs, pids, camera_ids, clean_imgs, augmented = batch
                clean_imgs = clean_imgs.to(self.device)
                augmented = augmented.to(self.device, dtype=torch.bool)
            elif len(batch) == 4:
                imgs, pids, camera_ids, anatomical_targets = batch
                clean_imgs = None
                augmented = torch.zeros(
                    len(imgs),
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                imgs, pids, camera_ids = batch
                clean_imgs = None
                augmented = torch.zeros(len(imgs), dtype=torch.bool, device=self.device)
            if isinstance(anatomical_targets, dict):
                clean_anatomical_targets = anatomical_targets.get("_clean_view")
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            camera_ids = camera_ids.to(self.device)
            if sample_indices is not None:
                sample_indices = sample_indices.to(self.device, dtype=torch.long)
            if clean_imgs is not None and self.clean_student_consistency_weight <= 0:
                max_augmented = int(math.floor(len(imgs) * (1.0 - self.pav_mosaic_min_unaltered)))
                active_indices = torch.nonzero(augmented, as_tuple=False).flatten()
                if active_indices.numel() > max_augmented:
                    order = torch.randperm(active_indices.numel(), device=self.device)
                    reverted = active_indices[order[max_augmented:]]
                    imgs[reverted] = clean_imgs[reverted]
                    augmented[reverted] = False
                active_indices = torch.nonzero(augmented, as_tuple=False).flatten()
                if self.pav_consistency_weight > 0 and active_indices.numel() == 1:
                    # The clean-view forward traverses training-mode BN layers.
                    # Retain either zero or at least two paired augmentations so
                    # sparse warm-up batches cannot send a singleton through a
                    # BNNeck, and every retained mosaic still receives its clean
                    # ID and consistency supervision.
                    imgs[active_indices] = clean_imgs[active_indices]
                    augmented[active_indices] = False
            if self.same_id_part_mosaic:
                imgs = cross_camera_same_id_part_mosaic(
                    imgs,
                    pids,
                    camera_ids,
                    probability=self.same_id_part_mosaic_probability,
                    max_regions=self.same_id_part_mosaic_max_regions,
                    min_replaced_area=self.same_id_part_mosaic_min_area,
                    max_replaced_area=self.same_id_part_mosaic_max_area,
                    boundary_jitter=self.same_id_part_mosaic_boundary_jitter,
                    cross_camera_rate=self.same_id_part_mosaic_cross_camera_rate,
                    min_unaltered_fraction=self.same_id_part_mosaic_min_unaltered,
                )
                imgs = apply_independent_random_erasing(imgs, self.random_erasing)

            if privileged_retrieval_scale > 0:
                (
                    semantic_drop_imgs,
                    semantic_drop_indices,
                    semantic_drop_parts,
                    semantic_drop_confidence,
                ) = self._build_hpgrd_semantic_drop_view(
                    imgs,
                    pids,
                    anatomical_targets,
                )
            else:
                semantic_drop_imgs = None
                semantic_drop_indices = None
                semantic_drop_parts = None
                semantic_drop_confidence = None

            model_forward_kwargs = (
                self._anatomical_forward_kwargs(
                    anatomical_targets,
                    dtype=imgs.dtype,
                )
                if anatomical_training_active
                else {}
            )
            if self.multilevel_suppression:
                # PIDs are deliberately limited to the primary student
                # forward. Clean/PAV/EMA/no-grad forwards stay suppression-free.
                model_forward_kwargs["pids"] = pids

            with torch.amp.autocast("cuda", enabled=use_amp):
                if self.device.type == "cuda":
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                    output = model(imgs, **model_forward_kwargs)
                    forward_end.record()
                    forward_events.append((forward_start, forward_end))
                elif self.device.type == "mps" and hasattr(torch.mps, "Event"):
                    forward_start = torch.mps.Event(enable_timing=True)
                    forward_end = torch.mps.Event(enable_timing=True)
                    forward_start.record()
                    output = model(imgs, **model_forward_kwargs)
                    forward_end.record()
                    forward_events.append((forward_start, forward_end))
                else:
                    forward_started_at = time.monotonic()
                    output = model(imgs, **model_forward_kwargs)
                    forward_elapsed_s += time.monotonic() - forward_started_at
                logits, features = self._split_model_output(output)
                teacher_part_features = None
                if self._part_relation_enabled():
                    if ema_model is None:
                        raise RuntimeError("Part-relation supervision requires its EMA teacher")
                    teacher_part_features = self._ema_part_teacher_features(
                        ema_model,
                        imgs,
                    )
                loss_pav_consistency = torch.tensor(0.0, device=self.device)
                loss_clean_student_consistency = torch.tensor(
                    0.0,
                    device=self.device,
                )
                clean_logits = None
                clean_features = None
                hpgrd_background_features = None
                hpgrd_background_indices = None
                semantic_drop_features = None
                active_indices = torch.nonzero(augmented, as_tuple=False).flatten()
                if self.pav_consistency_weight > 0 and clean_imgs is not None and active_indices.numel() > 0:
                    clean_output = model(clean_imgs[active_indices])
                    clean_logits, clean_features = self._split_model_output(clean_output)
                elif self.clean_student_consistency_weight > 0:
                    if clean_imgs is None or clean_anatomical_targets is None:
                        raise RuntimeError(
                            "Clean-student consistency requires paired clean "
                            "images and aligned clean anatomical targets"
                        )
                    if active_indices.numel() != len(imgs):
                        raise RuntimeError(
                            "Clean-student consistency expects one clean view for every augmented student view"
                        )
                    clean_forward_kwargs = (
                        self._anatomical_forward_kwargs(
                            clean_anatomical_targets,
                            dtype=clean_imgs.dtype,
                        )
                        if anatomical_training_active
                        else {}
                    )
                    with torch.no_grad():
                        clean_output = model(
                            clean_imgs,
                            **clean_forward_kwargs,
                        )
                    _, clean_features = self._split_model_output(clean_output)
                elif (
                    self.hpgrd_background_weight > 0
                    and privileged_retrieval_scale > 0
                    and clean_imgs is not None
                    and active_indices.numel() > 0
                ):
                    clean_output = self._hpgrd_intervention_forward(
                        model,
                        clean_imgs,
                        detached=True,
                    )
                    _, clean_features = self._split_model_output(clean_output)

                if self.hpgrd_background_weight > 0 and privileged_retrieval_scale > 0 and clean_features is not None:
                    hpgrd_background_features = clean_features
                    if self._deployment_descriptor(hpgrd_background_features).shape[0] != len(imgs):
                        if clean_imgs is None:
                            raise RuntimeError("HP-GRD background control requires clean views")
                        hpgrd_background_output = self._hpgrd_intervention_forward(
                            model,
                            clean_imgs,
                            detached=True,
                        )
                        _, hpgrd_background_features = self._split_model_output(hpgrd_background_output)
                    hpgrd_background_indices = active_indices
                if semantic_drop_imgs is not None:
                    semantic_drop_output = self._hpgrd_intervention_forward(
                        model,
                        semantic_drop_imgs,
                        detached=False,
                    )
                    _, semantic_drop_features = self._split_model_output(semantic_drop_output)

                # ID loss — CE uses model logits; margin classifiers use embeddings.
                if self.classifier_loss == "ce":
                    loss_id = self._classification_loss_for_logits(criterion_id, logits, pids, epoch, features)
                else:
                    cls_features = self._classification_features(features)
                    if cls_features is None:
                        raise RuntimeError(
                            f"classifier_loss={self.classifier_loss} requires embedding features; "
                            f"model loss contract is {self._model_loss_type()}"
                        )
                    loss_id = criterion_id(cls_features, pids)
                loss = id_loss_weight * loss_id
                loss_multilevel_suppression = self._multilevel_suppression_loss(
                    criterion_id,
                    features,
                    pids,
                )
                multilevel_suppression_diagnostics = self._multilevel_suppression_diagnostics(features)
                loss = loss + (self._effective_multilevel_suppression_loss_weight() * loss_multilevel_suppression)
                if self.pav_consistency_weight > 0 and clean_logits is not None:
                    clean_pids = pids[active_indices]
                    if self.classifier_loss == "ce":
                        clean_id_loss = self._classification_loss_for_logits(
                            criterion_id,
                            clean_logits,
                            clean_pids,
                            epoch,
                            clean_features,
                        )
                    else:
                        clean_classification_features = self._classification_features(clean_features)
                        if clean_classification_features is None:
                            raise RuntimeError("PAV consistency requires clean-view embedding features")
                        clean_id_loss = criterion_id(
                            clean_classification_features,
                            clean_pids,
                        )
                    mosaic_descriptor = self._pav_consistency_descriptor(features)
                    clean_descriptor = self._pav_consistency_descriptor(clean_features)
                    if not torch.is_tensor(mosaic_descriptor) or not torch.is_tensor(clean_descriptor):
                        raise RuntimeError("PAV consistency requires tensor retrieval descriptors")
                    loss_pav_consistency = (
                        1.0
                        - F.cosine_similarity(
                            mosaic_descriptor[active_indices],
                            clean_descriptor,
                            dim=1,
                        ).mean()
                    )
                    loss_id = loss_id + clean_id_loss
                    loss = loss + id_loss_weight * clean_id_loss + self.pav_consistency_weight * loss_pav_consistency
                compact_id = self._compact_student_id_loss(criterion_id, features, pids)
                if self.compact_deployment_head:
                    loss_id = loss_id + compact_id
                    loss = loss + id_loss_weight * compact_id

                # Triplet loss — L2-normalize features so Euclidean distance in
                # triplet loss aligns with cosine distance used at evaluation.
                loss_tri = torch.tensor(0.0, device=self.device)
                if criterion_metric is not None and features is not None:
                    loss_tri = self._metric_loss_for_features(criterion_metric, features, pids)
                    loss = loss + self.metric_loss_weight * loss_tri

                loss_jpm_id, loss_jpm_metric = self._jpm_auxiliary_losses(
                    criterion_id,
                    criterion_metric,
                    features,
                    pids,
                )
                loss = loss + self.jpm_id_loss_weight * loss_jpm_id + self.jpm_metric_loss_weight * loss_jpm_metric

                loss_adasp = torch.tensor(0.0, device=self.device)
                if criterion_adasp is not None and features is not None:
                    loss_adasp = self._adasp_loss_for_features(
                        criterion_adasp,
                        features,
                        pids,
                    )
                    loss = loss + self.adasp_loss_weight * self.adasp_scale * loss_adasp

                (
                    loss_part_relation,
                    loss_part_to_global,
                ) = self._part_relation_losses(
                    features,
                    teacher_part_features,
                    pids,
                )
                loss = (
                    loss
                    + self.part_relation_weight * loss_part_relation
                    + self.part_to_global_weight * loss_part_to_global
                )

                loss_csmm = torch.tensor(0.0, device=self.device)
                if criterion_csmm is not None and csmm_loss_weight > 0:
                    loss_csmm = self._cross_scale_majority_margin_loss(
                        criterion_csmm,
                        features,
                        pids,
                    )
                    loss = loss + csmm_loss_weight * loss_csmm

                loss_treeboost = torch.tensor(0.0, device=self.device)
                if criterion_treeboost is not None and treeboost_loss_weight > 0:
                    loss_treeboost = self._treeboost_ap_loss(
                        criterion_treeboost,
                        features,
                        pids,
                        camera_ids,
                    )
                    loss = loss + treeboost_loss_weight * loss_treeboost

                loss_global_ap = torch.tensor(0.0, device=self.device)
                global_ap_weight = 0.0
                if self._global_ap is not None:
                    if sample_indices is None:
                        raise RuntimeError("GlobalAP batch is missing stable sample indices")
                    loss_global_ap, global_ap_weight = self._global_ap_objective(
                        features,
                        sample_indices,
                        pids,
                        epoch=epoch,
                    )
                    loss = loss + global_ap_weight * loss_global_ap

                loss_late_interaction = torch.tensor(0.0, device=self.device)
                loss_late_interaction_distill = torch.tensor(0.0, device=self.device)
                if self.hierarchical_late_interaction and late_interaction_scale > 0:
                    loss_late_interaction, loss_late_interaction_distill = self._hierarchical_late_interaction_losses(
                        model,
                        features,
                        pids,
                        camera_ids,
                    )
                    loss = loss + (late_interaction_scale * self.late_interaction_loss_weight * loss_late_interaction)
                    loss = loss + (
                        late_interaction_scale * self.late_interaction_distill_weight * loss_late_interaction_distill
                    )

                compact_tri, compact_cosine, compact_pairwise = self._compact_student_losses(
                    criterion_metric,
                    features,
                    pids,
                )
                if self.compact_deployment_head:
                    weighted_compact_tri = self.compact_metric_loss_weight * compact_tri
                    loss_tri = loss_tri + weighted_compact_tri
                    loss = loss + weighted_compact_tri
                    loss = loss + self.compact_cosine_distill_weight * compact_cosine
                    loss = loss + self.compact_pairwise_distill_weight * compact_pairwise

                loss_evidence = self._evidence_auxiliary_loss(features, pids)
                loss = loss + loss_evidence
                loss_identity_register_diversity = self._identity_register_diversity_loss(features)
                loss = loss + self.identity_register_diversity_weight * loss_identity_register_diversity
                loss_mcpt, mcpt_components = self._mcpt_auxiliary_loss(
                    features,
                    epoch=epoch,
                )
                loss = loss + loss_mcpt
                if anatomical_training_active and not self._hpgrd_owns_anatomical_runtime():
                    (
                        loss_anatomical,
                        anatomical_components,
                    ) = self._anatomical_auxiliary_loss(
                        features,
                        anatomical_targets,
                        pids,
                        camera_ids,
                        epoch=epoch,
                        return_components=True,
                    )
                else:
                    (
                        loss_anatomical,
                        anatomical_components,
                    ) = self._zero_anatomical_loss(pids)
                loss = loss + loss_anatomical
                if self.clean_student_consistency_weight > 0:
                    loss_clean_student_consistency = self._clean_teacher_student_consistency_loss(
                        features,
                        clean_features,
                        clean_anatomical_targets,
                        active_indices,
                    )
                    loss = loss + self.clean_student_consistency_weight * loss_clean_student_consistency
                if self.anatomical_deployment:
                    (
                        anatomical_deployment_id,
                        anatomical_deployment_metric,
                    ) = self._anatomical_deployment_losses(
                        criterion_id,
                        features,
                        pids,
                        camera_ids,
                    )
                else:
                    anatomical_deployment_id = loss.new_zeros(())
                    anatomical_deployment_metric = loss.new_zeros(())
                weighted_deployment_id = self.anatomical_deployment_id_weight * anatomical_deployment_id
                weighted_deployment_metric = self.anatomical_deployment_metric_weight * anatomical_deployment_metric
                loss_id = loss_id + weighted_deployment_id
                loss_tri = loss_tri + weighted_deployment_metric
                loss = (
                    loss
                    + id_loss_weight * weighted_deployment_id
                    + self.metric_loss_weight * weighted_deployment_metric
                )

                # Center loss — only on embeddings, never on logits
                center_features, center_pids, center_loss_scale = self._center_loss_inputs(features, pids)
                loss_cen = torch.tensor(0.0, device=self.device)
                if center_features is not None and center_loss_weight > 0 and not head_warmup_active:
                    effective_center_loss_weight = center_loss_weight * center_loss_scale
                    loss_cen = criterion_center(center_features, center_pids) * effective_center_loss_weight
                    loss = loss + loss_cen

                loss_hpgrd = torch.tensor(0.0, device=self.device)
                hpgrd_components = {
                    "global_relational": loss_hpgrd,
                    "part_relational": loss_hpgrd,
                    "background_consistency": loss_hpgrd,
                    "semantic_drop_relational": loss_hpgrd,
                }
                hpgrd_gradient_scale = torch.tensor(0.0, device=self.device)
                if self._hpgrd_enabled():
                    if sample_indices is None:
                        raise RuntimeError("HP-GRD batch is missing stable sample indices")
                    student_packet = self._hpgrd_student_packet(
                        features,
                        background_features=hpgrd_background_features,
                        background_indices=hpgrd_background_indices,
                        semantic_drop_features=semantic_drop_features,
                        semantic_drop_indices=semantic_drop_indices,
                        semantic_drop_parts=semantic_drop_parts,
                        semantic_drop_confidence=semantic_drop_confidence,
                        anatomical_targets=anatomical_targets,
                    )
                    loss_hpgrd, hpgrd_components, hpgrd_gradient_scale = self._hpgrd_objective(
                        loss,
                        student_packet,
                        sample_indices,
                        pids,
                        epoch=epoch,
                    )
                    loss = loss + loss_hpgrd

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, batch {n_batches + 1}: "
                    f"loss={loss.detach().item():.6g}, "
                    f"id_loss={loss_id.detach().item():.6g}, "
                    "multilevel_suppression_loss="
                    f"{loss_multilevel_suppression.detach().item():.6g}, "
                    f"triplet_loss={loss_tri.detach().item():.6g}, "
                    f"adasp_loss={loss_adasp.detach().item():.6g}, "
                    f"part_relation={loss_part_relation.detach().item():.6g}, "
                    f"part_to_global={loss_part_to_global.detach().item():.6g}, "
                    f"jpm_id={loss_jpm_id.detach().item():.6g}, "
                    f"jpm_metric={loss_jpm_metric.detach().item():.6g}, "
                    f"csmm_loss={loss_csmm.detach().item():.6g}, "
                    f"treeboost_loss={loss_treeboost.detach().item():.6g}, "
                    f"global_ap_loss={loss_global_ap.detach().item():.6g}, "
                    f"hpgrd_loss={loss_hpgrd.detach().item():.6g}, "
                    f"late_interaction_loss={loss_late_interaction.detach().item():.6g}, "
                    f"late_interaction_distill={loss_late_interaction_distill.detach().item():.6g}, "
                    f"pav_consistency={loss_pav_consistency.detach().item():.6g}, "
                    "clean_student_consistency="
                    f"{loss_clean_student_consistency.detach().item():.6g}, "
                    f"compact_cosine={compact_cosine.detach().item():.6g}, "
                    f"compact_pairwise={compact_pairwise.detach().item():.6g}, "
                    f"evidence_loss={loss_evidence.detach().item():.6g}, "
                    "identity_register_diversity="
                    f"{loss_identity_register_diversity.detach().item():.6g}, "
                    f"mcpt_loss={loss_mcpt.detach().item():.6g}, "
                    f"anatomical_loss={loss_anatomical.detach().item():.6g}, "
                    f"center_loss={loss_cen.detach().item():.6g}"
                )

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping (transformer training stability)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    if "non-finite" not in str(exc).lower():
                        raise
                    nonfinite_parameters = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ]
                    preview = ", ".join(nonfinite_parameters[:12])
                    if len(nonfinite_parameters) > 12:
                        preview += f", ... (+{len(nonfinite_parameters) - 12} more)"
                    raise RuntimeError(
                        "Non-finite gradient before optimizer step at "
                        f"epoch {epoch}, batch {n_batches + 1}: "
                        f"parameters=[{preview}]"
                    ) from exc

            scaler.step(optimizer)

            # Center loss has its own optimizer with special LR
            if center_features is not None and center_loss_weight > 0 and not head_warmup_active:
                scaler.unscale_(optimizer_center)
                for param in criterion_center.parameters():
                    if param.grad is not None:
                        # Keep the center-table optimizer independent of both
                        # the user-facing loss weight and a model-requested
                        # branch aggregation scale. The model gradients retain
                        # the complete weighted objective above.
                        param.grad.data *= 1.0 / (center_loss_weight * center_loss_scale)
                scaler.step(optimizer_center)

            scaler.update()
            if sample_indices is not None:
                self._update_global_ap_memory(
                    features,
                    sample_indices,
                    pids,
                )

            if (
                anatomical_training_active
                and self.anatomical_auxiliary
                and not self._hpgrd_owns_anatomical_runtime()
                and self.anatomical_target_type in EMA_ANATOMICAL_TARGET_TYPES
            ):
                head = self._model_head(model)
                update_teacher = getattr(
                    head,
                    "update_anatomical_teacher",
                    None,
                )
                if not callable(update_teacher):
                    raise RuntimeError("Selected EMA anatomy teacher is missing its update hook")
                update_teacher(self.anatomical_teacher_momentum)
            if (
                anatomical_training_active
                and self.anatomical_auxiliary
                and not self._hpgrd_owns_anatomical_runtime()
                and self.anatomical_target_type == "body_slot_privileged_ema"
            ):
                unwrapped = model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                update_teacher = getattr(
                    unwrapped,
                    "update_body_slot_teacher",
                    None,
                )
                if not callable(update_teacher):
                    raise RuntimeError("Body-slot model is missing its EMA teacher update hook")
                update_teacher(self.anatomical_teacher_momentum)

            # EMA update (parameters + buffers)
            # Float buffers (BN running_mean/var) are EMA'd so their
            # statistics match the EMA model's feature distribution.
            # Integer buffers (num_batches_tracked, index tensors) are copied.
            if ema_model is not None:
                ema_update = (epoch - 1) * len(loader) + n_batches + 1
                self._update_ema_model(
                    ema_model,
                    model,
                    self._ema_decay_for_update(
                        self._effective_ema_decay(),
                        ema_update,
                    ),
                )

            running_losses.add_(
                torch.stack(
                    (
                        loss.detach(),
                        loss_id.detach(),
                        loss_tri.detach(),
                        loss_cen.detach(),
                        loss_csmm.detach(),
                        loss_treeboost.detach(),
                        loss_late_interaction.detach(),
                        loss_late_interaction_distill.detach(),
                        loss_pav_consistency.detach(),
                        loss_clean_student_consistency.detach(),
                        loss_anatomical.detach(),
                        anatomical_components["distill"].detach(),
                        anatomical_components["attention"].detach(),
                        anatomical_components["visibility"].detach(),
                        anatomical_components["contrastive"].detach(),
                        anatomical_components["descriptor_distill"].detach(),
                        anatomical_components["branch_distill"].detach(),
                        anatomical_components["branch_global"].detach(),
                        anatomical_components["branch_coarse"].detach(),
                        anatomical_components["branch_fine"].detach(),
                        anatomical_components["pose_teacher"].detach(),
                        anatomical_components.get(
                            "semantic_foreground",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components.get(
                            "semantic_part",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components["local_scale"].detach(),
                        anatomical_components["fine_scale"].detach(),
                        anatomical_components["cross_scale"].detach(),
                        anatomical_components["valid_part_fraction"].detach(),
                        anatomical_components["cross_camera_anchor_fraction"].detach(),
                        anatomical_components.get(
                            "query_distill",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components.get(
                            "query_relational_distill",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components.get(
                            "query_diversity",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components.get(
                            "part_triplet",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        anatomical_components.get(
                            "accessory_valid_fraction",
                            loss_anatomical.detach() * 0.0,
                        ).detach(),
                        loss_identity_register_diversity.detach(),
                        loss_mcpt.detach(),
                        mcpt_components["smoothness"].detach(),
                        mcpt_components["identity"].detach(),
                        mcpt_components["mean_abs_displacement"].detach(),
                        mcpt_components["boundary_1"].detach(),
                        mcpt_components["boundary_2"].detach(),
                        mcpt_components["boundary_3"].detach(),
                        mcpt_components["boundary_std"].detach(),
                        mcpt_components["cap_fraction"].detach(),
                        mcpt_components["local_gate"].detach(),
                        mcpt_components["fine_gate"].detach(),
                        loss_adasp.detach(),
                        loss_part_relation.detach(),
                        loss_part_to_global.detach(),
                        loss_jpm_id.detach(),
                        loss_jpm_metric.detach(),
                    )
                ).float()
            )
            running_multilevel_suppression_loss.add_(loss_multilevel_suppression.detach().float())
            running_multilevel_suppression_diagnostics.add_(multilevel_suppression_diagnostics)
            running_privileged_losses.add_(
                torch.stack(
                    (
                        loss_global_ap.detach(),
                        loss_hpgrd.detach(),
                        hpgrd_components["global_relational"].detach(),
                        hpgrd_components["part_relational"].detach(),
                        hpgrd_components["background_consistency"].detach(),
                        hpgrd_components["semantic_drop_relational"].detach(),
                        hpgrd_gradient_scale.detach(),
                    )
                ).float()
            )
            n_batches += 1
            if n_batches % 20 == 0:
                batch_bar.set_postfix(loss=f"{(running_losses[0] / n_batches).item():.4f}")

        if gradual_backbone_original_lrs is not None:
            for group, lr in zip(optimizer.param_groups, gradual_backbone_original_lrs):
                group["lr"] = lr
        if head_warmup_original_lrs is not None:
            for group, lr in zip(optimizer.param_groups, head_warmup_original_lrs):
                group["lr"] = lr

        # Scheduler step
        if epoch > self.warmup_epochs:
            scheduler.step()

        if forward_events:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elif self.device.type == "mps":
                torch.mps.synchronize()
            forward_elapsed_s = sum(float(start.elapsed_time(end)) for start, end in forward_events) / 1000.0

        elapsed = time.monotonic() - t0
        average_losses = (running_losses / max(n_batches, 1)).cpu().tolist()
        average_privileged_losses = (running_privileged_losses / max(n_batches, 1)).cpu().tolist()
        return TrainMetrics(
            epoch=epoch,
            loss=average_losses[0],
            id_loss=average_losses[1],
            triplet_loss=average_losses[2],
            center_loss=average_losses[3],
            global_ap_loss=average_privileged_losses[0],
            hpgrd_loss=average_privileged_losses[1],
            hpgrd_global_loss=average_privileged_losses[2],
            hpgrd_part_loss=average_privileged_losses[3],
            hpgrd_background_loss=average_privileged_losses[4],
            hpgrd_part_drop_loss=average_privileged_losses[5],
            hpgrd_gradient_scale=average_privileged_losses[6],
            csmm_loss=average_losses[4],
            treeboost_loss=average_losses[5],
            late_interaction_loss=average_losses[6],
            late_interaction_distill_loss=average_losses[7],
            pav_consistency_loss=average_losses[8],
            clean_student_consistency_loss=average_losses[9],
            anatomical_loss=average_losses[10],
            anatomical_distill_loss=average_losses[11],
            anatomical_attention_loss=average_losses[12],
            anatomical_visibility_loss=average_losses[13],
            anatomical_contrastive_loss=average_losses[14],
            anatomical_descriptor_distill_loss=average_losses[15],
            anatomical_branch_distill_loss=average_losses[16],
            anatomical_branch_global_loss=average_losses[17],
            anatomical_branch_coarse_loss=average_losses[18],
            anatomical_branch_fine_loss=average_losses[19],
            anatomical_pose_teacher_loss=average_losses[20],
            anatomical_semantic_foreground_loss=average_losses[21],
            anatomical_semantic_part_loss=average_losses[22],
            anatomical_local_scale_loss=average_losses[23],
            anatomical_fine_scale_loss=average_losses[24],
            anatomical_cross_scale_loss=average_losses[25],
            anatomical_valid_part_fraction=average_losses[26],
            anatomical_cross_camera_anchor_fraction=average_losses[27],
            anatomical_query_distill_loss=average_losses[28],
            anatomical_query_relational_distill_loss=average_losses[29],
            anatomical_query_diversity_loss=average_losses[30],
            anatomical_part_triplet_loss=average_losses[31],
            anatomical_accessory_valid_fraction=average_losses[32],
            identity_register_diversity_loss=average_losses[33],
            mcpt_loss=average_losses[34],
            mcpt_smoothness=average_losses[35],
            mcpt_identity=average_losses[36],
            mcpt_mean_abs_displacement=average_losses[37],
            mcpt_boundary_1=average_losses[38],
            mcpt_boundary_2=average_losses[39],
            mcpt_boundary_3=average_losses[40],
            mcpt_boundary_std=average_losses[41],
            mcpt_cap_fraction=average_losses[42],
            mcpt_local_gate=average_losses[43],
            mcpt_fine_gate=average_losses[44],
            adasp_loss=average_losses[45],
            part_relation_loss=average_losses[46],
            part_to_global_loss=average_losses[47],
            jpm_id_loss=average_losses[48],
            jpm_metric_loss=average_losses[49],
            multilevel_suppression_loss=(running_multilevel_suppression_loss / max(n_batches, 1)).item(),
            multilevel_suppression_weight=(self.multilevel_suppression_loss_weight * multilevel_suppression_progress),
            multilevel_suppression_effective_ratio=(
                running_multilevel_suppression_diagnostics[0] / max(n_batches, 1)
            ).item(),
            multilevel_suppression_coarse_erased_fraction=(
                running_multilevel_suppression_diagnostics[1] / max(n_batches, 1)
            ).item(),
            multilevel_suppression_fine_erased_fraction=(
                running_multilevel_suppression_diagnostics[2] / max(n_batches, 1)
            ).item(),
            multilevel_suppression_global_cam_active_fraction=(
                running_multilevel_suppression_diagnostics[3] / max(n_batches, 1)
            ).item(),
            multilevel_suppression_coarse_cam_active_fraction=(
                running_multilevel_suppression_diagnostics[4] / max(n_batches, 1)
            ).item(),
            lr=epoch_lr,
            elapsed_s=elapsed,
            forward_elapsed_s=forward_elapsed_s,
            backbone_lr=epoch_backbone_lr,
            head_lr=epoch_head_lr,
        )
