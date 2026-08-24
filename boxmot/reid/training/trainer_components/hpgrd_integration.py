"""Trainer integration for GlobalAP and human-privileged graph distillation.

This mixin owns only training state.  It consumes outputs that the existing
CSL-TinyViT head already produces and never adds a parameter or operation to
the deployed forward path.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from boxmot.reid.datasets.anatomical import ANATOMICAL_PARTS
from boxmot.reid.training.trainer_components.global_ap import IdentityGlobalAP
from boxmot.reid.training.trainer_components.privileged_graph import (
    BACKGROUND_CONFIDENCE_KEY,
    BACKGROUND_DESCRIPTOR_KEY,
    BACKGROUND_INDICES_KEY,
    DEPLOYED_DESCRIPTOR_KEY,
    PART_DESCRIPTOR_KEY,
    PART_RELIABILITY_KEY,
    SEMANTIC_DROP_CONFIDENCE_KEY,
    SEMANTIC_DROP_DESCRIPTOR_KEY,
    SEMANTIC_DROP_INDICES_KEY,
    SEMANTIC_DROP_PARTS_KEY,
    PrivilegedGraphLoss,
    PrivilegedGraphTeacherBatch,
    PrivilegedGraphTeacherCache,
    dataset_samples_sha256,
    fuse_privileged_confidence,
    scale_auxiliary_loss_to_gradient_budget,
)
from boxmot.utils import logger as LOGGER


class _HumanPrivilegedRetrievalMixin:
    """Connect training-only retrieval teachers to the existing trainer."""

    def _hpgrd_enabled(self) -> bool:
        """Return whether at least one human-privileged objective is enabled."""
        return any(
            weight > 0
            for weight in (
                float(getattr(self, "hpgrd_global_weight", 0.0)),
                float(getattr(self, "hpgrd_part_weight", 0.0)),
                float(getattr(self, "hpgrd_background_weight", 0.0)),
                float(getattr(self, "hpgrd_part_drop_weight", 0.0)),
            )
        )

    def _hpgrd_owns_anatomical_runtime(self) -> bool:
        """Return whether anatomy exists solely to expose HP-GRD part tokens."""
        if (
            float(getattr(self, "hpgrd_part_weight", 0.0)) <= 0
            and float(getattr(self, "hpgrd_part_drop_weight", 0.0)) <= 0
        ):
            return False
        intrinsic_weights = (
            "anatomical_distill_weight",
            "anatomical_attention_weight",
            "anatomical_foreground_weight",
            "anatomical_semantic_part_weight",
            "anatomical_visibility_weight",
            "anatomical_contrastive_weight",
            "anatomical_descriptor_distill_weight",
            "anatomical_branch_distill_weight",
            "anatomical_pose_teacher_weight",
            "anatomical_query_distill_weight",
            "anatomical_query_relational_distill_weight",
            "anatomical_query_diversity_weight",
            "anatomical_part_triplet_weight",
            "anatomical_cross_scale_weight",
            "clean_student_consistency_weight",
        )
        return not getattr(self, "anatomical_deployment", False) and not any(
            float(getattr(self, name, 0.0)) > 0 for name in intrinsic_weights
        )

    def _anatomical_training_active(self, epoch: int | None) -> bool:
        """Align a part-packet-only anatomy path with the HP-GRD schedule."""
        active = super()._anatomical_training_active(epoch)
        if not self._hpgrd_owns_anatomical_runtime() or epoch is None:
            return active
        return self._retrieval_auxiliary_schedule_scale(epoch) > 0

    def _set_anatomical_runtime_active(
        self,
        model: torch.nn.Module,
        active: bool,
    ) -> None:
        """Toggle both legacy anatomy and fixed HP-GRD packet exposure."""
        # The canonical HP-GRD recipe keeps the prior V20 checkpoint schema,
        # but bypasses its learned anatomy adapter because every intrinsic
        # anatomy objective is zero.  Dense masks instead pool the shared map
        # below with no trainable intermediary.
        super()._set_anatomical_runtime_active(
            model,
            active and not self._hpgrd_owns_anatomical_runtime(),
        )
        head = self._model_head(model)
        retrieval_setter = getattr(head, "set_retrieval_packet_active", None)
        retrieval_active = float(getattr(self, "global_ap_loss_weight", 0.0)) > 0 or self._hpgrd_enabled()
        if callable(retrieval_setter):
            retrieval_setter(retrieval_active)
        elif retrieval_active and head is not None:
            # Non-CSL toy/custom models may already return a mapping and need
            # no hook; CSL heads must provide the explicit contract.
            if head.__class__.__module__.startswith("boxmot.reid.backbones.families.csl_tinyvit"):
                raise RuntimeError("Retrieval objectives require CSL training-packet support")
        setter = getattr(head, "set_hpgrd_part_packet_active", None)
        part_active = active and float(getattr(self, "hpgrd_part_weight", 0.0)) > 0
        if callable(setter):
            setter(part_active)
        elif part_active:
            raise RuntimeError("HP-GRD part supervision requires a head with fixed part-packet support")

    def _deployment_feature_dim(self, model: torch.nn.Module, fallback: int) -> int:
        """Resolve the statically declared deployed descriptor width."""
        head = self._model_head(model)
        resolver = getattr(head, "_declared_feature_dim", None)
        if callable(resolver):
            declared = resolver(DEPLOYED_DESCRIPTOR_KEY)
            if declared is not None:
                return int(declared)
        return int(fallback)

    def _initialize_retrieval_training_components(self, deployment_dim: int) -> None:
        """Build the memory loss and load the immutable offline teacher cache."""
        self._global_ap = None
        self._privileged_graph_cache = None
        self._privileged_graph_loss = None
        self._hpgrd_manifest_sha256 = None
        self._retrieval_dataset_sha256 = None

        retrieval_enabled = self.global_ap_loss_weight > 0 or self._hpgrd_enabled()
        samples = tuple(getattr(self, "_train_samples", ()))
        if retrieval_enabled:
            if not samples:
                raise RuntimeError("GlobalAP/HP-GRD requires a non-empty indexed training set")
            self._retrieval_dataset_sha256 = dataset_samples_sha256(samples)

        if self.global_ap_loss_weight > 0:
            sample_count = int(getattr(self, "_train_sample_count", 0))
            if sample_count <= 0:
                raise RuntimeError("GlobalAP requires a non-empty indexed training set")
            if self.global_ap_memory_size < sample_count:
                raise ValueError(
                    "global_ap_memory_size must cover every stable training sample index: "
                    f"capacity={self.global_ap_memory_size}, samples={sample_count}"
                )
            self._global_ap = IdentityGlobalAP(
                memory_size=self.global_ap_memory_size,
                feature_dim=int(deployment_dim),
                top_k=self.global_ap_topk,
                temperature=self.global_ap_temperature,
                max_age=None if self.global_ap_max_age == 0 else self.global_ap_max_age,
                memory_momentum=self.global_ap_momentum,
                strict_metadata=True,
            ).to(self.device)
            LOGGER.info(
                "GlobalAP memory enabled on norm_concat_bn: "
                f"weight={self.global_ap_loss_weight:g}, rows={self.global_ap_memory_size}, "
                f"topk={self.global_ap_topk}, schedule="
                f"{self.global_ap_start_epoch}-{self.global_ap_ramp_end_epoch}-"
                f"{self.global_ap_decay_start_epoch}-{self.global_ap_decay_end_epoch}"
            )

        if self._hpgrd_enabled():
            cache_path = self._resolve_hpgrd_cache_path(self.hpgrd_cache_dir)
            expected_part_names = (
                ANATOMICAL_PARTS if self.hpgrd_part_weight > 0 or self.hpgrd_part_drop_weight > 0 else None
            )
            self._privileged_graph_cache = PrivilegedGraphTeacherCache.load(
                cache_path,
                expected_dataset_sha256=self._retrieval_dataset_sha256,
                expected_part_names=expected_part_names,
            )
            if len(self._privileged_graph_cache) != len(samples):
                raise ValueError(
                    "HP-GRD cache must contain exactly one row per training sample: "
                    f"cache={len(self._privileged_graph_cache)}, dataset={len(samples)}"
                )
            manifest = self._privileged_graph_cache.manifest or {}
            self._hpgrd_manifest_sha256 = manifest.get("manifest_sha256")
            if self.hpgrd_part_weight > 0 or self.hpgrd_part_drop_weight > 0:
                cached_parts = int(manifest.get("part_count", 0))
                if cached_parts != len(ANATOMICAL_PARTS):
                    raise ValueError(
                        "HP-GRD cache/model semantic-part mismatch: "
                        f"cache={cached_parts}, model={len(ANATOMICAL_PARTS)}"
                    )
                cached_names = tuple(manifest.get("part_names", ()))
                if cached_names != tuple(ANATOMICAL_PARTS):
                    raise ValueError(
                        "HP-GRD cache semantic-part names/order do not match the model: "
                        f"cache={cached_names}, model={tuple(ANATOMICAL_PARTS)}"
                    )
            if self.hpgrd_part_drop_weight > 0 and manifest.get("leave_part_out_dim") is None:
                raise ValueError("HP-GRD semantic part dropout requires cached leave_part_out_descriptors")
            self._privileged_graph_loss = PrivilegedGraphLoss(
                global_weight=self.hpgrd_global_weight,
                part_weight=self.hpgrd_part_weight,
                background_weight=self.hpgrd_background_weight,
                semantic_drop_weight=self.hpgrd_part_drop_weight,
            ).to(self.device)
            LOGGER.info(
                "HP-GRD offline teacher enabled: "
                f"cache={cache_path}, rows={len(self._privileged_graph_cache)}, "
                f"teacher={str(manifest.get('teacher_sha256', 'unknown'))[:12]}, "
                f"weights=(global={self.hpgrd_global_weight:g}, part={self.hpgrd_part_weight:g}, "
                f"background={self.hpgrd_background_weight:g}, drop={self.hpgrd_part_drop_weight:g})"
            )

    @staticmethod
    def _resolve_hpgrd_cache_path(value: str | Path | None) -> Path:
        if value is None:
            raise ValueError("HP-GRD requires a privileged teacher cache")
        path = Path(value).expanduser()
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"HP-GRD cache does not exist: {path}")
        preferred = tuple(path / name for name in ("privileged_graph.pt", "cache.pt"))
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        candidates = sorted(path.glob("*.pt"))
        if len(candidates) != 1:
            raise ValueError(
                f"HP-GRD cache directory must contain privileged_graph.pt, cache.pt, or exactly one .pt file: {path}"
            )
        return candidates[0]

    def _retrieval_auxiliary_schedule_scale(self, epoch: int) -> float:
        """Warm up, hold, and retire privileged retrieval supervision."""
        if epoch <= self.global_ap_start_epoch:
            return 0.0
        if epoch < self.global_ap_ramp_end_epoch:
            return (epoch - self.global_ap_start_epoch) / max(
                self.global_ap_ramp_end_epoch - self.global_ap_start_epoch,
                1,
            )
        if epoch <= self.global_ap_decay_start_epoch:
            return 1.0
        if epoch < self.global_ap_decay_end_epoch:
            return 1.0 - (epoch - self.global_ap_decay_start_epoch) / max(
                self.global_ap_decay_end_epoch - self.global_ap_decay_start_epoch,
                1,
            )
        return 0.0

    @staticmethod
    def _deployment_descriptor(features: Mapping[str, Any] | None) -> torch.Tensor:
        if not isinstance(features, Mapping):
            raise RuntimeError("Privileged retrieval losses require a dictionary feature packet")
        descriptor = features.get(DEPLOYED_DESCRIPTOR_KEY)
        if not torch.is_tensor(descriptor) or descriptor.ndim != 2:
            raise RuntimeError(f"Model training output must contain {DEPLOYED_DESCRIPTOR_KEY!r} with shape [B,D]")
        return descriptor

    def _global_ap_objective(
        self,
        features: Mapping[str, Any] | None,
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, float]:
        """Return the raw GlobalAP loss and its scheduled scalar weight."""
        if self._global_ap is None:
            zero = pids.sum() * 0.0
            return zero, 0.0
        descriptor = self._deployment_descriptor(features)
        scale = self._retrieval_auxiliary_schedule_scale(epoch)
        if scale <= 0:
            return descriptor.sum() * 0.0, 0.0
        return (
            self._global_ap(descriptor, sample_indices, pids),
            self.global_ap_loss_weight * scale,
        )

    @torch.no_grad()
    def _update_global_ap_memory(
        self,
        features: Mapping[str, Any] | None,
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
    ) -> None:
        if self._global_ap is None:
            return
        self._global_ap.update(
            self._deployment_descriptor(features),
            sample_indices,
            pids,
        )

    def _confidence_filtered_teacher(
        self,
        teacher: PrivilegedGraphTeacherBatch,
    ) -> PrivilegedGraphTeacherBatch:
        """Zero teacher reliabilities below the configured confidence floor."""
        visibility = teacher.part_visibility
        confidence = teacher.part_confidence
        fused = fuse_privileged_confidence(visibility, confidence)
        valid = fused >= self.hpgrd_min_confidence
        global_confidence = teacher.global_confidence
        if global_confidence is not None:
            global_confidence = torch.where(
                global_confidence >= self.hpgrd_min_confidence,
                global_confidence,
                torch.zeros_like(global_confidence),
            )
        return PrivilegedGraphTeacherBatch(
            sample_indices=teacher.sample_indices,
            global_descriptors=teacher.global_descriptors,
            part_descriptors=teacher.part_descriptors,
            part_visibility=torch.where(valid, visibility, torch.zeros_like(visibility)),
            part_confidence=torch.where(valid, confidence, torch.zeros_like(confidence)),
            global_confidence=global_confidence,
            leave_part_out_descriptors=teacher.leave_part_out_descriptors,
        )

    def _hpgrd_objective(
        self,
        base_loss: torch.Tensor,
        student_packet: Mapping[str, torch.Tensor],
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Return scheduled, gradient-budgeted HP-GRD and raw diagnostics."""
        zero = base_loss.detach() * 0.0
        empty = {
            "global_relational": zero,
            "part_relational": zero,
            "background_consistency": zero,
            "semantic_drop_relational": zero,
        }
        if self._privileged_graph_loss is None or self._privileged_graph_cache is None:
            return zero, empty, zero
        schedule = self._retrieval_auxiliary_schedule_scale(epoch)
        if schedule <= 0:
            return zero, empty, zero
        teacher = self._privileged_graph_cache.lookup(
            sample_indices,
            device=self.device,
            # Cached teacher geometry remains FP32 even under student AMP.
            dtype=None,
        )
        teacher = self._confidence_filtered_teacher(teacher)
        result = self._privileged_graph_loss(student_packet, teacher, pids)
        auxiliary = result.total * schedule
        if not auxiliary.requires_grad or float(auxiliary.detach().abs().item()) == 0.0:
            return auxiliary, dict(result.components), auxiliary.detach() * 0.0
        shared_reference = student_packet.get("_hpgrd_gradient_reference")
        budget_references: list[torch.Tensor] = [
            shared_reference if torch.is_tensor(shared_reference) else self._deployment_descriptor(student_packet)
        ]
        semantic_descriptor = student_packet.get(SEMANTIC_DROP_DESCRIPTOR_KEY)
        if torch.is_tensor(semantic_descriptor):
            # The alternate forward is not upstream of the primary map. Add
            # its descriptor so its intervention gradient remains budgeted;
            # the base objective correctly contributes zero on this branch.
            budget_references.append(semantic_descriptor)
        budget = scale_auxiliary_loss_to_gradient_budget(
            base_loss,
            auxiliary,
            tuple(budget_references),
            max_ratio=self.hpgrd_gradient_fraction,
        )
        return budget.scaled_loss, dict(result.components), budget.scale.detach()

    def _hpgrd_student_packet(
        self,
        features: Mapping[str, Any],
        *,
        background_features: Mapping[str, Any] | None = None,
        background_indices: torch.Tensor | None = None,
        semantic_drop_features: Mapping[str, Any] | None = None,
        semantic_drop_indices: torch.Tensor | None = None,
        semantic_drop_parts: torch.Tensor | None = None,
        semantic_drop_confidence: torch.Tensor | None = None,
        anatomical_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        packet: dict[str, torch.Tensor] = {
            DEPLOYED_DESCRIPTOR_KEY: self._deployment_descriptor(features),
        }
        gradient_reference = features.get("_hpgrd_feature_map")
        if not torch.is_tensor(gradient_reference):
            gradient_reference = features.get("raw_concat")
        if torch.is_tensor(gradient_reference):
            packet["_hpgrd_gradient_reference"] = gradient_reference
        if self.hpgrd_part_weight > 0:
            feature_map = features.get("_hpgrd_feature_map")
            if torch.is_tensor(feature_map) and anatomical_targets is not None:
                parts = self._fixed_mask_pooled_parts(
                    feature_map,
                    anatomical_targets,
                )
                runtime_reliability = anatomical_targets.get("reliability")
                if not torch.is_tensor(runtime_reliability):
                    runtime_masks = anatomical_targets.get("masks")
                    if not torch.is_tensor(runtime_masks):
                        raise RuntimeError("HP-GRD part pooling requires masks or reliability")
                    runtime_reliability = (runtime_masks.flatten(2).amax(dim=-1) > 1e-6).float()
                if runtime_reliability.shape != parts.shape[:2]:
                    raise RuntimeError("HP-GRD runtime part reliability must match [B,P]")
                packet[PART_RELIABILITY_KEY] = runtime_reliability.to(
                    device=parts.device,
                    dtype=torch.float32,
                )
            else:
                # Retain the generic packet contract for heads that already
                # expose part descriptors.  The canonical 7M recipe takes the
                # fixed-pooling branch above and has no disposable adapter.
                parts = features.get(PART_DESCRIPTOR_KEY)
                if not torch.is_tensor(parts):
                    raise RuntimeError("HP-GRD part loss requires '_hpgrd_feature_map' plus dense anatomical masks")
            packet[PART_DESCRIPTOR_KEY] = parts
        if background_features is not None:
            # The clean/background-control branch is a training-time target.
            # Keeping it detached makes the deployed view the only optimized
            # side of this consistency edge, including when another feature
            # path supplied the clean packet.
            background = self._deployment_descriptor(background_features)
            if background_indices is not None:
                if background.shape[0] == packet[DEPLOYED_DESCRIPTOR_KEY].shape[0]:
                    background = background.index_select(
                        0,
                        background_indices.to(device=background.device),
                    )
                elif background.shape[0] != background_indices.numel():
                    raise RuntimeError("Background intervention rows do not align with their indices")
            packet[BACKGROUND_DESCRIPTOR_KEY] = background.detach()
            if background_indices is not None:
                packet[BACKGROUND_INDICES_KEY] = background_indices
                packet[BACKGROUND_CONFIDENCE_KEY] = torch.ones(
                    background_indices.numel(),
                    device=background_indices.device,
                    dtype=packet[BACKGROUND_DESCRIPTOR_KEY].dtype,
                )
        if semantic_drop_features is not None:
            if semantic_drop_indices is None or semantic_drop_parts is None:
                raise RuntimeError("Semantic-drop descriptors require row and part indices")
            semantic_drop = self._deployment_descriptor(semantic_drop_features)
            if semantic_drop.shape[0] == packet[DEPLOYED_DESCRIPTOR_KEY].shape[0]:
                semantic_drop = semantic_drop.index_select(
                    0,
                    semantic_drop_indices.to(device=semantic_drop.device),
                )
            elif semantic_drop.shape[0] != semantic_drop_indices.numel():
                raise RuntimeError("Semantic-drop intervention rows do not align with their indices")
            packet[SEMANTIC_DROP_DESCRIPTOR_KEY] = semantic_drop
            packet[SEMANTIC_DROP_INDICES_KEY] = semantic_drop_indices
            packet[SEMANTIC_DROP_PARTS_KEY] = semantic_drop_parts
            if semantic_drop_confidence is not None:
                packet[SEMANTIC_DROP_CONFIDENCE_KEY] = semantic_drop_confidence
        return packet

    @staticmethod
    def _fixed_mask_pooled_parts(
        feature_map: torch.Tensor,
        anatomical_targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool semantic parts from a deployed map with no learned adapter."""
        masks = anatomical_targets.get("masks")
        if not torch.is_tensor(masks):
            raise RuntimeError("HP-GRD fixed part pooling requires dense anatomical masks")
        if feature_map.ndim != 4 or masks.ndim != 4:
            raise RuntimeError("HP-GRD feature maps and masks must have shapes [B,C,H,W] and [B,P,H,W]")
        if feature_map.shape[0] != masks.shape[0]:
            raise RuntimeError("HP-GRD feature maps and masks must have the same batch size")
        masks = masks.to(device=feature_map.device, dtype=torch.float32).clamp(0, 1)
        if masks.shape[-2:] != feature_map.shape[-2:]:
            masks = F.interpolate(
                masks,
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        features = feature_map.float().unsqueeze(1)
        weights = masks.unsqueeze(2)
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        return (features * weights).sum(dim=(-2, -1)) / denominator

    def _build_hpgrd_semantic_drop_view(
        self,
        imgs: torch.Tensor,
        pids: torch.Tensor,
        anatomical_targets: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Mask one shared semantic part while retaining identity graph groups."""
        if self.hpgrd_part_drop_weight <= 0 or anatomical_targets is None:
            return None, None, None, None
        masks = anatomical_targets.get("masks")
        reliability = anatomical_targets.get("reliability")
        if not torch.is_tensor(masks) or not torch.is_tensor(reliability):
            raise RuntimeError("HP-GRD semantic dropout requires dense masks and reliability")
        masks = masks.to(device=imgs.device, dtype=imgs.dtype)
        reliability = reliability.to(device=imgs.device, dtype=imgs.dtype).clamp(0, 1)
        if masks.ndim != 4 or reliability.shape != masks.shape[:2]:
            raise RuntimeError("HP-GRD semantic masks must have [B,P,H,W] with [B,P] reliability")
        visible = (reliability >= self.hpgrd_min_confidence) & (masks.flatten(2).amax(dim=-1) > 1e-6)
        candidate_parts: list[int] = []
        for part in range(masks.shape[1]):
            rows = torch.nonzero(visible[:, part], as_tuple=False).flatten()
            if rows.numel() < 3:
                continue
            row_pids = pids.index_select(0, rows)
            if row_pids.unique().numel() < 2:
                continue
            positive = row_pids[:, None] == row_pids[None, :]
            positive.fill_diagonal_(False)
            if bool(positive.any()):
                candidate_parts.append(part)
        if not candidate_parts:
            return None, None, None, None
        selected_part = candidate_parts[int(torch.randint(len(candidate_parts), (), device=imgs.device).item())]
        eligible_rows = torch.nonzero(visible[:, selected_part], as_tuple=False).flatten()
        eligible_pids = pids.index_select(0, eligible_rows)
        selected_pid_values = []
        for pid in eligible_pids.unique(sorted=True):
            if bool(torch.rand((), device=imgs.device) < self.hpgrd_part_drop_probability):
                selected_pid_values.append(pid)
        if len(selected_pid_values) < 2:
            return None, None, None, None
        selected_pid_tensor = torch.stack(selected_pid_values)
        selected = (eligible_pids[:, None] == selected_pid_tensor[None, :]).any(dim=1)
        base_indices = eligible_rows[selected]
        if base_indices.numel() < 3:
            return None, None, None, None
        selected_pids = pids.index_select(0, base_indices)
        positive = selected_pids[:, None] == selected_pids[None, :]
        positive.fill_diagonal_(False)
        if not bool(positive.any()) or selected_pids.unique().numel() < 2:
            return None, None, None, None

        part_masks = masks[base_indices, selected_part].unsqueeze(1).clamp(0, 1)
        if part_masks.shape[-2:] != imgs.shape[-2:]:
            part_masks = F.interpolate(
                part_masks,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        dropped = imgs.clone()
        dropped[base_indices] = imgs.index_select(0, base_indices) * (1.0 - part_masks)
        parts = torch.full_like(base_indices, selected_part)
        confidence = reliability[base_indices, selected_part]
        return dropped, base_indices, parts, confidence

    def _hpgrd_intervention_forward(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        *,
        detached: bool,
    ) -> Any:
        """Forward in the primary BN domain without persisting BN updates."""
        batch_norms = [module for module in model.modules() if isinstance(module, _BatchNorm)]
        snapshots = [
            (
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
                (None if module.num_batches_tracked is None else module.num_batches_tracked.detach().clone()),
            )
            for module in batch_norms
        ]
        context = torch.no_grad() if detached else nullcontext()
        try:
            with context:
                return model(images)
        finally:
            with torch.no_grad():
                for module, (running_mean, running_var, batches) in zip(
                    batch_norms,
                    snapshots,
                    strict=True,
                ):
                    if running_mean is not None:
                        module.running_mean.copy_(running_mean)
                    if running_var is not None:
                        module.running_var.copy_(running_var)
                    if batches is not None:
                        module.num_batches_tracked.copy_(batches)

    def _training_auxiliary_state(self) -> dict[str, Any] | None:
        """Return resumable training-only state, excluding immutable caches."""
        if self._global_ap is None:
            return None
        return {"global_ap_state_dict": self._global_ap.state_dict()}

    def _restore_training_auxiliary_state(self, state: Mapping[str, Any] | None) -> None:
        """Strictly restore the GlobalAP memory for exact continuation."""
        if self._global_ap is None:
            return
        if not isinstance(state, Mapping) or "global_ap_state_dict" not in state:
            raise ValueError("Resumable GlobalAP checkpoint is missing its memory state")
        self._global_ap.load_state_dict(state["global_ap_state_dict"], strict=True)
