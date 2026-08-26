"""Shared anatomical schedules, relations, and deployment losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AnatomicalCommonMixin:
    @staticmethod
    def _cross_camera_part_contrastive_loss(
        tokens: torch.Tensor,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        reliability: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Supervised contrastive loss over corresponding visible body parts."""
        if tokens.ndim != 3 or reliability.shape != tokens.shape[:2]:
            raise ValueError("anatomical contrastive inputs must have shapes [B,P,D] and [B,P]")
        contrastive_dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
        tokens_for_loss = tokens.to(dtype=contrastive_dtype)
        reliability = reliability.to(
            device=tokens.device,
            dtype=contrastive_dtype,
        )
        normalized = F.normalize(tokens_for_loss, p=2, dim=-1)
        batch_size, part_count, _ = normalized.shape
        identity_match = pids[:, None] == pids[None, :]
        cross_camera = camera_ids[:, None] != camera_ids[None, :]
        not_self = ~torch.eye(
            batch_size,
            dtype=torch.bool,
            device=tokens.device,
        )
        losses = []
        weights = []
        for part_index in range(part_count):
            part_reliability = reliability[:, part_index].clamp(0, 1)
            part_valid = part_reliability > 0
            candidate_mask = part_valid[:, None] & part_valid[None, :] & not_self & ~(identity_match & ~cross_camera)
            positive_mask = candidate_mask & identity_match & cross_camera
            anchors = part_valid & (positive_mask.sum(dim=1) > 0)
            if not anchors.any():
                continue

            similarities = (normalized[:, part_index] @ normalized[:, part_index].transpose(0, 1)) / temperature
            anchor_similarities = similarities[anchors]
            anchor_candidate_mask = candidate_mask[anchors]
            anchor_positive_mask = positive_mask[anchors]
            denominator_logits = anchor_similarities.masked_fill(
                ~anchor_candidate_mask,
                float("-inf"),
            )
            log_probabilities = anchor_similarities - torch.logsumexp(
                denominator_logits,
                dim=1,
                keepdim=True,
            )
            pair_weights = anchor_positive_mask.to(contrastive_dtype) * torch.sqrt(
                part_reliability[anchors, None] * part_reliability[None, :]
            )
            pair_weight_sums = pair_weights.sum(dim=1).clamp_min(1e-6)
            anchor_losses = (
                -(log_probabilities.masked_fill(~anchor_positive_mask, 0) * pair_weights).sum(dim=1) / pair_weight_sums
            )
            losses.append(anchor_losses)
            weights.append(part_reliability[anchors])
        if not losses:
            return tokens_for_loss.sum() * 0
        concatenated_losses = torch.cat(losses)
        concatenated_weights = torch.cat(weights)
        return (concatenated_losses * concatenated_weights).sum() / concatenated_weights.sum().clamp_min(1e-6)

    def _anatomical_deployment_losses(
        self,
        criterion_id: nn.Module,
        features,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep deployed RGB part tokens discriminative after pose loss decay."""
        zero = torch.zeros((), device=self.device)
        if not self.anatomical_deployment:
            return zero, zero
        if not isinstance(features, dict):
            raise RuntimeError("anatomical deployment requires the head feature dictionary")
        raw_parts = features.get("_anatomical_deployment_parts")
        logits = features.get("_anatomical_deployment_logits")
        visibility = features.get("_anatomical_deployment_visibility")
        if (
            not torch.is_tensor(raw_parts)
            or raw_parts.ndim != 3
            or not isinstance(logits, tuple)
            or len(logits) != raw_parts.shape[1]
            or not torch.is_tensor(visibility)
            or visibility.shape != raw_parts.shape[:2]
            or any(not torch.is_tensor(value) for value in logits)
        ):
            raise RuntimeError("anatomical deployment head returned an invalid training packet")

        detached_visibility = visibility.detach().clamp(0, 1)
        epsilon = float(getattr(criterion_id, "epsilon", 0.0))
        weighted_id_sum = raw_parts.sum() * 0.0
        id_weight_sum = raw_parts.new_zeros(())
        for part_index, part_logits in enumerate(logits):
            log_probabilities = F.log_softmax(part_logits, dim=1)
            nll = -log_probabilities.gather(
                1,
                pids[:, None],
            ).squeeze(1)
            smooth = -log_probabilities.mean(dim=1)
            values = (1.0 - epsilon) * nll + epsilon * smooth
            weights = detached_visibility[:, part_index].to(dtype=values.dtype)
            weighted_id_sum = weighted_id_sum + (values * weights).sum()
            id_weight_sum = id_weight_sum + weights.sum()
        id_loss = weighted_id_sum / id_weight_sum.clamp_min(1.0)

        metric_loss = self._cross_camera_part_contrastive_loss(
            raw_parts,
            pids,
            camera_ids,
            detached_visibility,
            self.anatomical_temperature,
        )
        return id_loss, metric_loss

    @staticmethod
    def _cross_camera_relational_distill_loss(
        student: torch.Tensor,
        teacher: torch.Tensor,
        reliability: torch.Tensor,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Match teacher/student relations with balanced cross-camera pairs."""
        if student.ndim != 2 or teacher.ndim != 2:
            raise ValueError("Branch relational distillation expects [B,D] descriptors")
        if student.shape[0] != teacher.shape[0]:
            raise ValueError("Branch student and teacher batch dimensions must match")
        if reliability.shape != (student.shape[0],):
            raise ValueError(f"Branch reliability must have shape [B], got {tuple(reliability.shape)}")
        loss_dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        student = F.normalize(
            student.to(dtype=loss_dtype),
            p=2,
            dim=1,
        )
        teacher = F.normalize(
            teacher.detach().to(
                device=student.device,
                dtype=loss_dtype,
            ),
            p=2,
            dim=1,
        )
        reliability = reliability.to(
            device=student.device,
            dtype=loss_dtype,
        ).clamp(0, 1)
        student_similarity = student @ student.transpose(0, 1)
        teacher_similarity = teacher @ teacher.transpose(0, 1)
        not_self = ~torch.eye(
            student.shape[0],
            device=student.device,
            dtype=torch.bool,
        )
        cross_camera = camera_ids[:, None] != camera_ids[None, :]
        reliable = reliability > 0
        valid = reliable[:, None] & reliable[None, :] & cross_camera & not_self
        same_identity = pids[:, None] == pids[None, :]
        pair_weights = (reliability[:, None] * reliability[None, :]).sqrt()
        squared_error = (student_similarity - teacher_similarity).square()
        pair_losses = []
        pair_active = []
        for pair_mask in (
            valid & same_identity,
            valid & ~same_identity,
        ):
            weights = pair_weights * pair_mask.to(loss_dtype)
            weight_sum = weights.sum()
            pair_losses.append((squared_error * weights).sum() / weight_sum.clamp_min(1e-6))
            pair_active.append((weight_sum > 0).to(loss_dtype))
        active = torch.stack(pair_active)
        return (torch.stack(pair_losses) * active).sum() / active.sum().clamp_min(1.0)

    @staticmethod
    def _query_relational_distill_loss(
        student: torch.Tensor,
        teacher: torch.Tensor,
        reliability: torch.Tensor,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Match per-query teacher identity geometry across cameras.

        Each semantic query gets its own batch cosine-similarity matrix. Same-
        identity and different-identity pairs are averaged independently so
        the much larger negative set cannot dominate the treatment.
        """
        if student.ndim != 3 or teacher.ndim != 3:
            raise ValueError("Query relational distillation expects [B,P,D] tokens")
        if student.shape != teacher.shape:
            raise ValueError("Query student and teacher tokens must have the same shape")
        if reliability.shape != student.shape[:2]:
            raise ValueError(f"Query reliability must have shape [B,P], got {tuple(reliability.shape)}")
        loss_dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        student = F.normalize(student.to(dtype=loss_dtype), p=2, dim=-1)
        teacher = F.normalize(
            teacher.detach().to(
                device=student.device,
                dtype=loss_dtype,
            ),
            p=2,
            dim=-1,
        )
        reliability = reliability.to(
            device=student.device,
            dtype=loss_dtype,
        ).clamp(0, 1)
        pids = pids.to(device=student.device)
        camera_ids = camera_ids.to(device=student.device)
        not_self = ~torch.eye(
            student.shape[0],
            device=student.device,
            dtype=torch.bool,
        )
        cross_camera = camera_ids[:, None] != camera_ids[None, :]
        same_identity = pids[:, None] == pids[None, :]

        part_losses: list[torch.Tensor] = []
        for part_index in range(student.shape[1]):
            part_reliability = reliability[:, part_index]
            valid_sample = part_reliability > 0
            valid_pair = valid_sample[:, None] & valid_sample[None, :] & cross_camera & not_self
            pair_weights = (part_reliability[:, None] * part_reliability[None, :]).sqrt()
            student_similarity = student[:, part_index] @ student[:, part_index].transpose(0, 1)
            teacher_similarity = teacher[:, part_index] @ teacher[:, part_index].transpose(0, 1)
            errors = F.smooth_l1_loss(
                student_similarity,
                teacher_similarity,
                reduction="none",
            )
            group_losses: list[torch.Tensor] = []
            for group_mask in (
                valid_pair & same_identity,
                valid_pair & ~same_identity,
            ):
                weights = pair_weights * group_mask.to(loss_dtype)
                if weights.sum() > 0:
                    group_losses.append((errors * weights).sum() / weights.sum().clamp_min(1e-6))
            if group_losses:
                part_losses.append(torch.stack(group_losses).mean())
        if not part_losses:
            return student.sum() * 0.0
        return torch.stack(part_losses).mean()

    def _anatomical_schedule_scales(
        self,
        epoch: int | None,
    ) -> tuple[float, float]:
        """Return the shared student and global decay scales."""
        if epoch is None:
            return 1.0, 1.0

        start_epoch = int(getattr(self, "anatomical_student_start_epoch", 0))
        ramp_end_epoch = int(getattr(self, "anatomical_student_ramp_end_epoch", 0))
        decay_start_epoch = int(getattr(self, "anatomical_decay_start_epoch", 0))
        decay_end_epoch = int(getattr(self, "anatomical_decay_end_epoch", 0))

        if start_epoch == 0 and ramp_end_epoch == 0:
            student_scale = 1.0
        elif epoch <= start_epoch:
            student_scale = 0.0
        elif ramp_end_epoch > start_epoch and epoch < ramp_end_epoch:
            student_scale = (epoch - start_epoch) / (ramp_end_epoch - start_epoch)
        else:
            student_scale = 1.0

        decay_scale = 1.0
        if decay_end_epoch > decay_start_epoch and epoch > decay_start_epoch:
            decay_scale = max(
                0.0,
                1.0 - (epoch - decay_start_epoch) / (decay_end_epoch - decay_start_epoch),
            )
        return student_scale * decay_scale, decay_scale

    def _anatomical_fine_schedule_scale(
        self,
        epoch: int | None,
        *,
        student_scale: float,
        decay_scale: float,
    ) -> float:
        """Return the fine-map scale, falling back to the shared student schedule."""
        start_epoch = int(getattr(self, "anatomical_fine_start_epoch", 0))
        ramp_end_epoch = int(getattr(self, "anatomical_fine_ramp_end_epoch", 0))
        if start_epoch == 0 and ramp_end_epoch == 0:
            return student_scale
        if epoch is None:
            return 1.0
        if epoch <= start_epoch:
            fine_scale = 0.0
        elif ramp_end_epoch > start_epoch and epoch < ramp_end_epoch:
            fine_scale = (epoch - start_epoch) / (ramp_end_epoch - start_epoch)
        else:
            fine_scale = 1.0
        return fine_scale * decay_scale

    def _anatomical_training_active(self, epoch: int | None) -> bool:
        """Return whether scheduled pose-concatenation work affects the loss.

        Other anatomical modes retain their existing execution contract. The
        compact non-semantic EMA modes can be shut down completely once their
        shared student, fine and teacher decay scales all reach zero.
        """
        if not getattr(self, "anatomical_auxiliary", False):
            return False
        # Deployment ID/metric supervision is intentionally unscheduled and
        # consumes the anatomical student packet after the auxiliary decay.
        if getattr(self, "anatomical_deployment", False):
            return True
        if getattr(self, "anatomical_target_type", "") not in {
            "learned_pose_concat_ema",
            "learned_pose_semantic_ema",
            "learned_pose_semantic_fused_ema",
        }:
            return True
        student_scale, decay_scale = self._anatomical_schedule_scales(epoch)
        fine_scale = self._anatomical_fine_schedule_scale(
            epoch,
            student_scale=student_scale,
            decay_scale=decay_scale,
        )
        return any(scale > 0 for scale in (student_scale, fine_scale, decay_scale))

    def _set_anatomical_runtime_active(
        self,
        model: nn.Module,
        active: bool,
    ) -> None:
        """Apply the non-persistent auxiliary schedule to a wrapped model."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        setter = getattr(
            unwrapped,
            "set_anatomical_auxiliary_active",
            None,
        )
        if not callable(setter):
            setter = getattr(
                getattr(unwrapped, "head", None),
                "set_anatomical_auxiliary_active",
                None,
            )
        if callable(setter):
            setter(active)
        elif (
            getattr(self, "anatomical_auxiliary", False)
            and not active
        ):
            raise RuntimeError(
                "Anatomical model is missing its runtime schedule hook"
            )

    @staticmethod
    def _zero_anatomical_loss(
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the complete zero-valued anatomy metrics packet."""
        zero = torch.zeros(
            (),
            device=reference.device,
            dtype=torch.float32,
        )
        keys = (
            "distill",
            "attention",
            "visibility",
            "contrastive",
            "descriptor_distill",
            "branch_distill",
            "branch_global",
            "branch_coarse",
            "branch_fine",
            "pose_teacher",
            "semantic_foreground",
            "semantic_part",
            "local_scale",
            "fine_scale",
            "cross_scale",
            "valid_part_fraction",
            "cross_camera_anchor_fraction",
            "query_distill",
            "query_relational_distill",
            "query_diversity",
            "part_triplet",
            "accessory_valid_fraction",
        )
        return zero, {key: zero for key in keys}
