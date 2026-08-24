"""Privileged body-slot supervision."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class _BodySlotAnatomicalMixin:
    @staticmethod
    def _body_slot_teacher_masks(
        targets: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the eight weak semantic role masks used only by the teacher.

        The roles anchor slot identity without constraining the RGB student's
        inference-time attention: head, upper torso, lower torso, left side,
        right side, legs/footwear, accessory, and residual person evidence.
        """
        missing = [key for key in ("masks", "foreground_mask", "accessory_mask") if key not in targets]
        if missing:
            raise RuntimeError(f"Body-slot teacher requires anatomical targets: {missing}")
        parts = targets["masks"].to(device=device, dtype=dtype)
        foreground = (
            targets["foreground_mask"]
            .to(
                device=device,
                dtype=dtype,
            )
            .clamp(0, 1)
        )
        accessory = (
            targets["accessory_mask"]
            .to(
                device=device,
                dtype=dtype,
            )
            .clamp(0, 1)
        )
        if parts.ndim != 4 or parts.shape[1] != 6:
            raise RuntimeError(f"Body-slot teacher expects six pose-derived part masks, got {tuple(parts.shape)}")
        if foreground.shape != parts[:, :1].shape:
            raise RuntimeError("Body-slot foreground target does not match part masks")
        if accessory.shape != foreground.shape:
            raise RuntimeError("Body-slot accessory target does not match foreground mask")

        parts = parts.clamp(0, 1) * foreground
        head, torso, left_arm, right_arm, left_leg, right_leg = (parts[:, index : index + 1] for index in range(6))
        height = parts.shape[-2]
        y = torch.linspace(
            0,
            1,
            height,
            device=device,
            dtype=dtype,
        ).view(1, 1, height, 1)
        upper_gate = (y <= 0.62).to(dtype)
        lower_gate = (y >= 0.38).to(dtype)
        leg_top_gate = (y <= 0.72).to(dtype)
        upper_torso = (torso * upper_gate + left_arm + right_arm).clamp(0, 1)
        lower_torso = (torso * lower_gate + (left_leg + right_leg) * leg_top_gate).clamp(0, 1)
        left_side = (left_arm + left_leg).clamp(0, 1)
        right_side = (right_arm + right_leg).clamp(0, 1)
        legs = (left_leg + right_leg).clamp(0, 1)
        anchored = torch.cat(
            (
                head,
                upper_torso,
                lower_torso,
                left_side,
                right_side,
                legs,
            ),
            dim=1,
        )
        residual = (foreground - anchored.amax(dim=1, keepdim=True)).clamp_min(0)
        return torch.cat(
            (anchored, accessory, residual),
            dim=1,
        ).clamp(0, 1)

    def _body_slot_privileged_loss(
        self,
        features,
        targets: dict[str, torch.Tensor] | None,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        epoch: int | None,
        return_components: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Distill recurrent RGB slots from masked EMA teacher pooling."""
        zero = torch.zeros((), device=pids.device, dtype=torch.float32)
        components = {
            "distill": zero,
            "attention": zero,
            "visibility": zero,
            "contrastive": zero,
            "descriptor_distill": zero,
            "branch_distill": zero,
            "branch_global": zero,
            "branch_coarse": zero,
            "branch_fine": zero,
            "pose_teacher": zero,
            "semantic_foreground": zero,
            "semantic_part": zero,
            "local_scale": zero,
            "fine_scale": zero,
            "cross_scale": zero,
            "valid_part_fraction": zero,
            "cross_camera_anchor_fraction": zero,
            "query_distill": zero,
            "query_diversity": zero,
            "part_triplet": zero,
            "accessory_valid_fraction": zero,
        }
        if not self.anatomical_auxiliary:
            return (zero, components) if return_components else zero
        if targets is None:
            raise RuntimeError("Body-slot supervision is enabled but the batch has no pose/mask targets")
        if not isinstance(features, dict):
            raise RuntimeError("Body-slot supervision requires dictionary model features")
        required = (
            "_body_slot_stage_slots",
            "_body_slot_stage_attentions",
            "_body_slot_stage_visibility_logits",
            "_body_slot_teacher_slots",
            "_body_slot_teacher_valid",
            "_body_slot_teacher_attentions",
        )
        missing = [key for key in required if key not in features]
        if missing:
            raise RuntimeError(f"Model did not return body-slot teacher outputs: {missing}")
        stage_slots = features["_body_slot_stage_slots"]
        stage_attentions = features["_body_slot_stage_attentions"]
        stage_visibility = features["_body_slot_stage_visibility_logits"]
        teacher_slots = features["_body_slot_teacher_slots"]
        teacher_valid = features["_body_slot_teacher_valid"]
        teacher_attentions = features["_body_slot_teacher_attentions"]
        stage_outputs = (
            stage_slots,
            stage_attentions,
            stage_visibility,
            teacher_slots,
            teacher_valid,
            teacher_attentions,
        )
        if any(not isinstance(values, tuple) or len(values) != 3 for values in stage_outputs):
            raise RuntimeError("Body-slot supervision requires Stage-0/2/3 outputs")

        loss_dtype = torch.float64 if stage_slots[-1].dtype == torch.float64 else torch.float32
        foreground = (
            targets["foreground_mask"]
            .to(
                device=pids.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )

        def weighted_mean(
            values: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        stage_losses = []
        stage_components = []
        for (
            student,
            attention,
            visibility_logits,
            teacher,
            valid,
            teacher_attention,
        ) in zip(*stage_outputs, strict=True):
            student = student.to(dtype=loss_dtype)
            attention = attention.to(dtype=loss_dtype).clamp_min(1e-8)
            teacher = teacher.detach().to(dtype=loss_dtype)
            valid = valid.to(device=pids.device, dtype=torch.bool)
            teacher_attention = teacher_attention.detach().to(dtype=loss_dtype)
            valid_weights = valid.to(loss_dtype)

            distill = weighted_mean(
                1.0
                - F.cosine_similarity(
                    F.normalize(student, p=2, dim=-1),
                    F.normalize(teacher, p=2, dim=-1),
                    dim=-1,
                ),
                valid_weights,
            )
            target_attention = teacher_attention / (teacher_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            attention_kl = (target_attention * (target_attention.clamp_min(1e-8).log() - attention.log())).sum(dim=-1)
            target_support = (teacher_attention > 0).to(loss_dtype)
            outside_mass = (attention * (1.0 - target_support)).sum(dim=-1)
            attention_loss = weighted_mean(
                0.5 * attention_kl + 0.5 * outside_mass,
                valid_weights,
            )
            visibility_loss = F.binary_cross_entropy_with_logits(
                visibility_logits.to(dtype=loss_dtype),
                valid_weights,
            )

            normalized_attention = F.normalize(
                attention,
                p=2,
                dim=-1,
            )
            similarity = normalized_attention @ (normalized_attention.transpose(1, 2))
            slot_count = attention.shape[1]
            upper = torch.triu(
                torch.ones(
                    slot_count,
                    slot_count,
                    device=attention.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            pair_weights = valid_weights[:, :, None] * valid_weights[:, None, :] * upper[None].to(loss_dtype)
            # Side and torso roles intentionally overlap. Downweight, but do
            # not remove, their diversity pressure so this regularizer cannot
            # directly fight the privileged attention target.
            teacher_overlap = F.normalize(target_attention, p=2, dim=-1) @ F.normalize(
                target_attention,
                p=2,
                dim=-1,
            ).transpose(1, 2)
            pair_weights = pair_weights * (1.0 - 0.75 * teacher_overlap).clamp(0.25, 1.0)
            diversity = weighted_mean(
                F.relu(similarity - self.anatomical_query_diversity_margin),
                pair_weights,
            )

            spatial_size = attention.shape[-1]
            stage_height = max(
                1,
                round(math.sqrt(spatial_size * foreground.shape[-2] / foreground.shape[-1])),
            )
            while spatial_size % stage_height:
                stage_height -= 1
            stage_width = spatial_size // stage_height
            foreground_stage = F.interpolate(
                foreground,
                size=(stage_height, stage_width),
                mode="area",
            )
            visible_attention = attention * valid_weights[..., None]
            visible_attention = visible_attention / (visible_attention.amax(dim=-1, keepdim=True).clamp_min(1e-6))
            coverage = F.l1_loss(
                visible_attention.amax(dim=1, keepdim=True).reshape_as(foreground_stage),
                foreground_stage,
            )
            stage_total = (
                self.anatomical_distill_weight * distill
                + self.anatomical_attention_weight * attention_loss
                + self.anatomical_visibility_weight * visibility_loss
                + self.anatomical_query_diversity_weight * diversity
                + self.anatomical_foreground_weight * coverage
            )
            stage_losses.append(stage_total)
            stage_components.append(
                (
                    distill,
                    attention_loss,
                    visibility_loss,
                    diversity,
                    coverage,
                )
            )

        final_slots = F.normalize(
            stage_slots[-1].to(dtype=loss_dtype),
            p=2,
            dim=-1,
        )
        final_valid = teacher_valid[-1].to(
            device=pids.device,
            dtype=torch.bool,
        )
        triplet_values = []
        for slot_index in range(final_slots.shape[1]):
            similarities = final_slots[:, slot_index] @ final_slots[:, slot_index].transpose(0, 1)
            valid = final_valid[:, slot_index]
            positive = (
                (pids[:, None] == pids[None, :])
                & (camera_ids[:, None] != camera_ids[None, :])
                & valid[:, None]
                & valid[None, :]
            )
            negative = (pids[:, None] != pids[None, :]) & valid[:, None] & valid[None, :]
            usable = positive.any(dim=1) & negative.any(dim=1)
            if not usable.any():
                continue
            hard_positive = (
                similarities.masked_fill(
                    ~positive,
                    float("inf"),
                )
                .min(dim=1)
                .values
            )
            hard_negative = (
                similarities.masked_fill(
                    ~negative,
                    float("-inf"),
                )
                .max(dim=1)
                .values
            )
            triplet_values.append(F.relu(self.margin + hard_negative - hard_positive)[usable])
        part_triplet = torch.cat(triplet_values).mean() if triplet_values else final_slots.sum() * 0.0
        cross_scale_values = []
        cross_scale_weights = []
        for left_index, right_index in ((0, 1), (1, 2)):
            values = 1.0 - F.cosine_similarity(
                stage_slots[left_index].to(dtype=loss_dtype),
                stage_slots[right_index].to(dtype=loss_dtype),
                dim=-1,
            )
            weights = teacher_valid[left_index].to(
                device=pids.device,
                dtype=loss_dtype,
            ) * teacher_valid[right_index].to(
                device=pids.device,
                dtype=loss_dtype,
            )
            cross_scale_values.append(values)
            cross_scale_weights.append(weights)
        cross_scale = weighted_mean(
            torch.cat(cross_scale_values, dim=1),
            torch.cat(cross_scale_weights, dim=1),
        )

        stage_weights = torch.tensor(
            (
                self.anatomical_branch_fine_coefficient,
                self.anatomical_branch_coarse_coefficient,
                self.anatomical_branch_global_coefficient,
            ),
            device=pids.device,
            dtype=loss_dtype,
        )
        stage_weights = stage_weights / stage_weights.sum().clamp_min(1e-6)
        stage_total = sum(
            weight * value
            for weight, value in zip(
                stage_weights,
                stage_losses,
                strict=True,
            )
        )
        student_scale, _ = self._anatomical_schedule_scales(epoch)
        total = student_scale * (
            stage_total
            + self.anatomical_part_triplet_weight * part_triplet
            + self.anatomical_cross_scale_weight * cross_scale
        )
        averaged = [
            sum(
                weight * values[index]
                for weight, values in zip(
                    stage_weights,
                    stage_components,
                    strict=True,
                )
            )
            for index in range(5)
        ]

        same_identity = pids[:, None] == pids[None, :]
        cross_camera = camera_ids[:, None] != camera_ids[None, :]
        positive_available = (
            same_identity[..., None] & cross_camera[..., None] & final_valid[:, None, :] & final_valid[None, :, :]
        ).any(dim=1)
        final_valid_count = final_valid.sum().clamp_min(1)
        cross_camera_fraction = (positive_available & final_valid).sum().to(loss_dtype) / final_valid_count
        components.update(
            {
                "distill": averaged[0],
                "attention": averaged[1],
                "visibility": averaged[2],
                "contrastive": part_triplet,
                "branch_global": stage_losses[2],
                "branch_coarse": stage_losses[1],
                "branch_fine": stage_losses[0],
                "pose_teacher": averaged[4],
                "semantic_foreground": averaged[4],
                "local_scale": stage_losses[1],
                "fine_scale": stage_losses[0],
                "cross_scale": cross_scale,
                "valid_part_fraction": torch.stack([valid.to(loss_dtype).mean() for valid in teacher_valid]).mean(),
                "cross_camera_anchor_fraction": (cross_camera_fraction),
                "query_diversity": averaged[3],
                "part_triplet": part_triplet,
                "accessory_valid_fraction": teacher_valid[-1][:, 6].to(loss_dtype).mean(),
            }
        )
        return (total, components) if return_components else total
