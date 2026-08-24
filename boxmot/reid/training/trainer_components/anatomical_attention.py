"""Privileged mask-and-pose attention supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from boxmot.reid.training.trainer_components.helpers import (
    _cross_scale_role_relation_loss,
)


class _PrivilegedAttentionMixin:
    def _privileged_mask_pose_attention_loss(
        self,
        features,
        targets: dict[str, torch.Tensor] | None,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        epoch: int | None,
        return_components: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Teach RGB attention where to look using pose and person masks."""
        zero = torch.zeros((), device=pids.device, dtype=torch.float32)
        zero_components = {
            "distill": zero,
            "attention": zero,
            "visibility": zero,
            "contrastive": zero,
            "descriptor_distill": zero,
            "branch_distill": zero,
            "branch_global": zero,
            "branch_coarse": zero,
            "branch_fine": zero,
            # Keep the existing metric slot backward-compatible while this
            # target type uses it to report foreground supervision.
            "pose_teacher": zero,
            "local_scale": zero,
            "fine_scale": zero,
            "cross_scale": zero,
            "valid_part_fraction": zero,
            "cross_camera_anchor_fraction": zero,
        }
        if not self.anatomical_auxiliary:
            return (zero, zero_components) if return_components else zero
        student_scale, decay_scale = self._anatomical_schedule_scales(epoch)
        fine_scale = self._anatomical_fine_schedule_scale(
            epoch,
            student_scale=student_scale,
            decay_scale=decay_scale,
        )
        if student_scale <= 0 and fine_scale <= 0:
            return (zero, zero_components) if return_components else zero
        if targets is None or not isinstance(features, dict):
            raise RuntimeError(
                "Privileged mask-pose attention requires model features and transformed anatomical targets"
            )
        required = (
            "_anatomical_feature_map",
            "_anatomical_student_tokens",
            "_anatomical_attention",
            "_anatomical_visibility_logits",
            "_anatomical_foreground_logits",
            "_anatomical_fine_feature_map",
            "_anatomical_fine_student_tokens",
            "_anatomical_fine_attention",
            "_anatomical_fine_visibility_logits",
            "_anatomical_fine_foreground_logits",
        )
        missing = [key for key in required if key not in features]
        if missing:
            raise RuntimeError(f"Model did not return privileged attention outputs: {missing}")

        loss_dtype = torch.float32
        masks = targets["masks"].to(
            device=pids.device,
            dtype=loss_dtype,
        )
        foreground = targets["foreground_mask"].to(
            device=pids.device,
            dtype=loss_dtype,
        )
        mask_valid = targets.get(
            "mask_valid",
            targets["valid"],
        ).to(device=pids.device, dtype=torch.bool)
        pose_valid = targets.get(
            "pose_valid",
            targets["valid"],
        ).to(device=pids.device, dtype=torch.bool)
        reliability = (
            targets["reliability"]
            .to(
                device=pids.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        visibility_target = (
            targets["visibility"]
            .to(
                device=pids.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        # Mask-backed visibility tracks real occlusion and every spatial
        # augmentation. Pose-only records have no segmentation visibility, so
        # their reliability is already reduced by pose_only_reliability.
        spatial_reliability = reliability * torch.where(
            mask_valid[:, None],
            visibility_target,
            torch.ones_like(visibility_target),
        )

        def foreground_loss(logits: torch.Tensor) -> torch.Tensor:
            target = F.interpolate(
                foreground,
                size=logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            logits = logits.float()
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            weights = mask_valid.to(loss_dtype)
            return ((0.5 * bce + 0.5 * dice) * weights).sum() / weights.sum().clamp_min(1.0)

        def part_attention_loss(attention: torch.Tensor) -> torch.Tensor:
            target = F.interpolate(
                masks,
                size=attention.shape[-2:],
                mode="area",
            ).clamp_min(0)
            mass = target.sum(dim=(-1, -2), keepdim=True)
            target = target / mass.clamp_min(1e-8)
            values = (target * (target.clamp_min(1e-8).log() - attention.float().clamp_min(1e-8).log())).sum(
                dim=(-1, -2)
            )
            valid = pose_valid[:, None] & (mass.squeeze(-1).squeeze(-1) > 1e-6) & (spatial_reliability > 0)
            weights = spatial_reliability * valid.to(loss_dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        def visibility_loss(logits: torch.Tensor) -> torch.Tensor:
            values = F.binary_cross_entropy_with_logits(
                logits.float(),
                visibility_target,
                reduction="none",
            )
            valid = pose_valid[:, None] & (reliability > 0)
            weights = reliability * valid.to(loss_dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        def token_distill_loss(
            tokens: torch.Tensor,
            feature_map: torch.Tensor,
        ) -> torch.Tensor:
            """Match RGB tokens to stop-gradient pose/mask region averages."""
            target = F.interpolate(
                masks,
                size=feature_map.shape[-2:],
                mode="area",
            ).clamp_min(0)
            mass = target.sum(dim=(-1, -2), keepdim=True)
            routing = target / mass.clamp_min(1e-8)
            teacher_tokens = torch.einsum(
                "bphw,bchw->bpc",
                routing,
                feature_map.detach().to(dtype=loss_dtype),
            )
            values = 1.0 - F.cosine_similarity(
                tokens.to(dtype=loss_dtype),
                teacher_tokens,
                dim=-1,
            )
            valid = pose_valid[:, None] & (mass.squeeze(-1).squeeze(-1) > 1e-6) & (spatial_reliability > 0)
            weights = spatial_reliability * valid.to(loss_dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        local_tokens = features["_anatomical_student_tokens"]
        fine_tokens = features["_anatomical_fine_student_tokens"]
        local_distill = token_distill_loss(
            local_tokens,
            features["_anatomical_feature_map"],
        )
        fine_distill = token_distill_loss(
            fine_tokens,
            features["_anatomical_fine_feature_map"],
        )
        local_foreground = foreground_loss(features["_anatomical_foreground_logits"])
        fine_foreground = foreground_loss(features["_anatomical_fine_foreground_logits"])
        local_attention = part_attention_loss(features["_anatomical_attention"])
        fine_attention = part_attention_loss(features["_anatomical_fine_attention"])
        local_visibility = visibility_loss(features["_anatomical_visibility_logits"])
        fine_visibility = visibility_loss(features["_anatomical_fine_visibility_logits"])
        local_contrastive = self._cross_camera_part_contrastive_loss(
            local_tokens,
            pids,
            camera_ids,
            spatial_reliability,
            self.anatomical_temperature,
        )
        fine_contrastive = self._cross_camera_part_contrastive_loss(
            fine_tokens,
            pids,
            camera_ids,
            spatial_reliability,
            self.anatomical_temperature,
        )
        cross_scale = _cross_scale_role_relation_loss(
            local_tokens,
            fine_tokens,
            spatial_reliability,
        )

        local_total = (
            self.anatomical_distill_weight * local_distill
            + self.anatomical_foreground_weight * local_foreground
            + self.anatomical_attention_weight * local_attention
            + self.anatomical_visibility_weight * local_visibility
            + self.anatomical_contrastive_weight * local_contrastive
        )
        fine_total = (
            self.anatomical_distill_weight * fine_distill
            + self.anatomical_foreground_weight * fine_foreground
            + self.anatomical_attention_weight * fine_attention
            + self.anatomical_visibility_weight * fine_visibility
            + self.anatomical_contrastive_weight * fine_contrastive
        )
        local_weight = self.anatomical_local_scale_weight
        fine_weight = self.anatomical_fine_scale_weight
        total = student_scale * local_weight * local_total + fine_scale * (
            fine_weight * fine_total + self.anatomical_cross_scale_weight * cross_scale
        )
        part_valid = pose_valid[:, None] & (spatial_reliability > 0)
        same_identity_cross_camera = (pids[:, None] == pids[None, :]) & (camera_ids[:, None] != camera_ids[None, :])
        positive_availability = (
            same_identity_cross_camera[:, :, None] & part_valid[:, None, :] & part_valid[None, :, :]
        ).any(dim=1)
        cross_camera_anchor_fraction = (
            positive_availability.to(loss_dtype) * part_valid.to(loss_dtype)
        ).sum() / part_valid.sum().clamp_min(1)
        components = {
            **zero_components,
            "distill": (local_weight * local_distill + fine_weight * fine_distill),
            "attention": (local_weight * local_attention + fine_weight * fine_attention),
            "visibility": (local_weight * local_visibility + fine_weight * fine_visibility),
            "contrastive": (local_weight * local_contrastive + fine_weight * fine_contrastive),
            "pose_teacher": (local_weight * local_foreground + fine_weight * fine_foreground),
            "local_scale": local_total,
            "fine_scale": fine_total,
            "cross_scale": cross_scale,
            "valid_part_fraction": part_valid.to(loss_dtype).mean(),
            "cross_camera_anchor_fraction": cross_camera_anchor_fraction,
        }
        return (total, components) if return_components else total
