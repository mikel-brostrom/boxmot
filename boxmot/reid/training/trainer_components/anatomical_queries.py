"""Decoupled pose/parsing query supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _QueryAnatomicalMixin:
    def _decoupled_pose_parsing_query_loss(
        self,
        features: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        epoch: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train unrestricted RGB queries from masked parsing queries."""
        zero = torch.zeros((), device=pids.device, dtype=torch.float32)
        components = {
            "query_distill": zero,
            "query_relational_distill": zero,
            "query_diversity": zero,
            "part_triplet": zero,
            "query_foreground": zero,
            "query_visibility": zero,
            "accessory_valid_fraction": zero,
        }
        if targets is None:
            raise RuntimeError("Decoupled pose-parsing supervision requires batch targets")
        required = (
            "_anatomical_query_student_tokens",
            "_anatomical_query_teacher_tokens",
            "_anatomical_query_teacher_valid",
            "_anatomical_query_visibility_logits",
            "_anatomical_query_foreground_logits",
            "_anatomical_query_part_logits",
            "_anatomical_query_fine_student_tokens",
            "_anatomical_query_fine_teacher_tokens",
            "_anatomical_query_fine_teacher_valid",
            "_anatomical_query_fine_visibility_logits",
            "_anatomical_query_fine_foreground_logits",
            "_anatomical_query_fine_part_logits",
        )
        missing = [key for key in required if key not in features]
        if missing:
            raise RuntimeError(f"Model did not return decoupled query outputs: {missing}")

        local_student = features["_anatomical_query_student_tokens"]
        loss_dtype = torch.float64 if local_student.dtype == torch.float64 else torch.float32
        device = local_student.device
        masks = targets["masks"].to(device=device, dtype=loss_dtype)
        foreground = targets["foreground_mask"].to(
            device=device,
            dtype=loss_dtype,
        )
        masks = masks * foreground
        visibility = (
            targets["visibility"]
            .to(
                device=device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        reliability = (targets["reliability"].to(device=device, dtype=loss_dtype) * visibility).clamp(0, 1)
        mask_valid = targets.get(
            "mask_valid",
            targets["valid"],
        ).to(device=device, dtype=torch.bool)
        reliability = reliability * mask_valid[:, None].to(loss_dtype) * (masks.sum(dim=(-1, -2)) > 1e-6).to(loss_dtype)
        visibility_valid = mask_valid[:, None].expand_as(visibility)

        if self.anatomical_accessory_query:
            accessory_mask = targets["accessory_mask"].to(
                device=device,
                dtype=loss_dtype,
            )
            accessory_visibility = (
                targets["accessory_visibility"]
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
                .reshape(-1, 1)
                .clamp(0, 1)
            )
            accessory_reliability = (
                targets["accessory_reliability"]
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
                .reshape(-1, 1)
                .clamp(0, 1)
            )
            accessory_valid = (
                targets["accessory_valid"]
                .to(
                    device=device,
                    dtype=torch.bool,
                )
                .reshape(-1, 1)
            )
            masks = torch.cat((masks, accessory_mask), dim=1)
            visibility = torch.cat(
                (visibility, accessory_visibility),
                dim=1,
            )
            reliability = torch.cat(
                (
                    reliability,
                    accessory_reliability
                    * accessory_visibility
                    * accessory_valid.to(loss_dtype)
                    * (accessory_mask.sum(dim=(-1, -2)) > 1e-6).to(loss_dtype),
                ),
                dim=1,
            )
            visibility_valid = torch.cat(
                (visibility_valid, accessory_valid),
                dim=1,
            )
            components["accessory_valid_fraction"] = accessory_valid.to(loss_dtype).mean()

        expected_shape = local_student.shape[:2]
        if masks.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Query target count does not match model outputs: {tuple(masks.shape[:2])} != {tuple(expected_shape)}"
            )

        def weighted_mean(
            values: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        def scale_losses(
            student: torch.Tensor,
            teacher: torch.Tensor,
            teacher_valid: torch.Tensor,
            visibility_logits: torch.Tensor,
            foreground_logits: torch.Tensor,
            part_logits: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            student = student.to(dtype=loss_dtype)
            teacher = teacher.detach().to(dtype=loss_dtype)
            valid_weights = reliability * teacher_valid.to(device=device, dtype=loss_dtype)
            distill = weighted_mean(
                1.0 - F.cosine_similarity(student, teacher, dim=-1),
                valid_weights,
            )
            if (
                getattr(
                    self,
                    "anatomical_query_relational_distill_weight",
                    0.0,
                )
                > 0
            ):
                relational_distill = self._query_relational_distill_loss(
                    student,
                    teacher,
                    valid_weights,
                    pids,
                    camera_ids,
                )
            else:
                relational_distill = student.sum() * 0.0

            normalized = F.normalize(student, p=2, dim=-1)
            pair_similarity = torch.einsum(
                "bpd,bqd->bpq",
                normalized,
                normalized,
            )
            num_parts = student.shape[1]
            upper = torch.triu(
                torch.ones(
                    num_parts,
                    num_parts,
                    device=device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            pair_weights = (reliability[:, :, None] * reliability[:, None, :]).sqrt()
            semantic_coefficients = torch.ones(
                num_parts,
                num_parts,
                device=device,
                dtype=loss_dtype,
            )
            for left, right in ((2, 3), (4, 5)):
                if right < num_parts:
                    semantic_coefficients[left, right] = 0.5
                    semantic_coefficients[right, left] = 0.5
            pair_weights = pair_weights * semantic_coefficients[None] * upper[None].to(loss_dtype)
            diversity = weighted_mean(
                F.relu(pair_similarity - self.anatomical_query_diversity_margin),
                pair_weights,
            )

            triplet_values = []
            triplet_weights = []
            for part_index in range(num_parts):
                part_valid = reliability[:, part_index] > 0
                similarities = normalized[:, part_index] @ normalized[:, part_index].transpose(0, 1)
                positive_mask = (
                    (pids[:, None] == pids[None, :])
                    & (camera_ids[:, None] != camera_ids[None, :])
                    & part_valid[:, None]
                    & part_valid[None, :]
                )
                negative_mask = (pids[:, None] != pids[None, :]) & part_valid[:, None] & part_valid[None, :]
                usable = positive_mask.any(dim=1) & negative_mask.any(dim=1)
                if not usable.any():
                    continue
                hard_positive, positive_index = similarities.masked_fill(
                    ~positive_mask,
                    float("inf"),
                ).min(dim=1)
                hard_negative, negative_index = similarities.masked_fill(
                    ~negative_mask,
                    float("-inf"),
                ).max(dim=1)
                anchor_reliability = reliability[:, part_index]
                pair_reliability = (
                    anchor_reliability
                    * reliability[positive_index, part_index].clamp_min(0).sqrt()
                    * reliability[negative_index, part_index].clamp_min(0).sqrt()
                )
                triplet_values.append(F.relu(self.margin + hard_negative - hard_positive)[usable])
                triplet_weights.append(pair_reliability[usable])
            if triplet_values:
                triplet = weighted_mean(
                    torch.cat(triplet_values),
                    torch.cat(triplet_weights),
                )
            else:
                triplet = student.sum() * 0.0

            target_foreground = F.interpolate(
                foreground,
                size=foreground_logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            foreground_logits = foreground_logits.to(dtype=loss_dtype)
            foreground_bce = F.binary_cross_entropy_with_logits(
                foreground_logits,
                target_foreground,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            foreground_probability = foreground_logits.sigmoid()
            foreground_dice = 1.0 - (2.0 * (foreground_probability * target_foreground).sum(dim=(1, 2, 3)) + 1.0) / (
                foreground_probability.sum(dim=(1, 2, 3)) + target_foreground.sum(dim=(1, 2, 3)) + 1.0
            )
            foreground_loss = weighted_mean(
                0.5 * foreground_bce + 0.5 * foreground_dice,
                mask_valid.to(loss_dtype),
            )

            target_parts = F.interpolate(
                masks,
                size=part_logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            part_logits = part_logits.to(dtype=loss_dtype)
            part_bce = F.binary_cross_entropy_with_logits(
                part_logits,
                target_parts,
                reduction="none",
            ).mean(dim=(-1, -2))
            part_probability = part_logits.sigmoid()
            part_dice = 1.0 - (2.0 * (part_probability * target_parts).sum(dim=(-1, -2)) + 1.0) / (
                part_probability.sum(dim=(-1, -2)) + target_parts.sum(dim=(-1, -2)) + 1.0
            )
            parsing = 0.5 * foreground_loss + 0.5 * weighted_mean(
                0.5 * part_bce + 0.5 * part_dice,
                reliability,
            )

            visibility_values = F.binary_cross_entropy_with_logits(
                visibility_logits.to(dtype=loss_dtype),
                visibility,
                reduction="none",
            )
            visibility_loss = weighted_mean(
                visibility_values,
                visibility_valid.to(loss_dtype),
            )
            return (
                distill,
                relational_distill,
                diversity,
                triplet,
                parsing,
                visibility_loss,
            )

        local = scale_losses(
            local_student,
            features["_anatomical_query_teacher_tokens"],
            features["_anatomical_query_teacher_valid"],
            features["_anatomical_query_visibility_logits"],
            features["_anatomical_query_foreground_logits"],
            features["_anatomical_query_part_logits"],
        )
        fine = scale_losses(
            features["_anatomical_query_fine_student_tokens"],
            features["_anatomical_query_fine_teacher_tokens"],
            features["_anatomical_query_fine_teacher_valid"],
            features["_anatomical_query_fine_visibility_logits"],
            features["_anatomical_query_fine_foreground_logits"],
            features["_anatomical_query_fine_part_logits"],
        )
        scale_weights = (
            self.anatomical_local_scale_weight,
            self.anatomical_fine_scale_weight,
        )
        combined = tuple(
            scale_weights[0] * local[index] + scale_weights[1] * fine[index] for index in range(len(local))
        )
        (
            query_distill,
            query_relational_distill,
            query_diversity,
            part_triplet,
            parsing_loss,
            query_visibility,
        ) = combined

        _, decay_scale = self._anatomical_schedule_scales(epoch)
        if epoch is None:
            query_scale = decay_scale
        elif epoch <= self.anatomical_query_start_epoch:
            query_scale = 0.0
        elif (
            self.anatomical_query_ramp_end_epoch <= self.anatomical_query_start_epoch
            or epoch >= self.anatomical_query_ramp_end_epoch
        ):
            query_scale = decay_scale
        else:
            query_scale = decay_scale * (
                (epoch - self.anatomical_query_start_epoch)
                / (self.anatomical_query_ramp_end_epoch - self.anatomical_query_start_epoch)
            )
        total = decay_scale * (
            self.anatomical_foreground_weight * parsing_loss + self.anatomical_visibility_weight * query_visibility
        ) + query_scale * (
            self.anatomical_query_distill_weight * query_distill
            + getattr(
                self,
                "anatomical_query_relational_distill_weight",
                0.0,
            )
            * query_relational_distill
            + self.anatomical_query_diversity_weight * query_diversity
            + self.anatomical_part_triplet_weight * part_triplet
        )
        components.update(
            {
                "query_distill": query_distill,
                "query_relational_distill": query_relational_distill,
                "query_diversity": query_diversity,
                "part_triplet": part_triplet,
                "query_foreground": parsing_loss,
                "query_visibility": query_visibility,
            }
        )
        return total, components
