"""EMA-teacher anatomical supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from boxmot.reid.backbones.anatomical_registry import (
    SEMANTIC_ANATOMICAL_TARGET_TYPES,
)
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_CANONICAL_CELLS,
    ANATOMICAL_CANONICAL_GRID_SIZE,
)
from boxmot.reid.training.trainer_components.helpers import (
    _bilinear_sample_2d,
)


class _EmaAnatomicalMixin:
    def _ema_anatomical_auxiliary_loss(
        self,
        features,
        targets: dict[str, torch.Tensor] | None,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        epoch: int | None,
        return_components: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the A11v8 pose-concatenation EMA teacher objective."""
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
            "pose_teacher": zero,
            "semantic_foreground": zero,
            "semantic_part": zero,
            "local_scale": zero,
            "fine_scale": zero,
            "cross_scale": zero,
            "valid_part_fraction": zero,
            "cross_camera_anchor_fraction": zero,
        }
        if not self.anatomical_auxiliary:
            return (zero, zero_components) if return_components else zero
        student_scale, teacher_scale = self._anatomical_schedule_scales(epoch)
        fine_student_scale = self._anatomical_fine_schedule_scale(
            epoch,
            student_scale=student_scale,
            decay_scale=teacher_scale,
        )
        if student_scale <= 0 and fine_student_scale <= 0 and teacher_scale <= 0:
            return (zero, zero_components) if return_components else zero
        if targets is None:
            raise RuntimeError("Anatomical supervision is enabled but the batch has no targets")
        if not isinstance(features, dict):
            raise RuntimeError("Anatomical supervision requires dictionary model features")
        required_keys = (
            "_anatomical_feature_map",
            "_anatomical_teacher_feature_map",
            "_anatomical_online_teacher_feature_map",
            "_anatomical_student_tokens",
            "_anatomical_attention",
            "_anatomical_visibility_logits",
        )
        multiscale_enabled = bool(self.anatomical_multiscale)
        if multiscale_enabled:
            required_keys += (
                "_anatomical_fine_feature_map",
                "_anatomical_fine_student_tokens",
                "_anatomical_fine_attention",
                "_anatomical_fine_visibility_logits",
            )
        semantic_teacher_enabled = self.anatomical_target_type in SEMANTIC_ANATOMICAL_TARGET_TYPES
        semantic_target_fusion = self.anatomical_target_type == "learned_pose_semantic_fused_ema"
        semantic_foreground_weight = self.anatomical_foreground_weight if semantic_teacher_enabled else 0.0
        semantic_part_weight = self.anatomical_semantic_part_weight if semantic_teacher_enabled else 0.0
        if semantic_teacher_enabled:
            required_keys += (
                "_anatomical_semantic_foreground_logits",
                "_anatomical_semantic_part_logits",
                "_anatomical_semantic_fine_foreground_logits",
                "_anatomical_semantic_fine_part_logits",
            )
        if getattr(self, "anatomical_branch_distill_weight", 0.0) > 0:
            required_keys += ("_anatomical_branch_features",)
        missing = [key for key in required_keys if key not in features]
        if missing:
            raise RuntimeError(f"Model did not return EMA anatomical outputs: {missing}")

        feature_map = features["_anatomical_feature_map"]
        teacher_feature_map = features["_anatomical_teacher_feature_map"]
        online_teacher_feature_map = features["_anatomical_online_teacher_feature_map"]
        student_tokens = features["_anatomical_student_tokens"]
        student_attention = features["_anatomical_attention"]
        visibility_logits = features["_anatomical_visibility_logits"]
        fine_feature_map = features.get("_anatomical_fine_feature_map")
        fine_student_tokens = features.get("_anatomical_fine_student_tokens")
        fine_student_attention = features.get("_anatomical_fine_attention")
        fine_visibility_logits = features.get("_anatomical_fine_visibility_logits")
        loss_dtype = torch.float64 if feature_map.dtype == torch.float64 else torch.float32
        semantic_masks = None
        semantic_foreground = None
        if semantic_teacher_enabled:
            if "masks" not in targets:
                raise RuntimeError(
                    "Semantic anatomy supervision requires dense part masks"
                )
            semantic_masks = targets["masks"].to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
            semantic_foreground = targets.get(
                "foreground_mask",
                semantic_masks.amax(dim=1, keepdim=True),
            ).to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
        mask_present = targets.get("mask_present")
        if mask_present is None:
            dense_masks = targets.get("masks")
            if dense_masks is None:
                raise RuntimeError(
                    "Non-semantic anatomy supervision requires mask_present"
                )
            mask_present = dense_masks.flatten(2).amax(dim=-1) > 1e-6
        mask_present = mask_present.to(
            device=feature_map.device,
            dtype=torch.bool,
        )
        visibility = (
            targets["visibility"]
            .to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        mask_reliability = (
            targets["reliability"]
            .to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        if mask_present.shape != visibility.shape:
            raise ValueError(
                "mask_present must match anatomical visibility shape, "
                f"got {tuple(mask_present.shape)} and {tuple(visibility.shape)}"
            )
        mask_metadata_valid = targets.get(
            "mask_valid",
            targets["valid"],
        ).to(
            device=feature_map.device,
            dtype=torch.bool,
        )
        pose_metadata_valid = targets.get(
            "pose_valid",
            targets["valid"],
        ).to(
            device=feature_map.device,
            dtype=torch.bool,
        )
        pose_reliability = (
            targets.get(
                "pose_reliability",
                targets["reliability"],
            )
            .to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        pose_mask_agreement = (
            targets.get(
                "pose_mask_agreement",
                torch.ones_like(mask_metadata_valid, dtype=loss_dtype),
            )
            .to(
                device=feature_map.device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        agreement_weight = 0.25 + 0.75 * pose_mask_agreement
        canonical_grid = targets["canonical_grid"].to(
            device=teacher_feature_map.device,
            dtype=loss_dtype,
        )
        canonical_grid_valid = targets["canonical_grid_valid"].to(
            device=teacher_feature_map.device,
            dtype=torch.bool,
        )
        canonical_grid_pose_valid = targets.get(
            "canonical_grid_pose_valid",
            targets["canonical_grid_valid"],
        ).to(
            device=teacher_feature_map.device,
            dtype=torch.bool,
        )
        expected_grid_shape = (*ANATOMICAL_CANONICAL_GRID_SIZE, 2)
        if canonical_grid.ndim != 5 or tuple(canonical_grid.shape[2:]) != expected_grid_shape:
            raise ValueError("canonical anatomical grids must have shape [B,P,4,2,2]")
        if canonical_grid_valid.shape != canonical_grid.shape[:-1]:
            raise ValueError("canonical grid validity must match canonical grid cells")
        if canonical_grid_pose_valid.shape != canonical_grid.shape[:-1]:
            raise ValueError("pose-grid validity must match canonical grid cells")
        batch_size, num_parts = canonical_grid.shape[:2]
        canonical_grid = canonical_grid.flatten(2, 3)
        canonical_grid_valid = canonical_grid_valid.flatten(2, 3)
        canonical_grid_pose_valid = canonical_grid_pose_valid.flatten(2, 3)
        if student_attention.shape[:3] != (
            batch_size,
            num_parts,
            ANATOMICAL_CANONICAL_CELLS,
        ):
            raise RuntimeError(
                f"Anatomical student attention must have shape [B,P,8,H,W], got {tuple(student_attention.shape)}"
            )
        if multiscale_enabled:
            if fine_student_tokens.shape != student_tokens.shape:
                raise RuntimeError(
                    "Fine/local anatomical tokens must have matching shapes, "
                    f"got {tuple(fine_student_tokens.shape)} and "
                    f"{tuple(student_tokens.shape)}"
                )
            if fine_student_attention.shape[:3] != (
                batch_size,
                num_parts,
                ANATOMICAL_CANONICAL_CELLS,
            ):
                raise RuntimeError(
                    f"Fine anatomical attention must have shape [B,P,8,H,W], got {tuple(fine_student_attention.shape)}"
                )

        finite_grid = torch.isfinite(canonical_grid).all(dim=-1)
        canonical_grid_valid = canonical_grid_valid & finite_grid
        canonical_grid_pose_valid = canonical_grid_pose_valid & finite_grid
        mask_part_valid = (
            mask_metadata_valid[:, None]
            & (visibility > 0)
            & (mask_reliability > 0)
            & mask_present
            & canonical_grid_valid.any(dim=-1)
        )
        pose_part_valid = pose_metadata_valid[:, None] & (pose_reliability > 0) & canonical_grid_pose_valid.any(dim=-1)
        if self.anatomical_pose_teacher_weight > 0:
            use_mask = mask_metadata_valid[:, None]
            supervision_grid_valid = torch.where(
                use_mask[:, :, None],
                canonical_grid_valid,
                canonical_grid_pose_valid,
            )
            part_valid = torch.where(
                use_mask,
                mask_part_valid,
                pose_part_valid,
            )
            mask_confidence = visibility * mask_reliability * agreement_weight[:, None]
            pose_only_confidence = pose_reliability * self.anatomical_pose_only_reliability
            part_confidence = torch.where(
                use_mask,
                mask_confidence,
                pose_only_confidence,
            ) * part_valid.to(visibility.dtype)
        else:
            supervision_grid_valid = canonical_grid_valid
            part_valid = mask_part_valid
            part_confidence = (
                visibility * mask_reliability * agreement_weight[:, None] * part_valid.to(visibility.dtype)
            )
        valid_cell_fraction = supervision_grid_valid.to(visibility.dtype).mean(dim=-1)
        reliability = part_confidence * valid_cell_fraction
        cell_weights = part_confidence[:, :, None] * supervision_grid_valid.to(part_confidence.dtype)

        def sample_cells(
            source: torch.Tensor,
            *,
            differentiable: bool,
        ) -> list[torch.Tensor]:
            source = source.to(dtype=loss_dtype)
            source_height, source_width = source.shape[-2:]
            expanded = (
                source[:, None]
                .expand(
                    -1,
                    num_parts,
                    -1,
                    -1,
                    -1,
                )
                .reshape(
                    batch_size * num_parts,
                    source.shape[1],
                    source_height,
                    source_width,
                )
            )
            grid = canonical_grid.reshape(
                batch_size * num_parts,
                *ANATOMICAL_CANONICAL_GRID_SIZE,
                2,
            )
            sampled = (
                _bilinear_sample_2d(expanded, grid)
                if differentiable
                else F.grid_sample(
                    expanded.detach(),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            ).reshape(
                batch_size,
                num_parts,
                source.shape[1],
                ANATOMICAL_CANONICAL_CELLS,
            )
            chunks = torch.tensor_split(
                sampled,
                ANATOMICAL_CANONICAL_CELLS,
                dim=2,
            )
            return [chunk[..., cell_index] for cell_index, chunk in enumerate(chunks)]

        teacher_cell_tokens = sample_cells(
            teacher_feature_map,
            differentiable=False,
        )
        online_teacher_cell_tokens = sample_cells(
            online_teacher_feature_map,
            differentiable=True,
        )

        branch_distill_loss = zero
        branch_global_loss = zero
        branch_coarse_loss = zero
        branch_fine_loss = zero
        if getattr(self, "anatomical_branch_distill_weight", 0.0) > 0:
            branch_features = features["_anatomical_branch_features"]
            if not isinstance(branch_features, (tuple, list)) or len(branch_features) != 3:
                raise RuntimeError("Anatomical branch features must contain global, coarse, and fine levels")
            global_feature, coarse_features, fine_features = branch_features
            branch_levels = (
                (global_feature,),
                tuple(coarse_features),
                tuple(fine_features),
            )
            expected_counts = (1, 2, 4)
            for granularity, level, expected_count in zip(
                expected_counts,
                branch_levels,
                expected_counts,
                strict=True,
            ):
                if len(level) != expected_count or any(
                    not torch.is_tensor(branch_feature)
                    or branch_feature.ndim != 2
                    or branch_feature.shape[0] != batch_size
                    for branch_feature in level
                ):
                    raise RuntimeError(
                        "Anatomical branch level "
                        f"{granularity} must contain {expected_count} "
                        "descriptors with shape [B,D]"
                    )

            valid_cells = supervision_grid_valid.to(loss_dtype)
            grid_y = torch.nan_to_num(
                canonical_grid[..., 1],
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

            def teacher_targets_for_level(
                granularity: int,
            ) -> tuple[
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
            ]:
                if granularity == 1:
                    assignments = torch.ones(
                        *grid_y.shape,
                        1,
                        device=grid_y.device,
                        dtype=loss_dtype,
                    )
                else:
                    stripe_centers = (
                        -1.0
                        + (
                            2.0
                            * torch.arange(
                                granularity,
                                device=grid_y.device,
                                dtype=loss_dtype,
                            )
                            + 1.0
                        )
                        / granularity
                    )
                    distances = (grid_y[..., None] - stripe_centers).abs()
                    assignments = torch.softmax(
                        -distances / (1.0 / granularity),
                        dim=-1,
                    )
                assignments = assignments * valid_cells[..., None]
                teacher_targets = []
                branch_reliabilities = []
                for stripe_index in range(granularity):
                    assignment = assignments[..., stripe_index]
                    effective_weights = cell_weights * assignment
                    weighted_cells = torch.cat(
                        [
                            teacher_token.detach()
                            * effective_weights[
                                :,
                                :,
                                cell_index,
                                None,
                            ]
                            .clamp_min(0)
                            .sqrt()
                            for cell_index, teacher_token in enumerate(teacher_cell_tokens)
                        ],
                        dim=-1,
                    )
                    teacher_targets.append(
                        F.normalize(
                            weighted_cells.flatten(1),
                            p=2,
                            dim=1,
                        )
                    )
                    assignment_mass = assignment.sum(dim=(1, 2))
                    branch_reliabilities.append(effective_weights.sum(dim=(1, 2)) / assignment_mass.clamp_min(1e-6))
                return (
                    tuple(teacher_targets),
                    tuple(branch_reliabilities),
                )

            level_losses = []
            for granularity, student_level in zip(
                expected_counts,
                branch_levels,
                strict=True,
            ):
                teacher_level, reliability_level = teacher_targets_for_level(granularity)
                losses = [
                    self._cross_camera_relational_distill_loss(
                        student_branch,
                        teacher_branch,
                        branch_reliability,
                        pids,
                        camera_ids,
                    )
                    for (
                        student_branch,
                        teacher_branch,
                        branch_reliability,
                    ) in zip(
                        student_level,
                        teacher_level,
                        reliability_level,
                        strict=True,
                    )
                ]
                level_losses.append(torch.stack(losses).mean())
            (
                branch_global_loss,
                branch_coarse_loss,
                branch_fine_loss,
            ) = level_losses
            branch_distill_loss = (
                self.anatomical_branch_global_coefficient * branch_global_loss
                + self.anatomical_branch_coarse_coefficient * branch_coarse_loss
                + self.anatomical_branch_fine_coefficient * branch_fine_loss
            )

        def token_distill_loss(tokens: torch.Tensor) -> torch.Tensor:
            student_chunks = torch.tensor_split(
                tokens.to(dtype=loss_dtype),
                ANATOMICAL_CANONICAL_CELLS,
                dim=-1,
            )
            values = torch.stack(
                [
                    1.0
                    - F.cosine_similarity(
                        student_chunk,
                        teacher_token,
                        dim=-1,
                    )
                    for student_chunk, teacher_token in zip(
                        student_chunks,
                        teacher_cell_tokens,
                        strict=True,
                    )
                ],
                dim=-1,
            )
            return (values * cell_weights).sum() / cell_weights.sum().clamp_min(1e-6)

        local_distill_loss = token_distill_loss(student_tokens)
        fine_distill_loss = token_distill_loss(fine_student_tokens) if multiscale_enabled else zero
        online_consistency_values = torch.stack(
            [
                1.0
                - F.cosine_similarity(
                    online_token,
                    ema_token.detach(),
                    dim=-1,
                )
                for online_token, ema_token in zip(
                    online_teacher_cell_tokens,
                    teacher_cell_tokens,
                    strict=True,
                )
            ],
            dim=-1,
        )
        online_consistency_loss = (online_consistency_values * cell_weights).sum() / cell_weights.sum().clamp_min(1e-6)
        online_part_tokens = torch.cat(
            [
                token
                * supervision_grid_valid[
                    :,
                    :,
                    cell_index,
                    None,
                ].to(token.dtype)
                for cell_index, token in enumerate(online_teacher_cell_tokens)
            ],
            dim=-1,
        )
        online_identity_loss = self._cross_camera_part_contrastive_loss(
            online_part_tokens,
            pids,
            camera_ids,
            reliability,
            self.anatomical_temperature,
        )
        pose_teacher_loss = online_identity_loss + 0.25 * online_consistency_loss
        sample_weights = (mask_metadata_valid.to(visibility.dtype) * agreement_weight.to(visibility.dtype))[:, None]

        def spatial_attention_loss(
            attention: torch.Tensor,
            source: torch.Tensor,
        ) -> torch.Tensor:
            attention = attention.to(dtype=loss_dtype)
            attention_height, attention_width = attention.shape[-2:]
            grid_x = (canonical_grid[..., 0] + 1.0) * attention_width * 0.5 - 0.5
            grid_y = (canonical_grid[..., 1] + 1.0) * attention_height * 0.5 - 0.5
            spatial_x = torch.arange(
                attention_width,
                device=source.device,
                dtype=loss_dtype,
            )
            spatial_y = torch.arange(
                attention_height,
                device=source.device,
                dtype=loss_dtype,
            )
            horizontal_weights = (
                1.0 - (spatial_x[None, None, None, None, :] - grid_x[..., None, None]).abs()
            ).clamp_min(0)
            vertical_weights = (1.0 - (spatial_y[None, None, None, :, None] - grid_y[..., None, None]).abs()).clamp_min(
                0
            )
            target_attention = vertical_weights * horizontal_weights
            target_attention = target_attention / target_attention.sum(
                dim=(-1, -2),
                keepdim=True,
            ).clamp_min(1e-8)
            if semantic_target_fusion:
                semantic_target = F.interpolate(
                    semantic_masks,
                    size=(attention_height, attention_width),
                    mode="area",
                ).clamp_min(0)
                person_target = F.interpolate(
                    semantic_foreground,
                    size=(attention_height, attention_width),
                    mode="area",
                ).clamp(0, 1)
                semantic_target = semantic_target * person_target
                semantic_mass = semantic_target.sum(
                    dim=(-1, -2),
                    keepdim=True,
                )
                semantic_target = (semantic_target / semantic_mass.clamp_min(1e-8))[:, :, None]
                pose_confidence = pose_reliability * pose_metadata_valid[:, None].to(loss_dtype)
                parsing_confidence = (
                    visibility * agreement_weight[:, None] * mask_metadata_valid[:, None].to(loss_dtype)
                )
                pose_fraction = pose_confidence / (pose_confidence + parsing_confidence).clamp_min(1e-6)
                pose_fraction = torch.where(
                    semantic_mass.squeeze(-1).squeeze(-1) > 1e-6,
                    pose_fraction,
                    torch.ones_like(pose_fraction),
                )[:, :, None, None, None]
                target_attention = pose_fraction * target_attention + (1.0 - pose_fraction) * semantic_target
                target_attention = target_attention / target_attention.sum(
                    dim=(-1, -2),
                    keepdim=True,
                ).clamp_min(1e-8)
            values = (
                target_attention * (target_attention.clamp_min(1e-8).log() - attention.clamp_min(1e-8).log())
            ).sum(dim=(-1, -2))
            return (values * cell_weights).sum() / cell_weights.sum().clamp_min(1e-6)

        def part_visibility_loss(logits: torch.Tensor) -> torch.Tensor:
            values = F.binary_cross_entropy_with_logits(
                logits.to(dtype=loss_dtype),
                visibility,
                reduction="none",
            )
            return (values * sample_weights).sum() / (sample_weights.sum() * values.shape[1]).clamp_min(1.0)

        def semantic_foreground_loss(
            logits: torch.Tensor,
        ) -> torch.Tensor:
            target = F.interpolate(
                semantic_foreground,
                size=logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            logits = logits.to(dtype=loss_dtype)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            weights = mask_metadata_valid.to(loss_dtype)
            return ((0.5 * bce + 0.5 * dice) * weights).sum() / weights.sum().clamp_min(1.0)

        def semantic_part_loss(logits: torch.Tensor) -> torch.Tensor:
            target = F.interpolate(
                semantic_masks,
                size=logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            target = target * F.interpolate(
                semantic_foreground,
                size=logits.shape[-2:],
                mode="area",
            ).clamp(0, 1)
            logits = logits.to(dtype=loss_dtype)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            ).mean(dim=(-1, -2))
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(-1, -2))
            denominator = probability.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            valid = mask_metadata_valid[:, None] & (target.sum(dim=(-1, -2)) > 1e-6)
            weights = visibility * agreement_weight[:, None] * valid.to(loss_dtype)
            return ((0.5 * bce + 0.5 * dice) * weights).sum() / weights.sum().clamp_min(1e-6)

        local_attention_loss = spatial_attention_loss(
            student_attention,
            feature_map,
        )
        local_visibility_loss = part_visibility_loss(visibility_logits)
        local_contrastive_loss = self._cross_camera_part_contrastive_loss(
            student_tokens,
            pids,
            camera_ids,
            reliability,
            self.anatomical_temperature,
        )
        local_semantic_foreground_loss = zero
        local_semantic_part_loss = zero
        fine_semantic_foreground_loss = zero
        fine_semantic_part_loss = zero
        if semantic_teacher_enabled:
            local_semantic_foreground_loss = semantic_foreground_loss(
                features["_anatomical_semantic_foreground_logits"]
            )
            local_semantic_part_loss = semantic_part_loss(features["_anatomical_semantic_part_logits"])
            fine_semantic_foreground_loss = semantic_foreground_loss(
                features["_anatomical_semantic_fine_foreground_logits"]
            )
            fine_semantic_part_loss = semantic_part_loss(features["_anatomical_semantic_fine_part_logits"])
        if multiscale_enabled:
            fine_attention_loss = spatial_attention_loss(
                fine_student_attention,
                fine_feature_map,
            )
            fine_visibility_loss = part_visibility_loss(fine_visibility_logits)
            fine_contrastive_loss = self._cross_camera_part_contrastive_loss(
                fine_student_tokens,
                pids,
                camera_ids,
                reliability,
                self.anatomical_temperature,
            )
            cross_scale_values = 1.0 - F.cosine_similarity(
                student_tokens.to(dtype=loss_dtype),
                fine_student_tokens.to(dtype=loss_dtype),
                dim=-1,
            )
            cross_scale_loss = (cross_scale_values * reliability).sum() / reliability.sum().clamp_min(1e-6)
        else:
            fine_attention_loss = zero
            fine_visibility_loss = zero
            fine_contrastive_loss = zero
            cross_scale_loss = zero

        descriptor_distill_loss = student_tokens.to(dtype=loss_dtype).sum() * 0.0
        if self.anatomical_descriptor_distill_weight > 0:
            projected_final = features.get("_anatomical_final_student")
            final_descriptor = features.get("norm_concat_bn")
            if not torch.is_tensor(projected_final) or not torch.is_tensor(final_descriptor):
                raise RuntimeError(
                    "Anatomical descriptor distillation requires projected and deployed final descriptors"
                )
            teacher_part_tokens = torch.cat(
                [
                    token
                    * supervision_grid_valid[
                        :,
                        :,
                        cell_index,
                        None,
                    ].to(token.dtype)
                    for cell_index, token in enumerate(teacher_cell_tokens)
                ],
                dim=-1,
            )
            teacher_part_tokens = F.normalize(
                teacher_part_tokens,
                p=2,
                dim=-1,
            )
            teacher_descriptor = F.normalize(
                (teacher_part_tokens * reliability.clamp_min(0).sqrt()[:, :, None]).flatten(1),
                p=2,
                dim=1,
            ).detach()
            projected_final = F.normalize(
                projected_final.to(dtype=loss_dtype),
                p=2,
                dim=1,
            )
            final_descriptor = F.normalize(
                final_descriptor.to(dtype=loss_dtype),
                p=2,
                dim=1,
            )
            descriptor_valid = part_valid.any(dim=-1)
            descriptor_weights = descriptor_valid.to(projected_final.dtype)
            alignment_values = 1.0 - F.cosine_similarity(
                projected_final,
                teacher_descriptor,
                dim=1,
            )
            alignment_loss = (alignment_values * descriptor_weights).sum() / descriptor_weights.sum().clamp_min(1.0)
            teacher_similarity = teacher_descriptor @ teacher_descriptor.transpose(0, 1)
            student_similarity = final_descriptor @ final_descriptor.transpose(0, 1)
            pair_valid = (
                descriptor_valid[:, None]
                & descriptor_valid[None, :]
                & ~torch.eye(
                    batch_size,
                    device=descriptor_valid.device,
                    dtype=torch.bool,
                )
            )
            sample_reliability = reliability.sum(dim=1) / part_valid.sum(dim=1).clamp_min(1)
            pair_weights = (sample_reliability[:, None] * sample_reliability[None, :]).sqrt() * pair_valid.to(
                final_descriptor.dtype
            )
            relational_loss = (
                (student_similarity - teacher_similarity).square() * pair_weights
            ).sum() / pair_weights.sum().clamp_min(1e-6)
            descriptor_distill_loss = alignment_loss + 0.5 * relational_loss

        same_identity_cross_camera = (pids[:, None] == pids[None, :]) & (camera_ids[:, None] != camera_ids[None, :])
        positive_availability = (
            same_identity_cross_camera[:, :, None] & part_valid[:, None, :] & part_valid[None, :, :]
        ).any(dim=1)
        valid_part_fraction = part_valid.to(loss_dtype).mean()
        cross_camera_anchor_fraction = (
            positive_availability.to(loss_dtype) * part_valid.to(loss_dtype)
        ).sum() / part_valid.sum().clamp_min(1)
        local_scale_loss = (
            self.anatomical_distill_weight * local_distill_loss
            + self.anatomical_attention_weight * local_attention_loss
            + self.anatomical_visibility_weight * local_visibility_loss
            + self.anatomical_contrastive_weight * local_contrastive_loss
            + semantic_foreground_weight * local_semantic_foreground_loss
            + semantic_part_weight * local_semantic_part_loss
        )
        if multiscale_enabled:
            fine_scale_loss = (
                self.anatomical_distill_weight * fine_distill_loss
                + self.anatomical_attention_weight * fine_attention_loss
                + self.anatomical_visibility_weight * fine_visibility_loss
                + self.anatomical_contrastive_weight * fine_contrastive_loss
                + semantic_foreground_weight * fine_semantic_foreground_loss
                + semantic_part_weight * fine_semantic_part_loss
            )
            local_scale_weight = self.anatomical_local_scale_weight
            fine_scale_weight = self.anatomical_fine_scale_weight
            distill_loss = local_scale_weight * local_distill_loss + fine_scale_weight * fine_distill_loss
            attention_loss = local_scale_weight * local_attention_loss + fine_scale_weight * fine_attention_loss
            visibility_loss = local_scale_weight * local_visibility_loss + fine_scale_weight * fine_visibility_loss
            contrastive_loss = local_scale_weight * local_contrastive_loss + fine_scale_weight * fine_contrastive_loss
            semantic_foreground_loss_value = (
                local_scale_weight * local_semantic_foreground_loss + fine_scale_weight * fine_semantic_foreground_loss
            )
            semantic_part_loss_value = (
                local_scale_weight * local_semantic_part_loss + fine_scale_weight * fine_semantic_part_loss
            )
            local_student_total = (
                local_scale_weight * local_scale_loss
                + self.anatomical_descriptor_distill_weight * descriptor_distill_loss
                + self.anatomical_branch_distill_weight * branch_distill_loss
            )
            fine_student_total = (
                fine_scale_weight * fine_scale_loss + self.anatomical_cross_scale_weight * cross_scale_loss
            )
        else:
            fine_scale_loss = zero
            distill_loss = local_distill_loss
            attention_loss = local_attention_loss
            visibility_loss = local_visibility_loss
            contrastive_loss = local_contrastive_loss
            semantic_foreground_loss_value = local_semantic_foreground_loss
            semantic_part_loss_value = local_semantic_part_loss
            local_student_total = (
                local_scale_loss
                + self.anatomical_descriptor_distill_weight * descriptor_distill_loss
                + self.anatomical_branch_distill_weight * branch_distill_loss
            )
            fine_student_total = zero
        total = (
            student_scale * local_student_total
            + fine_student_scale * fine_student_total
            + teacher_scale * self.anatomical_pose_teacher_weight * pose_teacher_loss
        )
        components = {
            "distill": distill_loss,
            "attention": attention_loss,
            "visibility": visibility_loss,
            "contrastive": contrastive_loss,
            "descriptor_distill": descriptor_distill_loss,
            "branch_distill": branch_distill_loss,
            "branch_global": branch_global_loss,
            "branch_coarse": branch_coarse_loss,
            "branch_fine": branch_fine_loss,
            "pose_teacher": pose_teacher_loss,
            "semantic_foreground": semantic_foreground_loss_value,
            "semantic_part": semantic_part_loss_value,
            "local_scale": local_scale_loss,
            "fine_scale": fine_scale_loss,
            "cross_scale": cross_scale_loss,
            "valid_part_fraction": valid_part_fraction,
            "cross_camera_anchor_fraction": cross_camera_anchor_fraction,
        }
        return (total, components) if return_components else total
