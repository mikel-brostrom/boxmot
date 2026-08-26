"""Anatomical objective dispatch and training-loop integration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
    EMA_ANATOMICAL_TARGET_TYPES,
)
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_CANONICAL_CELLS,
    ANATOMICAL_CANONICAL_GRID_SIZE,
)
from boxmot.reid.training.trainer_components.helpers import (
    _cross_scale_role_relation_loss,
    _scale_aware_anatomical_targets,
)


class _AnatomicalIntegrationMixin:
    def _anatomical_auxiliary_loss(
        self,
        features,
        targets: dict[str, torch.Tensor] | None,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        epoch: int | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute deterministic geometry-routed anatomical supervision."""
        device_type = pids.device.type
        if torch.is_autocast_enabled(device_type):
            # Re-enter once with autocast disabled. The auxiliary target math
            # contains probability normalization and logarithms that must not
            # inherit FP16 from the RGB model forward.
            with torch.amp.autocast(device_type, enabled=False):
                return self._anatomical_auxiliary_loss(
                    features,
                    targets,
                    pids,
                    camera_ids,
                    epoch=epoch,
                    return_components=return_components,
                )
        target_type = getattr(
            self,
            "anatomical_target_type",
            DEFAULT_ANATOMICAL_TARGET_TYPE,
        )
        if target_type == "body_slot_privileged_ema":
            return self._body_slot_privileged_loss(
                features,
                targets,
                pids,
                camera_ids,
                epoch=epoch,
                return_components=return_components,
            )
        if target_type == "privileged_mask_pose_attention":
            return self._privileged_mask_pose_attention_loss(
                features,
                targets,
                pids,
                camera_ids,
                epoch=epoch,
                return_components=return_components,
            )
        if target_type == "decoupled_pose_parsing_teacher":
            base_loss, base_components = self._ema_anatomical_auxiliary_loss(
                features,
                targets,
                pids,
                camera_ids,
                epoch=epoch,
                return_components=True,
            )
            query_loss, query_components = self._decoupled_pose_parsing_query_loss(
                features,
                targets,
                pids,
                camera_ids,
                epoch=epoch,
            )
            base_components.update(query_components)
            base_components["semantic_foreground"] = query_components["query_foreground"]
            base_components["visibility"] = base_components["visibility"] + query_components["query_visibility"]
            total = base_loss + query_loss
            return (total, base_components) if return_components else total
        if target_type in EMA_ANATOMICAL_TARGET_TYPES:
            return self._ema_anatomical_auxiliary_loss(
                features,
                targets,
                pids,
                camera_ids,
                epoch=epoch,
                return_components=return_components,
            )
        zero = torch.tensor(0.0, device=pids.device)
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
            "local_scale": zero,
            "fine_scale": zero,
            "cross_scale": zero,
            "valid_part_fraction": zero,
            "cross_camera_anchor_fraction": zero,
        }
        if not self.anatomical_auxiliary:
            return (zero, zero_components) if return_components else zero
        student_scale, decay_scale = self._anatomical_schedule_scales(epoch)
        fine_student_scale = self._anatomical_fine_schedule_scale(
            epoch,
            student_scale=student_scale,
            decay_scale=decay_scale,
        )
        if student_scale <= 0 and fine_student_scale <= 0:
            return (zero, zero_components) if return_components else zero
        if targets is None:
            raise RuntimeError("Anatomical supervision is enabled but the batch has no targets")
        if not isinstance(features, dict):
            raise RuntimeError("Anatomical supervision requires dictionary model features")
        required_keys = (
            "_anatomical_feature_map",
            "_anatomical_student_tokens",
            "_anatomical_attention",
            "_anatomical_visibility_logits",
        )
        multiscale_enabled = bool(getattr(self, "anatomical_multiscale", False))
        if multiscale_enabled:
            required_keys += (
                "_anatomical_fine_feature_map",
                "_anatomical_fine_student_tokens",
                "_anatomical_fine_attention",
                "_anatomical_fine_visibility_logits",
            )
        missing = [key for key in required_keys if key not in features]
        if missing:
            raise RuntimeError(f"Model did not return anatomical outputs: {missing}")

        feature_map = features["_anatomical_feature_map"]
        student_tokens = features["_anatomical_student_tokens"]
        student_attention = features["_anatomical_attention"]
        visibility_logits = features["_anatomical_visibility_logits"]
        fine_feature_map = features.get("_anatomical_fine_feature_map")
        fine_student_tokens = features.get("_anatomical_fine_student_tokens")
        fine_student_attention = features.get("_anatomical_fine_attention")
        fine_visibility_logits = features.get("_anatomical_fine_visibility_logits")
        loss_dtype = torch.float64 if feature_map.dtype == torch.float64 else torch.float32
        masks = targets["masks"].to(
            device=feature_map.device,
            dtype=loss_dtype,
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
        # Keep noisy pose/mask pairs useful, but prevent them from dominating
        # the privileged signal. Pose-only records have agreement=1.
        agreement_weight = 0.25 + 0.75 * pose_mask_agreement
        canonical_grid = targets["canonical_grid"].to(
            device=feature_map.device,
            dtype=loss_dtype,
        )
        canonical_grid_valid = targets["canonical_grid_valid"].to(
            device=feature_map.device,
            dtype=torch.bool,
        )
        canonical_grid_pose_valid = targets.get(
            "canonical_grid_pose_valid",
            targets["canonical_grid_valid"],
        ).to(
            device=feature_map.device,
            dtype=torch.bool,
        )
        expected_grid_shape = (
            *ANATOMICAL_CANONICAL_GRID_SIZE,
            2,
        )
        if canonical_grid.ndim != 5 or tuple(canonical_grid.shape[2:]) != expected_grid_shape:
            raise ValueError("canonical anatomical grids must have shape [B,P,4,2,2]")
        if canonical_grid_valid.shape != canonical_grid.shape[:-1]:
            raise ValueError("canonical grid validity must match canonical grid cells")
        if canonical_grid_pose_valid.shape != canonical_grid.shape[:-1]:
            raise ValueError("pose-grid validity must match canonical grid cells")
        batch_size, num_parts = canonical_grid.shape[:2]
        canonical_grid = canonical_grid.flatten(2, 3)
        canonical_grid_valid = canonical_grid_valid.flatten(2, 3)
        canonical_grid_pose_valid = canonical_grid_pose_valid.flatten(
            2,
            3,
        )
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
        mask_mass = masks.sum(dim=(-1, -2))
        canonical_grid_valid = canonical_grid_valid & torch.isfinite(canonical_grid).all(dim=-1)
        canonical_grid_pose_valid = canonical_grid_pose_valid & torch.isfinite(canonical_grid).all(dim=-1)
        mask_part_valid = (
            mask_metadata_valid[:, None]
            & (visibility > 0)
            & (mask_reliability > 0)
            & (mask_mass > 1e-6)
            & canonical_grid_valid.any(dim=-1)
        )
        pose_part_valid = pose_metadata_valid[:, None] & (pose_reliability > 0) & canonical_grid_pose_valid.any(dim=-1)
        if self.anatomical_pose_teacher_weight > 0:
            # Person masks provide the stricter spatial validity signal. Only
            # records without a mask fall back to the looser pose-only grid,
            # and those targets are explicitly down-weighted.
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
            pose_only_confidence = pose_reliability * float(
                getattr(
                    self,
                    "anatomical_pose_only_reliability",
                    0.35,
                )
            )
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

        sample_weights = (mask_metadata_valid.to(visibility.dtype) * agreement_weight.to(visibility.dtype))[:, None]

        def token_distill_loss(
            tokens: torch.Tensor,
            teacher_cell_tokens: torch.Tensor,
            cell_weights: torch.Tensor,
        ) -> torch.Tensor:
            if tokens.shape[-1] % ANATOMICAL_CANONICAL_CELLS:
                raise RuntimeError("Anatomical token width must be divisible by the canonical cell count")
            student_cell_tokens = tokens.unflatten(
                -1,
                (
                    ANATOMICAL_CANONICAL_CELLS,
                    tokens.shape[-1] // ANATOMICAL_CANONICAL_CELLS,
                ),
            )
            if student_cell_tokens.shape != teacher_cell_tokens.shape:
                raise RuntimeError(
                    "Student and geometry-teacher cell tokens must match, "
                    f"got {tuple(student_cell_tokens.shape)} and "
                    f"{tuple(teacher_cell_tokens.shape)}"
                )
            student_cell_tokens = student_cell_tokens.to(
                dtype=teacher_cell_tokens.dtype,
            )
            values = 1.0 - F.cosine_similarity(
                student_cell_tokens,
                teacher_cell_tokens,
                dim=-1,
            )
            return (values * cell_weights).sum() / cell_weights.sum().clamp_min(1e-6)

        def spatial_attention_loss(
            attention: torch.Tensor,
            target_attention: torch.Tensor,
            cell_weights: torch.Tensor,
        ) -> torch.Tensor:
            attention = attention.to(dtype=target_attention.dtype)
            values = (
                target_attention * (target_attention.clamp_min(1e-8).log() - attention.clamp_min(1e-8).log())
            ).sum(dim=(-1, -2))
            return (values * cell_weights).sum() / cell_weights.sum().clamp_min(1e-6)

        def dense_geometry_loss(
            attention: torch.Tensor,
            dense_target: torch.Tensor,
            routing_valid: torch.Tensor,
        ) -> torch.Tensor:
            attention = attention.to(dtype=dense_target.dtype)
            valid = routing_valid.to(attention.dtype)
            aggregated = (attention * valid[..., None, None]).sum(dim=2)
            aggregated = aggregated / aggregated.sum(
                dim=(-1, -2),
                keepdim=True,
            ).clamp_min(1e-8)
            values = (dense_target * (dense_target.clamp_min(1e-8).log() - aggregated.clamp_min(1e-8).log())).sum(
                dim=(-1, -2)
            )
            weights = part_confidence * routing_valid.any(dim=-1).to(part_confidence.dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1e-6)

        (
            local_routing,
            local_dense_target,
            local_routing_valid,
            local_teacher_cell_tokens,
        ) = _scale_aware_anatomical_targets(
            feature_map,
            masks,
            canonical_grid,
            supervision_grid_valid,
            mask_metadata_valid,
            fine_scale=False,
        )
        local_cell_weights = part_confidence[:, :, None] * local_routing_valid.to(part_confidence.dtype)
        local_distill_loss = token_distill_loss(
            student_tokens,
            local_teacher_cell_tokens,
            local_cell_weights,
        )
        local_attention_loss = spatial_attention_loss(
            student_attention,
            local_routing,
            local_cell_weights,
        )
        local_geometry_loss = dense_geometry_loss(
            student_attention,
            local_dense_target,
            local_routing_valid,
        )

        fine_teacher_cell_tokens = None
        if multiscale_enabled:
            (
                fine_routing,
                fine_dense_target,
                fine_routing_valid,
                fine_teacher_cell_tokens,
            ) = _scale_aware_anatomical_targets(
                fine_feature_map,
                masks,
                canonical_grid,
                supervision_grid_valid,
                mask_metadata_valid,
                fine_scale=True,
            )
            fine_cell_weights = part_confidence[:, :, None] * fine_routing_valid.to(part_confidence.dtype)
            fine_distill_loss = token_distill_loss(
                fine_student_tokens,
                fine_teacher_cell_tokens,
                fine_cell_weights,
            )
            fine_attention_loss = spatial_attention_loss(
                fine_student_attention,
                fine_routing,
                fine_cell_weights,
            )
            fine_geometry_loss = dense_geometry_loss(
                fine_student_attention,
                fine_dense_target,
                fine_routing_valid,
            )
        else:
            fine_distill_loss = zero
            fine_attention_loss = zero
            fine_geometry_loss = zero

        def part_visibility_loss(logits: torch.Tensor) -> torch.Tensor:
            logits = logits.to(dtype=visibility.dtype)
            values = F.binary_cross_entropy_with_logits(
                logits,
                visibility,
                reduction="none",
            )
            return (values * sample_weights).sum() / (sample_weights.sum() * values.shape[1]).clamp_min(1.0)

        local_visibility_loss = part_visibility_loss(visibility_logits)
        local_contrastive_loss = self._cross_camera_part_contrastive_loss(
            student_tokens,
            pids,
            camera_ids,
            reliability,
            self.anatomical_temperature,
        )
        if multiscale_enabled:
            fine_visibility_loss = part_visibility_loss(fine_visibility_logits)
            fine_contrastive_loss = self._cross_camera_part_contrastive_loss(
                fine_student_tokens,
                pids,
                camera_ids,
                reliability,
                self.anatomical_temperature,
            )
            cross_scale_loss = _cross_scale_role_relation_loss(
                student_tokens,
                fine_student_tokens,
                reliability,
            )
        else:
            fine_visibility_loss = zero
            fine_contrastive_loss = zero
            cross_scale_loss = zero
        descriptor_distill_loss = (
            student_tokens.to(
                dtype=loss_dtype,
            ).sum()
            * 0.0
        )
        if self.anatomical_descriptor_distill_weight > 0:
            projected_final = features.get("_anatomical_final_student")
            final_descriptor = features.get("norm_concat_bn")
            if not torch.is_tensor(projected_final) or not torch.is_tensor(final_descriptor):
                raise RuntimeError(
                    "Anatomical descriptor distillation requires projected and deployed final descriptors"
                )
            # The local map is the semantic anchor for the deployed descriptor.
            # Fine tokens remain free to preserve additional spatial detail.
            teacher_part_tokens = (
                local_teacher_cell_tokens * local_routing_valid[..., None].to(local_teacher_cell_tokens.dtype)
            ).flatten(2)
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
                projected_final.to(dtype=teacher_descriptor.dtype),
                p=2,
                dim=1,
            )
            final_descriptor = F.normalize(
                final_descriptor.to(dtype=teacher_descriptor.dtype),
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
        valid_part_fraction = part_valid.to(feature_map.dtype).mean()
        cross_camera_anchor_fraction = (
            positive_availability.to(feature_map.dtype) * part_valid.to(feature_map.dtype)
        ).sum() / part_valid.sum().clamp_min(1)
        local_scale_loss = (
            self.anatomical_distill_weight * local_distill_loss
            + self.anatomical_attention_weight * local_attention_loss
            + self.anatomical_visibility_weight * local_visibility_loss
            + self.anatomical_contrastive_weight * local_contrastive_loss
            + self.anatomical_pose_teacher_weight * local_geometry_loss
        )
        if multiscale_enabled:
            fine_scale_loss = (
                self.anatomical_distill_weight * fine_distill_loss
                + self.anatomical_attention_weight * fine_attention_loss
                + self.anatomical_visibility_weight * fine_visibility_loss
                + self.anatomical_contrastive_weight * fine_contrastive_loss
                + self.anatomical_pose_teacher_weight * fine_geometry_loss
            )
            local_scale_weight = float(self.anatomical_local_scale_weight)
            fine_scale_weight = float(self.anatomical_fine_scale_weight)
            distill_loss = local_scale_weight * local_distill_loss + fine_scale_weight * fine_distill_loss
            attention_loss = local_scale_weight * local_attention_loss + fine_scale_weight * fine_attention_loss
            visibility_loss = local_scale_weight * local_visibility_loss + fine_scale_weight * fine_visibility_loss
            contrastive_loss = local_scale_weight * local_contrastive_loss + fine_scale_weight * fine_contrastive_loss
            pose_teacher_loss = local_scale_weight * local_geometry_loss + fine_scale_weight * fine_geometry_loss
            local_student_total = (
                local_scale_weight * local_scale_loss
                + self.anatomical_descriptor_distill_weight * descriptor_distill_loss
            )
            fine_student_total = (
                fine_scale_weight * fine_scale_loss
                + float(
                    getattr(
                        self,
                        "anatomical_cross_scale_weight",
                        0.0,
                    )
                )
                * cross_scale_loss
            )
        else:
            fine_scale_loss = zero
            distill_loss = local_distill_loss
            attention_loss = local_attention_loss
            visibility_loss = local_visibility_loss
            contrastive_loss = local_contrastive_loss
            pose_teacher_loss = local_geometry_loss
            local_student_total = local_scale_loss + self.anatomical_descriptor_distill_weight * descriptor_distill_loss
            fine_student_total = zero
        total = student_scale * local_student_total + fine_student_scale * fine_student_total
        components = {
            "distill": distill_loss,
            "attention": attention_loss,
            "visibility": visibility_loss,
            "contrastive": contrastive_loss,
            "descriptor_distill": descriptor_distill_loss,
            "pose_teacher": pose_teacher_loss,
            "local_scale": local_scale_loss,
            "fine_scale": fine_scale_loss,
            "cross_scale": cross_scale_loss,
            "valid_part_fraction": valid_part_fraction,
            "cross_camera_anchor_fraction": cross_camera_anchor_fraction,
        }
        return (total, components) if return_components else total

    def _anatomical_forward_kwargs(
        self,
        targets: dict[str, torch.Tensor] | None,
        *,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Build pose/mask inputs for one geometrically aligned RGB view."""
        if not self.anatomical_auxiliary:
            return {}

        kwargs: dict[str, torch.Tensor] = {}
        if self.anatomical_target_type in EMA_ANATOMICAL_TARGET_TYPES:
            if targets is None or "pose_keypoints" not in targets:
                raise RuntimeError(
                    "Pose-conditioned anatomy teacher requires cached pose_keypoints in every training batch"
                )
            kwargs["anatomical_pose"] = targets["pose_keypoints"].to(
                device=self.device,
                dtype=dtype,
            )
        if self.anatomical_target_type == "decoupled_pose_parsing_teacher":
            if targets is None or "masks" not in targets or "foreground_mask" not in targets:
                raise RuntimeError(
                    "Decoupled pose-parsing teacher requires part and foreground masks in every training batch"
                )
            query_masks = targets["masks"] * targets["foreground_mask"]
            if self.anatomical_accessory_query:
                if "accessory_mask" not in targets:
                    raise RuntimeError("Accessory query requires accessory_mask targets")
                query_masks = torch.cat(
                    (query_masks, targets["accessory_mask"]),
                    dim=1,
                )
            kwargs["anatomical_query_masks"] = query_masks.to(
                device=self.device,
                dtype=dtype,
            )
        if self.anatomical_target_type == "body_slot_privileged_ema":
            if targets is None:
                raise RuntimeError("Body-slot teacher requires pose/mask targets in every training batch")
            kwargs["anatomical_query_masks"] = self._body_slot_teacher_masks(
                targets,
                device=self.device,
                dtype=dtype,
            )
        return kwargs

    def _clean_teacher_student_consistency_loss(
        self,
        student_features: dict[str, torch.Tensor],
        clean_features: dict[str, torch.Tensor],
        clean_targets: dict[str, torch.Tensor],
        student_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Distil clean masked queries and retrieval geometry into RGB views."""
        if not isinstance(student_features, dict) or not isinstance(
            clean_features,
            dict,
        ):
            raise RuntimeError("Clean-student consistency requires dictionary model features")
        scale_keys = (
            (
                "_anatomical_query_student_tokens",
                "_anatomical_query_teacher_tokens",
                "_anatomical_query_teacher_valid",
                self.anatomical_local_scale_weight,
            ),
            (
                "_anatomical_query_fine_student_tokens",
                "_anatomical_query_fine_teacher_tokens",
                "_anatomical_query_fine_teacher_valid",
                self.anatomical_fine_scale_weight,
            ),
        )
        missing = []
        for student_key, teacher_key, valid_key, _ in scale_keys:
            if student_key not in student_features:
                missing.append(student_key)
            for clean_key in (teacher_key, valid_key):
                if clean_key not in clean_features:
                    missing.append(clean_key)
        if missing:
            raise RuntimeError(f"Clean-student consistency is missing query outputs: {missing}")

        local_student = student_features[scale_keys[0][0]][student_indices]
        loss_dtype = torch.float64 if local_student.dtype == torch.float64 else torch.float32
        device = local_student.device
        masks = clean_targets["masks"].to(
            device=device,
            dtype=loss_dtype,
        )
        foreground = clean_targets["foreground_mask"].to(
            device=device,
            dtype=loss_dtype,
        )
        masks = masks * foreground
        visibility = (
            clean_targets["visibility"]
            .to(
                device=device,
                dtype=loss_dtype,
            )
            .clamp(0, 1)
        )
        reliability = (
            clean_targets["reliability"].to(
                device=device,
                dtype=loss_dtype,
            )
            * visibility
        ).clamp(0, 1)
        mask_valid = clean_targets.get(
            "mask_valid",
            clean_targets["valid"],
        ).to(device=device, dtype=torch.bool)
        reliability = reliability * mask_valid[:, None].to(loss_dtype) * (masks.sum(dim=(-1, -2)) > 1e-6).to(loss_dtype)
        if self.anatomical_accessory_query:
            accessory_mask = clean_targets["accessory_mask"].to(
                device=device,
                dtype=loss_dtype,
            )
            accessory_weight = (
                clean_targets["accessory_reliability"]
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
                .reshape(-1, 1)
                * clean_targets["accessory_visibility"]
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
                .reshape(-1, 1)
                * clean_targets["accessory_valid"]
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
                .reshape(-1, 1)
                * (accessory_mask.sum(dim=(-1, -2)) > 1e-6).to(loss_dtype)
            )
            reliability = torch.cat(
                (reliability, accessory_weight),
                dim=1,
            )

        query_loss = local_student.sum() * 0.0
        for student_key, teacher_key, valid_key, scale_weight in scale_keys:
            student = student_features[student_key][student_indices].to(dtype=loss_dtype)
            teacher = (
                clean_features[teacher_key]
                .detach()
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
            )
            valid = (
                clean_features[valid_key]
                .detach()
                .to(
                    device=device,
                    dtype=loss_dtype,
                )
            )
            weights = reliability * valid
            values = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
            scale_loss = (values * weights).sum() / weights.sum().clamp_min(1e-6)
            query_loss = query_loss + float(scale_weight) * scale_loss

        student_descriptor = self._pav_consistency_descriptor(student_features)
        clean_descriptor = self._pav_consistency_descriptor(clean_features)
        if not torch.is_tensor(student_descriptor) or not torch.is_tensor(clean_descriptor):
            raise RuntimeError("Clean-student consistency requires retrieval descriptors")
        descriptor_loss = (
            1.0
            - F.cosine_similarity(
                student_descriptor[student_indices],
                clean_descriptor.detach(),
                dim=1,
            ).mean()
        )
        return 0.5 * (query_loss + descriptor_loss)
