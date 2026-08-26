"""Metric, branch, transport, evidence, and classification objectives."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from boxmot.reid.training.losses import (
    AdaSPLoss,
    CrossScaleMajorityMarginLoss,
    TreeBoostAPLoss,
)
from boxmot.utils import logger as LOGGER


class _ObjectiveMixin:
    def _metric_loss_for_features(self, criterion_metric, features, pids: torch.Tensor) -> torch.Tensor:
        """Compute metric loss for a tensor feature or branch feature dict."""
        if isinstance(features, dict):
            explicit_features = features.get("_metric_features")
            if isinstance(explicit_features, (list, tuple)):
                valid_features = [feature for feature in explicit_features if torch.is_tensor(feature)]
                if not valid_features:
                    return torch.zeros((), device=self.device, requires_grad=True)
                losses = [
                    criterion_metric(F.normalize(feature, p=2, dim=1), pids)
                    for feature in valid_features
                ]
                if features.get("_metric_loss_aggregation") == "sum":
                    return sum(losses)
                return sum(losses) / len(losses)

            if not self.branch_aware_metric:
                key = self._effective_metric_feature()
                selected = features.get(key, features["raw_mean"])
                return criterion_metric(F.normalize(selected, p=2, dim=1), pids)

            part_supervision_weights = self._part_supervision_weights(features)
            if torch.is_tensor(part_supervision_weights):
                global_features = features.get("global", features.get("raw_mean"))
                global_loss = criterion_metric(F.normalize(global_features, p=2, dim=1), pids)
                part_losses = []
                part_weights = []
                for index, key in enumerate(self._sorted_part_feature_keys(features)):
                    if key not in features or index >= part_supervision_weights.shape[1]:
                        continue
                    branch_features = F.normalize(features[key], p=2, dim=1)
                    part_losses.append(criterion_metric(branch_features, pids))
                    part_weights.append(part_supervision_weights[:, index].mean().clamp(min=0.0))
                if not part_losses:
                    return global_loss
                weights = torch.stack(part_weights)
                part_loss = sum(loss * weight for loss, weight in zip(part_losses, weights)) / weights.sum().clamp(
                    min=1e-12
                )
                return global_loss + self.branch_metric_part_weight * part_loss

            weighted_losses = []
            total_weight = 0.0
            branch_weights = [("global", 1.0)]
            metric_key = self._effective_metric_feature()
            if metric_key == "raw_concat" and metric_key in features:
                branch_weights.append((metric_key, 1.0))
            branch_weights += [
                (key, self.branch_metric_part_weight) for key in self._sorted_part_feature_keys(features)
            ]
            for key, weight in branch_weights:
                if key in features and weight > 0:
                    branch_features = F.normalize(features[key], p=2, dim=1)
                    weighted_losses.append(criterion_metric(branch_features, pids) * weight)
                    total_weight += weight
            if weighted_losses and total_weight > 0:
                return sum(weighted_losses) / total_weight
            return torch.zeros((), device=self.device, requires_grad=True)

        if isinstance(features, (list, tuple)):
            valid_features = [feat for feat in features if isinstance(feat, torch.Tensor)]
            if not valid_features:
                return torch.zeros((), device=self.device, requires_grad=True)
            losses = [criterion_metric(F.normalize(feat, p=2, dim=1), pids) for feat in valid_features]
            return self._reduce_branch_losses(losses)

        return criterion_metric(F.normalize(features, p=2, dim=1), pids)

    def _jpm_auxiliary_losses(
        self,
        criterion_id,
        criterion_metric,
        features,
        pids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean identity and triplet losses over JPM local groups."""
        zero = torch.zeros((), device=self.device)
        if not self.jpm:
            return zero, zero
        if not isinstance(features, dict):
            raise RuntimeError("JPM requires the auxiliary feature dictionary")
        logits = features.get("_jpm_logits")
        local_features = features.get("_jpm_features")
        if not isinstance(logits, tuple) or not isinstance(local_features, tuple):
            raise RuntimeError("Enabled JPM did not return local logits/features")
        if len(logits) != self.jpm_num_groups or len(local_features) != self.jpm_num_groups:
            raise RuntimeError(f"JPM output count does not match jpm_num_groups={self.jpm_num_groups}")
        if criterion_metric is None:
            raise RuntimeError("JPM metric supervision requires a triplet criterion")
        id_losses = []
        metric_losses = []
        for group_logits, group_features in zip(
            logits,
            local_features,
            strict=True,
        ):
            if not torch.is_tensor(group_logits) or not torch.is_tensor(group_features):
                raise RuntimeError("JPM group outputs must be tensors")
            id_losses.append(criterion_id(group_logits, pids))
            metric_losses.append(criterion_metric(F.normalize(group_features, p=2, dim=1), pids))
        return torch.stack(id_losses).mean(), torch.stack(metric_losses).mean()

    def _adasp_loss_for_features(
        self,
        criterion_adasp: AdaSPLoss,
        features,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply AdaSP once to the complete normalized retrieval descriptor."""
        if isinstance(features, dict):
            key = self._effective_metric_feature()
            descriptor = features.get(key, features.get("raw_mean"))
        else:
            descriptor = features
        if not torch.is_tensor(descriptor) or descriptor.ndim != 2:
            raise RuntimeError("AdaSP requires one full [batch, features] descriptor")
        return criterion_adasp(
            F.normalize(descriptor, p=2, dim=1),
            pids,
        )

    @torch.no_grad()
    def _ema_part_teacher_features(
        self,
        ema_model: nn.Module,
        imgs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract an eval-stable training packet from the EMA teacher."""
        ema_model.eval()
        head = self._model_head(ema_model)
        if head is None:
            raise RuntimeError("Part-relation EMA teacher has no ReID head")
        head.train()
        for module in head.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        try:
            output = ema_model(imgs)
        finally:
            ema_model.eval()
        _, features = self._split_model_output(output)
        if not isinstance(features, dict):
            raise RuntimeError("Part-relation EMA teacher did not return branch features")
        return features

    def _part_relation_losses(
        self,
        student_features,
        teacher_features,
        pids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match cross-ID fine-part geometry and distil it into global RGB."""
        zero = torch.zeros((), device=self.device)
        if not self._part_relation_enabled():
            return zero, zero
        if not isinstance(student_features, dict) or not isinstance(
            teacher_features,
            dict,
        ):
            raise RuntimeError("Part-relation supervision requires branch feature dictionaries")
        part_keys = self._sorted_part_feature_keys(student_features)
        if len(part_keys) != 6:
            raise RuntimeError("Part-relation supervision expects two coarse and four fine branches")
        fine_keys = part_keys[-4:]
        if any(key not in teacher_features for key in fine_keys):
            raise RuntimeError("EMA teacher is missing corresponding fine-part features")
        cross_identity = pids[:, None] != pids[None, :]
        if not cross_identity.any():
            anchor = student_features[fine_keys[0]]
            return anchor.sum() * 0.0, anchor.sum() * 0.0

        def neighborhood_kl(
            student_similarity: torch.Tensor,
            teacher_similarity: torch.Tensor,
        ) -> torch.Tensor:
            student_logits = (student_similarity / self.part_relation_temperature).masked_fill(
                ~cross_identity, float("-inf")
            )
            teacher_logits = (teacher_similarity / self.part_relation_temperature).masked_fill(
                ~cross_identity, float("-inf")
            )
            student_log_prob = student_logits - torch.logsumexp(
                student_logits,
                dim=1,
                keepdim=True,
            )
            teacher_log_prob = teacher_logits - torch.logsumexp(
                teacher_logits,
                dim=1,
                keepdim=True,
            )
            student_log_prob = torch.where(
                cross_identity,
                student_log_prob,
                torch.zeros_like(student_log_prob),
            )
            teacher_log_prob = torch.where(
                cross_identity,
                teacher_log_prob,
                torch.zeros_like(teacher_log_prob),
            )
            teacher_prob = torch.where(
                cross_identity,
                teacher_log_prob.exp(),
                torch.zeros_like(teacher_log_prob),
            )
            terms = teacher_prob * (teacher_log_prob - student_log_prob)
            return (
                torch.where(
                    cross_identity,
                    terms,
                    torch.zeros_like(terms),
                )
                .sum(dim=1)
                .mean()
            )

        teacher_part_similarities = []
        part_losses = []
        for key in fine_keys:
            student = F.normalize(student_features[key].float(), p=2, dim=1)
            teacher = F.normalize(
                teacher_features[key].detach().float(),
                p=2,
                dim=1,
            )
            student_similarity = student @ student.transpose(0, 1)
            teacher_similarity = teacher @ teacher.transpose(0, 1)
            teacher_part_similarities.append(teacher_similarity)
            part_losses.append(neighborhood_kl(student_similarity, teacher_similarity))
        part_relation = torch.stack(part_losses).mean()

        global_student = F.normalize(
            student_features["global"].float(),
            p=2,
            dim=1,
        )
        global_similarity = global_student @ global_student.transpose(0, 1)
        aggregate_teacher_similarity = torch.stack(
            teacher_part_similarities,
            dim=0,
        ).mean(dim=0)
        part_to_global = neighborhood_kl(
            global_similarity,
            aggregate_teacher_similarity,
        )
        return part_relation, part_to_global

    def _cross_scale_majority_margin_loss(
        self,
        criterion_csmm: CrossScaleMajorityMarginLoss | None,
        features,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CSMM from the training-only global/coarse/fine groups."""
        if criterion_csmm is None:
            return torch.zeros((), device=self.device)
        if not isinstance(features, dict):
            raise RuntimeError("CSMM training requires the CSL-TinyViT feature dictionary")
        scale_features = features.get("_cross_scale_features")
        if not isinstance(scale_features, tuple) or len(scale_features) != 3:
            raise RuntimeError("CSL-TinyViT head did not return three CSMM scale groups")
        return criterion_csmm(scale_features, pids, mining_descriptor=features.get("raw_concat"))

    def _treeboost_ap_loss(
        self,
        criterion_treeboost: TreeBoostAPLoss | None,
        features,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TreeBoost-AP from training-only post-BN hierarchy features."""
        if criterion_treeboost is None:
            return torch.zeros((), device=self.device)
        if not isinstance(features, dict):
            raise RuntimeError("TreeBoost-AP training requires the CSL-TinyViT feature dictionary")
        hierarchy_features = features.get("_treeboost_features")
        if not isinstance(hierarchy_features, tuple) or len(hierarchy_features) != 3:
            raise RuntimeError("CSL-TinyViT head did not return TreeBoost hierarchy features")
        return criterion_treeboost(hierarchy_features, pids, camera_ids)

    @staticmethod
    def _model_head(model: nn.Module) -> nn.Module | None:
        """Return the underlying ReID head through optional wrapper modules."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return getattr(unwrapped, "head", None)

    def _set_mcpt_epoch(self, model: nn.Module, epoch: int) -> None:
        """Apply the deterministic MCPT gate schedule to a model copy."""
        if self.mcpt_mode == "none":
            return
        head = self._model_head(model)
        setter = getattr(head, "set_mcpt_epoch", None)
        if not callable(setter):
            raise RuntimeError("Enabled MCPT model is missing its epoch schedule hook")
        setter(epoch)

    def _set_mcpt_force_disabled(self, model: nn.Module, disabled: bool) -> None:
        """Toggle the evaluation-only MCPT control path."""
        head = self._model_head(model)
        setter = getattr(head, "set_mcpt_force_disabled", None)
        if not callable(setter):
            raise RuntimeError("Enabled MCPT model is missing its disable hook")
        setter(disabled)

    def _mcpt_identity_weight_for_epoch(self, epoch: int) -> float:
        """Linearly remove the identity prior after transport activation."""
        if self.mcpt_mode == "none" or epoch >= self.mcpt_identity_decay_epoch:
            return 0.0
        if epoch <= self.mcpt_start_epoch:
            return self.mcpt_identity_weight
        remaining = self.mcpt_identity_decay_epoch - epoch
        duration = self.mcpt_identity_decay_epoch - self.mcpt_start_epoch
        return self.mcpt_identity_weight * remaining / max(duration, 1)

    def _mcpt_auxiliary_loss(
        self,
        features,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return weak warp regularization and alignment health diagnostics."""
        zero = torch.zeros((), device=self.device)
        defaults = {
            "smoothness": zero,
            "identity": zero,
            "mean_abs_displacement": zero,
            "boundary_1": zero + 0.25,
            "boundary_2": zero + 0.50,
            "boundary_3": zero + 0.75,
            "boundary_std": zero,
            "cap_fraction": zero,
            "local_gate": zero,
            "fine_gate": zero,
        }
        if self.mcpt_mode == "none":
            return zero, defaults
        if not isinstance(features, dict):
            raise RuntimeError("Enabled MCPT requires the CSL-TinyViT feature dictionary")
        required = {
            "smoothness": "_mcpt_smoothness",
            "identity": "_mcpt_identity",
            "mean_abs_displacement": "_mcpt_mean_abs_displacement",
            "boundary": "_mcpt_boundary_mean",
            "boundary_std": "_mcpt_boundary_std",
            "cap_fraction": "_mcpt_cap_fraction",
            "local_gate": "_mcpt_local_gate",
            "fine_gate": "_mcpt_fine_gate",
        }
        missing = [key for key in required.values() if key not in features]
        if missing:
            raise RuntimeError(f"Enabled MCPT output is missing diagnostics: {missing}")
        boundaries = features[required["boundary"]].reshape(-1)
        if boundaries.numel() != 3:
            raise RuntimeError("MCPT must report exactly three nested fine boundaries")
        components = {
            "smoothness": features[required["smoothness"]],
            "identity": features[required["identity"]],
            "mean_abs_displacement": features[required["mean_abs_displacement"]],
            "boundary_1": boundaries[0],
            "boundary_2": boundaries[1],
            "boundary_3": boundaries[2],
            "boundary_std": features[required["boundary_std"]],
            "cap_fraction": features[required["cap_fraction"]],
            "local_gate": features[required["local_gate"]],
            "fine_gate": features[required["fine_gate"]],
        }
        identity_weight = self._mcpt_identity_weight_for_epoch(epoch)
        loss = self.mcpt_smoothness_weight * components["smoothness"] + identity_weight * components["identity"]
        return loss, components

    @staticmethod
    def _save_mcpt_energy_maps(
        save_dir: Path,
        epoch: int,
        captured: dict[str, torch.Tensor] | None,
    ) -> None:
        """Save a contact sheet of up to 100 before/after MCPT energy maps."""
        if not captured or "local_before" not in captured:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            LOGGER.warning(f"Could not render MCPT energy maps: {exc}")
            return

        def contact_sheet(maps: torch.Tensor) -> np.ndarray:
            values = maps[:100].numpy()
            count, height, width = values.shape
            columns = min(10, max(count, 1))
            rows = math.ceil(count / columns)
            canvas = np.zeros((rows * height, columns * width), dtype=np.float32)
            for index, value in enumerate(values):
                low = float(value.min())
                high = float(np.percentile(value, 99.0))
                normalized = np.clip(
                    (value - low) / max(high - low, 1e-12),
                    0.0,
                    1.0,
                )
                row, column = divmod(index, columns)
                canvas[
                    row * height : (row + 1) * height,
                    column * width : (column + 1) * width,
                ] = normalized
            return canvas

        scales = ["local"]
        if "fine_before" in captured and "fine_after" in captured:
            scales.append("fine")
        figure, axes = plt.subplots(
            len(scales),
            2,
            figsize=(14, 5 * len(scales)),
            dpi=140,
            squeeze=False,
        )
        for row, scale in enumerate(scales):
            for column, state in enumerate(("before", "after")):
                maps = captured[f"{scale}_{state}"]
                axes[row, column].imshow(
                    contact_sheet(maps),
                    cmap="inferno",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="auto",
                )
                axes[row, column].set_title(f"{scale.title()} {state} MCPT ({maps.shape[0]} samples)")
                axes[row, column].axis("off")
        figure.suptitle(f"MCPT feature energy, epoch {epoch}")
        figure.tight_layout()
        path = save_dir / f"mcpt_energy_epoch_{epoch:04d}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        LOGGER.info(f"Saved MCPT feature-energy maps to {path}")

    def _hierarchical_late_interaction_losses(
        self,
        model: nn.Module,
        features,
        pids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return multi-positive matcher and stop-gradient ranking distillation losses."""
        zero = torch.zeros((), device=self.device)
        if not self.hierarchical_late_interaction:
            return zero, zero
        if not isinstance(features, dict):
            raise RuntimeError("hierarchical late interaction requires the CSL-TinyViT feature dictionary")
        hierarchy = features.get("_late_interaction_features")
        base_descriptor = features.get("norm_concat_bn")
        if not isinstance(hierarchy, tuple) or len(hierarchy) != 3 or not torch.is_tensor(base_descriptor):
            raise RuntimeError("CSL-TinyViT head did not return late-interaction evidence features")
        head = self._model_head(model)
        matcher = getattr(head, "late_interaction_matcher", None)
        if matcher is None:
            raise RuntimeError("hierarchical late-interaction matcher is missing from the model head")

        normalized_base = F.normalize(base_descriptor, p=2, dim=1)
        mining_similarity = (normalized_base.detach() @ normalized_base.detach().transpose(0, 1)).clone()
        mining_similarity.fill_diagonal_(-torch.inf)
        query_indices: list[torch.Tensor] = []
        gallery_indices: list[torch.Tensor] = []
        segments: list[tuple[int, int, int]] = []
        cursor = 0

        for anchor in range(pids.shape[0]):
            positive_indices = torch.nonzero(
                (pids == pids[anchor]) & (camera_ids != camera_ids[anchor]),
                as_tuple=False,
            ).flatten()
            if positive_indices.numel() == 0:
                continue

            negative_representatives = []
            negative_scores = []
            for identity in torch.unique(pids[pids != pids[anchor]]):
                identity_indices = torch.nonzero(pids == identity, as_tuple=False).flatten()
                identity_similarities = mining_similarity[anchor, identity_indices]
                best_offset = identity_similarities.argmax()
                negative_representatives.append(identity_indices[best_offset])
                negative_scores.append(identity_similarities[best_offset])
            if not negative_representatives:
                continue
            representatives = torch.stack(negative_representatives)
            representative_scores = torch.stack(negative_scores)
            negative_count = min(self.late_interaction_negative_identities, representatives.numel())
            selected_negatives = representatives[representative_scores.topk(negative_count).indices]
            candidates = torch.cat((positive_indices, selected_negatives))
            query_indices.append(torch.full_like(candidates, anchor))
            gallery_indices.append(candidates)
            end = cursor + candidates.numel()
            segments.append((cursor, end, positive_indices.numel()))
            cursor = end

        if not query_indices:
            return normalized_base.sum() * 0.0, normalized_base.sum() * 0.0

        query_index = torch.cat(query_indices)
        gallery_index = torch.cat(gallery_indices)

        def select_hierarchy(indices: torch.Tensor):
            global_feature, coarse_features, fine_features = hierarchy
            return (
                global_feature[indices],
                tuple(feature[indices] for feature in coarse_features),
                tuple(feature[indices] for feature in fine_features),
            )

        pair_scores = matcher.score_pairs(
            select_hierarchy(query_index),
            select_hierarchy(gallery_index),
            normalized_base[query_index],
            normalized_base[gallery_index],
        )
        base_scores = torch.sum(
            normalized_base[query_index] * normalized_base[gallery_index],
            dim=1,
        )
        matcher_losses = []
        distillation_losses = []
        temperature = self.late_interaction_temperature
        for start, end, positive_count in segments:
            logits = pair_scores[start:end] / temperature
            matcher_losses.append(torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[:positive_count], dim=0))
            teacher_distribution = F.softmax(logits.detach(), dim=0)
            student_log_distribution = F.log_softmax(base_scores[start:end] / temperature, dim=0)
            distillation_losses.append(F.kl_div(student_log_distribution, teacher_distribution, reduction="sum"))
        return torch.stack(matcher_losses).mean(), torch.stack(distillation_losses).mean()

    def _compact_student_losses(
        self,
        criterion_metric,
        features,
        pids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return compact triplet, cosine, and pairwise distillation losses."""
        zero = torch.zeros((), device=self.device)
        if not self.compact_deployment_head:
            return zero, zero, zero
        if not isinstance(features, dict):
            raise RuntimeError("Compact deployment training requires the head feature dictionary")
        required = {
            "_compact_student",
            "_compact_student_bn",
            "_compact_teacher",
            "_compact_decoded",
        }
        missing = sorted(required.difference(features))
        if missing:
            raise RuntimeError(f"Compact deployment head did not return training features: {missing}")

        student = F.normalize(features["_compact_student"], p=2, dim=1)
        student_bn = F.normalize(features["_compact_student_bn"], p=2, dim=1)
        teacher = F.normalize(features["_compact_teacher"].detach(), p=2, dim=1)
        decoded = F.normalize(features["_compact_decoded"], p=2, dim=1)

        metric = zero
        if criterion_metric is not None and self.compact_metric_loss_weight > 0:
            metric = criterion_metric(student, pids)
        cosine = 1.0 - torch.sum(decoded * teacher, dim=1).mean()

        if student.shape[0] < 2:
            pairwise = zero
        else:
            student_distances = 1.0 - student_bn @ student_bn.transpose(0, 1)
            teacher_distances = 1.0 - teacher @ teacher.transpose(0, 1)
            off_diagonal = ~torch.eye(
                student.shape[0],
                device=student.device,
                dtype=torch.bool,
            )
            pairwise = F.smooth_l1_loss(
                student_distances[off_diagonal],
                teacher_distances[off_diagonal],
            )
        return metric, cosine, pairwise

    def _compact_student_id_loss(
        self,
        criterion_id: nn.Module,
        features,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the existing ID criterion without changing teacher CE allocation."""
        if not self.compact_deployment_head:
            return torch.zeros((), device=self.device)
        if not isinstance(features, dict) or not torch.is_tensor(features.get("_compact_logits")):
            raise RuntimeError("Compact deployment head did not return student classifier logits")
        return criterion_id(features["_compact_logits"], pids)

    def _evidence_auxiliary_loss(self, features, pids: torch.Tensor) -> torch.Tensor:
        """Compute IET evidence alignment, null-token, and diversity losses."""
        if not isinstance(features, dict):
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        if self.evidence_alignment_loss_weight > 0:
            total = total + self.evidence_alignment_loss_weight * self._evidence_alignment_loss(features, pids)
        if self.evidence_null_loss_weight > 0:
            total = total + self.evidence_null_loss_weight * self._evidence_null_loss(features)
        if self.evidence_diversity_loss_weight > 0:
            total = total + self.evidence_diversity_loss_weight * self._evidence_diversity_loss(features)
        return total

    def _identity_register_diversity_loss(
        self,
        features,
    ) -> torch.Tensor:
        """Discourage Stage-2/3 identity registers from collapsing."""
        zero = torch.zeros((), device=self.device)
        if not self.identity_registers:
            return zero
        if not isinstance(features, dict):
            raise RuntimeError("Identity register training requires feature dictionaries")
        register_tokens = features.get("_identity_register_tokens")
        if not isinstance(register_tokens, tuple) or len(register_tokens) != 2:
            raise RuntimeError("Identity register model did not return Stage-2/3 tokens")
        losses = []
        for tokens in register_tokens:
            if not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[1] != self.identity_register_count:
                raise RuntimeError("Identity register tokens must have shape [B,R,D]")
            normalized = F.normalize(tokens.float(), p=2, dim=-1)
            similarities = torch.einsum(
                "brd,bsd->brs",
                normalized,
                normalized,
            )
            upper = torch.triu(
                torch.ones(
                    self.identity_register_count,
                    self.identity_register_count,
                    device=tokens.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            losses.append(F.relu(similarities[:, upper] - self.identity_register_diversity_margin).mean())
        return torch.stack(losses).mean()

    def _evidence_alignment_loss(self, features: dict, pids: torch.Tensor) -> torch.Tensor:
        """Batch evidence-set OT loss over same-ID positives and different-ID negatives."""
        part_keys = self._sorted_part_feature_keys(features)
        required = ("_visibility", "_rarity", "_role_logits", "_nullness")
        if not part_keys or any(key not in features for key in required):
            return torch.zeros((), device=self.device)

        part_features = torch.stack(
            [F.normalize(features[key], p=2, dim=1) for key in part_keys],
            dim=1,
        )
        visibility = features["_visibility"].to(dtype=part_features.dtype)
        rarity = features["_rarity"].to(dtype=part_features.dtype)
        role_probs = F.softmax(features["_role_logits"].to(dtype=part_features.dtype), dim=-1)
        nullness = features["_nullness"].to(dtype=part_features.dtype)
        if visibility.shape[1] != part_features.shape[1]:
            return torch.zeros((), device=self.device)

        evidence_mass = (visibility * rarity * (1.0 - nullness)).clamp_min(1e-6)
        evidence_mass = evidence_mass / evidence_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)

        part_similarity = torch.einsum("bkd,cld->bckl", part_features, part_features)
        role_compatibility = torch.einsum("bkr,clr->bckl", role_probs, role_probs)
        scores = part_similarity * role_compatibility
        alignment = self._sinkhorn_alignment(
            scores,
            evidence_mass,
            iters=self.evidence_sinkhorn_iters,
            temperature=self.evidence_sinkhorn_temperature,
        )

        same_id = pids[:, None].eq(pids[None, :])
        eye = torch.eye(pids.shape[0], device=pids.device, dtype=torch.bool)
        positive_mask = same_id & ~eye
        negative_mask = ~same_id

        losses = []
        if positive_mask.any():
            losses.append((1.0 - alignment[positive_mask]).mean())
        if negative_mask.any():
            losses.append(F.relu(alignment[negative_mask] - self.evidence_alignment_margin).mean())
        if not losses:
            return torch.zeros((), device=self.device)
        return sum(losses) / len(losses)

    @staticmethod
    def _sinkhorn_alignment(
        scores: torch.Tensor,
        evidence_mass: torch.Tensor,
        *,
        iters: int,
        temperature: float,
    ) -> torch.Tensor:
        """Return pairwise optimal-transport evidence alignment scores."""
        eps = torch.finfo(scores.dtype).eps
        temperature = max(float(temperature), float(eps))
        logits = (scores - scores.amax(dim=(-1, -2), keepdim=True)) / temperature
        kernel = logits.exp().clamp_min(eps)
        row_mass = evidence_mass[:, None, :]
        col_mass = evidence_mass[None, :, :]
        u = torch.ones_like(row_mass).expand(scores.shape[0], scores.shape[1], scores.shape[2])
        v = torch.ones_like(col_mass).expand(scores.shape[0], scores.shape[1], scores.shape[3])
        for _ in range(max(int(iters), 1)):
            u = row_mass / torch.einsum("bckl,bcl->bck", kernel, v).clamp_min(eps)
            v = col_mass / torch.einsum("bckl,bck->bcl", kernel, u).clamp_min(eps)
        plan = u[..., :, None] * kernel * v[..., None, :]
        return (plan * scores).sum(dim=(-1, -2))

    def _evidence_null_loss(self, features: dict) -> torch.Tensor:
        """Supervise the final semantic evidence token as the explicit null slot."""
        nullness = features.get("_nullness")
        if not torch.is_tensor(nullness) or nullness.shape[1] < 2:
            return torch.zeros((), device=self.device)
        target = torch.zeros_like(nullness)
        target[:, -1] = 1.0
        return F.binary_cross_entropy(nullness.clamp(1e-6, 1.0 - 1e-6), target)

    def _evidence_diversity_loss(self, features: dict) -> torch.Tensor:
        """Encourage evidence roles and descriptors to avoid token collapse."""
        part_keys = self._sorted_part_feature_keys(features)
        role_logits = features.get("_role_logits")
        if len(part_keys) < 2 or not torch.is_tensor(role_logits):
            return torch.zeros((), device=self.device)
        role_probs = F.softmax(role_logits, dim=-1)
        role_overlap = torch.einsum("bkr,blr->bkl", role_probs, role_probs)
        part_features = torch.stack(
            [F.normalize(features[key], p=2, dim=1) for key in part_keys],
            dim=1,
        )
        feature_overlap = torch.einsum("bkd,bld->bkl", part_features, part_features).clamp_min(0.0)
        mask = ~torch.eye(len(part_keys), device=role_overlap.device, dtype=torch.bool)
        return 0.5 * (role_overlap[:, mask].mean() + feature_overlap[:, mask].mean())

    def _reduce_branch_losses(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate branch losses using mean (default) or sum."""
        if not losses:
            return torch.zeros((), device=self.device, requires_grad=True)
        if self.branch_loss_agg == "sum":
            return sum(losses)
        return sum(losses) / len(losses)

    def _classification_loss_for_logits(
        self,
        criterion_id: nn.Module,
        logits,
        pids: torch.Tensor,
        epoch: int,
        features=None,
    ) -> torch.Tensor:
        """Compute global CE plus relatively weighted auxiliary-head CE."""
        if not isinstance(logits, list):
            return criterion_id(logits, pids)
        losses = [criterion_id(logit, pids) for logit in logits]
        if len(losses) == 1:
            return losses[0]
        if isinstance(features, dict) and features.get("_classification_loss_aggregation") == "sum":
            return sum(losses)
        aux_weight = self._aux_ce_weight_for_epoch(epoch)
        if self.scale_balanced_branches:
            return self._scale_balanced_classification_loss(losses, aux_weight)
        weights = [torch.ones((), device=losses[0].device, dtype=losses[0].dtype)]
        part_supervision_weights = self._part_supervision_weights(features) if isinstance(features, dict) else None
        if torch.is_tensor(part_supervision_weights):
            for index in range(1, len(losses)):
                part_index = index - 1
                if part_index < part_supervision_weights.shape[1]:
                    weights.append(aux_weight * part_supervision_weights[:, part_index].mean().clamp(min=0.0))
                else:
                    weights.append(torch.as_tensor(aux_weight, device=losses[0].device, dtype=losses[0].dtype))
        else:
            weights.extend(
                torch.as_tensor(aux_weight, device=losses[0].device, dtype=losses[0].dtype) for _ in losses[1:]
            )
        weighted = sum(loss * weight for loss, weight in zip(losses, weights))
        normalizer = torch.stack(weights).sum().clamp(min=1e-12)
        return weighted / normalizer

    def _scale_balanced_classification_loss(
        self,
        losses: list[torch.Tensor],
        aux_weight: float,
    ) -> torch.Tensor:
        """Average CE within each granularity before averaging spatial scales."""
        if self.head_type == "body_slot":
            if len(losses) != 9:
                raise RuntimeError(f"body_slot CE requires one global and eight slot classifiers, got {len(losses)}")
            dtype = losses[0].dtype
            device = losses[0].device
            global_weight = torch.as_tensor(
                self.body_slot_alpha,
                dtype=dtype,
                device=device,
            )
            slot_weight = torch.as_tensor(
                (1.0 - self.body_slot_alpha) * aux_weight,
                dtype=dtype,
                device=device,
            )
            return (global_weight * losses[0] + slot_weight * (sum(losses[1:]) / 8.0)) / (
                global_weight + slot_weight
            ).clamp_min(1e-12)

        local_granularities = tuple(granularity for granularity in self.head_parts if granularity > 1)
        branch_count = 1 + sum(local_granularities)
        if len(losses) < branch_count:
            raise RuntimeError(
                "scale-balanced CE received fewer classifier outputs than head_parts requires: "
                f"got {len(losses)}, expected at least {branch_count} for {self.head_parts}"
            )

        dtype = losses[0].dtype
        device = losses[0].device
        aux = torch.as_tensor(aux_weight, dtype=dtype, device=device)
        if self.head_type == "multiscale_channel2":
            channel_losses = losses[branch_count:]
            if len(channel_losses) != 6:
                raise RuntimeError(
                    "multiscale_channel2 CE requires two channel "
                    "classifiers for each of global/coarse/fine, got "
                    f"{len(channel_losses)}"
                )
            spatial_groups = [
                losses[:1],
                losses[1:3],
                losses[3:7],
            ]
            channel_groups = [channel_losses[index : index + 2] for index in range(0, 6, 2)]
            channel_power = torch.as_tensor(
                self.multiscale_channel_alpha**2,
                dtype=dtype,
                device=device,
            )
            spatial_power = 1.0 - channel_power
            terms = []
            weights = []
            for scale_index, (spatial, channel) in enumerate(zip(spatial_groups, channel_groups, strict=True)):
                scale_weight = torch.ones((), dtype=dtype, device=device) if scale_index == 0 else aux
                terms.append(sum(spatial) / len(spatial))
                weights.append(spatial_power * scale_weight)
                terms.append(sum(channel) / len(channel))
                weights.append(channel_power * scale_weight)
            weighted = sum(loss * weight for loss, weight in zip(terms, weights, strict=True))
            return weighted / torch.stack(weights).sum().clamp(min=1e-12)

        # Classifiers after the configured main global/stripe branches are
        # auxiliary specialists (dropped-global or Stage-2 G/P/C branches).
        # Average that group before combining scales so adding specialists
        # cannot inflate the normalized ID-loss magnitude.
        global_losses = losses[:1] + losses[branch_count:]
        global_weights = [torch.ones((), dtype=dtype, device=device)] + [aux for _ in global_losses[1:]]
        global_loss = sum(loss * weight for loss, weight in zip(global_losses, global_weights, strict=True))
        global_loss = global_loss / torch.stack(global_weights).sum().clamp(min=1e-12)

        scale_losses = [global_loss]
        scale_weights = [torch.ones((), dtype=dtype, device=device)]
        offset = 1
        for granularity in local_granularities:
            group = losses[offset : offset + granularity]
            scale_losses.append(sum(group) / granularity)
            scale_weight = aux
            if granularity == 2:
                scale_weight = scale_weight * self.coarse_branch_ce_weight
            elif granularity == 4:
                scale_weight = scale_weight * self.fine_branch_ce_weight
            scale_weights.append(scale_weight)
            offset += granularity

        weighted = sum(loss * weight for loss, weight in zip(scale_losses, scale_weights, strict=True))
        return weighted / torch.stack(scale_weights).sum().clamp(min=1e-12)

    @staticmethod
    def _part_supervision_weights(features: dict) -> torch.Tensor | None:
        """Detached part weights for auxiliary CE/metric supervision.

        Visibility decides whether a part should contribute. Nullness removes
        explicit null/background evidence tokens from identity supervision.
        Detaching prevents the part metadata heads from minimizing ID/metric
        losses by simply lowering their own weights.
        """
        visibility = features.get("_visibility")
        if not torch.is_tensor(visibility):
            return None
        weights = visibility
        nullness = features.get("_nullness")
        if torch.is_tensor(nullness) and nullness.shape == visibility.shape:
            weights = weights * (1.0 - nullness)
        return weights.detach().clamp(min=0.0)

    @staticmethod
    def _sorted_part_feature_keys(features: dict) -> list[str]:
        """Return part feature keys sorted by numeric suffix: part0, part1, ..."""

        def part_index(key: str) -> int:
            try:
                return int(key[4:])
            except ValueError:
                return 10**9

        return sorted(
            (key for key in features if key.startswith("part") and key[4:].isdigit()),
            key=part_index,
        )

    def _center_features(self, features):
        """Select the center-loss descriptor for the active branch policy."""
        if isinstance(features, dict):
            explicit_features = features.get("_center_features")
            if isinstance(explicit_features, (list, tuple)):
                valid_features = [feature for feature in explicit_features if torch.is_tensor(feature)]
                return torch.cat(valid_features, dim=0) if valid_features else None
            if self.scale_balanced_branches:
                key = self._effective_metric_feature()
                return features.get(key, features.get("raw_mean", features.get("global")))
            return features.get("global", features.get("raw_mean"))
        if isinstance(features, (list, tuple)):
            return features[0] if len(features) > 0 else None
        return features

    def _center_loss_inputs(
        self,
        features,
        pids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, float]:
        """Return center features, matching labels, and model-requested loss scale."""
        center_features = self._center_features(features)
        if center_features is None:
            return None, pids, 1.0
        if center_features.shape[0] == pids.shape[0]:
            center_pids = pids
        else:
            if pids.shape[0] == 0 or center_features.shape[0] % pids.shape[0] != 0:
                raise RuntimeError(
                    "Center feature batch must equal or be an integer multiple of the PID batch: "
                    f"{center_features.shape[0]} vs {pids.shape[0]}"
                )
            center_pids = pids.repeat(center_features.shape[0] // pids.shape[0])
        scale = features.get("_center_loss_scale", 1.0) if isinstance(features, dict) else 1.0
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Center-loss scale must be finite and positive, got {scale}")
        return center_features, center_pids, scale

    def _classification_features(self, features):
        """Select embeddings for margin-based classifier losses."""
        if isinstance(features, dict):
            key = self._effective_metric_feature()
            return features.get(key, features.get("raw_mean", features.get("global")))
        if isinstance(features, (list, tuple)):
            return features[0] if len(features) > 0 else None
        return features

    def _pav_consistency_descriptor(self, features):
        """Select the retrieval descriptor used for clean/mosaic consistency."""
        if isinstance(features, dict):
            descriptor = features.get("norm_concat_bn")
            if torch.is_tensor(descriptor):
                return descriptor
        return self._classification_features(features)
