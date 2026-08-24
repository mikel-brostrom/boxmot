"""Batch-normalization calibration and ranking evaluation."""

from __future__ import annotations

import torch

from boxmot.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
    visibility_part_count,
)
from boxmot.reid.training.trainer_components.types import (
    ValMetrics,
)


class _ValidationMixin:
    @torch.no_grad()
    def _calibrate_bn(self, model, data_loader, num_batches: int = 50):
        """Run forward passes to calibrate BN running stats for the EMA model."""
        model.train()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                imgs = batch[0]
                imgs = imgs.to(self.device)
                model(imgs)
        model.eval()

    def _validate(self, epoch, model, query_loader, gallery_loader) -> ValMetrics:
        recipe = self._resolve_training_recipe(model)
        use_flip = self.flip_tta if self.flip_tta is not None else recipe.default_flip_tta
        visibility_distance = self.inference_feature == "visibility_weighted_parts"
        evidence_distance = self.inference_feature == "evidence_sinkhorn"
        late_interaction_distance = self.hierarchical_late_interaction
        structured_distance = visibility_distance or evidence_distance or late_interaction_distance
        head = self._model_head(model)
        matcher = getattr(head, "late_interaction_matcher", None)
        if late_interaction_distance and matcher is None:
            raise RuntimeError("late-interaction validation requires the matcher in the model head")
        previous_packet_mode = getattr(head, "emit_late_interaction_packet", False)
        if late_interaction_distance:
            head.emit_late_interaction_packet = True
        try:
            q_feats, q_pids, q_camids = extract_features(
                model,
                query_loader,
                self.device,
                desc="Query",
                flip_tta=use_flip,
                normalize=not structured_distance,
            )
            g_feats, g_pids, g_camids = extract_features(
                model,
                gallery_loader,
                self.device,
                desc="Gallery",
                flip_tta=use_flip,
                normalize=not structured_distance,
            )
        finally:
            if late_interaction_distance:
                head.emit_late_interaction_packet = previous_packet_mode
        distmat = compute_distance_matrix(
            q_feats,
            g_feats,
            metric=(
                "evidence_sinkhorn"
                if evidence_distance
                else "hierarchical_late_interaction"
                if late_interaction_distance
                else "visibility_weighted_parts"
                if visibility_distance
                else "cosine"
            ),
            part_dim=self.feat_dim if structured_distance else None,
            part_count=visibility_part_count(self.head_parts) if structured_distance else None,
            role_count=self.evidence_num_roles if evidence_distance else None,
            beta=self.branch_metric_part_weight,
            topk=(
                self.evidence_rerank_topk
                if evidence_distance
                else self.late_interaction_rerank_topk
                if late_interaction_distance
                else None
            ),
            sinkhorn_iters=self.evidence_sinkhorn_iters,
            sinkhorn_temperature=self.evidence_sinkhorn_temperature,
            late_interaction_matcher=matcher if late_interaction_distance else None,
            base_dim=3 * self.feat_dim if late_interaction_distance else None,
            branch_dims=(
                (
                    self.feat_dim,
                    self.feat_dim // 2,
                    self.feat_dim // 2,
                    self.feat_dim // 4,
                    self.feat_dim // 4,
                    self.feat_dim // 4,
                    self.feat_dim // 4,
                )
                if late_interaction_distance
                else None
            ),
        )
        del q_feats, g_feats
        cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)
        del distmat, q_pids, g_pids, q_camids, g_camids
        return ValMetrics(
            epoch=epoch,
            mAP=mAP,
            rank1=float(cmc[0]) if len(cmc) > 0 else 0.0,
            rank5=float(cmc[4]) if len(cmc) > 4 else 0.0,
            rank10=float(cmc[9]) if len(cmc) > 9 else 0.0,
        )
