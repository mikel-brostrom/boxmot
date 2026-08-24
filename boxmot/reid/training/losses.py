"""Loss functions for ReID training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross-entropy loss with label smoothing.

    Reference:
        Szegedy et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. "In Defense of the Triplet Loss for Person Re-Identification." arXiv 2017.

    Args:
        margin: Margin offset applied to both hard-margin and soft-margin forms.
        soft_margin: Use ``log(1 + exp(d_ap - d_an + margin))`` instead of
            ``max(0, d_ap - d_an + margin)``.  Provides smoother gradients.
    """

    def __init__(self, margin: float = 0.3, soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        if not soft_margin:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)
        # Pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        if self.soft_margin:
            return F.softplus(dist_ap - dist_an + self.margin).mean()

        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class WeightedRegularizedTripletLoss(nn.Module):
    """Triplet loss with soft weighting over every valid in-batch pair.

    Harder positives receive larger weights according to their distance, while
    closer negatives receive larger weights according to their inverse
    distance. Unlike batch-hard mining, this keeps informative gradients from
    every pair and prevents one outlier from solely determining an anchor's
    metric-learning update.

    Reference:
        Ye et al. "Deep Learning for Person Re-identification: A Survey and
        Outlook." TPAMI 2022 (AGW weighted regularization triplet loss).
    """

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"inputs must have shape (batch, features), got {tuple(inputs.shape)}")
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("targets must have shape (batch,) and match inputs")

        distances = torch.cdist(inputs, inputs, p=2)
        same_identity = targets[:, None].eq(targets[None, :])
        positive_mask = same_identity.clone()
        positive_mask.fill_diagonal_(False)
        negative_mask = ~same_identity

        valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid_anchors.any():
            # Preserve a differentiable zero so unusual batches can still run
            # through backward without special handling in the trainer.
            return inputs.sum() * 0.0

        distances = distances[valid_anchors]
        positive_mask = positive_mask[valid_anchors]
        negative_mask = negative_mask[valid_anchors]

        positive_weights = F.softmax(
            distances.masked_fill(~positive_mask, -torch.inf),
            dim=1,
        )
        negative_weights = F.softmax(
            (-distances).masked_fill(~negative_mask, -torch.inf),
            dim=1,
        )
        weighted_positive = (positive_weights * distances).sum(dim=1)
        weighted_negative = (negative_weights * distances).sum(dim=1)
        return F.softplus(weighted_positive - weighted_negative).mean()


class CrossScaleMajorityMarginLoss(nn.Module):
    """Require a majority of three descriptor scales to rank each pair correctly.

    Triplets are mined once with the complete equal-energy descriptor. The
    resulting positive and negative pairs are then shared by the global,
    coarse, and fine scales, and the median scale margin is optimized.
    """

    def __init__(
        self,
        margin: float = 0.10,
        temperature: float = 0.05,
        topk_negatives: int = 8,
    ) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if topk_negatives < 1:
            raise ValueError("topk_negatives must be positive")
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.topk_negatives = int(topk_negatives)

    def forward(
        self,
        scale_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        mining_descriptor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(scale_features) != 3:
            raise ValueError("CSMM requires exactly global, coarse, and fine scale features")
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape (batch,), got {tuple(labels.shape)}")

        batch_size = labels.shape[0]
        for feature in scale_features:
            if feature.ndim != 2 or feature.shape[0] != batch_size:
                raise ValueError(
                    "Each CSMM scale must have shape (batch, features) and match labels; "
                    f"got {tuple(feature.shape)} for batch={batch_size}"
                )
        if batch_size < 2:
            return scale_features[0].sum() * 0.0

        scales = tuple(F.normalize(feature, p=2, dim=1) for feature in scale_features)
        scale_dims = {feature.shape[1] for feature in scales}
        if len(scale_dims) != 1:
            raise ValueError(
                "CSMM scale groups must have equal descriptor dimensions, got "
                f"{tuple(feature.shape[1] for feature in scales)}"
            )
        if mining_descriptor is None:
            full_descriptor = F.normalize(torch.cat(scales, dim=1), p=2, dim=1)
        else:
            if mining_descriptor.ndim != 2 or mining_descriptor.shape[0] != batch_size:
                raise ValueError(
                    "mining_descriptor must have shape (batch, features) and match labels; "
                    f"got {tuple(mining_descriptor.shape)} for batch={batch_size}"
                )
            full_descriptor = F.normalize(mining_descriptor, p=2, dim=1)

        same_identity = labels[:, None].eq(labels[None, :])
        positive_mask = same_identity.clone()
        positive_mask.fill_diagonal_(False)
        negative_mask = ~same_identity
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return full_descriptor.sum() * 0.0

        valid_rows = torch.where(valid)[0]
        with torch.no_grad():
            full_similarity = full_descriptor.detach() @ full_descriptor.detach().T
            positive_indices = full_similarity.masked_fill(
                ~positive_mask,
                torch.inf,
            ).argmin(dim=1)
            available_negatives = int(negative_mask[valid].sum(dim=1).min().item())
            num_negatives = min(self.topk_negatives, available_negatives)
            negative_indices = full_similarity.masked_fill(
                ~negative_mask,
                -torch.inf,
            ).topk(k=num_negatives, dim=1).indices

        positive_indices = positive_indices[valid]
        negative_indices = negative_indices[valid]
        scale_margins = []
        for scale in scales:
            similarity = scale @ scale.T
            positive_similarity = similarity[valid_rows, positive_indices].unsqueeze(1)
            negative_similarity = similarity[valid_rows].gather(1, negative_indices)
            scale_margins.append(positive_similarity - negative_similarity)

        majority_margin = torch.stack(scale_margins, dim=-1).median(dim=-1).values
        return (
            self.temperature
            * F.softplus((self.margin - majority_margin) / self.temperature)
        ).mean()


class TreeBoostAPLoss(nn.Module):
    """Train a global → two-part → four-part hierarchy as residual rank refinements.

    Global, coarse, and fine retrieval scores are supervised stagewise with a
    camera-valid SmoothAP approximation. Coarser score matrices are detached
    in later stages so each finer level learns only the residual ranking errors
    left by its parent level.
    """

    def __init__(
        self,
        coarse_coefficient: float = 1.0,
        fine_coefficient: float = 1.0,
        node_coefficient: float = 0.25,
        regression_coefficient: float = 0.10,
        difficulty_floor: float = 0.25,
        regression_tolerance: float = 0.02,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        for name, value in (
            ("coarse_coefficient", coarse_coefficient),
            ("fine_coefficient", fine_coefficient),
            ("node_coefficient", node_coefficient),
            ("regression_coefficient", regression_coefficient),
            ("regression_tolerance", regression_tolerance),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= difficulty_floor <= 1:
            raise ValueError("difficulty_floor must satisfy 0 <= value <= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.coarse_coefficient = float(coarse_coefficient)
        self.fine_coefficient = float(fine_coefficient)
        self.node_coefficient = float(node_coefficient)
        self.regression_coefficient = float(regression_coefficient)
        self.difficulty_floor = float(difficulty_floor)
        self.regression_tolerance = float(regression_tolerance)
        self.temperature = float(temperature)

    def _camera_valid_soft_ap(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-anchor SmoothAP losses and a cross-camera-valid mask."""
        batch_size = labels.shape[0]
        same_identity = labels[:, None].eq(labels[None, :])
        different_camera = camera_ids[:, None].ne(camera_ids[None, :])
        positive = same_identity & different_camera
        positive.fill_diagonal_(False)
        negative = ~same_identity
        valid = positive.any(dim=1) & negative.any(dim=1)

        # Dimensions are [anchor, candidate, ranked-positive]. Same-camera
        # positives are evaluation junk and are absent from both masks.
        differences = (scores[:, :, None] - scores[:, None, :]) / self.temperature
        soft_precedes = torch.sigmoid(differences)
        off_diagonal = ~torch.eye(batch_size, device=scores.device, dtype=torch.bool)
        ranked_positive = positive[:, None, :]
        candidate = (positive | negative)[:, :, None]
        comparison_mask = off_diagonal[None, :, :]

        rank_all = 1.0 + (
            soft_precedes
            * (candidate & ranked_positive & comparison_mask).to(soft_precedes.dtype)
        ).sum(dim=1)
        rank_positive = 1.0 + (
            soft_precedes
            * (positive[:, :, None] & ranked_positive & comparison_mask).to(soft_precedes.dtype)
        ).sum(dim=1)
        positive_count = positive.sum(dim=1).clamp_min(1)
        average_precision = (
            (rank_positive / rank_all) * positive.to(rank_all.dtype)
        ).sum(dim=1) / positive_count
        losses = (1.0 - average_precision).masked_fill(~valid, 0.0)
        return losses, valid

    @staticmethod
    def _cosine_matrix(features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(features, p=2, dim=1)
        return normalized @ normalized.T

    def forward(
        self,
        hierarchy_features: tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        labels: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        if len(hierarchy_features) != 3:
            raise ValueError("TreeBoost-AP requires global, coarse, and fine hierarchy features")
        global_feature, coarse_features, fine_features = hierarchy_features
        if len(coarse_features) != 2 or len(fine_features) != 4:
            raise ValueError("TreeBoost-AP requires exactly one global, two coarse, and four fine branches")
        if labels.ndim != 1 or camera_ids.ndim != 1 or labels.shape != camera_ids.shape:
            raise ValueError("labels and camera_ids must have matching shape (batch,)")
        batch_size = labels.shape[0]
        all_features = (global_feature, *coarse_features, *fine_features)
        if any(feature.ndim != 2 or feature.shape[0] != batch_size for feature in all_features):
            raise ValueError("Every TreeBoost-AP feature must have shape (batch, features)")

        sim_global = self._cosine_matrix(global_feature)
        sim_coarse_branches = tuple(self._cosine_matrix(feature) for feature in coarse_features)
        sim_fine_branches = tuple(self._cosine_matrix(feature) for feature in fine_features)
        sim_coarse = sum(sim_coarse_branches) / 2.0
        sim_fine = sum(sim_fine_branches) / 4.0

        score_global = sim_global
        score_coarse = 0.5 * sim_global.detach() + 0.5 * sim_coarse
        score_fine = sim_global.detach() / 3.0 + sim_coarse.detach() / 3.0 + sim_fine / 3.0

        loss_global, valid = self._camera_valid_soft_ap(score_global, labels, camera_ids)
        if not valid.any():
            return sum(feature.sum() for feature in all_features) * 0.0
        loss_coarse, _ = self._camera_valid_soft_ap(score_coarse, labels, camera_ids)
        loss_fine, _ = self._camera_valid_soft_ap(score_fine, labels, camera_ids)

        difficulty_global = self.difficulty_floor + (1.0 - self.difficulty_floor) * loss_global.detach()
        difficulty_coarse = self.difficulty_floor + (1.0 - self.difficulty_floor) * loss_coarse.detach()
        stage_loss = (
            loss_global
            + self.coarse_coefficient * difficulty_global * loss_coarse
            + self.fine_coefficient * difficulty_coarse * loss_fine
        )[valid].mean()

        node_losses = []
        for parent_similarity, child_similarities in zip(
            sim_coarse_branches,
            (sim_fine_branches[:2], sim_fine_branches[2:]),
            strict=True,
        ):
            child_similarity = sum(child_similarities) / 2.0
            parent_loss, _ = self._camera_valid_soft_ap(
                parent_similarity.detach(),
                labels,
                camera_ids,
            )
            node_score = 0.5 * parent_similarity.detach() + 0.5 * child_similarity
            node_loss, _ = self._camera_valid_soft_ap(node_score, labels, camera_ids)
            node_difficulty = self.difficulty_floor + (1.0 - self.difficulty_floor) * parent_loss.detach()
            node_losses.append((node_difficulty * node_loss)[valid].mean())
        node_loss = torch.stack(node_losses).mean()

        regression_loss = (
            F.relu(loss_coarse - loss_global.detach() - self.regression_tolerance)
            + F.relu(loss_fine - loss_coarse.detach() - self.regression_tolerance)
        )[valid].mean()
        return (
            stage_loss
            + self.node_coefficient * node_loss
            + self.regression_coefficient * regression_loss
        )


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity loss for metric learning.

    Reference:
        Wang et al. "Multi-Similarity Loss with General Pair Weighting
        for Deep Metric Learning." CVPR 2019.

    Exploits self-similarity and relative similarity via soft pair weighting,
    combining the strengths of contrastive, triplet, and lifted-structure losses.

    Args:
        alpha: Scale for positive pair weighting (higher → focus on harder positives).
        beta: Scale for negative pair weighting (higher → focus on harder negatives).
        thresh: Threshold (lambda) in the exp weighting formula.
        mining_margin: Margin epsilon for informative pair mining.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        thresh: float = 0.5,
        mining_margin: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.thresh = thresh
        self.mining_margin = mining_margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: L2-normalized embeddings (N, D)
        batch_size = inputs.size(0)
        # Cosine similarity matrix
        sim = inputs @ inputs.t()

        epsilon = 1e-5
        loss = []

        for i in range(batch_size):
            # Positive similarities (excluding self, which has sim ≈ 1)
            pos_sim = sim[i][targets == targets[i]]
            pos_sim = pos_sim[pos_sim < 1 - epsilon]

            # Negative similarities
            neg_sim = sim[i][targets != targets[i]]

            if pos_sim.numel() == 0 or neg_sim.numel() == 0:
                continue

            # Multi-Similarity mining: select informative pairs
            # Negatives closer than the easiest positive (with margin)
            neg_pairs = neg_sim[neg_sim + self.mining_margin > pos_sim.min()]
            # Positives farther than the closest negative (with margin)
            pos_pairs = pos_sim[pos_sim - self.mining_margin < neg_sim.max()]

            if neg_pairs.numel() < 1 or pos_pairs.numel() < 1:
                continue

            # Positive term: (1/alpha) * log[1 + sum exp(-alpha * (s_p - thresh))]
            pos_term = (1.0 / self.alpha) * torch.log(
                1.0 + torch.exp(-self.alpha * (pos_pairs - self.thresh)).sum()
            )
            # Negative term: (1/beta) * log[1 + sum exp(beta * (s_n - thresh))]
            neg_term = (1.0 / self.beta) * torch.log(
                1.0 + torch.exp(self.beta * (neg_pairs - self.thresh)).sum()
            )

            loss.append(pos_term + neg_term)

        if len(loss) == 0:
            return torch.zeros([], device=inputs.device, requires_grad=True)

        return sum(loss) / batch_size


class CircleLoss(nn.Module):
    """Circle loss for pair-similarity optimization.

    Reference:
        Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity
        Optimization." CVPR 2020.

    Args:
        margin: Similarity margin ``m``.
        gamma: Logit scale ``gamma``.
    """

    def __init__(self, margin: float = 0.25, gamma: float = 64.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.softplus = nn.Softplus()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.normalize(inputs, p=2, dim=1)
        sim = inputs @ inputs.t()
        targets = targets.view(-1, 1)
        pos_mask = targets.eq(targets.t())
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)

        losses = []
        delta_p = 1.0 - self.margin
        delta_n = self.margin
        for i in range(inputs.size(0)):
            sp = sim[i][pos_mask[i]]
            sn = sim[i][neg_mask[i]]
            if sp.numel() == 0 or sn.numel() == 0:
                continue

            alpha_p = torch.clamp_min(-sp.detach() + 1.0 + self.margin, 0.0)
            alpha_n = torch.clamp_min(sn.detach() + self.margin, 0.0)
            logit_p = -self.gamma * alpha_p * (sp - delta_p)
            logit_n = self.gamma * alpha_n * (sn - delta_n)
            losses.append(self.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)))

        if not losses:
            return torch.zeros([], device=inputs.device, requires_grad=True)
        return torch.stack(losses).mean()


class AdaSPLoss(nn.Module):
    """Numerically stable adaptive sparse pairwise loss.

    This is the adaptive SP formulation from Zhou et al., CVPR 2023. It
    operates on one descriptor per image and forms one sparse loss item per
    identity. Unlike the reference implementation, identities need not be
    contiguous in the batch and class sizes may differ.
    """

    def __init__(self, temperature: float = 0.04) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("AdaSP temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("AdaSP inputs must have shape [batch, features]")
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("AdaSP targets must have shape [batch]")

        working_inputs = (
            inputs.float()
            if inputs.dtype in {torch.float16, torch.bfloat16}
            else inputs
        )
        features = F.normalize(working_inputs, p=2, dim=1)
        similarities = features @ features.transpose(0, 1)
        scaled = similarities / self.temperature
        identities = torch.unique(targets, sorted=True)
        class_indices = [
            torch.nonzero(targets == identity, as_tuple=False).flatten()
            for identity in identities
        ]
        if len(class_indices) < 2 or any(
            indices.numel() < 2 for indices in class_indices
        ):
            return inputs.sum() * 0.0

        hard_hard_log_positives = []
        hard_easy_log_positives = []
        negative_logits: list[list[torch.Tensor]] = []
        for anchor_indices in class_indices:
            within_class = scaled[anchor_indices][:, anchor_indices]
            hard_hard_log_positives.append(
                -torch.logsumexp(-within_class.flatten(), dim=0)
            )
            per_anchor_harmonic = -torch.logsumexp(
                -within_class,
                dim=1,
            )
            hard_easy_log_positives.append(
                torch.logsumexp(per_anchor_harmonic, dim=0)
            )
            row = []
            for gallery_indices in class_indices:
                row.append(
                    torch.logsumexp(
                        scaled[anchor_indices][:, gallery_indices].flatten(),
                        dim=0,
                    )
                )
            negative_logits.append(row)

        hard_hard_log = torch.stack(hard_hard_log_positives)
        hard_easy_log = torch.stack(hard_easy_log_positives)
        hard_hard_similarity = hard_hard_log * self.temperature
        hard_easy_similarity = hard_easy_log * self.temperature
        denominator = hard_hard_similarity + hard_easy_similarity
        epsilon = torch.finfo(denominator.dtype).eps
        adaptive_weight = torch.where(
            denominator.abs() > epsilon,
            2.0
            * hard_hard_similarity
            * hard_easy_similarity
            / denominator,
            torch.zeros_like(denominator),
        )
        adaptive_weight = torch.where(
            hard_hard_similarity < 0,
            torch.zeros_like(adaptive_weight),
            adaptive_weight,
        ).detach()
        adaptive_log_positive = (
            adaptive_weight * hard_hard_log
            + (1.0 - adaptive_weight) * hard_easy_log
        )

        losses = []
        for class_index, log_positive in enumerate(adaptive_log_positive):
            competing = [log_positive]
            competing.extend(
                negative_logits[class_index][other_index]
                for other_index in range(len(class_indices))
                if other_index != class_index
            )
            losses.append(torch.logsumexp(torch.stack(competing), dim=0) - log_positive)
        return torch.stack(losses).mean().to(dtype=inputs.dtype)


class ArcFaceLoss(nn.Module):
    """Additive angular-margin classifier loss.

    The classifier weights are local to the training criterion. They are not
    needed for ReID inference, which uses the backbone embedding directly.
    """

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(inputs, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt((1.0 - cosine.square()).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, targets)


class CosFaceLoss(nn.Module):
    """Additive cosine-margin classifier loss."""

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(inputs, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        logits = (cosine - one_hot * self.margin) * self.scale
        return F.cross_entropy(logits, targets)


# Registry of metric losses (beyond CE).  Maps name → (class, default kwargs).
METRIC_LOSS_REGISTRY: dict[str, type] = {
    "triplet": TripletLoss,
    "wrt": WeightedRegularizedTripletLoss,
    "ms": MultiSimilarityLoss,
    "circle": CircleLoss,
}


class CenterLoss(nn.Module):
    """Center loss for discriminative feature learning.

    Reference:
        Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition." ECCV 2016.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_centers = self.centers.index_select(0, targets.long())
        distances = (inputs - target_centers).square().sum(dim=1).clamp_min(1e-12)
        return distances.mean()
