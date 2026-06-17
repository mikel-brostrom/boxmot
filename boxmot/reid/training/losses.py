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
        batch_size = inputs.size(0)
        # Compute pairwise distance between features and centers
        dist = (
            torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        dist.addmm_(inputs, self.centers.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)

        # Gather center distances for the correct classes
        classes = torch.arange(self.num_classes, dtype=torch.long, device=inputs.device)
        labels = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        loss = dist * mask.float()
        return loss.sum() / batch_size
