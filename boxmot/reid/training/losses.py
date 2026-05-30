"""Loss functions for ReID training."""

from __future__ import annotations

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
        margin: Hard margin threshold (ignored when ``soft_margin=True``).
        soft_margin: Use ``log(1 + exp(d_ap - d_an))`` instead of
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
            return F.softplus(dist_ap - dist_an).mean()

        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


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
