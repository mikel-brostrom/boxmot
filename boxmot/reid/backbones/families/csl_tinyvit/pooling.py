# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ActivatedGeM",
    "DSELitePool",
    "GeM",
    "LearnedPartTokenPool",
    "PatternAdapter",
    "SemanticVisibilityPartPool",
    "SpatialTopDrop",
    "StripeVisibilityGate",
]


class GeM(nn.Module):
    """Generalized mean pooling with optional spatial output size."""

    def __init__(self, output_size: tuple[int, int], p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.output_size = output_size
        initial_p = max(float(p), 1.0 + eps)
        self.raw_p = nn.Parameter(torch.tensor([math.log(math.expm1(initial_p - 1.0))]))
        self.eps = eps

    def effective_p(self) -> torch.Tensor:
        return (1.0 + F.softplus(self.raw_p)).clamp(max=8.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        old_key = f"{prefix}p"
        new_key = f"{prefix}raw_p"
        if old_key in state_dict and new_key not in state_dict:
            p = state_dict[old_key].clamp(min=1.0 + self.eps, max=8.0)
            state_dict[new_key] = torch.log(torch.expm1(p - 1.0))
            del state_dict[old_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.effective_p()
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, self.output_size)
        return x.pow(1.0 / p)


class ActivatedGeM(nn.Sequential):
    """Apply an activation before GeM pooling."""

    def __init__(self, activation: nn.Module, output_size: tuple[int, int]):
        super().__init__(activation, GeM(output_size))


class SpatialTopDrop(nn.Module):
    """Drop top-activation rows in a feature map during training."""

    def __init__(self, h_ratio: float = 0.33):
        super().__init__()
        self.h_ratio = h_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, h, w = x.size()
        rh = max(1, min(h, round(self.h_ratio * h)))
        act = (x**2).sum(1)
        max_act, _ = act.max(2)
        top_rows = torch.argsort(max_act, dim=1)[:, -rh:]
        mask = x.new_ones((b, h))
        for i in range(b):
            mask[i, top_rows[i]] = 0
        mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, c, -1, w)
        return x * mask


class PatternAdapter(nn.Module):
    """Zero-initialized residual adapter for pattern-specific feature maps."""

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"pattern_adapter_dim must be positive, got {hidden_dim}")
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.projection[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.projection(x)


class LearnedPartTokenPool(nn.Module):
    """Pool spatial features with learned queries and a trainable band prior."""

    def __init__(self, channels: int, num_parts: int):
        super().__init__()
        if num_parts < 1:
            raise ValueError(f"num_part_tokens must be positive, got {num_parts}")
        self.channels = channels
        self.num_parts = num_parts
        self.queries = nn.Parameter(torch.empty(num_parts, channels))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(channels)
        self.token_norm = nn.LayerNorm(channels)

        centers = (torch.arange(num_parts, dtype=torch.float32) + 0.5) / num_parts
        initial_width = 0.5 / num_parts
        self.band_centers = nn.Parameter(centers)
        self.band_log_widths = nn.Parameter(torch.full((num_parts,), math.log(math.expm1(initial_width))))
        self.band_log_strength = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))

    def _band_bias(self, height: int, width: int) -> torch.Tensor:
        rows = (torch.arange(height, device=self.queries.device, dtype=self.queries.dtype) + 0.5) / height
        centers = self.band_centers.clamp(1e-3, 1.0 - 1e-3)
        widths = F.softplus(self.band_log_widths).clamp_min(1e-3)
        strength = F.softplus(self.band_log_strength)
        bias = -0.5 * ((rows[None, :] - centers[:, None]) / widths[:, None]).square()
        return (strength * bias)[:, :, None].expand(-1, -1, width).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        keys = self.token_norm(tokens)
        queries = self.query_norm(self.queries)
        logits = torch.einsum("kc,bnc->bkn", queries, keys) / math.sqrt(channels)
        logits = logits + self._band_bias(height, width)[None, :, :]
        attention = logits.softmax(dim=-1)
        pooled = torch.einsum("bkn,bnc->bkc", attention, tokens)
        return pooled.reshape(batch_size, self.num_parts, channels, 1, 1)


class SemanticVisibilityPartPool(nn.Module):
    """Learn semantic-ish evidence tokens with visibility, rarity, and roles."""

    def __init__(self, channels: int, num_parts: int, num_roles: int = 8):
        super().__init__()
        if num_parts < 1:
            raise ValueError(f"num_parts must be positive, got {num_parts}")
        if num_roles < 1:
            raise ValueError(f"num_roles must be positive, got {num_roles}")
        self.num_parts = int(num_parts)
        self.num_roles = int(num_roles)
        self.pool = LearnedPartTokenPool(channels, num_parts)
        self.metadata_norm = nn.LayerNorm(channels)
        self.visibility_predictor = nn.Linear(channels, 1)
        self.rarity_predictor = nn.Linear(channels, 1)
        self.role_predictor = nn.Linear(channels, self.num_roles)
        self.null_predictor = nn.Linear(channels, 1)
        self.reset_metadata_initialization()

    def reset_metadata_initialization(self) -> None:
        """Restore evidence metadata priors after model-wide Linear init."""
        nn.init.zeros_(self.visibility_predictor.weight)
        nn.init.constant_(self.visibility_predictor.bias, math.log(9.0))
        nn.init.zeros_(self.rarity_predictor.weight)
        nn.init.constant_(self.rarity_predictor.bias, 0.0)
        nn.init.trunc_normal_(self.role_predictor.weight, std=0.02)
        nn.init.zeros_(self.role_predictor.bias)
        nn.init.zeros_(self.null_predictor.weight)
        nn.init.constant_(self.null_predictor.bias, math.log(1.0 / 9.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool(x)
        part_tokens = pooled.squeeze(-1).squeeze(-1)
        metadata = self.metadata_norm(part_tokens)
        visibility = torch.sigmoid(self.visibility_predictor(metadata)).squeeze(-1)
        rarity = torch.sigmoid(self.rarity_predictor(metadata)).squeeze(-1)
        role_logits = self.role_predictor(metadata)
        nullness = torch.sigmoid(self.null_predictor(metadata)).squeeze(-1)
        return pooled, visibility, rarity, role_logits, nullness


class DSELitePool(nn.Module):
    """DSE-lite weighted spatial pooling without token pruning or merging."""

    def __init__(self, output_size: tuple[int, int], eps: float = 1e-6):
        super().__init__()
        if len(output_size) != 2 or output_size[1] != 1:
            raise ValueError(f"DSE-lite pooling expects output_size=(parts, 1), got {output_size}")
        self.output_size = tuple(int(value) for value in output_size)
        self.eps = float(eps)

    def _center_gaussian_prior(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = self.output_size[0]
        rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        centers = (torch.arange(parts, device=device, dtype=dtype) + 0.5) / parts
        sigma = max(0.5 / parts, self.eps)
        prior = torch.exp(-0.5 * ((rows[None, :] - centers[:, None]) / sigma) ** 2)
        return prior[:, :, None].expand(parts, height, width)

    def _entropy_inverse_attention_score(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels <= 1:
            return x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        probabilities = torch.softmax(x.float().square(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
        inverse = 1.0 - entropy / math.log(channels)
        return inverse.clamp_min(self.eps).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        prior = self._center_gaussian_prior(height, width, x.device, x.dtype)
        token_score = self._entropy_inverse_attention_score(x)
        weights = token_score[:, None, :, :, :] * prior[None, :, None, :, :]
        denominator = weights.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        weights = (weights / denominator).squeeze(2)
        pooled = torch.einsum("bghw,bchw->bcg", weights, x)
        return pooled.reshape(batch_size, x.shape[1], self.output_size[0], 1)


class StripeVisibilityGate(nn.Module):
    """Predict a confidence for each pooled stripe from its local feature."""

    def __init__(self, channels: int, num_stripes: int):
        super().__init__()
        if num_stripes < 1:
            raise ValueError(f"num_stripes must be positive, got {num_stripes}")
        self.num_stripes = int(num_stripes)
        self.norm = nn.LayerNorm(channels)
        self.predictor = nn.Linear(channels, 1)
        self.reset_visibility_initialization()

    def reset_visibility_initialization(self) -> None:
        """Restore high initial stripe visibility after model-wide Linear init."""
        nn.init.zeros_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias, math.log(9.0))

    def forward(self, pooled_stripes: torch.Tensor) -> torch.Tensor:
        """Return sigmoid confidences with shape ``(batch, num_stripes)``."""
        if pooled_stripes.ndim != 4 or pooled_stripes.shape[2] != self.num_stripes:
            raise ValueError(
                f"Expected pooled stripes shaped (B, C, {self.num_stripes}, 1), got {tuple(pooled_stripes.shape)}"
            )
        stripe_tokens = pooled_stripes.squeeze(-1).transpose(1, 2)
        return torch.sigmoid(self.predictor(self.norm(stripe_tokens))).squeeze(-1)
