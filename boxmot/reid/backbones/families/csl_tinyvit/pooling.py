# BoxMOT AGPL-3.0 license

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ActivatedGeM",
    "AnatomicalAuxiliaryPool",
    "DSELitePool",
    "EMAAnatomicalAuxiliaryPool",
    "GeM",
    "LearnedPartTokenPool",
    "PatternAdapter",
    "PrivilegedMaskPoseAttentionAdapter",
    "SemanticVisibilityPartPool",
    "SpatialTopDrop",
    "SpatialTopSuppression",
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


class PrivilegedMaskPoseAttentionAdapter(nn.Module):
    """Learn RGB-only foreground/part attention from training annotations.

    The pose and person mask are never inputs to this module. They supervise
    its outputs in the trainer, after which the adapter uses RGB features alone
    to apply a bounded residual spatial gate.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_parts: int = 6,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if channels < 1 or num_parts < 1:
            raise ValueError("attention adapter channels and part count must be positive")
        if not 0 < max_scale <= 1:
            raise ValueError("attention adapter max_scale must satisfy 0 < value <= 1")
        hidden_channels = min(64, max(16, channels // 4))
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.num_parts = int(num_parts)
        self.max_scale = float(max_scale)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.foreground_predictor = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )
        self.part_predictor = nn.Conv2d(
            hidden_channels,
            self.num_parts,
            kernel_size=1,
            bias=False,
        )
        self.visibility_norm = nn.LayerNorm(channels)
        self.visibility_predictor = nn.Linear(channels, 1)
        self.foreground_gate_logit = nn.Parameter(torch.zeros(()))
        self.part_gate_logit = nn.Parameter(torch.zeros(()))
        self.gate_active = True
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Start with the exact ungated RGB feature map."""
        with torch.no_grad():
            self.foreground_gate_logit.zero_()
            self.part_gate_logit.zero_()
        nn.init.zeros_(self.visibility_predictor.weight)
        nn.init.constant_(self.visibility_predictor.bias, math.log(4.0))

    def set_gate_active(self, active: bool) -> None:
        """Enable retrieval gating without disabling attention prediction."""
        self.gate_active = bool(active)

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.encoder(feature_map)
        foreground_logits = self.foreground_predictor(hidden)
        part_logits = self.part_predictor(hidden)
        part_attention = part_logits.flatten(2).softmax(dim=-1).reshape_as(
            part_logits
        )
        spatial_tokens = feature_map.flatten(2).transpose(1, 2)
        part_tokens = torch.einsum(
            "bpn,bnc->bpc",
            part_attention.flatten(2),
            spatial_tokens,
        )
        visibility_logits = self.visibility_predictor(
            self.visibility_norm(part_tokens)
        ).squeeze(-1)

        foreground_probability = foreground_logits.sigmoid()
        # The spatial-softmax target determines each part map only up to an
        # additive per-part logit offset. Remove that unconstrained offset
        # before using the logits for retrieval gating, otherwise a predictor
        # bias invisible to the pose loss can arbitrarily change inference.
        centered_part_logits = part_logits - part_logits.mean(
            dim=(-1, -2),
            keepdim=True,
        )
        part_visibility = visibility_logits.sigmoid()[:, :, None, None]
        # Invisible parts contribute neutral evidence (0.5), while visible
        # parts expose the spatial contrast learned from pose supervision.
        part_evidence = 0.5 + (
            centered_part_logits.sigmoid() - 0.5
        ) * part_visibility
        part_probability = part_evidence.amax(
            dim=1,
            keepdim=True,
        )
        foreground_signal = 2.0 * foreground_probability - 1.0
        part_signal = 2.0 * part_probability - 1.0
        # Learn mask and pose contributions independently while sharing one
        # L1 residual budget. Either cue can use the full budget on its own;
        # when both are strong, normalization keeps their combined absolute
        # contribution bounded by max_scale.
        raw_foreground_scale = self.foreground_gate_logit.tanh()
        raw_part_scale = self.part_gate_logit.tanh()
        scale_normalizer = (
            raw_foreground_scale.abs() + raw_part_scale.abs()
        ).clamp_min(1.0)
        foreground_gate_scale = (
            self.max_scale
            * raw_foreground_scale
            / scale_normalizer
        )
        part_gate_scale = (
            self.max_scale * raw_part_scale / scale_normalizer
        )
        if not self.gate_active:
            foreground_gate_scale = foreground_gate_scale * 0.0
            part_gate_scale = part_gate_scale * 0.0
        gate_delta = (
            foreground_gate_scale * foreground_signal
            + part_gate_scale * part_signal
        )
        gated = feature_map * (1.0 + gate_delta)
        gate_scale = foreground_gate_scale + part_gate_scale
        return (
            gated,
            part_tokens,
            part_attention,
            visibility_logits,
            foreground_logits,
            gate_scale,
        )


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


class SpatialTopSuppression(nn.Module):
    """Suppress the most active horizontal rows in both train and eval modes."""

    def __init__(self, h_ratio: float = 0.25):
        super().__init__()
        if not 0 < h_ratio <= 1:
            raise ValueError(f"h_ratio must satisfy 0 < value <= 1, got {h_ratio}")
        self.h_ratio = float(h_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        rows_to_suppress = max(1, min(height, round(self.h_ratio * height)))
        row_energy = x.square().sum(dim=1).amax(dim=2)
        suppressed_rows = row_energy.topk(rows_to_suppress, dim=1).indices
        row_mask = x.new_ones((x.shape[0], height))
        row_mask.scatter_(1, suppressed_rows, 0)
        return x * row_mask[:, None, :, None]


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


class AnatomicalAuxiliaryPool(nn.Module):
    """Pool scale-adapted anatomical cell tokens from RGB feature maps.

    Pose and mask metadata supervise the returned attention maps in the
    trainer. Keeping privileged geometry outside this module prevents the RGB
    student from bypassing pose through a learned concatenation projection.
    """

    _INITIAL_CENTERS = (0.12, 0.38, 0.40, 0.40, 0.72, 0.72)
    _INITIAL_WIDTHS = (0.16, 0.24, 0.28, 0.28, 0.30, 0.30)
    _CANONICAL_GRID_SIZE = (4, 2)
    _CANONICAL_CELLS = 8
    _SCALE_COUNT = 2

    def __init__(
        self,
        channels: int,
        token_dim: int = 128,
        num_parts: int = 6,
        descriptor_dim: int | None = None,
        *,
        fine_channels: int | None = None,
        multiscale: bool = False,
    ) -> None:
        super().__init__()
        if num_parts != len(self._INITIAL_CENTERS):
            raise ValueError("The anatomical auxiliary pool currently requires six ordered body parts")
        if channels < 1 or token_dim < 2 * self._CANONICAL_CELLS:
            raise ValueError(
                "anatomical channels must be positive and token_dim must "
                "provide at least two channels for each canonical grid cell"
            )
        if token_dim % self._CANONICAL_CELLS:
            raise ValueError(
                "anatomical token_dim must be divisible by the eight "
                "canonical grid cells"
            )
        self.num_parts = int(num_parts)
        self.token_dim = int(token_dim)
        self.cell_dim = self.token_dim // self._CANONICAL_CELLS
        self.multiscale_enabled = bool(multiscale)
        fine_channels = channels if fine_channels is None else int(fine_channels)
        if fine_channels < 1:
            raise ValueError("fine anatomical channels must be positive")
        self.feature_projection = nn.Conv2d(
            channels,
            self.cell_dim,
            kernel_size=1,
            bias=False,
        )
        self.role_queries = nn.Parameter(
            torch.empty(num_parts, self.cell_dim)
        )
        self.cell_embeddings = nn.Parameter(
            torch.empty(self._CANONICAL_CELLS, self.cell_dim)
        )
        # The shared role/cell basis preserves correspondence. Zero-initialized
        # offsets let local and fine maps specialize only when their different
        # resolutions provide evidence that specialization is useful.
        self.scale_query_offsets = nn.Parameter(
            torch.zeros(
                self._SCALE_COUNT,
                num_parts,
                self.cell_dim,
            )
        )
        self.query_norms = nn.ModuleList(
            nn.LayerNorm(self.cell_dim)
            for _ in range(self._SCALE_COUNT)
        )
        self.key_norms = nn.ModuleList(
            nn.LayerNorm(self.cell_dim)
            for _ in range(self._SCALE_COUNT)
        )
        self.visibility_norms = nn.ModuleList(
            nn.LayerNorm(token_dim)
            for _ in range(self._SCALE_COUNT)
        )
        self.visibility_predictors = nn.ModuleList(
            nn.Linear(token_dim, 1)
            for _ in range(self._SCALE_COUNT)
        )
        self.descriptor_projection = (
            nn.Linear(
                int(descriptor_dim),
                num_parts * token_dim,
                bias=False,
            )
            if descriptor_dim is not None
            else None
        )
        nn.init.trunc_normal_(self.role_queries, std=0.02)
        nn.init.trunc_normal_(self.cell_embeddings, std=0.02)
        self.band_centers = nn.Parameter(torch.tensor(self._INITIAL_CENTERS, dtype=torch.float32))
        widths = torch.tensor(self._INITIAL_WIDTHS, dtype=torch.float32)
        self.band_log_widths = nn.Parameter(torch.log(torch.expm1(widths)))
        self.band_log_strength = nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))
        self.reset_visibility_initialization()
        self.fine_feature_projection = (
            nn.Conv2d(
                fine_channels,
                self.cell_dim,
                kernel_size=1,
                bias=False,
            )
            if self.multiscale_enabled
            else None
        )

    def reset_visibility_initialization(self) -> None:
        """Initialize visibility as mostly present without blocking learning."""
        for predictor in self.visibility_predictors:
            nn.init.zeros_(predictor.weight)
            nn.init.constant_(
                predictor.bias,
                math.log(4.0),
            )

    def _band_bias(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        centers = self.band_centers.to(dtype=dtype).clamp(1e-3, 1.0 - 1e-3)
        widths = F.softplus(self.band_log_widths.to(dtype=dtype)).clamp_min(1e-3)
        strength = F.softplus(self.band_log_strength.to(dtype=dtype))
        bias = -0.5 * ((rows[None, :] - centers[:, None]) / widths[:, None]).square()
        return (strength * bias)[:, :, None].expand(-1, -1, width)

    def _student_outputs(
        self,
        projected: torch.Tensor,
        *,
        scale_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool explicit role/cell tokens from one spatial feature scale."""
        if not 0 <= scale_index < self._SCALE_COUNT:
            raise ValueError(f"invalid anatomical scale index: {scale_index}")
        batch_size, _, height, width = projected.shape
        band_bias = self._band_bias(
            height,
            width,
            device=projected.device,
            dtype=projected.dtype,
        ).flatten(1)
        spatial_tokens = projected.flatten(2).transpose(1, 2)
        keys = self.key_norms[scale_index](spatial_tokens)
        queries = (
            self.role_queries[:, None, :]
            + self.cell_embeddings[None, :, :]
            + self.scale_query_offsets[scale_index, :, None, :]
        )
        queries = self.query_norms[scale_index](queries)
        logits = torch.einsum(
            "pkc,bnc->bpkn",
            queries,
            keys,
        ) / math.sqrt(self.cell_dim)
        canonical_attention = (
            logits + band_bias[None, :, None, :]
        ).softmax(dim=-1)
        cell_tokens = torch.einsum(
            "bpkn,bnc->bpkc",
            canonical_attention,
            spatial_tokens,
        )
        student_tokens = cell_tokens.flatten(2)
        visibility_logits = self.visibility_predictors[scale_index](
            self.visibility_norms[scale_index](student_tokens)
        ).squeeze(-1)
        return (
            student_tokens,
            canonical_attention.reshape(
                batch_size,
                self.num_parts,
                self._CANONICAL_CELLS,
                height,
                width,
            ),
            visibility_logits,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        fine_x: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        projected = self.feature_projection(x)
        student_tokens, canonical_attention, visibility_logits = (
            self._student_outputs(
                projected,
                scale_index=0,
            )
        )
        fine_projected = None
        fine_student_tokens = None
        fine_attention = None
        fine_visibility_logits = None
        if self.fine_feature_projection is not None:
            if fine_x is None:
                raise RuntimeError(
                    "Multi-scale anatomy requires a fine feature map"
                )
            fine_projected = self.fine_feature_projection(fine_x)
            (
                fine_student_tokens,
                fine_attention,
                fine_visibility_logits,
            ) = self._student_outputs(
                fine_projected,
                scale_index=1,
            )
        return (
            projected,
            student_tokens,
            canonical_attention,
            visibility_logits,
            fine_projected,
            fine_student_tokens,
            fine_attention,
            fine_visibility_logits,
        )

    def project_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Project the deployed descriptor into anatomical-token space."""
        if self.descriptor_projection is None:
            raise RuntimeError("Anatomical descriptor distillation is not configured")
        return self.descriptor_projection(descriptor)


class AnatomicalSemanticPredictionHead(nn.Module):
    """Predict training-only foreground and six semantic-part masks."""

    def __init__(self, channels: int, num_parts: int = 6) -> None:
        super().__init__()
        if channels < 1 or num_parts < 1:
            raise ValueError(
                "semantic prediction channels and part count must be positive"
            )
        hidden_channels = min(64, max(16, channels // 2))
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.foreground_predictor = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )
        self.part_predictor = nn.Conv2d(
            hidden_channels,
            num_parts,
            kernel_size=1,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(feature_map)
        return (
            self.foreground_predictor(hidden),
            self.part_predictor(hidden),
        )


class DecoupledMaskedQueryTeacher(nn.Module):
    """Pool unrestricted RGB queries beside parsing-masked teacher queries."""

    def __init__(
        self,
        local_channels: int,
        fine_channels: int,
        token_dim: int,
        *,
        num_parts: int = 6,
    ) -> None:
        super().__init__()
        if min(local_channels, fine_channels, token_dim, num_parts) < 1:
            raise ValueError(
                "decoupled query teacher dimensions must be positive"
            )
        self.num_parts = int(num_parts)
        self.token_dim = int(token_dim)
        self.queries = nn.Parameter(
            torch.empty(self.num_parts, self.token_dim)
        )
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.key_norm = nn.LayerNorm(self.token_dim)
        self.parsing_adapters = nn.ModuleList(
            self._parsing_adapter(channels)
            for channels in (local_channels, fine_channels)
        )
        self.foreground_predictors = nn.ModuleList(
            nn.Conv2d(self.token_dim, 1, kernel_size=1)
            for _ in range(2)
        )
        self.part_predictors = nn.ModuleList(
            nn.Conv2d(self.token_dim, self.num_parts, kernel_size=1)
            for _ in range(2)
        )
        self.visibility_norm = nn.LayerNorm(self.token_dim)
        self.visibility_predictor = nn.Linear(self.token_dim, 1)
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.zeros_(self.visibility_predictor.weight)
        nn.init.constant_(
            self.visibility_predictor.bias,
            math.log(4.0),
        )

    def _parsing_adapter(self, channels: int) -> nn.Sequential:
        hidden_channels = self.token_dim
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )

    def _scale_outputs(
        self,
        student_map: torch.Tensor,
        parsing_map: torch.Tensor,
        part_masks: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if student_map.shape[1] != self.token_dim:
            raise ValueError(
                "decoupled student map channel mismatch: "
                f"expected {self.token_dim}, got {student_map.shape[1]}"
            )
        masks = F.interpolate(
            part_masks.to(
                device=student_map.device,
                dtype=student_map.dtype,
            ),
            size=student_map.shape[-2:],
            mode="area",
        ).clamp(0, 1)
        if masks.shape[1] != self.num_parts:
            raise ValueError(
                "decoupled query masks must match the configured query count: "
                f"expected {self.num_parts}, got {masks.shape[1]}"
            )

        queries = self.query_norm(self.queries)
        student_values = student_map.flatten(2).transpose(1, 2)
        parsing_values = parsing_map.flatten(2).transpose(1, 2)
        student_logits = torch.einsum(
            "pd,bnd->bpn",
            queries,
            self.key_norm(student_values),
        ) / math.sqrt(self.token_dim)
        teacher_logits = torch.einsum(
            "pd,bnd->bpn",
            queries,
            self.key_norm(parsing_values),
        ) / math.sqrt(self.token_dim)

        student_attention = student_logits.softmax(dim=-1)
        student_tokens = torch.einsum(
            "bpn,bnd->bpd",
            student_attention,
            student_values,
        )

        flat_masks = masks.flatten(2)
        teacher_weights = (
            teacher_logits
            - teacher_logits.amax(dim=-1, keepdim=True)
        ).exp() * flat_masks
        teacher_mass = teacher_weights.sum(dim=-1, keepdim=True)
        teacher_attention = (
            teacher_weights / teacher_mass.clamp_min(1e-8)
        )
        teacher_tokens = torch.einsum(
            "bpn,bnd->bpd",
            teacher_attention,
            parsing_values,
        )
        teacher_valid = (
            flat_masks.sum(dim=-1) > 1e-6
        )
        teacher_tokens = (
            teacher_tokens
            * teacher_valid[:, :, None].to(teacher_tokens.dtype)
        )
        visibility_logits = self.visibility_predictor(
            self.visibility_norm(student_tokens)
        ).squeeze(-1)
        return (
            student_tokens,
            teacher_tokens,
            teacher_valid,
            visibility_logits,
        )

    def forward(
        self,
        local_student_map: torch.Tensor,
        fine_student_map: torch.Tensor,
        local_source: torch.Tensor,
        fine_source: torch.Tensor,
        part_masks: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return local/fine student and masked-teacher query outputs."""
        parsing_maps = tuple(
            adapter(source)
            for adapter, source in zip(
                self.parsing_adapters,
                (local_source, fine_source),
                strict=True,
            )
        )
        local_outputs = self._scale_outputs(
            local_student_map,
            parsing_maps[0],
            part_masks,
        )
        fine_outputs = self._scale_outputs(
            fine_student_map,
            parsing_maps[1],
            part_masks,
        )
        return (
            *local_outputs,
            self.foreground_predictors[0](parsing_maps[0]),
            self.part_predictors[0](parsing_maps[0]),
            *fine_outputs,
            self.foreground_predictors[1](parsing_maps[1]),
            self.part_predictors[1](parsing_maps[1]),
        )


class EMAAnatomicalAuxiliaryPool(nn.Module):
    """Learn RGB tokens beside a pose-conditioned online/EMA teacher."""

    _INITIAL_CENTERS = (0.12, 0.38, 0.40, 0.40, 0.72, 0.72)
    _INITIAL_WIDTHS = (0.16, 0.24, 0.28, 0.28, 0.30, 0.30)
    _CANONICAL_GRID_SIZE = (4, 2)
    _CANONICAL_CELLS = 8
    _POSE_LIMBS = (
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )
    _POSE_CHANNELS = 17 + len(_POSE_LIMBS)
    _POSE_EMBED_DIM = 32

    def __init__(
        self,
        channels: int,
        token_dim: int = 128,
        num_parts: int = 6,
        descriptor_dim: int | None = None,
        *,
        pose_teacher: bool = True,
        teacher_channels: int | None = None,
        multiscale: bool = False,
        semantic_teacher: bool = False,
        decoupled_query_teacher: bool = False,
        accessory_query: bool = False,
    ) -> None:
        super().__init__()
        if num_parts != len(self._INITIAL_CENTERS):
            raise ValueError("The anatomical auxiliary pool currently requires six ordered body parts")
        if channels < 1 or token_dim < 2 * self._CANONICAL_CELLS:
            raise ValueError(
                "anatomical channels must be positive and token_dim must "
                "provide at least two channels for each canonical grid cell"
            )
        self.num_parts = int(num_parts)
        self.token_dim = int(token_dim)
        self.pose_teacher_enabled = bool(pose_teacher)
        self.multiscale_enabled = bool(multiscale)
        self.semantic_teacher_enabled = bool(semantic_teacher)
        self.decoupled_query_teacher_enabled = bool(
            decoupled_query_teacher
        )
        self.accessory_query_enabled = bool(accessory_query)
        if (
            self.decoupled_query_teacher_enabled
            and not self.multiscale_enabled
        ):
            raise ValueError(
                "decoupled query teacher requires multi-scale anatomy"
            )
        teacher_channels = channels if teacher_channels is None else int(teacher_channels)
        if teacher_channels < 1:
            raise ValueError("anatomical teacher channels must be positive")
        base_chunk_size, remainder = divmod(
            self.token_dim,
            self._CANONICAL_CELLS,
        )
        self.chunk_sizes = tuple(
            base_chunk_size + (index < remainder)
            for index in range(self._CANONICAL_CELLS)
        )
        self.feature_projection = nn.Conv2d(
            channels,
            token_dim,
            kernel_size=1,
            bias=False,
        )
        self.teacher_projection = nn.Conv2d(
            channels,
            token_dim,
            kernel_size=1,
            bias=False,
        )
        self.teacher_projection.requires_grad_(False)
        self.queries = nn.Parameter(torch.empty(num_parts, token_dim))
        self.query_norms = nn.ModuleList(
            nn.LayerNorm(chunk_size)
            for chunk_size in self.chunk_sizes
        )
        self.key_norms = nn.ModuleList(
            nn.LayerNorm(chunk_size)
            for chunk_size in self.chunk_sizes
        )
        self.visibility_norm = nn.LayerNorm(token_dim)
        self.visibility_predictor = nn.Linear(token_dim, 1)
        self.descriptor_projection = (
            nn.Linear(
                int(descriptor_dim),
                num_parts * token_dim,
                bias=False,
            )
            if descriptor_dim is not None
            else None
        )
        if self.pose_teacher_enabled:
            self.online_pose_encoder = nn.Sequential(
                nn.Conv2d(
                    self._POSE_CHANNELS,
                    self._POSE_EMBED_DIM,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, self._POSE_EMBED_DIM),
                nn.GELU(),
                nn.Conv2d(
                    self._POSE_EMBED_DIM,
                    self._POSE_EMBED_DIM,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, self._POSE_EMBED_DIM),
                nn.GELU(),
            )
            self.online_pose_projection = nn.Conv2d(
                teacher_channels + self._POSE_EMBED_DIM,
                token_dim,
                kernel_size=1,
                bias=False,
            )
            self.ema_pose_encoder = copy.deepcopy(self.online_pose_encoder)
            self.ema_pose_projection = copy.deepcopy(self.online_pose_projection)
            self.ema_pose_encoder.requires_grad_(False)
            self.ema_pose_projection.requires_grad_(False)
        else:
            self.online_pose_encoder = None
            self.online_pose_projection = None
            self.ema_pose_encoder = None
            self.ema_pose_projection = None

        nn.init.trunc_normal_(self.queries, std=0.02)
        self.band_centers = nn.Parameter(torch.tensor(self._INITIAL_CENTERS, dtype=torch.float32))
        widths = torch.tensor(self._INITIAL_WIDTHS, dtype=torch.float32)
        self.band_log_widths = nn.Parameter(torch.log(torch.expm1(widths)))
        self.band_log_strength = nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))
        self.reset_visibility_initialization()
        self.reset_teacher_initialization()
        self.fine_feature_projection = (
            nn.Conv2d(
                teacher_channels,
                token_dim,
                kernel_size=1,
                bias=False,
            )
            if self.multiscale_enabled
            else None
        )
        self.semantic_prediction_heads = (
            nn.ModuleList(
                AnatomicalSemanticPredictionHead(
                    token_dim,
                    num_parts=num_parts,
                )
                for _ in range(2)
            )
            if self.semantic_teacher_enabled
            else None
        )
        self.decoupled_query_teacher = (
            DecoupledMaskedQueryTeacher(
                channels,
                teacher_channels,
                token_dim,
                num_parts=num_parts + int(self.accessory_query_enabled),
            )
            if self.decoupled_query_teacher_enabled
            else None
        )

    def reset_visibility_initialization(self) -> None:
        """Initialize visibility as mostly present without blocking learning."""
        nn.init.zeros_(self.visibility_predictor.weight)
        nn.init.constant_(
            self.visibility_predictor.bias,
            math.log(4.0),
        )

    @torch.no_grad()
    def reset_teacher_initialization(self) -> None:
        """Start every EMA target from its matching online projection."""
        self.teacher_projection.weight.copy_(self.feature_projection.weight)
        self.teacher_projection.requires_grad_(False)
        if self.online_pose_encoder is not None:
            self.ema_pose_encoder.load_state_dict(self.online_pose_encoder.state_dict())
            self.ema_pose_projection.load_state_dict(self.online_pose_projection.state_dict())
            self.ema_pose_encoder.requires_grad_(False)
            self.ema_pose_projection.requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def _ema_module(
        target: nn.Module,
        online: nn.Module,
        momentum: float,
    ) -> None:
        for target_parameter, online_parameter in zip(
            target.parameters(),
            online.parameters(),
            strict=True,
        ):
            target_parameter.mul_(momentum).add_(
                online_parameter.detach(),
                alpha=1.0 - momentum,
            )
        for target_buffer, online_buffer in zip(
            target.buffers(),
            online.buffers(),
            strict=True,
        ):
            target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Update the stop-gradient projection used for privileged targets."""
        if not 0 <= momentum < 1:
            raise ValueError("anatomical teacher momentum must be in [0, 1)")
        self.teacher_projection.weight.mul_(momentum).add_(
            self.feature_projection.weight.detach(),
            alpha=1.0 - momentum,
        )
        if self.online_pose_encoder is not None:
            self._ema_module(
                self.ema_pose_encoder,
                self.online_pose_encoder,
                momentum,
            )
            self._ema_module(
                self.ema_pose_projection,
                self.online_pose_projection,
                momentum,
            )

    @classmethod
    def pose_heatmaps(
        cls,
        pose_keypoints: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Rasterize confidence-aware COCO joints and limbs."""
        if pose_keypoints.ndim != 3 or pose_keypoints.shape[1:] != (17, 3):
            raise ValueError("pose keypoints must have shape [B,17,3]")
        dtype = pose_keypoints.dtype
        device = pose_keypoints.device
        x = (pose_keypoints[..., 0] + 1.0) * width * 0.5 - 0.5
        y = (pose_keypoints[..., 1] + 1.0) * height * 0.5 - 0.5
        confidence = pose_keypoints[..., 2].clamp(0, 1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        spatial_grid = torch.stack((grid_x, grid_y), dim=-1)
        points = torch.stack((x, y), dim=-1)
        sigma = max(min(height, width) * 0.06, 1.0)
        joint_distance = (
            spatial_grid[None, None]
            - points[:, :, None, None]
        ).square().sum(dim=-1)
        joint_heatmaps = (
            torch.exp(-0.5 * joint_distance / (sigma * sigma))
            * confidence[:, :, None, None]
        )

        limb_indices = torch.tensor(
            cls._POSE_LIMBS,
            device=device,
            dtype=torch.long,
        )
        starts = points[:, limb_indices[:, 0]]
        ends = points[:, limb_indices[:, 1]]
        segment = ends - starts
        segment_length = segment.square().sum(dim=-1).clamp_min(1e-6)
        relative = spatial_grid[None, None] - starts[:, :, None, None]
        along = (
            relative * segment[:, :, None, None]
        ).sum(dim=-1) / segment_length[:, :, None, None]
        along = along.clamp(0, 1)
        closest = (
            starts[:, :, None, None]
            + along[..., None] * segment[:, :, None, None]
        )
        limb_distance = (
            spatial_grid[None, None] - closest
        ).square().sum(dim=-1)
        limb_confidence = torch.minimum(
            confidence[:, limb_indices[:, 0]],
            confidence[:, limb_indices[:, 1]],
        )
        limb_heatmaps = (
            torch.exp(-0.5 * limb_distance / (sigma * sigma))
            * limb_confidence[:, :, None, None]
        )
        return torch.cat((joint_heatmaps, limb_heatmaps), dim=1)

    def _band_bias(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        centers = self.band_centers.to(dtype=dtype).clamp(1e-3, 1.0 - 1e-3)
        widths = F.softplus(self.band_log_widths.to(dtype=dtype)).clamp_min(1e-3)
        strength = F.softplus(self.band_log_strength.to(dtype=dtype))
        bias = -0.5 * ((rows[None, :] - centers[:, None]) / widths[:, None]).square()
        return (strength * bias)[:, :, None].expand(-1, -1, width)

    def _student_outputs(
        self,
        projected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool shared anatomical roles from one spatial feature scale."""
        batch_size, _, height, width = projected.shape
        band_bias = self._band_bias(
            height,
            width,
            device=projected.device,
            dtype=projected.dtype,
        ).flatten(1)
        projected_chunks = projected.split(self.chunk_sizes, dim=1)
        query_chunks = self.queries.split(self.chunk_sizes, dim=1)
        attention_chunks = []
        student_token_chunks = []
        for projected_chunk, query_chunk, query_norm, key_norm in zip(
            projected_chunks,
            query_chunks,
            self.query_norms,
            self.key_norms,
            strict=True,
        ):
            spatial_tokens = projected_chunk.flatten(2).transpose(1, 2)
            keys = key_norm(spatial_tokens)
            queries = query_norm(query_chunk)
            logits = torch.einsum(
                "pc,bnc->bpn",
                queries,
                keys,
            ) / math.sqrt(projected_chunk.shape[1])
            attention = (logits + band_bias[None]).softmax(dim=-1)
            attention_chunks.append(attention)
            student_token_chunks.append(
                torch.einsum(
                    "bpn,bnc->bpc",
                    attention,
                    spatial_tokens,
                )
            )
        canonical_attention = torch.stack(
            attention_chunks,
            dim=2,
        )
        student_tokens = torch.cat(
            student_token_chunks,
            dim=-1,
        )
        visibility_logits = self.visibility_predictor(
            self.visibility_norm(student_tokens)
        ).squeeze(-1)
        return (
            student_tokens,
            canonical_attention.reshape(
                batch_size,
                self.num_parts,
                self._CANONICAL_CELLS,
                height,
                width,
            ),
            visibility_logits,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        teacher_x: torch.Tensor | None = None,
        pose_keypoints: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        teacher_source = x if teacher_x is None else teacher_x
        (
            projected,
            student_tokens,
            canonical_attention,
            visibility_logits,
            fine_projected,
            fine_student_tokens,
            fine_attention,
            fine_visibility_logits,
        ) = self.student_forward(
            x,
            fine_x=teacher_source,
        )
        online_teacher_projected = None
        if self.pose_teacher_enabled:
            if pose_keypoints is None:
                raise RuntimeError("Pose-conditioned anatomy teacher requires cached pose keypoints")
            pose_heatmaps = self.pose_heatmaps(
                pose_keypoints.to(
                    device=teacher_source.device,
                    dtype=teacher_source.dtype,
                ),
                height=teacher_source.shape[-2],
                width=teacher_source.shape[-1],
            )
            online_pose = self.online_pose_encoder(pose_heatmaps)
            online_teacher_projected = self.online_pose_projection(
                torch.cat((teacher_source.detach(), online_pose), dim=1)
            )
            with torch.no_grad():
                ema_pose = self.ema_pose_encoder(pose_heatmaps.detach())
                teacher_projected = self.ema_pose_projection(
                    torch.cat(
                        (teacher_source.detach(), ema_pose),
                        dim=1,
                    )
                )
        else:
            with torch.no_grad():
                teacher_projected = self.teacher_projection(teacher_source.detach())
        return (
            projected,
            teacher_projected,
            online_teacher_projected,
            student_tokens,
            canonical_attention,
            visibility_logits,
            fine_projected,
            fine_student_tokens,
            fine_attention,
            fine_visibility_logits,
        )

    def student_forward(
        self,
        x: torch.Tensor,
        *,
        fine_x: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return RGB-only anatomical students without traversing the pose teacher."""
        projected = self.feature_projection(x)
        student_tokens, canonical_attention, visibility_logits = (
            self._student_outputs(projected)
        )
        fine_projected = None
        fine_student_tokens = None
        fine_attention = None
        fine_visibility_logits = None
        if self.fine_feature_projection is not None:
            fine_source = x if fine_x is None else fine_x
            fine_projected = self.fine_feature_projection(fine_source)
            (
                fine_student_tokens,
                fine_attention,
                fine_visibility_logits,
            ) = self._student_outputs(fine_projected)
        return (
            projected,
            student_tokens,
            canonical_attention,
            visibility_logits,
            fine_projected,
            fine_student_tokens,
            fine_attention,
            fine_visibility_logits,
        )

    def semantic_predictions(
        self,
        local_feature_map: torch.Tensor,
        fine_feature_map: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return local/fine semantic logits from projected RGB features."""
        if self.semantic_prediction_heads is None:
            raise RuntimeError("Semantic anatomical prediction is not enabled")
        if fine_feature_map is None:
            raise RuntimeError(
                "Multi-scale semantic prediction requires a fine feature map"
            )
        local_foreground, local_parts = self.semantic_prediction_heads[0](
            local_feature_map
        )
        fine_foreground, fine_parts = self.semantic_prediction_heads[1](
            fine_feature_map
        )
        return (
            local_foreground,
            local_parts,
            fine_foreground,
            fine_parts,
        )

    def decoupled_query_outputs(
        self,
        local_feature_map: torch.Tensor,
        fine_feature_map: torch.Tensor | None,
        local_source: torch.Tensor,
        fine_source: torch.Tensor | None,
        part_masks: torch.Tensor,
    ):
        """Return training-only masked-teacher and RGB-student queries."""
        if self.decoupled_query_teacher is None:
            raise RuntimeError(
                "Decoupled anatomical query teacher is not enabled"
            )
        if fine_feature_map is None or fine_source is None:
            raise RuntimeError(
                "Decoupled anatomical queries require a fine feature map"
            )
        return self.decoupled_query_teacher(
            local_feature_map,
            fine_feature_map,
            local_source,
            fine_source,
            part_masks,
        )

    def project_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Project the deployed descriptor into teacher-token space."""
        if self.descriptor_projection is None:
            raise RuntimeError("Anatomical descriptor distillation is not configured")
        return self.descriptor_projection(descriptor)


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
