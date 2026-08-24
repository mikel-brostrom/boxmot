"""Monotonic RGB-conditioned vertical alignment for ordered ReID stripes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MCPT_MODES = frozenset(
    {
        "none",
        "dataset_boundaries",
        "foreground_aware_shared_multiscale",
        "per_image_stage2",
        "shared_multiscale",
    }
)


@dataclass(frozen=True)
class MCPTDiagnostics:
    """Differentiable regularizers and detached alignment diagnostics."""

    smoothness: torch.Tensor
    identity: torch.Tensor
    mean_abs_displacement: torch.Tensor
    boundary_mean: torch.Tensor
    boundary_std: torch.Tensor
    cap_fraction: torch.Tensor
    local_gate: torch.Tensor
    fine_gate: torch.Tensor


class MonotonicCanonicalPartTransport(nn.Module):
    """Warp ordered local maps into a learned canonical vertical coordinate.

    Positive source-interval lengths define a cumulative piecewise-linear map,
    so vertical order cannot fold or permute. An epoch-controlled residual gate
    keeps the complete module exactly identical to fixed stripes through the
    configured start epoch while avoiding the dead-gradient combination of an
    identity warp and a zero learnable gate.
    """

    def __init__(
        self,
        channels: int,
        *,
        fine_channels: int | None = None,
        mode: str = "none",
        hidden_dim: int = 64,
        max_displacement: float = 0.15,
        start_epoch: int = 10,
        ramp_end_epoch: int = 40,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in MCPT_MODES - {"none"}:
            raise ValueError(
                f"MCPT mode must be one of {sorted(MCPT_MODES - {'none'})}, "
                f"got {mode!r}"
            )
        self.channels = int(channels)
        self.fine_channels = int(
            self.channels if fine_channels is None else fine_channels
        )
        self.hidden_dim = int(hidden_dim)
        self.max_displacement = float(max_displacement)
        self.start_epoch = int(start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        if self.channels < 1 or self.fine_channels < 1 or self.hidden_dim < 1:
            raise ValueError(
                "MCPT channels, fine_channels, and hidden_dim must be positive"
            )
        if not 0 < self.max_displacement < 0.5:
            raise ValueError("MCPT max_displacement must satisfy 0 < value < 0.5")
        if self.start_epoch < 0 or self.ramp_end_epoch <= self.start_epoch:
            raise ValueError("MCPT ramp_end_epoch must be greater than start_epoch")

        if self.mode == "dataset_boundaries":
            self.dataset_row_logits = nn.Parameter(
                torch.zeros(4)
            )
            self.row_norm = None
            self.row_projection = None
            self.row_mixer = None
            self.row_output = None
        else:
            self.dataset_row_logits = None
            self.row_norm = nn.LayerNorm(self.channels)
            self.row_projection = nn.Linear(
                self.channels,
                self.hidden_dim,
                bias=False,
            )
            self.row_mixer = nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=5,
                padding=2,
                groups=self.hidden_dim,
                bias=True,
            )
            self.row_output = nn.Conv1d(
                self.hidden_dim,
                1,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.row_output.weight)
            nn.init.zeros_(self.row_output.bias)

        if self.mode == "foreground_aware_shared_multiscale":
            # A bounded residual over uniform width pooling keeps every column
            # represented while allowing the predictor to suppress side
            # background. Separate attention logits accommodate the different
            # semantics and resolutions of the local and fine maps.
            self.local_foreground_attention = nn.Conv2d(
                self.channels,
                1,
                kernel_size=1,
                bias=True,
            )
            self.fine_foreground_attention = nn.Conv2d(
                self.fine_channels,
                1,
                kernel_size=1,
                bias=True,
            )
            self.fine_row_norm = nn.LayerNorm(self.fine_channels)
            self.fine_row_projection = nn.Linear(
                self.fine_channels,
                self.hidden_dim,
                bias=False,
            )
            # Channel-wise residual fusion starts at the proven local-only
            # predictor. tanh bounds the learned fine contribution without a
            # dead gradient at its exact-zero initialization.
            self.fine_fusion_gate_delta = nn.Parameter(
                torch.zeros(self.hidden_dim)
            )
        else:
            self.local_foreground_attention = None
            self.fine_foreground_attention = None
            self.fine_row_norm = None
            self.fine_row_projection = None
            self.fine_fusion_gate_delta = None

        # The epoch schedule supplies the initial non-zero learning path after
        # start_epoch. These parameters learn scale-specific deviations from
        # that nominal schedule without changing epoch-zero behavior.
        self.local_gate_delta = nn.Parameter(torch.zeros(()))
        self.fine_gate_delta = nn.Parameter(torch.zeros(()))
        # Runtime controls stay as Python scalars. Keeping them off-device
        # avoids a GPU/MPS synchronization from Tensor.item() on every batch.
        self._schedule_scale = 0.0
        self._schedule_was_explicit = False
        self._force_disabled = False
        self._visualization_limit = 0
        self._visualization_count = 0
        self._visualization_batches: list[dict[str, torch.Tensor]] = []
        self.reset_predictor_initialization()
        self.reset_identity_initialization()

    @property
    def applies_to_fine(self) -> bool:
        return self.mode in {
            "dataset_boundaries",
            "foreground_aware_shared_multiscale",
            "shared_multiscale",
        }

    @torch.no_grad()
    def reset_predictor_initialization(self) -> None:
        """Apply the shared TinyViT predictor initialization in every host."""
        for projection in (self.row_projection, self.fine_row_projection):
            if projection is not None:
                nn.init.trunc_normal_(projection.weight, std=0.02)
                if projection.bias is not None:
                    projection.bias.zero_()
        for normalization in (self.row_norm, self.fine_row_norm):
            if normalization is not None:
                if normalization.weight is not None:
                    normalization.weight.fill_(1.0)
                if normalization.bias is not None:
                    normalization.bias.zero_()

    @torch.no_grad()
    def reset_identity_initialization(self) -> None:
        """Restore the uniform transform after model-wide initialization."""
        if self.dataset_row_logits is not None:
            self.dataset_row_logits.zero_()
        if self.row_output is not None:
            self.row_output.weight.zero_()
            self.row_output.bias.zero_()
        if self.local_foreground_attention is not None:
            self.local_foreground_attention.weight.zero_()
            self.local_foreground_attention.bias.zero_()
        if self.fine_foreground_attention is not None:
            self.fine_foreground_attention.weight.zero_()
            self.fine_foreground_attention.bias.zero_()
        if self.fine_fusion_gate_delta is not None:
            self.fine_fusion_gate_delta.zero_()
        self.local_gate_delta.zero_()
        self.fine_gate_delta.zero_()
        self._schedule_scale = 0.0
        self._schedule_was_explicit = False
        self._force_disabled = False

    def set_epoch(self, epoch: int) -> None:
        """Set the residual gate schedule for the next train/eval forward."""
        epoch = int(epoch)
        if epoch <= self.start_epoch:
            scale = 0.0
        elif epoch >= self.ramp_end_epoch:
            scale = 1.0
        else:
            scale = (epoch - self.start_epoch) / (
                self.ramp_end_epoch - self.start_epoch
            )
        self._schedule_scale = float(scale)
        self._schedule_was_explicit = True

    def set_force_disabled(self, disabled: bool) -> None:
        """Disable transport without changing parameters for control eval."""
        self._force_disabled = bool(disabled)

    def train(self, mode: bool = True):
        """Enable the learned transport for standalone inference by default.

        The trainer always calls :meth:`set_epoch`, so scheduled validation is
        preserved. A model reconstructed directly from a deployment checkpoint
        has no epoch lifecycle; its first ``eval()`` therefore selects the full
        learned transform instead of silently deploying fixed stripes.
        """
        super().train(mode)
        if not mode and not self._schedule_was_explicit:
            self._schedule_scale = 1.0
        return self

    def enable_visualization_capture(self, limit: int = 100) -> None:
        """Capture bounded before/after feature-energy maps on later forwards."""
        self._visualization_limit = max(int(limit), 0)
        self._visualization_count = 0
        self._visualization_batches.clear()

    def pop_visualization_capture(self) -> dict[str, torch.Tensor] | None:
        """Return and clear captured feature-energy maps."""
        if not self._visualization_batches:
            self._visualization_limit = 0
            return None
        keys = self._visualization_batches[0]
        captured = {
            key: torch.cat(
                [batch[key] for batch in self._visualization_batches],
                dim=0,
            )
            for key in keys
        }
        self._visualization_limit = 0
        self._visualization_count = 0
        self._visualization_batches.clear()
        return captured

    def _capture_visualization(
        self,
        local_before: torch.Tensor,
        local_after: torch.Tensor,
        fine_before: torch.Tensor | None,
        fine_after: torch.Tensor | None,
    ) -> None:
        remaining = self._visualization_limit - self._visualization_count
        if remaining <= 0:
            return
        count = min(local_before.shape[0], remaining)

        def energy(feature: torch.Tensor) -> torch.Tensor:
            return feature[:count].detach().float().square().mean(dim=1).cpu()

        batch = {
            "local_before": energy(local_before),
            "local_after": energy(local_after),
        }
        if fine_before is not None and fine_after is not None:
            batch.update(
                fine_before=energy(fine_before),
                fine_after=energy(fine_after),
            )
        self._visualization_batches.append(batch)
        self._visualization_count += count

    @staticmethod
    def _foreground_weighted_row_tokens(
        feature: torch.Tensor,
        attention: nn.Conv2d,
    ) -> torch.Tensor:
        """Return width-aware row tokens with a collapse-resistant residual."""
        attention_weights = torch.softmax(
            attention(feature).float(),
            dim=-1,
        ).to(feature.dtype)
        # Retain 25% uniform pooling so a narrow attention peak cannot erase
        # identity evidence or make the boundary predictor depend on one pixel.
        attention_weights = 0.75 * attention_weights + (
            0.25 / feature.shape[-1]
        )
        return (feature * attention_weights).sum(dim=-1).transpose(1, 2)

    def _row_logits(
        self,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, height, _ = local_feature.shape
        if self.dataset_row_logits is not None:
            # Four positive segment lengths directly parameterize the three
            # nested fine-stripe boundaries shared by the whole dataset.
            return self.dataset_row_logits.view(1, 4).expand(batch_size, -1)

        if self.mode == "foreground_aware_shared_multiscale":
            if fine_feature is None:
                raise ValueError(
                    "Foreground-aware MCPT requires a fine feature map"
                )
            if fine_feature.shape[1] != self.fine_channels:
                raise ValueError(
                    f"MCPT expected {self.fine_channels} fine channels, got "
                    f"{fine_feature.shape[1]}"
                )
            row_tokens = self._foreground_weighted_row_tokens(
                local_feature,
                self.local_foreground_attention,
            )
            fine_row_tokens = self._foreground_weighted_row_tokens(
                fine_feature,
                self.fine_foreground_attention,
            )
        else:
            row_tokens = local_feature.mean(dim=-1).transpose(1, 2)
            fine_row_tokens = None

        mixed = self.row_projection(self.row_norm(row_tokens)).transpose(1, 2)
        if fine_row_tokens is not None:
            fine_mixed = self.fine_row_projection(
                self.fine_row_norm(fine_row_tokens)
            ).transpose(1, 2)
            fine_mixed = F.adaptive_avg_pool1d(fine_mixed, height)
            fine_gate = torch.tanh(self.fine_fusion_gate_delta).view(
                1,
                -1,
                1,
            )
            mixed = mixed + fine_gate.to(mixed.dtype) * fine_mixed
        mixed = F.gelu(self.row_mixer(mixed))
        return self.row_output(mixed).squeeze(1)

    def _source_edges(
        self,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._row_logits(local_feature, fine_feature).float()
        # Smoothly bound the interval logits before softplus. This prevents a
        # rare inf/inf normalization failure during a long mixed-precision run
        # while retaining more than enough dynamic range for a 15% warp.
        logits = 8.0 * torch.tanh(logits / 8.0)
        lengths = F.softplus(logits) + 1e-6
        lengths = lengths / lengths.sum(dim=1, keepdim=True)
        raw_edges = F.pad(lengths.cumsum(dim=1), (1, 0), value=0.0)
        height = lengths.shape[1]
        uniform_edges = torch.linspace(
            0.0,
            1.0,
            height + 1,
            device=lengths.device,
            dtype=lengths.dtype,
        ).unsqueeze(0)
        raw_displacement = raw_edges - uniform_edges
        edges = uniform_edges + self.max_displacement * torch.tanh(
            raw_displacement / self.max_displacement
        )
        # Preserve exact endpoints despite finite-precision normalization.
        edges = torch.cat(
            (
                torch.zeros_like(edges[:, :1]),
                edges[:, 1:-1],
                torch.ones_like(edges[:, -1:]),
            ),
            dim=1,
        )
        return edges, uniform_edges, raw_displacement

    @staticmethod
    def _sample_positions(
        edges: torch.Tensor,
        output_height: int,
    ) -> torch.Tensor:
        source_intervals = edges.shape[1] - 1
        output_positions = (
            torch.arange(
                output_height,
                device=edges.device,
                dtype=edges.dtype,
            )
            + 0.5
        ) / output_height
        interval_position = output_positions * source_intervals
        interval_index = interval_position.floor().long().clamp(
            max=source_intervals - 1
        )
        interval_fraction = (
            interval_position - interval_index.to(interval_position.dtype)
        )
        left = edges[:, interval_index]
        right = edges[:, interval_index + 1]
        return left + interval_fraction.unsqueeze(0) * (right - left)

    @classmethod
    def _interpolation_matrix(
        cls,
        feature: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, _ = feature.shape
        source_y = cls._sample_positions(edges, height)
        source_pixels = (source_y * height - 0.5).clamp(0, height - 1)
        input_pixels = torch.arange(
            height,
            device=feature.device,
            dtype=source_pixels.dtype,
        ).view(1, 1, height)
        return (
            1.0 - (source_pixels.unsqueeze(-1) - input_pixels).abs()
        ).clamp_min(0.0)

    @staticmethod
    def _apply_interpolation(
        feature: torch.Tensor,
        interpolation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        # A batched matrix product implements the same border-padded linear
        # interpolation without grid_sample. Its backward is available on MPS
        # and respects deterministic-algorithm mode on CUDA.
        flattened = feature.permute(0, 2, 1, 3).reshape(
            batch_size,
            height,
            -1,
        )
        transported = torch.bmm(interpolation.to(feature.dtype), flattened)
        return transported.reshape(
            batch_size,
            height,
            -1,
            width,
        ).permute(0, 2, 1, 3)

    @classmethod
    def _warp_y(
        cls,
        feature: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        interpolation = cls._interpolation_matrix(feature, edges)
        return cls._apply_interpolation(feature, interpolation)

    @classmethod
    def _transport_y(
        cls,
        feature: torch.Tensor,
        edges: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the residual gate in interpolation space to save memory."""
        interpolation = cls._interpolation_matrix(feature, edges)
        height = feature.shape[2]
        identity = torch.eye(
            height,
            device=feature.device,
            dtype=interpolation.dtype,
        ).unsqueeze(0)
        residual_interpolation = identity + gate.float() * (
            interpolation - identity
        )
        return cls._apply_interpolation(feature, residual_interpolation)

    def _gate(self, delta: torch.Tensor) -> torch.Tensor:
        if self._force_disabled:
            return delta * 0.0
        # Keep the effective warp inside the configured displacement cap.
        # The external epoch schedule supplies an exact zero before activation;
        # sigmoid(delta) then gives the learnable gate a live gradient in [0, 1].
        return self._schedule_scale * torch.sigmoid(delta)

    def _identity_forward(
        self,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, MCPTDiagnostics]:
        """Return an exact fixed-stripe path with no MCPT parameter graph."""
        if self._visualization_count < self._visualization_limit:
            self._capture_visualization(
                local_feature,
                local_feature,
                fine_feature,
                fine_feature,
            )
        zero = local_feature.sum() * 0.0
        boundaries = torch.tensor(
            [0.25, 0.5, 0.75],
            device=local_feature.device,
            dtype=torch.float32,
        )
        diagnostics = MCPTDiagnostics(
            smoothness=zero,
            identity=zero,
            mean_abs_displacement=zero.detach(),
            boundary_mean=boundaries,
            boundary_std=zero.detach(),
            cap_fraction=zero.detach(),
            local_gate=zero.detach(),
            fine_gate=zero.detach(),
        )
        return local_feature, fine_feature, diagnostics

    def forward(
        self,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        MCPTDiagnostics,
    ]:
        """Return transported maps and regularization/health diagnostics."""
        if local_feature.ndim != 4:
            raise ValueError("MCPT local feature must have shape [B,C,H,W]")
        if local_feature.shape[1] != self.channels:
            raise ValueError(
                f"MCPT expected {self.channels} local channels, got "
                f"{local_feature.shape[1]}"
            )
        if self.applies_to_fine and fine_feature is None:
            raise ValueError(f"MCPT mode {self.mode!r} requires a fine map")
        if self._force_disabled or self._schedule_scale <= 0.0:
            return self._identity_forward(local_feature, fine_feature)

        edges, uniform_edges, _ = self._source_edges(
            local_feature,
            fine_feature,
        )
        local_gate = self._gate(self.local_gate_delta).to(local_feature.dtype)
        transported_local = self._transport_y(
            local_feature,
            edges,
            local_gate,
        )

        fine_gate = local_gate * 0.0
        transported_fine = fine_feature
        if self.applies_to_fine and fine_feature is not None:
            fine_gate = self._gate(self.fine_gate_delta).to(fine_feature.dtype)
            transported_fine = self._transport_y(
                fine_feature,
                edges,
                fine_gate,
            )

        if self._visualization_count < self._visualization_limit:
            self._capture_visualization(
                local_feature,
                transported_local,
                fine_feature,
                transported_fine,
            )

        displacement = edges - uniform_edges
        boundary_indices = torch.tensor(
            [
                edges.shape[1] // 4,
                edges.shape[1] // 2,
                3 * edges.shape[1] // 4,
            ],
            device=edges.device,
        ).clamp(max=edges.shape[1] - 2)
        boundaries = edges[:, boundary_indices]
        second_difference = (
            displacement[:, 2:]
            - 2.0 * displacement[:, 1:-1]
            + displacement[:, :-2]
        )
        diagnostics = MCPTDiagnostics(
            smoothness=second_difference.abs().mean(),
            identity=displacement.square().mean(),
            mean_abs_displacement=(
                displacement[:, 1:-1].abs().mean().detach()
            ),
            boundary_mean=boundaries.mean(dim=0).detach(),
            boundary_std=boundaries.std(dim=0, unbiased=False).mean().detach(),
            cap_fraction=(
                displacement[:, 1:-1].abs()
                >= 0.95 * self.max_displacement
            ).any(dim=1).float().mean().detach(),
            local_gate=local_gate.detach(),
            fine_gate=fine_gate.detach(),
        )
        return transported_local, transported_fine, diagnostics


__all__ = [
    "MCPTDiagnostics",
    "MCPT_MODES",
    "MonotonicCanonicalPartTransport",
]
