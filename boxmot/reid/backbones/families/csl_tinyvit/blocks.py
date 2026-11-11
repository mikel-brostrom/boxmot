# BoxMOT AGPL-3.0 license

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention

__all__ = [
    "BasicLayer",
    "BodySlotReadWrite",
    "Conv2d_BN",
    "ConvLayer",
    "DropPath",
    "IdentityRegisterCommunication",
    "LayerNorm2d",
    "MBConv",
    "NormPreservingWidthMerge",
    "PatchEmbed",
    "PatchMerging",
    "ReIDResidualAdapter",
    "RMSFeatureSuppression",
    "TinyViTBlock",
    "TinyViTMlp",
    "fuse_conv2d_bn_eval_",
]


class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of entire residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x.div(keep) * mask


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d (fused at deployment)."""

    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(in_ch, out_ch, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(out_ch)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and BN with MPS-safe activation strides."""
        x = self.c(x)
        if (
            self.training
            and x.device.type == "mps"
            and not x.is_contiguous()
        ):
            # Some depthwise shapes are emitted in a channels-last layout by
            # MPS. NativeBatchNormBackward currently assumes a view-compatible
            # NCHW tensor and otherwise fails during ReID-X's 48x8 stage.
            x = x.contiguous()
        return self.bn(x)

    def fuse(self) -> nn.Conv2d:
        """Return an equivalent inference-only convolution with a folded BN."""
        if self.training:
            raise RuntimeError("Conv2d_BN fusion requires eval mode")
        return nn.utils.fusion.fuse_conv_bn_eval(self.c, self.bn)


def fuse_conv2d_bn_eval_(module: nn.Module) -> int:
    """Recursively replace every CSL ``Conv2d_BN`` child with a fused convolution.

    The conversion is deliberately in-place and inference-only. It preserves
    the original training/checkpoint layout until deployment preparation and
    is idempotent because converted children are plain ``nn.Conv2d`` modules.
    """
    if module.training:
        raise RuntimeError("Conv2d_BN fusion requires the complete model to be in eval mode")

    fused_count = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, Conv2d_BN):
            setattr(module, name, child.fuse())
            fused_count += 1
        else:
            fused_count += fuse_conv2d_bn_eval_(child)
    return fused_count


class PatchEmbed(nn.Module):
    """Stride-4 convolutional patch embedding."""

    def __init__(self, in_chans, embed_dim, img_size, activation):
        super().__init__()
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv block."""

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    """Downsampling layer between stages."""

    def __init__(
        self,
        input_resolution,
        dim,
        out_dim,
        activation,
        stride: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        # TinyViT normally preserves resolution before its final stage. ReID
        # speed variants can explicitly downsample only that global path while
        # retaining the pre-merge Stage-2 tokens for local stripes.
        stride_c = (
            1 if out_dim in (320, 448, 576) else 2
        ) if stride is None else stride
        if isinstance(stride_c, tuple):
            stride_c = tuple(int(value) for value in stride_c)
            if (
                len(stride_c) != 2
                or any(value not in {1, 2} for value in stride_c)
            ):
                raise ValueError(
                    "PatchMerging tuple stride values must be 1 or 2, "
                    f"got {stride_c}"
                )
        else:
            stride_c = int(stride_c)
            if stride_c not in {1, 2}:
                raise ValueError(
                    f"PatchMerging stride must be 1 or 2, got {stride_c}"
                )
        self.stride = stride_c
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x, hw_size):
        if x.ndim == 3:
            H, W = hw_size
            B = x.shape[0]
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        out_size = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class NormPreservingWidthMerge(nn.Module):
    """Merge adjacent columns using activation-norm weights.

    The weighted average keeps the direction selected by the two input tokens,
    then restores the larger input norm so discriminative activations are not
    contracted merely because the spatial sequence was shortened.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        hw_size: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, tokens, channels = x.shape
        height, width = hw_size
        if tokens != height * width:
            raise ValueError(f"Expected {height * width} tokens for {hw_size}, got {tokens}")
        if width % 2 != 0:
            raise ValueError(f"Norm-preserving width merge requires an even width, got {width}")

        pairs = x.view(batch, height, width // 2, 2, channels)
        norms = torch.sqrt(torch.sum(pairs * pairs, dim=-1, keepdim=True))
        weights = norms / norms.sum(dim=3, keepdim=True).clamp_min(self.eps)
        merged = torch.sum(weights * pairs, dim=3)
        merged_norm = torch.sqrt(torch.sum(merged * merged, dim=-1, keepdim=True))
        target_norm = norms.amax(dim=3)
        merged = merged * (target_norm / merged_norm.clamp_min(self.eps))
        return merged.view(batch, height * (width // 2), channels), (height, width // 2)


class IdentityRegisterCommunication(nn.Module):
    """Exchange context through a compact recurrent-register bottleneck."""

    def __init__(
        self,
        dim: int,
        *,
        register_dim: int,
        num_registers: int,
        num_heads: int,
        window_size: tuple[int, int],
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            dim < 1
            or register_dim < 1
            or num_registers < 2
            or num_heads < 1
        ):
            raise ValueError(
                "Identity register dimensions and counts must be positive"
            )
        if register_dim % num_heads:
            raise ValueError(
                "Register bottleneck dimension "
                f"{register_dim} must divide num_heads={num_heads}"
            )
        if not 0 <= dropout < 1:
            raise ValueError("Identity register dropout must be in [0, 1)")
        self.spatial_dim = int(dim)
        self.register_dim = int(register_dim)
        self.num_registers = int(num_registers)
        self.window_size = tuple(int(value) for value in window_size)
        self.dropout = float(dropout)
        # The following projection can absorb LayerNorm's affine transform, so
        # learned scale/bias here add no capacity. Keeping this normalization
        # non-affine also avoids an MPS LayerNorm bug that can produce non-finite
        # affine gradients when its frozen-backbone input has no gradient path.
        self.summary_norm = nn.LayerNorm(
            self.spatial_dim,
            elementwise_affine=False,
        )
        self.summary_projection = nn.Linear(
            self.spatial_dim,
            self.register_dim,
        )
        self.register_norm = nn.LayerNorm(self.register_dim)
        self.register_attention = nn.MultiheadAttention(
            self.register_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.register_mlp = nn.Sequential(
            nn.LayerNorm(self.register_dim),
            nn.Linear(self.register_dim, 2 * self.register_dim),
            nn.GELU(),
            nn.Linear(2 * self.register_dim, self.register_dim),
        )
        self.broadcast_attention = nn.MultiheadAttention(
            self.register_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.broadcast_projection = nn.Linear(
            self.register_dim,
            self.spatial_dim,
        )
        self.broadcast_gate = nn.Parameter(
            torch.tensor(float(gate_init))
        )

    def _window_summaries(
        self,
        x: torch.Tensor,
        hw_size: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        batch, tokens, channels = x.shape
        height, width = hw_size
        if tokens != height * width:
            raise ValueError(
                f"Expected {height * width} register tokens, got {tokens}"
            )
        window_height = min(self.window_size[0], height)
        window_width = min(self.window_size[1], width)
        padded_height = (
            (height + window_height - 1) // window_height
        ) * window_height
        padded_width = (
            (width + window_width - 1) // window_width
        ) * window_width
        feature_map = x.view(batch, height, width, channels).permute(
            0,
            3,
            1,
            2,
        )
        feature_map = F.pad(
            feature_map,
            (0, padded_width - width, 0, padded_height - height),
        )
        rows = padded_height // window_height
        columns = padded_width // window_width
        summaries = (
            feature_map.view(
                batch,
                channels,
                rows,
                window_height,
                columns,
                window_width,
            )
            .mean(dim=(3, 5))
            .permute(0, 2, 3, 1)
            .reshape(batch, rows * columns, channels)
        )
        return summaries, (rows, columns), (
            window_height,
            window_width,
        )

    @staticmethod
    def _broadcast_windows(
        window_context: torch.Tensor,
        grid_size: tuple[int, int],
        window_size: tuple[int, int],
        hw_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, _, channels = window_context.shape
        rows, columns = grid_size
        window_height, window_width = window_size
        height, width = hw_size
        context = window_context.view(
            batch,
            rows,
            columns,
            channels,
        ).repeat_interleave(
            window_height,
            dim=1,
        ).repeat_interleave(
            window_width,
            dim=2,
        )
        return context[:, :height, :width].reshape(
            batch,
            height * width,
            channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        hw_size: tuple[int, int],
        registers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update registers from windows and broadcast context back."""
        summaries, grid_size, effective_window = (
            self._window_summaries(x, hw_size)
        )
        summary_tokens = self.summary_projection(
            self.summary_norm(summaries)
        )
        register_delta, _ = self.register_attention(
            self.register_norm(registers),
            summary_tokens,
            summary_tokens,
            need_weights=False,
        )
        registers = registers + register_delta
        registers = registers + self.register_mlp(registers)

        key_padding_mask = None
        broadcast_registers = registers
        if self.training and self.dropout > 0:
            keep = torch.rand(
                registers.shape[:2],
                device=registers.device,
            ) >= self.dropout
            keep[:, 0] = True
            key_padding_mask = ~keep
            broadcast_registers = (
                registers * keep[:, :, None].to(registers.dtype)
            )
        window_context, _ = self.broadcast_attention(
            summary_tokens,
            self.register_norm(broadcast_registers),
            self.register_norm(broadcast_registers),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        token_context = self._broadcast_windows(
            window_context,
            grid_size,
            effective_window,
            hw_size,
        )
        x = x + torch.tanh(self.broadcast_gate) * (
            self.broadcast_projection(token_context)
        )
        return x, registers


class BodySlotReadWrite(nn.Module):
    """Update persistent identity slots from one spatial backbone stage.

    The optional slot-to-spatial path is exactly disabled when its scalar gate
    is zero. EMA teacher buffers reuse the learned RGB memory projection while
    remaining outside the deployed forward path.
    """

    def __init__(
        self,
        spatial_dim: int,
        *,
        slot_dim: int = 128,
        num_slots: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        writeback: bool = False,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if min(spatial_dim, slot_dim, num_slots, num_heads) < 1:
            raise ValueError("Body-slot dimensions and counts must be positive")
        if slot_dim % num_heads:
            raise ValueError(
                f"Body-slot dimension {slot_dim} must divide num_heads={num_heads}"
            )
        if slot_dim % 4:
            raise ValueError(
                "Body-slot dimension must be divisible by four for 2D positions"
            )
        if mlp_ratio <= 0:
            raise ValueError("Body-slot MLP ratio must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Body-slot dropout must satisfy 0 <= value < 1")

        self.spatial_dim = int(spatial_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.writeback_enabled = bool(writeback)
        # The following projection already learns a per-channel scale, so an
        # affine LayerNorm here is redundant. Keeping this normalization
        # non-affine also avoids unstable MPS reductions for affine gradients
        # over the large Stage-0 token set.
        self.memory_norm = nn.LayerNorm(
            self.spatial_dim,
            elementwise_affine=False,
        )
        self.memory_projection = nn.Linear(
            self.spatial_dim,
            self.slot_dim,
            bias=False,
        )
        self.attention_memory_norm = nn.LayerNorm(
            self.slot_dim,
            elementwise_affine=False,
        )
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_attention = nn.MultiheadAttention(
            self.slot_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = max(1, round(self.slot_dim * mlp_ratio))
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.slot_dim),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 1),
        )
        if self.writeback_enabled:
            self.broadcast_attention = nn.MultiheadAttention(
                self.slot_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.broadcast_projection = nn.Linear(
                self.slot_dim,
                self.spatial_dim,
                bias=False,
            )
            self.broadcast_gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.broadcast_attention = None
            self.broadcast_projection = None
            self.register_parameter("broadcast_gate", None)

        self.register_buffer(
            "teacher_projection_weight",
            torch.empty(self.slot_dim, self.spatial_dim),
        )
        self.reset_teacher()

    @staticmethod
    def _position_encoding(
        height: int,
        width: int,
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        quarter_dim = dim // 4
        frequency = torch.arange(
            quarter_dim,
            device=device,
            dtype=torch.float32,
        )
        frequency = 1.0 / (
            10_000 ** (frequency / max(quarter_dim, 1))
        )
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        y = y / max(height - 1, 1) * (2 * torch.pi)
        x = x / max(width - 1, 1) * (2 * torch.pi)
        y_phase = y[:, None] * frequency[None]
        x_phase = x[:, None] * frequency[None]
        y_encoding = torch.cat((y_phase.sin(), y_phase.cos()), dim=1)
        x_encoding = torch.cat((x_phase.sin(), x_phase.cos()), dim=1)
        position = torch.cat(
            (
                y_encoding[:, None].expand(height, width, -1),
                x_encoding[None].expand(height, width, -1),
            ),
            dim=-1,
        )
        return position.reshape(1, height * width, dim).to(dtype=dtype)

    def reset_teacher(self) -> None:
        """Synchronize the training-only EMA projection with the online read path."""
        self.teacher_projection_weight.copy_(
            self.memory_projection.weight.detach()
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA-update the privileged masked-pooling projection."""
        if not 0 <= momentum < 1:
            raise ValueError("Body-slot teacher momentum must be in [0, 1)")
        self.teacher_projection_weight.mul_(momentum).add_(
            self.memory_projection.weight.detach(),
            alpha=1.0 - momentum,
        )

    def _teacher_slots(
        self,
        x: torch.Tensor,
        hw_size: tuple[int, int],
        teacher_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if teacher_masks is None:
            return None, None, None
        if teacher_masks.ndim != 4 or teacher_masks.shape[1] != self.num_slots:
            raise ValueError(
                "Body-slot teacher masks must have shape "
                f"[B,{self.num_slots},H,W], got {tuple(teacher_masks.shape)}"
            )
        masks = F.interpolate(
            teacher_masks.float(),
            size=hw_size,
            mode="area",
        ).clamp(0, 1)
        flat_masks = masks.flatten(2)
        mass = flat_masks.sum(dim=-1)
        valid = mass > 1e-4
        normalized_masks = flat_masks / mass.clamp_min(1e-6)[..., None]
        teacher_memory = F.layer_norm(
            F.linear(
                F.layer_norm(
                    x.detach().float(),
                    (self.spatial_dim,),
                    eps=self.memory_norm.eps,
                ),
                self.teacher_projection_weight.float(),
            ),
            (self.slot_dim,),
            eps=self.attention_memory_norm.eps,
        )
        teacher_slots = torch.einsum(
            "bkn,bnd->bkd",
            normalized_masks,
            teacher_memory,
        )
        return teacher_slots, valid, normalized_masks

    def forward(
        self,
        x: torch.Tensor,
        hw_size: tuple[int, int],
        slots: torch.Tensor,
        role_embeddings: torch.Tensor,
        *,
        teacher_masks: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Read spatial evidence, optionally write slots back, and emit teacher targets."""
        batch, tokens, channels = x.shape
        height, width = hw_size
        if tokens != height * width or channels != self.spatial_dim:
            raise ValueError(
                "Body-slot memory shape does not match its stage: "
                f"{tuple(x.shape)} vs HxW={hw_size}, C={self.spatial_dim}"
            )
        if slots.shape != (batch, self.num_slots, self.slot_dim):
            raise ValueError(
                "Body-slot state must have shape "
                f"[B,{self.num_slots},{self.slot_dim}], got {tuple(slots.shape)}"
            )
        if role_embeddings.shape[-2:] != (self.num_slots, self.slot_dim):
            raise ValueError(
                "Body-slot role embeddings have an incompatible shape"
            )

        memory = self.memory_projection(self.memory_norm(x))
        memory = memory + self._position_encoding(
            height,
            width,
            self.slot_dim,
            device=x.device,
            dtype=memory.dtype,
        )
        attention_memory = self.attention_memory_norm(memory)
        slot_delta, attention = self.slot_attention(
            self.slot_norm(slots + role_embeddings),
            attention_memory,
            attention_memory,
            need_weights=True,
            average_attn_weights=True,
        )
        slots = slots + slot_delta
        slots = slots + self.slot_mlp(slots)
        visibility_logits = self.visibility_head(slots).squeeze(-1)

        # Keep the privileged target independent of the student's optional
        # writeback so Tier C cannot improve its own target through a feedback
        # loop. The detached RGB memory is still stage-matched.
        teacher_slots, teacher_valid, teacher_attention = (
            self._teacher_slots(x, hw_size, teacher_masks)
        )
        if self.writeback_enabled:
            spatial_delta, _ = self.broadcast_attention(
                attention_memory,
                self.slot_norm(slots),
                self.slot_norm(slots),
                need_weights=False,
            )
            x = x + torch.tanh(self.broadcast_gate) * (
                self.broadcast_projection(spatial_delta)
            )

        return (
            x,
            slots,
            visibility_logits,
            attention,
            teacher_slots,
            teacher_valid,
            teacher_attention,
        )


class ConvLayer(nn.Module):
    """Convolutional stage (MBConv blocks)."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
        downsample_stride: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim, dim, conv_expand_ratio, activation, drop_path[i] if isinstance(drop_path, list) else drop_path
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation,
                stride=downsample_stride,
            )
        else:
            self.downsample = None

    def forward(self, x, out_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x, out_size = self.downsample(x, out_size)
        return x, out_size


class TinyViTMlp(nn.Module):
    """MLP with pre-norm for TinyViT blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class RMSFeatureSuppression(nn.Module):
    """Suppress spatial locations with high channel-wise RMS energy.

    The hard mask follows Hi-AFA's feature-suppression rule, but uses RMS
    energy instead of a signed channel mean so opposite activations cannot
    cancel one another.  Energy is min-max normalized independently for each
    sample before applying the threshold.
    """

    def __init__(self, tau: float = 0.7, eps: float = 1e-12) -> None:
        super().__init__()
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.tau = float(tau)
        self.eps = float(eps)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-sample BCHW keep mask from channel RMS energy."""
        if x.ndim != 4:
            raise ValueError(
                "RMSFeatureSuppression expects BCHW input, got "
                f"shape {tuple(x.shape)}"
            )
        energy = x.float().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        flat_energy = energy.flatten(2)
        minimum = flat_energy.amin(dim=2, keepdim=True).unsqueeze(-1)
        maximum = flat_energy.amax(dim=2, keepdim=True).unsqueeze(-1)
        normalized = (energy - minimum) / (maximum - minimum).clamp_min(self.eps)
        return (normalized <= self.tau).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask(x)


class ReIDResidualAdapter(nn.Module):
    """Zero-gated ReID adapter with optional lateral feature suppression."""

    def __init__(
        self,
        dim: int,
        reduction_ratio: int = 4,
        suppression_tau: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction_ratio < 1:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}")
        if not 0.0 <= suppression_tau <= 1.0:
            raise ValueError(
                "suppression_tau must be in [0, 1], got "
                f"{suppression_tau}"
            )
        hidden_dim = max(dim // int(reduction_ratio), 1)
        self.suppression_tau = float(suppression_tau)
        self.suppression = (
            RMSFeatureSuppression(self.suppression_tau)
            if self.suppression_tau > 0.0
            else nn.Identity()
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=hidden_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor, hw_size: tuple[int, int]) -> torch.Tensor:
        B, L, C = x.shape
        H, W = hw_size
        if L != H * W:
            raise ValueError(f"Adapter token count {L} does not match spatial size {hw_size}")
        spatial = x.transpose(1, 2).reshape(B, C, H, W)
        # Suppression is lateral-only: the untouched token stream remains the
        # residual bypass while only the adapter input is masked.
        adapted = self.adapter(self.suppression(spatial)).flatten(2).transpose(1, 2)
        return x + self.gamma * adapted


def _is_window_size(value) -> bool:
    return isinstance(value, int) or (
        isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) for part in value)
    )


def _to_2tuple(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or (height, width) tuple, got {value!r}")


def _expand_block_values(value, depth: int) -> list:
    if _is_window_size(value):
        return [value for _ in range(depth)]
    values = list(value)
    if len(values) != depth:
        raise ValueError(f"Expected {depth} block values, got {len(values)}: {value!r}")
    return values


def _shift_for_window(window_size) -> tuple[int, int]:
    window_h, window_w = _to_2tuple(window_size)
    return window_h // 2, window_w // 2


class TinyViTBlock(nn.Module):
    """TinyViT block: windowed attention + local depthwise conv + MLP."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
        shift_size=0,
        attention_bias: str = "absolute",
        attention_mask: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = _to_2tuple(window_size)
        self.shift_size = _to_2tuple(shift_size)
        if any(shift < 0 for shift in self.shift_size):
            raise ValueError(f"CSL-TinyViT shift_size must be non-negative, got {shift_size}")
        if any(shift >= window for shift, window in zip(self.shift_size, self.window_size, strict=True)):
            raise ValueError(
                f"CSL-TinyViT shift_size {self.shift_size} must be smaller than window_size {self.window_size}"
            )
        self.attention_mask = bool(attention_mask)
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        head_dim = dim // num_heads
        window_resolution = self.window_size
        self.attn = Attention(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution, bias_mode=attention_bias
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TinyViTMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    @staticmethod
    def _partition_windows(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
        window_h, window_w = window_size
        B, H, W, C = x.shape
        nH = H // window_h
        nW = W // window_w
        x = x.view(B, nH, window_h, nW, window_w, C)
        return x.transpose(2, 3).reshape(B * nH * nW, window_h * window_w, C)

    @staticmethod
    def _mask_slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
        if shift == 0:
            return (slice(0, size),)
        return (slice(0, -window), slice(-window, -shift), slice(-shift, None))

    def _window_attention_mask(
        self,
        *,
        batch_size: int,
        original_size: tuple[int, int],
        padded_size: tuple[int, int],
        device: torch.device,
        shift_size: tuple[int, int],
    ) -> torch.Tensor | None:
        window_h, window_w = self.window_size
        shift_h, shift_w = shift_size
        H, W = original_size
        pH, pW = padded_size
        nH = pH // window_h
        nW = pW // window_w
        num_windows = nH * nW
        num_tokens = window_h * window_w
        allowed: torch.Tensor | None = None

        if shift_h > 0 or shift_w > 0:
            region_mask = torch.zeros((1, pH, pW, 1), device=device, dtype=torch.long)
            counter = 0
            for h_slice in self._mask_slices(pH, window_h, shift_h):
                for w_slice in self._mask_slices(pW, window_w, shift_w):
                    region_mask[:, h_slice, w_slice, :] = counter
                    counter += 1
            mask_windows = self._partition_windows(region_mask, self.window_size).view(num_windows, num_tokens)
            allowed = mask_windows[:, :, None] == mask_windows[:, None, :]

        if self.attention_mask and (H != pH or W != pW):
            valid = torch.ones((1, H, W, 1), device=device, dtype=torch.bool)
            valid = F.pad(valid, (0, 0, 0, pW - W, 0, pH - H), value=False)
            if shift_h > 0 or shift_w > 0:
                valid = torch.roll(valid, shifts=(-shift_h, -shift_w), dims=(1, 2))
            valid_windows = self._partition_windows(valid, self.window_size).view(num_windows, num_tokens)
            valid_allowed = valid_windows[:, None, :].expand(num_windows, num_tokens, num_tokens)
            allowed = valid_allowed if allowed is None else allowed & valid_allowed

        if allowed is None:
            return None
        return allowed.repeat(batch_size, 1, 1)

    def forward(self, x, hw_size):
        B, L, C = x.shape
        H, W = hw_size
        assert L == H * W

        res_x = x
        window_h, window_w = self.window_size
        shift_h, shift_w = self.shift_size
        if H <= window_h:
            shift_h = 0
        if W <= window_w:
            shift_w = 0
        active_shift = (shift_h, shift_w)

        if H == window_h and W == window_w and active_shift == (0, 0):
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (window_h - H % window_h) % window_h
            pad_r = (window_w - W % window_w) % window_w
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            if active_shift != (0, 0):
                x = torch.roll(x, shifts=(-active_shift[0], -active_shift[1]), dims=(1, 2))

            nH = pH // window_h
            nW = pW // window_w
            attn_mask = self._window_attention_mask(
                batch_size=B,
                original_size=(H, W),
                padded_size=(pH, pW),
                device=x.device,
                shift_size=active_shift,
            )
            # Window partition
            x = x.view(B, nH, window_h, nW, window_w, C)
            x = x.transpose(2, 3).reshape(B * nH * nW, window_h * window_w, C)
            x = self.attn(x, attn_mask=attn_mask)
            # Window reverse
            x = x.view(B, nH, nW, window_h, window_w, C)
            x = x.transpose(2, 3).reshape(B, pH, pW, C)

            if active_shift != (0, 0):
                x = torch.roll(x, shifts=active_shift, dims=(1, 2))
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        # Local depthwise convolution
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        # MLP
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    """A stage of TinyViT blocks (windowed attention)."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
        shift_size=0,
        attention_bias: str = "absolute",
        attention_mask: bool = False,
        adapter_reduction_ratio: int | None = None,
        adapter_suppression_tau: float = 0.0,
        downsample_stride: int | tuple[int, int] | None = None,
        width_merge_after_blocks: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.width_merge_after_blocks = int(width_merge_after_blocks)
        if self.width_merge_after_blocks < 0 or self.width_merge_after_blocks >= depth:
            if self.width_merge_after_blocks != 0:
                raise ValueError(
                    "width_merge_after_blocks must be zero (disabled) or fall before the final block; "
                    f"got {self.width_merge_after_blocks} for depth={depth}"
                )
        block_window_sizes = _expand_block_values(window_size, depth)
        block_shift_sizes = _expand_block_values(shift_size, depth)

        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=block_window_sizes[i],
                    shift_size=block_shift_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                    attention_bias=attention_bias,
                    attention_mask=attention_mask,
                )
                for i in range(depth)
            ]
        )
        if adapter_reduction_ratio is not None:
            # ReID adapters are zero-gated treatments. Keep their private random
            # initialization from advancing the RNG stream used by the shared
            # backbone, neck, fusion, and head so same-seed ablations remain
            # initialization matched while adapter weights stay deterministic.
            with torch.random.fork_rng(devices=[]):
                reid_adapters = [
                    ReIDResidualAdapter(
                        dim,
                        adapter_reduction_ratio,
                        suppression_tau=adapter_suppression_tau,
                    )
                    for _ in range(depth)
                ]
        else:
            reid_adapters = []
        self.reid_adapters = nn.ModuleList(reid_adapters)

        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                activation=activation,
                stride=downsample_stride,
            )
        else:
            self.downsample = None
        self.width_merge = NormPreservingWidthMerge() if self.width_merge_after_blocks else None

    def forward(
        self,
        x,
        out_size,
        return_pre_downsample: bool = False,
        return_pre_width_merge: bool = False,
    ):
        pre_width_merge = None
        for index, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, out_size, use_reentrant=False)
            else:
                x = blk(x, out_size)
            if self.reid_adapters:
                x = self.reid_adapters[index](x, out_size)
            if self.width_merge is not None and index + 1 == self.width_merge_after_blocks:
                pre_width_merge = (x, out_size)
                x, out_size = self.width_merge(x, out_size)
        pre_downsample = (x, out_size)
        if self.downsample is not None:
            x, out_size = self.downsample(x, out_size)
        if return_pre_downsample:
            return x, out_size, pre_downsample[0], pre_downsample[1]
        if return_pre_width_merge:
            if pre_width_merge is None:
                raise RuntimeError("Requested pre-width-merge tokens from a layer without width merging")
            return x, out_size, pre_width_merge[0], pre_width_merge[1]
        return x, out_size


class LayerNorm2d(nn.Module):
    """LayerNorm for channel-first (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
