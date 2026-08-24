# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
    get_anatomical_target_spec,
)
from boxmot.reid.backbones.families.csl_tinyvit.jigsaw import (
    JigsawPatchAuxiliary,
)
from boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression import (
    MultilevelClassifierSuppression,
)
from boxmot.reid.backbones.families.csl_tinyvit.pooling import (
    ActivatedGeM,
    AnatomicalAuxiliaryPool,
    DSELitePool,
    EMAAnatomicalAuxiliaryPool,
    GeM,
    LearnedPartTokenPool,
    PatternAdapter,
    PrivilegedMaskPoseAttentionAdapter,
    SemanticVisibilityPartPool,
    SpatialTopDrop,
    SpatialTopSuppression,
    StripeVisibilityGate,
)
from boxmot.reid.backbones.families.csl_tinyvit.transport import (
    MCPT_MODES,
    MonotonicCanonicalPartTransport,
)
from boxmot.reid.backbones.head_registry import MULTI_BRANCH_HEAD_TYPES
from boxmot.reid.backbones.heads.bnneck import BNNeck, BNNeck3

__all__ = [
    "BodySlotFeatures",
    "BodySlotHead",
    "BranchSetAttention",
    "GPCLiteMultiBranchHead",
    "HierarchicalBranchAttention",
    "HierarchicalLateInteractionMatcher",
    "LMBNStyleMultiBranchHead",
    "MultiBranchHead",
    "ResidualMultiScaleQueryDecoder",
]


class BodySlotFeatures(NamedTuple):
    """Backbone packet consumed by the two-stream body-slot head."""

    global_map: torch.Tensor
    slots: torch.Tensor
    visibility_logits: torch.Tensor
    stage_slots: tuple[torch.Tensor, ...]
    stage_attentions: tuple[torch.Tensor, ...]
    stage_visibility_logits: tuple[torch.Tensor, ...]
    teacher_slots: tuple[torch.Tensor, ...] | None
    teacher_valid: tuple[torch.Tensor, ...] | None
    teacher_attentions: tuple[torch.Tensor, ...] | None


class BodySlotHead(nn.Module):
    """Build a 512-D global + eight 128-D persistent-slot descriptor."""

    GLOBAL_DIM = 512
    SLOT_DIM = 128
    NUM_SLOTS = 8

    def __init__(
        self,
        in_ch: int,
        *,
        num_classes: int,
        head_pool: str = "gelu_gem",
        alpha: float = 0.45,
        visibility_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if int(in_ch) < 1:
            raise ValueError("Body-slot global input width must be positive")
        if not 0 < alpha < 1:
            raise ValueError("Body-slot alpha must satisfy 0 < alpha < 1")
        if not 0 <= visibility_floor < 1:
            raise ValueError("Body-slot visibility floor must satisfy 0 <= value < 1")
        self.alpha = float(alpha)
        self.visibility_floor = float(visibility_floor)
        self.metric_feature = "raw_concat"
        self.inference_feature = "norm_concat_bn"
        self.branch_metric = False
        self.global_neck = BNNeck3(
            int(in_ch),
            num_classes,
            self.GLOBAL_DIM,
            return_f=True,
        )
        self.slot_necks = nn.ModuleList(
            BNNeck(self.SLOT_DIM, num_classes, return_f=True) for _ in range(self.NUM_SLOTS)
        )
        self.set_pooling(head_pool)

    @staticmethod
    def _make_pool(name: str) -> nn.Module:
        normalized = str(name).lower()
        if normalized == "avg":
            return nn.AdaptiveAvgPool2d((1, 1))
        if normalized == "gem":
            return GeM((1, 1))
        if normalized == "dse":
            return DSELitePool((1, 1))
        if normalized == "gelu_gem":
            return ActivatedGeM(nn.GELU(), (1, 1))
        if normalized == "relu_gem":
            return ActivatedGeM(nn.ReLU(inplace=False), (1, 1))
        if normalized == "softplus_gem":
            return ActivatedGeM(nn.Softplus(), (1, 1))
        raise ValueError(f"Unsupported CSL-TinyViT head_pool: {name}")

    def set_pooling(self, head_pool: str) -> None:
        """Replace only the global pooling operator."""
        self.global_pool = self._make_pool(head_pool)
        self.head_pool = str(head_pool).lower()

    def set_branch_metric(self, branch_metric: bool) -> None:
        """Keep the trainer's generic head hook while rejecting ambiguity."""
        self.branch_metric = bool(branch_metric)

    def reset_reid_initialization(self) -> None:
        """Restore BNNeck/classifier initialization after backbone init."""
        self.global_neck.reset_reid_initialization()
        for neck in self.slot_necks:
            neck.reset_reid_initialization()

    def forward(self, packet: BodySlotFeatures):
        if not isinstance(packet, BodySlotFeatures):
            raise TypeError("BodySlotHead expects a BodySlotFeatures backbone packet")
        slots = packet.slots
        if slots.shape[1:] != (self.NUM_SLOTS, self.SLOT_DIM):
            raise ValueError(f"Body-slot head expected slots [B,8,128], got {tuple(slots.shape)}")
        global_output = self.global_neck(self.global_pool(packet.global_map))
        slot_outputs = tuple(neck(slots[:, index]) for index, neck in enumerate(self.slot_necks))
        global_bn, global_logits, global_raw = global_output
        slot_bn = torch.stack([output[0] for output in slot_outputs], dim=1)
        slot_raw = torch.stack([output[2] for output in slot_outputs], dim=1)
        slot_logits = [output[1] for output in slot_outputs]

        visibility = packet.visibility_logits.sigmoid()
        descriptor_visibility = visibility.detach() if self.training else visibility
        slot_weights = descriptor_visibility.clamp_min(self.visibility_floor)
        slot_weights = slot_weights / slot_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        global_descriptor = F.normalize(global_bn, p=2, dim=1) * math.sqrt(self.alpha)
        slot_descriptor = F.normalize(slot_bn, p=2, dim=-1) * ((1.0 - self.alpha) * slot_weights).sqrt()[..., None]
        descriptor = F.normalize(
            torch.cat((global_descriptor, slot_descriptor.flatten(1)), dim=1),
            p=2,
            dim=1,
        )
        if not self.training:
            return descriptor

        features = {
            "global": global_raw,
            **{f"part{index}": slot_raw[:, index] for index in range(self.NUM_SLOTS)},
            "raw_mean": global_raw,
            "raw_concat": descriptor,
            "concat_bn": torch.cat((global_bn, slot_bn.flatten(1)), dim=1),
            "norm_concat_bn": descriptor,
            "_visibility": visibility,
            "_body_slot_final_slots": slots,
            "_body_slot_stage_slots": packet.stage_slots,
            "_body_slot_stage_attentions": packet.stage_attentions,
            "_body_slot_stage_visibility_logits": (packet.stage_visibility_logits),
        }
        if packet.teacher_slots is not None:
            features["_body_slot_teacher_slots"] = packet.teacher_slots
            features["_body_slot_teacher_valid"] = packet.teacher_valid
            features["_body_slot_teacher_attentions"] = packet.teacher_attentions
        return [global_logits, *slot_logits], features


class ResidualMultiScaleQueryDecoder(nn.Module):
    """Use seven pooled descriptors to retrieve evidence from three spatial maps."""

    def __init__(
        self,
        input_dim: int = 512,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError("query-decoder dimensions must be positive and token_dim divisible by num_heads")
        if token_dim % 4:
            raise ValueError("query-decoder token_dim must be divisible by four for 2D sine/cosine positions")
        if num_layers < 1 or mlp_ratio <= 0:
            raise ValueError("query-decoder layer count and MLP ratio must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("query-decoder dropout must satisfy 0 <= value < 1")

        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.query_projection = nn.Linear(input_dim, token_dim, bias=False)
        self.memory_projection = nn.Conv2d(input_dim, token_dim, kernel_size=1, bias=False)
        self.branch_embedding = nn.Parameter(torch.empty(1, 7, token_dim))
        self.scale_embedding = nn.Parameter(torch.empty(3, token_dim))
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=max(1, round(token_dim * mlp_ratio)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.output_projection = nn.Linear(token_dim, input_dim, bias=False)
        nn.init.trunc_normal_(self.branch_embedding, std=0.02)
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Make the decoder exactly preserve all pooled branches at initialization."""
        nn.init.zeros_(self.output_projection.weight)

    def has_identity_output(self) -> bool:
        """Return whether the residual output is still exactly zero."""
        return not bool(torch.count_nonzero(self.output_projection.weight.detach()).item())

    def _position_encoding(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        quarter_dim = self.token_dim // 4
        frequency = torch.arange(quarter_dim, device=device, dtype=torch.float32)
        frequency = 1.0 / (10_000 ** (frequency / max(quarter_dim, 1)))
        y_position = torch.arange(height, device=device, dtype=torch.float32)
        x_position = torch.arange(width, device=device, dtype=torch.float32)
        y_position = y_position / max(height - 1, 1) * (2 * math.pi)
        x_position = x_position / max(width - 1, 1) * (2 * math.pi)
        y_phase = y_position[:, None] * frequency[None, :]
        x_phase = x_position[:, None] * frequency[None, :]
        y_encoding = torch.cat((y_phase.sin(), y_phase.cos()), dim=1)
        x_encoding = torch.cat((x_phase.sin(), x_phase.cos()), dim=1)
        position = torch.cat(
            (
                y_encoding[:, None, :].expand(height, width, -1),
                x_encoding[None, :, :].expand(height, width, -1),
            ),
            dim=2,
        )
        return position.reshape(1, height * width, self.token_dim).to(dtype=dtype)

    def forward(
        self,
        branches: torch.Tensor,
        spatial_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if branches.ndim != 3 or branches.shape[1:] != (7, self.input_dim):
            raise ValueError(f"query decoder expects branches [B, 7, {self.input_dim}], got {tuple(branches.shape)}")
        if len(spatial_maps) != 3:
            raise ValueError(f"query decoder expects three spatial maps, got {len(spatial_maps)}")

        memory_tokens = []
        for scale_index, feature_map in enumerate(spatial_maps):
            if feature_map.ndim != 4 or feature_map.shape[1] != self.input_dim:
                raise ValueError(
                    f"query decoder expects [B, {self.input_dim}, H, W] maps, got {tuple(feature_map.shape)}"
                )
            height, width = feature_map.shape[-2:]
            projected = self.memory_projection(feature_map).flatten(2).transpose(1, 2)
            projected = projected + self._position_encoding(
                height,
                width,
                device=feature_map.device,
                dtype=projected.dtype,
            )
            projected = projected + self.scale_embedding[scale_index].view(1, 1, -1)
            memory_tokens.append(projected)
        memory = torch.cat(memory_tokens, dim=1)

        queries = self.query_projection(branches) + self.branch_embedding
        for layer in self.layers:
            queries = layer(queries, memory)
        correction = self.output_projection(queries)
        return branches + correction


class BranchSetAttention(nn.Module):
    """Unmasked residual self-attention over seven equal-width pooled branches."""

    def __init__(
        self,
        input_dim: int = 512,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError("branch-set dimensions must be positive and token_dim divisible by num_heads")
        if num_layers < 1 or mlp_ratio <= 0:
            raise ValueError("branch-set layer count and MLP ratio must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("branch-set dropout must satisfy 0 <= value < 1")

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, token_dim, bias=False)
        self.branch_embedding = nn.Parameter(torch.empty(1, 7, token_dim))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=max(1, round(token_dim * mlp_ratio)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.output_proj = nn.Linear(token_dim, input_dim, bias=False)
        nn.init.trunc_normal_(self.branch_embedding, std=0.02)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Make the branch set exactly equal to its input at initialization."""
        nn.init.zeros_(self.output_proj.weight)

    def has_identity_output(self) -> bool:
        """Return whether the residual output is still exactly zero."""
        return not bool(torch.count_nonzero(self.output_proj.weight.detach()).item())

    def forward(self, branches: torch.Tensor) -> torch.Tensor:
        if branches.ndim != 3 or branches.shape[1] != 7:
            raise ValueError(f"branch-set attention expects [B, 7, C], got {tuple(branches.shape)}")
        tokens = self.input_proj(self.input_norm(branches)) + self.branch_embedding
        refined_tokens = tokens
        for block in self.blocks:
            refined_tokens = block(refined_tokens)
        correction = self.output_proj(refined_tokens - tokens)
        return branches + correction


class HierarchicalBranchAttention(nn.Module):
    """Tree-masked descriptor-token attention with identity initialization."""

    LEVEL_IDS = (0, 1, 1, 2, 2, 2, 2)
    PARENT_IDS = (0, 1, 2, 1, 1, 2, 2)
    ALLOWED_ATTENTION = (
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 3, 4),
        (0, 2, 5, 6),
        (0, 1, 3, 4),
        (0, 1, 3, 4),
        (0, 2, 5, 6),
        (0, 2, 5, 6),
    )

    def __init__(
        self,
        global_dim: int,
        coarse_dim: int,
        fine_dim: int,
        *,
        token_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError("branch attention token_dim must be positive and divisible by num_heads")
        if num_layers < 1 or mlp_ratio <= 0:
            raise ValueError("branch attention num_layers and mlp_ratio must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("branch attention dropout must satisfy 0 <= value < 1")

        self.global_in = nn.Linear(global_dim, token_dim, bias=False)
        self.coarse_in = nn.Linear(coarse_dim, token_dim, bias=False)
        self.fine_in = nn.Linear(fine_dim, token_dim, bias=False)
        self.level_embed = nn.Parameter(torch.zeros(3, token_dim))
        self.position_embed = nn.Parameter(torch.zeros(7, token_dim))
        self.parent_embed = nn.Parameter(torch.zeros(3, token_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=max(1, round(token_dim * mlp_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.global_out = nn.Linear(token_dim, global_dim, bias=False)
        self.coarse_out = nn.Linear(token_dim, coarse_dim, bias=False)
        self.fine_out = nn.Linear(token_dim, fine_dim, bias=False)

        allowed = torch.zeros(7, 7, dtype=torch.bool)
        for query, keys in enumerate(self.ALLOWED_ATTENTION):
            allowed[query, list(keys)] = True
        self.register_buffer("attention_mask", ~allowed, persistent=False)
        self.register_buffer("level_ids", torch.tensor(self.LEVEL_IDS), persistent=False)
        self.register_buffer("parent_ids", torch.tensor(self.PARENT_IDS), persistent=False)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Make every residual correction exactly zero at initialization."""
        nn.init.zeros_(self.global_out.weight)
        nn.init.zeros_(self.coarse_out.weight)
        nn.init.zeros_(self.fine_out.weight)

    def has_identity_output(self) -> bool:
        """Return whether all hierarchy corrections are still exactly zero."""
        return not any(
            bool(torch.count_nonzero(layer.weight.detach()).item())
            for layer in (self.global_out, self.coarse_out, self.fine_out)
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        coarse_features: tuple[torch.Tensor, torch.Tensor],
        fine_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if len(coarse_features) != 2 or len(fine_features) != 4:
            raise ValueError("hierarchical branch attention requires one global, two coarse, and four fine branches")
        tokens = torch.stack(
            [
                self.global_in(global_feature),
                *[self.coarse_in(feature) for feature in coarse_features],
                *[self.fine_in(feature) for feature in fine_features],
            ],
            dim=1,
        )
        tokens = (
            tokens
            + self.level_embed[self.level_ids].unsqueeze(0)
            + self.position_embed.unsqueeze(0)
            + self.parent_embed[self.parent_ids].unsqueeze(0)
        )
        attended = self.encoder(tokens, mask=self.attention_mask)
        refined_global = global_feature + self.global_out(attended[:, 0])
        refined_coarse = tuple(
            feature + self.coarse_out(attended[:, index + 1]) for index, feature in enumerate(coarse_features)
        )
        refined_fine = tuple(
            feature + self.fine_out(attended[:, index + 3]) for index, feature in enumerate(fine_features)
        )
        return refined_global, refined_coarse, refined_fine


class HierarchicalLateInteractionMatcher(nn.Module):
    """Pair-conditioned, tree-biased Sinkhorn matcher for seven ReID branches."""

    LEVEL_IDS = (0, 1, 1, 2, 2, 2, 2)
    PARENT_IDS = (0, 1, 2, 1, 1, 2, 2)
    FINE_PARENT_TOKEN = (1, 1, 2, 2)

    def __init__(
        self,
        global_dim: int,
        coarse_dim: int,
        fine_dim: int,
        *,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        sinkhorn_iters: int = 5,
        null_tokens: int = 1,
        base_score_init: float = 0.9,
    ) -> None:
        super().__init__()
        if token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError("late-interaction token_dim must be positive and divisible by num_heads")
        if num_layers < 1 or sinkhorn_iters < 1:
            raise ValueError("late-interaction layer and Sinkhorn iteration counts must be positive")
        if null_tokens != 1:
            raise ValueError("hierarchical late interaction currently requires exactly one null token")
        if not 0 < base_score_init < 1:
            raise ValueError("late-interaction base_score_init must satisfy 0 < value < 1")

        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.null_tokens = int(null_tokens)
        self.num_real_tokens = 7
        self.num_tokens = self.num_real_tokens + self.null_tokens

        self.global_in = nn.Linear(global_dim, token_dim, bias=False)
        self.coarse_in = nn.Linear(coarse_dim, token_dim, bias=False)
        self.fine_in = nn.Linear(fine_dim, token_dim, bias=False)
        self.level_embed = nn.Parameter(torch.zeros(3, token_dim))
        self.position_embed = nn.Parameter(torch.zeros(self.num_real_tokens, token_dim))
        self.parent_embed = nn.Parameter(torch.zeros(3, token_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.query_projections = nn.ModuleList(nn.Linear(token_dim, token_dim, bias=False) for _ in range(num_layers))
        self.key_projections = nn.ModuleList(nn.Linear(token_dim, token_dim, bias=False) for _ in range(num_layers))
        self.value_projections = nn.ModuleList(nn.Linear(token_dim, token_dim, bias=False) for _ in range(num_layers))
        self.output_projections = nn.ModuleList(nn.Linear(token_dim, token_dim, bias=False) for _ in range(num_layers))
        self.token_norms = nn.ModuleList(nn.LayerNorm(token_dim) for _ in range(num_layers))

        self.tree_bias = nn.Parameter(self._initial_tree_bias())
        self.position_bias = nn.Parameter(torch.zeros(self.num_tokens, self.num_tokens))
        self.parent_context_scale = nn.Parameter(torch.tensor(0.25))
        self.global_context_scale = nn.Parameter(torch.tensor(0.25))
        initial_logit = math.log(base_score_init / (1.0 - base_score_init))
        self.base_score_logit = nn.Parameter(torch.tensor(initial_logit))

        self.register_buffer("level_ids", torch.tensor(self.LEVEL_IDS), persistent=False)
        self.register_buffer("parent_ids", torch.tensor(self.PARENT_IDS), persistent=False)
        self.register_buffer(
            "fine_parent_tokens",
            torch.tensor(self.FINE_PARENT_TOKEN),
            persistent=False,
        )

    def _initial_tree_bias(self) -> torch.Tensor:
        """Favor matching within the known global/upper/lower hierarchy."""
        bias = torch.full((self.num_tokens, self.num_tokens), -1.0)
        bias[0, :] = 0.0
        bias[:, 0] = 0.0
        bias[-1, :] = 0.0
        bias[:, -1] = 0.0
        families = (0, 1, 2, 1, 1, 2, 2)
        for query_index in range(self.num_real_tokens):
            for gallery_index in range(self.num_real_tokens):
                if families[query_index] == families[gallery_index]:
                    bias[query_index, gallery_index] = 0.0
                elif query_index >= 3 and gallery_index >= 3 and abs(query_index - gallery_index) == 1:
                    bias[query_index, gallery_index] = -0.25
        return bias

    def encode(
        self,
        global_feature: torch.Tensor,
        coarse_features: tuple[torch.Tensor, torch.Tensor],
        fine_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Project one image's seven evidence branches into the shared token space."""
        if len(coarse_features) != 2 or len(fine_features) != 4:
            raise ValueError("late interaction requires one global, two coarse, and four fine branches")
        tokens = torch.stack(
            [
                self.global_in(global_feature),
                *[self.coarse_in(feature) for feature in coarse_features],
                *[self.fine_in(feature) for feature in fine_features],
            ],
            dim=1,
        )
        tokens = (
            tokens
            + self.level_embed[self.level_ids].unsqueeze(0)
            + self.position_embed.unsqueeze(0)
            + self.parent_embed[self.parent_ids].unsqueeze(0)
        )
        null_token = self.null_token.expand(tokens.shape[0], -1, -1)
        return torch.cat((tokens, null_token), dim=1)

    def _pair_logits(
        self,
        query_tokens: torch.Tensor,
        gallery_tokens: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        head_dim = self.token_dim // self.num_heads

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(values.shape[0], self.num_tokens, self.num_heads, head_dim).transpose(1, 2)

        query_q = split_heads(self.query_projections[layer_index](query_tokens))
        query_k = split_heads(self.key_projections[layer_index](query_tokens))
        gallery_q = split_heads(self.query_projections[layer_index](gallery_tokens))
        gallery_k = split_heads(self.key_projections[layer_index](gallery_tokens))
        forward_logits = torch.einsum("bhid,bhjd->bhij", query_q, gallery_k)
        reverse_logits = torch.einsum("bhjd,bhid->bhji", gallery_q, query_k).transpose(-1, -2)
        logits = 0.5 * (forward_logits + reverse_logits)
        logits = logits.mean(dim=1) / math.sqrt(head_dim)

        tree_bias = 0.5 * (self.tree_bias + self.tree_bias.transpose(0, 1))
        position_bias = 0.5 * (self.position_bias + self.position_bias.transpose(0, 1))
        logits = logits + tree_bias.unsqueeze(0) + position_bias.unsqueeze(0)

        normalized_query = F.normalize(query_tokens[:, : self.num_real_tokens], p=2, dim=-1)
        normalized_gallery = F.normalize(gallery_tokens[:, : self.num_real_tokens], p=2, dim=-1)
        token_similarity = torch.einsum("bid,bjd->bij", normalized_query, normalized_gallery)
        parent_similarity = token_similarity[:, self.fine_parent_tokens[:, None], self.fine_parent_tokens[None, :]]
        global_similarity = token_similarity[:, 0, 0].view(-1, 1, 1)
        logits[:, 3:7, 3:7] = (
            logits[:, 3:7, 3:7]
            + self.parent_context_scale * parent_similarity
            + self.global_context_scale * global_similarity
        )
        return logits

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a differentiable doubly-stochastic alignment plan."""
        log_plan = logits - logits.amax(dim=(-1, -2), keepdim=True)
        for _ in range(self.sinkhorn_iters):
            log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
            log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True)
        return log_plan.exp()

    def score_pairs(
        self,
        query_hierarchy: tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        gallery_hierarchy: tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        query_base: torch.Tensor,
        gallery_base: torch.Tensor,
    ) -> torch.Tensor:
        """Score aligned query/gallery pairs with symmetric late interaction."""
        query_tokens = self.encode(*query_hierarchy)
        gallery_tokens = self.encode(*gallery_hierarchy)
        plan = None
        for layer_index in range(self.num_layers):
            logits = self._pair_logits(query_tokens, gallery_tokens, layer_index)
            plan = self._sinkhorn(logits)
            query_values = self.value_projections[layer_index](query_tokens)
            gallery_values = self.value_projections[layer_index](gallery_tokens)
            query_context = torch.bmm(plan, gallery_values)
            gallery_context = torch.bmm(plan.transpose(1, 2), query_values)
            query_tokens = self.token_norms[layer_index](
                query_tokens + self.output_projections[layer_index](query_context)
            )
            gallery_tokens = self.token_norms[layer_index](
                gallery_tokens + self.output_projections[layer_index](gallery_context)
            )

        if plan is None:
            raise RuntimeError("late-interaction matcher produced no alignment plan")
        value_similarity = torch.einsum(
            "bid,bjd->bij",
            F.normalize(query_tokens, p=2, dim=-1),
            F.normalize(gallery_tokens, p=2, dim=-1),
        )
        real_plan = plan[:, : self.num_real_tokens, : self.num_real_tokens] / self.num_tokens
        local_score = (real_plan * value_similarity[:, : self.num_real_tokens, : self.num_real_tokens]).sum(
            dim=(-1, -2)
        )
        base_score = torch.sum(
            F.normalize(query_base, p=2, dim=1) * F.normalize(gallery_base, p=2, dim=1),
            dim=1,
        )
        base_weight = self.base_score_logit.sigmoid()
        return base_weight * base_score + (1.0 - base_weight) * local_score


class MultiBranchHead(nn.Module):
    """Multi-granularity head with fixed stripes or learned part tokens.

    Produces:
      - Training: (cls_scores_list, features_tensor)
      - Inference: (B, feat_dim × num_branches) concatenated features
    """

    SPECIALIST_MODES = MULTI_BRANCH_HEAD_TYPES
    SPECIALIST_DIM = 128
    STAGE2_GLOBAL_WEIGHT = 0.25
    STAGE2_CHANNEL_WEIGHT = 0.20
    SUPPRESSED_GLOBAL_WEIGHT = 0.20
    SPECIALIST_GATE_INIT = 0.225

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        scale_balanced_branches: bool = False,
        head_parts: tuple[int, ...] = (1, 2),
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        stripe_visibility: bool = False,
        drop_global_aux: bool = False,
        drop_global_aux_ratio: float = 0.25,
        evidence_num_roles: int = 8,
        anatomical_auxiliary: bool = False,
        anatomical_token_dim: int = 128,
        anatomical_descriptor_distill: bool = False,
        anatomical_branch_distill: bool = False,
        anatomical_multiscale: bool = False,
        anatomical_target_type: str = DEFAULT_ANATOMICAL_TARGET_TYPE,
        anatomical_accessory_query: bool = False,
        anatomical_deployment: bool = False,
        anatomical_deployment_dim: int = 64,
        anatomical_deployment_alpha: float = 0.25,
        hierarchical_scales: bool = False,
        compact_deployment_head: bool = False,
        specialist_mode: str = "standard",
        multiscale_channel_alpha: float = 0.5,
        return_cross_scale_features: bool = False,
        return_treeboost_features: bool = False,
        return_auxiliary_features: bool = False,
        multilevel_suppression: bool = False,
        multilevel_suppression_ratio: float = 0.15,
        hierarchical_branch_attention: bool = False,
        branch_attention_token_dim: int = 96,
        branch_attention_num_heads: int = 4,
        branch_attention_num_layers: int = 1,
        branch_attention_mlp_ratio: float = 2.0,
        branch_attention_dropout: float = 0.0,
        branch_set_attention: bool = False,
        branch_set_attention_token_dim: int = 128,
        branch_set_attention_num_heads: int = 4,
        branch_set_attention_num_layers: int = 1,
        branch_set_attention_mlp_ratio: float = 2.0,
        branch_set_attention_dropout: float = 0.0,
        multiscale_query_decoder: bool = False,
        query_decoder_dim: int = 128,
        query_decoder_num_heads: int = 4,
        query_decoder_num_layers: int = 1,
        query_decoder_mlp_ratio: float = 2.0,
        query_decoder_dropout: float = 0.0,
        hierarchical_late_interaction: bool = False,
        late_interaction_dim: int = 128,
        late_interaction_num_heads: int = 4,
        late_interaction_num_layers: int = 1,
        late_interaction_sinkhorn_iters: int = 5,
        late_interaction_null_tokens: int = 1,
        late_interaction_base_score_init: float = 0.9,
        mcpt_mode: str = "none",
        mcpt_hidden_dim: int = 64,
        mcpt_max_displacement: float = 0.15,
        mcpt_start_epoch: int = 10,
        mcpt_ramp_end_epoch: int = 40,
        jpm: bool = False,
        jpm_num_groups: int = 4,
        jpm_shift: int = 5,
        jpm_token_dim: int = 96,
        jpm_num_heads: int = 4,
        jpm_mlp_ratio: float = 4.0,
        jpm_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(in_ch, int):
            global_in_ch = local_in_ch = fine_in_ch = int(in_ch)
        else:
            channel_values = tuple(int(value) for value in in_ch)
            if len(channel_values) != 3 or any(value < 1 for value in channel_values):
                raise ValueError(f"Expected positive (global, local, fine) input channels, got {in_ch!r}")
            global_in_ch, local_in_ch, fine_in_ch = channel_values
        self.branch_input_channels = (global_in_ch, local_in_ch, fine_in_ch)
        self.metric_feature = metric_feature
        self.inference_feature = inference_feature
        self.branch_metric = branch_metric
        self.scale_balanced_branches = bool(scale_balanced_branches)
        self.drop_global_aux_enabled = bool(drop_global_aux)
        self.drop_global_aux_ratio = float(drop_global_aux_ratio)
        if not 0 < self.drop_global_aux_ratio <= 1:
            raise ValueError(f"drop_global_aux_ratio must satisfy 0 < value <= 1, got {drop_global_aux_ratio}")
        self.part_pooling = str(part_pooling).lower()
        if self.part_pooling in {"soft_stripes", "overlapping_stripes"}:
            self.part_pooling = "overlap_stripes"
        if self.part_pooling in {"semantic", "semantic_tokens", "semantic_visibility"}:
            self.part_pooling = "semantic_parts"
        if self.part_pooling not in {"stripes", "overlap_stripes", "tokens", "semantic_parts"}:
            raise ValueError(f"Unsupported CSL-TinyViT part_pooling: {part_pooling}")
        self.num_part_tokens = int(num_part_tokens)
        self.evidence_num_roles = int(evidence_num_roles)
        self.anatomical_auxiliary_enabled = bool(anatomical_auxiliary)
        # Runtime-only schedule state. Keeping this as a plain attribute avoids
        # changing checkpoints while allowing expired auxiliary work to be
        # removed from the training graph.
        self.anatomical_auxiliary_runtime_active = True
        # HP-GRD may expose a shared RGB feature map to the trainer for fixed
        # mask pooling.  This plain runtime flag adds no parameter, buffer, or
        # inference operation to the deployed model.
        self.hpgrd_part_packet_runtime_active = False
        self.retrieval_packet_runtime_active = False
        self.anatomical_token_dim = int(anatomical_token_dim)
        self.anatomical_descriptor_distill_enabled = bool(anatomical_descriptor_distill)
        self.anatomical_branch_distill_enabled = bool(anatomical_branch_distill)
        self.anatomical_multiscale_enabled = bool(anatomical_multiscale)
        self.anatomical_deployment_enabled = bool(anatomical_deployment)
        self.anatomical_deployment_dim = int(anatomical_deployment_dim)
        self.anatomical_deployment_alpha = float(anatomical_deployment_alpha)
        self.anatomical_target_type = str(anatomical_target_type).lower()
        self.anatomical_accessory_query_enabled = bool(anatomical_accessory_query)
        anatomical_target = get_anatomical_target_spec(self.anatomical_target_type)
        self.anatomical_pose_teacher_enabled = anatomical_target.uses_ema_teacher
        self.anatomical_decoupled_query_teacher_enabled = anatomical_target.uses_decoupled_queries
        self.anatomical_semantic_teacher_enabled = anatomical_target.uses_semantic_teacher
        self.anatomical_privileged_attention_enabled = anatomical_target.uses_privileged_attention
        if self.anatomical_token_dim < 1:
            raise ValueError("anatomical_token_dim must be positive")
        if self.anatomical_descriptor_distill_enabled and not self.anatomical_auxiliary_enabled:
            raise ValueError("anatomical descriptor distillation requires the anatomical auxiliary branch")
        if self.anatomical_branch_distill_enabled and not self.anatomical_auxiliary_enabled:
            raise ValueError("anatomical branch distillation requires the anatomical auxiliary branch")
        if self.anatomical_multiscale_enabled and not self.anatomical_auxiliary_enabled:
            raise ValueError("multi-scale anatomy requires the anatomical auxiliary branch")
        if self.anatomical_privileged_attention_enabled and not self.anatomical_multiscale_enabled:
            raise ValueError("privileged mask-pose attention requires multi-scale anatomy")
        if self.anatomical_privileged_attention_enabled and self.anatomical_descriptor_distill_enabled:
            raise ValueError("privileged mask-pose attention does not use descriptor distillation")
        if self.anatomical_deployment_enabled:
            if not self.anatomical_auxiliary_enabled:
                raise ValueError("anatomical deployment requires the anatomical auxiliary branch")
            if not self.anatomical_pose_teacher_enabled:
                raise ValueError("anatomical deployment requires anatomical_target_type='learned_pose_concat_ema'")
            if not self.anatomical_multiscale_enabled:
                raise ValueError("anatomical deployment requires multi-scale anatomy")
            if self.anatomical_deployment_dim < 1:
                raise ValueError("anatomical_deployment_dim must be positive")
            if not 0 < self.anatomical_deployment_alpha <= 1:
                raise ValueError("anatomical_deployment_alpha must satisfy 0 < value <= 1")
            if self.anatomical_descriptor_distill_enabled:
                raise ValueError("anatomical deployment and descriptor distillation are independent treatments")
            if self.anatomical_branch_distill_enabled:
                raise ValueError("anatomical deployment and stripe branch distillation are independent treatments")
        self.hierarchical_scales = bool(hierarchical_scales)
        self.compact_deployment_head = bool(compact_deployment_head)
        self.return_cross_scale_features = bool(return_cross_scale_features)
        self.return_treeboost_features = bool(return_treeboost_features)
        self.return_auxiliary_features = bool(return_auxiliary_features)
        self.multilevel_suppression_enabled = bool(multilevel_suppression)
        self.multilevel_suppression_ratio = float(multilevel_suppression_ratio)
        self.head_parts = self._normalize_head_parts(head_parts)
        self.specialist_mode = str(specialist_mode).lower()
        if self.specialist_mode not in self.SPECIALIST_MODES:
            raise ValueError(f"Unsupported CSL-TinyViT specialist head: {specialist_mode}")
        self.multiscale_channel_alpha = float(multiscale_channel_alpha)
        if not 0 <= self.multiscale_channel_alpha <= 1:
            raise ValueError("multiscale_channel_alpha must be in [0, 1]")
        self.has_stage2_pg = self.specialist_mode in {
            "stage2_pg",
            "stage2_pg_gate",
            "stage2_gpc_lite",
            "stage2_gpc_lite_gate",
        }
        self.has_stage2_channels = self.specialist_mode in {
            "stage2_channel2",
            "stage2_gpc_lite",
            "stage2_gpc_lite_gate",
        }
        self.has_multiscale_channels = self.specialist_mode == "multiscale_channel2"
        self.has_specialist_gate = self.specialist_mode in {
            "stage2_pg_gate",
            "stage2_gpc_lite_gate",
        }
        self.has_suppressed_global = self.specialist_mode == "suppressed_global"
        self.has_specialists = self.specialist_mode != "standard"
        if self.anatomical_branch_distill_enabled:
            if not self.anatomical_pose_teacher_enabled:
                raise ValueError(
                    "Anatomical branch distillation requires anatomical_target_type='learned_pose_concat_ema'"
                )
            if not self.anatomical_multiscale_enabled:
                raise ValueError("Anatomical branch distillation requires multi-scale anatomy")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("Anatomical branch distillation requires hierarchical head_parts=(1, 2, 4)")
            if not self.scale_balanced_branches:
                raise ValueError("Anatomical branch distillation requires scale-balanced branches")
            if self.part_pooling != "stripes":
                raise ValueError("Anatomical branch distillation requires fixed stripe pooling")
            if self.has_specialists:
                raise ValueError("Anatomical branch distillation requires the standard head")
            if self.compact_deployment_head:
                raise ValueError("Anatomical branch distillation does not support the compact deployment descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("Anatomical branch distillation requires inference_feature='norm_concat_bn'")
        if self.anatomical_descriptor_distill_enabled and self.inference_feature != "norm_concat_bn":
            raise ValueError(
                "Anatomical descriptor distillation requires "
                "inference_feature='norm_concat_bn' so it supervises the "
                "deployed descriptor"
            )
        if self.anatomical_descriptor_distill_enabled and self.compact_deployment_head:
            raise ValueError(
                "Anatomical descriptor distillation does not yet support the compact deployment descriptor"
            )
        if self.anatomical_deployment_enabled:
            if self.compact_deployment_head:
                raise ValueError("anatomical deployment does not support the compact deployment head")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("anatomical deployment requires inference_feature='norm_concat_bn'")
        if self.has_specialists:
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("G/P/C specialists require hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("G/P/C specialists require fixed stripe pooling")
        if self.has_multiscale_channels and not self.scale_balanced_branches:
            raise ValueError("Multi-scale channels require scale-balanced branches")
        self.hierarchical_branch_attention_enabled = bool(hierarchical_branch_attention)
        if self.hierarchical_branch_attention_enabled:
            if self.has_specialists:
                raise ValueError("hierarchical branch attention requires the standard head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("hierarchical branch attention requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("hierarchical branch attention requires fixed stripe pooling")
            if self.compact_deployment_head:
                raise ValueError("hierarchical branch attention does not support the compact deployment head")
            if feat_dim % 4:
                raise ValueError("hierarchical branch attention requires feat_dim divisible by four")
        self.branch_set_attention_enabled = bool(branch_set_attention)
        if self.branch_set_attention_enabled:
            if self.hierarchical_branch_attention_enabled:
                raise ValueError("tree attention and branch-set attention are independent treatments")
            if self.has_specialists:
                raise ValueError("branch-set attention requires the standard head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("branch-set attention requires hierarchical head_parts=(1, 2, 4)")
            if not self.scale_balanced_branches:
                raise ValueError("branch-set attention requires scale-balanced branches")
            if self.part_pooling != "stripes":
                raise ValueError("branch-set attention requires fixed stripe pooling")
            if self.compact_deployment_head:
                raise ValueError("branch-set attention does not support the compact deployment head")
            if len(set(self.branch_input_channels)) != 1:
                raise ValueError("branch-set attention requires equal-width global, coarse, and fine pooled features")
        self.multiscale_query_decoder_enabled = bool(multiscale_query_decoder)
        if self.multiscale_query_decoder_enabled:
            if self.hierarchical_branch_attention_enabled or self.branch_set_attention_enabled:
                raise ValueError("descriptor attention and the multi-scale query decoder are independent treatments")
            if self.has_specialists:
                raise ValueError("multi-scale query decoder requires the standard head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("multi-scale query decoder requires hierarchical head_parts=(1, 2, 4)")
            if not self.scale_balanced_branches:
                raise ValueError("multi-scale query decoder requires scale-balanced branches")
            if self.part_pooling != "stripes":
                raise ValueError("multi-scale query decoder requires fixed stripe pooling")
            if self.compact_deployment_head:
                raise ValueError("multi-scale query decoder does not support the compact deployment head")
            if len(set(self.branch_input_channels)) != 1:
                raise ValueError("multi-scale query decoder requires equal-width global, coarse, and fine maps")
        self.hierarchical_late_interaction_enabled = bool(hierarchical_late_interaction)
        if self.hierarchical_late_interaction_enabled:
            if (
                self.hierarchical_branch_attention_enabled
                or self.branch_set_attention_enabled
                or self.multiscale_query_decoder_enabled
            ):
                raise ValueError("attention decoders and hierarchical late interaction are independent treatments")
            if self.has_specialists:
                raise ValueError("hierarchical late interaction requires the standard head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("hierarchical late interaction requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("hierarchical late interaction requires fixed stripe pooling")
            if self.compact_deployment_head:
                raise ValueError("hierarchical late interaction does not support the compact deployment head")
            if feat_dim % 4:
                raise ValueError("hierarchical late interaction requires feat_dim divisible by four")
        if self.evidence_num_roles < 1:
            raise ValueError(f"evidence_num_roles must be positive, got {evidence_num_roles}")
        if self.scale_balanced_branches and self.part_pooling not in {"stripes", "overlap_stripes"}:
            raise ValueError("scale-balanced branches require fixed or overlapping stripe pooling")
        if self.hierarchical_scales and self.head_parts != (1, 2, 4):
            raise ValueError("hierarchical FPN requires head_parts=(1, 2, 4)")
        self.mcpt_mode = str(mcpt_mode).lower()
        if self.mcpt_mode not in MCPT_MODES:
            raise ValueError(f"mcpt_mode must be one of {sorted(MCPT_MODES)}, got {mcpt_mode!r}")
        self.mcpt_enabled = self.mcpt_mode != "none"
        if self.mcpt_enabled:
            if self.has_specialists:
                raise ValueError("MCPT requires the standard retrieval head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("MCPT requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("MCPT requires fixed stripe pooling")
            if not self.scale_balanced_branches:
                raise ValueError("MCPT requires scale-balanced branches")
            if self.compact_deployment_head:
                raise ValueError("MCPT preserves the full fixed-stripe descriptor")
            if self.anatomical_auxiliary_enabled and (
                not self.anatomical_pose_teacher_enabled
                or not self.anatomical_multiscale_enabled
                or self.anatomical_deployment_enabled
                or self.anatomical_accessory_query_enabled
            ):
                raise ValueError("MCPT supports only the training-only multiscale V8 pose teacher")
        self.jpm_enabled = bool(jpm)
        if self.jpm_enabled:
            if self.has_specialists:
                raise ValueError("JPM requires the standard retrieval head")
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("JPM requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("JPM requires fixed stripe pooling")
            if not self.scale_balanced_branches:
                raise ValueError("JPM requires scale-balanced branches")
            if self.compact_deployment_head:
                raise ValueError("JPM preserves the full fixed-stripe descriptor")
            if self.mcpt_enabled:
                raise ValueError("JPM and MCPT are independent treatments")
            if self.anatomical_auxiliary_enabled:
                raise ValueError("JPM and privileged anatomy are independent treatments")
            if global_in_ch != local_in_ch:
                raise ValueError("JPM requires equal-width global and coarse feature maps")
        if self.multilevel_suppression_enabled:
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("Multilevel suppression requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("Multilevel suppression requires fixed stripe pooling")
            if not self.scale_balanced_branches:
                raise ValueError("Multilevel suppression requires scale-balanced branches")
            if self.has_specialists:
                raise ValueError("Multilevel suppression requires the standard head")
            if self.compact_deployment_head:
                raise ValueError("Multilevel suppression preserves the full descriptor")
            if (
                self.hierarchical_branch_attention_enabled
                or self.branch_set_attention_enabled
                or self.multiscale_query_decoder_enabled
                or self.mcpt_enabled
                or self.jpm_enabled
            ):
                raise ValueError(
                    "Multilevel suppression is an independent treatment from branch communication, MCPT, and JPM"
                )
        if self.compact_deployment_head:
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("compact deployment head requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("compact deployment head requires fixed stripe pooling")
        if self.return_cross_scale_features or self.return_treeboost_features:
            if self.head_parts != (1, 2, 4):
                raise ValueError("hierarchical loss features require head_parts=(1, 2, 4)")
            if self.part_pooling not in {"stripes", "overlap_stripes"}:
                raise ValueError("hierarchical loss features require fixed or overlapping stripe pooling")
        if self.part_pooling == "tokens":
            if self.num_part_tokens < 1:
                raise ValueError(f"num_part_tokens must be positive, got {num_part_tokens}")
            self.branch_specs = [
                ("global", 1, 0),
                *[(f"part{index}", 0, index) for index in range(self.num_part_tokens)],
            ]
            self.part_token_pool = LearnedPartTokenPool(local_in_ch, self.num_part_tokens)
            self.semantic_part_pool = None
        elif self.part_pooling == "semantic_parts":
            semantic_part_count = self._semantic_part_count(self.head_parts)
            self.branch_specs = [
                ("global", 1, 0),
                *[(f"part{index}", 0, index) for index in range(semantic_part_count)],
            ]
            self.part_token_pool = None
            self.semantic_part_pool = SemanticVisibilityPartPool(
                local_in_ch,
                semantic_part_count,
                num_roles=self.evidence_num_roles,
            )
        else:
            self.branch_specs = self._build_branch_specs(self.head_parts)
            self.part_token_pool = None
            self.semantic_part_pool = None
        self.part_keys = [key for key, granularity, _ in self.branch_specs if granularity > 1]
        if self.part_pooling in {"tokens", "semantic_parts"}:
            self.part_keys = [key for key, _, _ in self.branch_specs if key != "global"]
        self.decouple_patterns = bool(decouple_patterns)
        self.pattern_adapter_dim = int(pattern_adapter_dim)
        if self.decouple_patterns:
            self.global_adapter = PatternAdapter(global_in_ch, self.pattern_adapter_dim)
            self.local_adapter = PatternAdapter(local_in_ch, self.pattern_adapter_dim)
            self.fine_adapter = PatternAdapter(fine_in_ch, self.pattern_adapter_dim)
        else:
            self.global_adapter = nn.Identity()
            self.local_adapter = nn.Identity()
            self.fine_adapter = nn.Identity()
        self.stripe_visibility = bool(stripe_visibility)
        if self.stripe_visibility:
            if self.part_pooling != "stripes":
                raise ValueError("stripe_visibility requires fixed stripe pooling")
            local_specs = [spec for spec in self.branch_specs if spec[0] != "global"]
            granularities = {granularity for _, granularity, _ in local_specs}
            if len(granularities) != 1:
                raise ValueError(
                    f"stripe_visibility requires exactly one local stripe granularity, got head_parts={self.head_parts}"
                )
            self.visibility_granularity = granularities.pop()
            self.visibility_gate = StripeVisibilityGate(local_in_ch, len(local_specs))
        else:
            self.visibility_granularity = None
            self.visibility_gate = None
        self.dse_descriptor_pool = DSELitePool((1, 1))
        self.set_pooling(head_pool)

        for key, granularity, _ in self.branch_specs:
            branch_dim = feat_dim
            if self.hierarchical_scales:
                branch_dim = feat_dim if granularity == 1 else feat_dim // granularity
            branch_in_ch = global_in_ch if granularity == 1 else (fine_in_ch if granularity >= 4 else local_in_ch)
            setattr(self, self._bn_attr(key), BNNeck3(branch_in_ch, num_classes, branch_dim, return_f=True))
        # This training-only controller owns no parameters or buffers. Its
        # construction therefore preserves both shared RNG and checkpoint
        # state exactly when the treatment is toggled.
        self.multilevel_suppression = (
            MultilevelClassifierSuppression(ratio=self.multilevel_suppression_ratio)
            if self.multilevel_suppression_enabled
            else None
        )
        if self.drop_global_aux_enabled:
            self.drop_global_aux = SpatialTopDrop(h_ratio=self.drop_global_aux_ratio)
            self.bn_drop_global_aux = BNNeck3(global_in_ch, num_classes, feat_dim, return_f=True)
        else:
            self.drop_global_aux = None
            self.bn_drop_global_aux = None
        if self.has_stage2_pg:
            self.stage2_pg_activation = nn.GELU()
            self.stage2_pg_gem = GeM((1, 1))
            self.stage2_pg_max = nn.AdaptiveMaxPool2d((1, 1))
            self.bn_stage2_pg = BNNeck3(
                local_in_ch,
                num_classes,
                self.SPECIALIST_DIM,
                return_f=True,
            )
        else:
            self.stage2_pg_activation = None
            self.stage2_pg_gem = None
            self.stage2_pg_max = None
            self.bn_stage2_pg = None
        if self.has_stage2_channels:
            if local_in_ch % 2:
                raise ValueError(f"Stage-2 channel specialists require even channels, got {local_in_ch}")
            self.stage2_channel_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.stage2_channel_shared = nn.Sequential(
                nn.Conv2d(local_in_ch // 2, self.SPECIALIST_DIM, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.SPECIALIST_DIM),
                nn.GELU(),
            )
            # LightMBN-style sharing: both channel halves traverse the same
            # projection, BN neck, and classifier. They remain two CE terms.
            self.bn_stage2_channel_shared = BNNeck(
                self.SPECIALIST_DIM,
                num_classes,
                return_f=True,
            )
        else:
            self.stage2_channel_pool = None
            self.stage2_channel_shared = None
            self.bn_stage2_channel_shared = None
        if self.has_multiscale_channels:
            scale_channels = {
                "global": global_in_ch,
                "coarse": local_in_ch,
                "fine": fine_in_ch,
            }
            if any(channels % 2 for channels in scale_channels.values()):
                raise ValueError(f"Multi-scale channel specialists require even channels, got {scale_channels}")
            # Isolate optional-module construction so enabling this ablation
            # cannot perturb the v8 backbone/base-head initialization at the
            # same seed.
            with torch.random.fork_rng(devices=[]):
                self.multiscale_channel_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.multiscale_channel_projections = nn.ModuleDict(
                    {
                        scale: nn.Sequential(
                            nn.Conv2d(
                                channels // 2,
                                self.SPECIALIST_DIM,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.SPECIALIST_DIM),
                            nn.GELU(),
                        )
                        for scale, channels in scale_channels.items()
                    }
                )
                # The two channel halves share their projection, BN neck, and
                # classifier within a scale. Separate scale modules let
                # global/coarse/fine maps retain distinct feature statistics.
                self.multiscale_channel_necks = nn.ModuleDict(
                    {
                        scale: BNNeck(
                            self.SPECIALIST_DIM,
                            num_classes,
                            return_f=True,
                        )
                        for scale in scale_channels
                    }
                )
        else:
            self.multiscale_channel_pool = None
            self.multiscale_channel_projections = None
            self.multiscale_channel_necks = None
        if self.has_specialist_gate:
            self.specialist_gate = nn.Linear(global_in_ch + local_in_ch, 1)
        else:
            self.specialist_gate = None
        if self.has_suppressed_global:
            self.suppressed_global = SpatialTopSuppression(h_ratio=0.25)
            self.suppressed_global_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.bn_suppressed_global = BNNeck3(
                global_in_ch,
                num_classes,
                self.SPECIALIST_DIM,
                return_f=True,
            )
        else:
            self.suppressed_global = None
            self.suppressed_global_pool = None
            self.bn_suppressed_global = None
        if self.compact_deployment_head:
            compact_input_dim = sum(self.branch_input_channels)
            self.compact_reduction = nn.Linear(compact_input_dim, feat_dim, bias=False)
            self.compact_bn = nn.BatchNorm1d(feat_dim)
            self.compact_bn.bias.requires_grad_(False)
            self.compact_classifier = nn.Linear(feat_dim, num_classes, bias=False)
            # Training-only decoder makes 512-D student directions directly
            # comparable with the full 1536-D teacher descriptor. It is not
            # traversed in eval/export mode.
            self.compact_distill_decoder = nn.Linear(feat_dim, 3 * feat_dim, bias=False)
        else:
            self.compact_reduction = None
            self.compact_bn = None
            self.compact_classifier = None
            self.compact_distill_decoder = None
        if self.hierarchical_branch_attention_enabled:
            self.branch_attention = HierarchicalBranchAttention(
                feat_dim,
                feat_dim // 2,
                feat_dim // 4,
                token_dim=branch_attention_token_dim,
                num_heads=branch_attention_num_heads,
                num_layers=branch_attention_num_layers,
                mlp_ratio=branch_attention_mlp_ratio,
                dropout=branch_attention_dropout,
            )
        else:
            self.branch_attention = None
        if self.branch_set_attention_enabled:
            self.branch_set_attention = BranchSetAttention(
                global_in_ch,
                token_dim=branch_set_attention_token_dim,
                num_heads=branch_set_attention_num_heads,
                num_layers=branch_set_attention_num_layers,
                mlp_ratio=branch_set_attention_mlp_ratio,
                dropout=branch_set_attention_dropout,
            )
        else:
            self.branch_set_attention = None
        if self.multiscale_query_decoder_enabled:
            self.multiscale_query_decoder = ResidualMultiScaleQueryDecoder(
                global_in_ch,
                token_dim=query_decoder_dim,
                num_heads=query_decoder_num_heads,
                num_layers=query_decoder_num_layers,
                mlp_ratio=query_decoder_mlp_ratio,
                dropout=query_decoder_dropout,
            )
        else:
            self.multiscale_query_decoder = None
        if self.hierarchical_late_interaction_enabled:
            self.late_interaction_matcher = HierarchicalLateInteractionMatcher(
                feat_dim,
                feat_dim // 2,
                feat_dim // 4,
                token_dim=late_interaction_dim,
                num_heads=late_interaction_num_heads,
                num_layers=late_interaction_num_layers,
                sinkhorn_iters=late_interaction_sinkhorn_iters,
                null_tokens=late_interaction_null_tokens,
                base_score_init=late_interaction_base_score_init,
            )
        else:
            self.late_interaction_matcher = None
        self.emit_late_interaction_packet = False
        if self.specialist_gate is not None:
            nn.init.zeros_(self.specialist_gate.weight)
            initial_logit = math.log(self.SPECIALIST_GATE_INIT / (1.0 - self.SPECIALIST_GATE_INIT))
            nn.init.constant_(self.specialist_gate.bias, initial_logit)
        if self.branch_attention is not None:
            self.branch_attention.reset_identity_initialization()
        if self.branch_set_attention is not None:
            self.branch_set_attention.reset_identity_initialization()
        if self.multiscale_query_decoder is not None:
            self.multiscale_query_decoder.reset_identity_initialization()
        if self.mcpt_enabled:
            with torch.random.fork_rng(devices=[]):
                self.mcpt = MonotonicCanonicalPartTransport(
                    local_in_ch,
                    fine_channels=fine_in_ch,
                    mode=self.mcpt_mode,
                    hidden_dim=mcpt_hidden_dim,
                    max_displacement=mcpt_max_displacement,
                    start_epoch=mcpt_start_epoch,
                    ramp_end_epoch=mcpt_ramp_end_epoch,
                )
        else:
            self.mcpt = None
        if self.jpm_enabled:
            # Construction is RNG-isolated so the same-seed RGB backbone and
            # fixed-stripe head remain exact controls when JPM is enabled.
            with torch.random.fork_rng(devices=[]):
                self.jpm = JigsawPatchAuxiliary(
                    local_in_ch,
                    num_classes,
                    num_groups=jpm_num_groups,
                    shift=jpm_shift,
                    token_dim=jpm_token_dim,
                    num_heads=jpm_num_heads,
                    mlp_ratio=jpm_mlp_ratio,
                    dropout=jpm_dropout,
                )
        else:
            self.jpm = None
        self._shared_initialization_rng_state = torch.random.get_rng_state()
        # Keep the optional training branch last and isolate its constructor
        # RNG. Toggling anatomical supervision must not perturb any shared
        # backbone, fusion, BNNeck, or classifier initialization.
        if self.anatomical_auxiliary_enabled:
            descriptor_dim = None
            if self.anatomical_descriptor_distill_enabled:
                descriptor_dim = sum(
                    (feat_dim if not self.hierarchical_scales or granularity == 1 else feat_dim // granularity)
                    for _, granularity, _ in self.branch_specs
                )
                descriptor_dim += self.SPECIALIST_DIM * (
                    int(self.has_stage2_pg)
                    + 2 * int(self.has_stage2_channels)
                    + 6 * int(self.has_multiscale_channels)
                    + int(self.has_suppressed_global)
                )
            with torch.random.fork_rng(devices=[]):
                if self.anatomical_privileged_attention_enabled:
                    self.anatomical_auxiliary_pool = None
                    self.anatomical_attention_adapter = PrivilegedMaskPoseAttentionAdapter(local_in_ch)
                    self.anatomical_fine_attention_adapter = PrivilegedMaskPoseAttentionAdapter(fine_in_ch)
                elif self.anatomical_pose_teacher_enabled:
                    self.anatomical_auxiliary_pool = EMAAnatomicalAuxiliaryPool(
                        local_in_ch,
                        token_dim=self.anatomical_token_dim,
                        descriptor_dim=descriptor_dim,
                        pose_teacher=True,
                        teacher_channels=fine_in_ch,
                        multiscale=self.anatomical_multiscale_enabled,
                        semantic_teacher=(self.anatomical_semantic_teacher_enabled),
                        decoupled_query_teacher=(self.anatomical_decoupled_query_teacher_enabled),
                        accessory_query=(self.anatomical_accessory_query_enabled),
                    )
                    self.anatomical_attention_adapter = None
                    self.anatomical_fine_attention_adapter = None
                else:
                    self.anatomical_auxiliary_pool = AnatomicalAuxiliaryPool(
                        local_in_ch,
                        token_dim=self.anatomical_token_dim,
                        descriptor_dim=descriptor_dim,
                        fine_channels=fine_in_ch,
                        multiscale=self.anatomical_multiscale_enabled,
                    )
                    self.anatomical_attention_adapter = None
                    self.anatomical_fine_attention_adapter = None
                if self.anatomical_deployment_enabled:
                    self.anatomical_deployment_norm = nn.LayerNorm(2 * self.anatomical_token_dim)
                    self.anatomical_deployment_projection = nn.Linear(
                        2 * self.anatomical_token_dim,
                        self.anatomical_deployment_dim,
                        bias=False,
                    )
                    self.anatomical_deployment_necks = nn.ModuleList(
                        BNNeck(
                            self.anatomical_deployment_dim,
                            num_classes,
                            return_f=True,
                        )
                        for _ in range(6)
                    )
                else:
                    self.anatomical_deployment_norm = None
                    self.anatomical_deployment_projection = None
                    self.anatomical_deployment_necks = None
        else:
            self.anatomical_auxiliary_pool = None
            self.anatomical_attention_adapter = None
            self.anatomical_fine_attention_adapter = None
            self.anatomical_deployment_norm = None
            self.anatomical_deployment_projection = None
            self.anatomical_deployment_necks = None

    def reset_reid_initialization(self) -> None:
        """Restore ReID-specific head initialization after global model init."""
        # Model-wide initialization visits the optional auxiliary module before
        # this method runs. Restore the shared-head RNG boundary so its presence
        # cannot change classifier or projection initialization.
        torch.random.set_rng_state(self._shared_initialization_rng_state)
        for module in self.modules():
            if isinstance(module, (BNNeck, BNNeck3)):
                module.reset_reid_initialization()
        if self.semantic_part_pool is not None:
            self.semantic_part_pool.reset_metadata_initialization()
        if self.anatomical_auxiliary_pool is not None:
            self.anatomical_auxiliary_pool.reset_visibility_initialization()
            reset_teacher = getattr(
                self.anatomical_auxiliary_pool,
                "reset_teacher_initialization",
                None,
            )
            if callable(reset_teacher):
                reset_teacher()
        for adapter in (
            self.anatomical_attention_adapter,
            self.anatomical_fine_attention_adapter,
        ):
            if adapter is not None:
                adapter.reset_identity_initialization()
        if self.visibility_gate is not None:
            self.visibility_gate.reset_visibility_initialization()
        if self.compact_deployment_head:
            nn.init.kaiming_normal_(self.compact_reduction.weight, mode="fan_out")
            nn.init.constant_(self.compact_bn.weight, 1.0)
            nn.init.constant_(self.compact_bn.bias, 0.0)
            nn.init.normal_(self.compact_classifier.weight, std=0.001)
            nn.init.kaiming_normal_(self.compact_distill_decoder.weight, mode="fan_out")
            self.compact_bn.bias.requires_grad_(False)
        if self.specialist_gate is not None:
            nn.init.zeros_(self.specialist_gate.weight)
            initial_logit = math.log(self.SPECIALIST_GATE_INIT / (1.0 - self.SPECIALIST_GATE_INIT))
            nn.init.constant_(self.specialist_gate.bias, initial_logit)
        if self.branch_attention is not None:
            self.branch_attention.reset_identity_initialization()
        if self.branch_set_attention is not None:
            self.branch_set_attention.reset_identity_initialization()
        if self.multiscale_query_decoder is not None:
            self.multiscale_query_decoder.reset_identity_initialization()
        if self.mcpt is not None:
            self.mcpt.reset_identity_initialization()

    def set_mcpt_epoch(self, epoch: int) -> None:
        """Update the transport gate schedule without changing parameters."""
        if self.mcpt is not None:
            self.mcpt.set_epoch(epoch)

    def set_anatomical_auxiliary_active(self, active: bool) -> None:
        """Enable or bypass training-only anatomical computation."""
        self.anatomical_auxiliary_runtime_active = bool(active)

    def set_hpgrd_part_packet_active(self, active: bool) -> None:
        """Expose the shared feature map for parameter-free HP-GRD pooling."""
        self.hpgrd_part_packet_runtime_active = bool(active)

    def set_retrieval_packet_active(self, active: bool) -> None:
        """Return the training feature dictionary for retrieval objectives."""
        self.retrieval_packet_runtime_active = bool(active)

    def set_multilevel_suppression_progress(
        self,
        progress: float,
    ) -> None:
        """Set the training-only classifier suppression schedule."""
        if self.multilevel_suppression is not None:
            self.multilevel_suppression.set_progress(progress)

    def set_mcpt_force_disabled(self, disabled: bool) -> None:
        """Force fixed-stripe evaluation for the MCPT control diagnostic."""
        if self.mcpt is not None:
            self.mcpt.set_force_disabled(disabled)

    def enable_mcpt_visualization_capture(self, limit: int = 100) -> None:
        """Capture a bounded MCPT feature-energy contact sheet."""
        if self.mcpt is not None:
            self.mcpt.enable_visualization_capture(limit)

    def pop_mcpt_visualization_capture(
        self,
    ) -> dict[str, torch.Tensor] | None:
        """Return and clear captured MCPT feature-energy maps."""
        if self.mcpt is None:
            return None
        return self.mcpt.pop_visualization_capture()

    @torch.no_grad()
    def update_anatomical_teacher(self, momentum: float) -> None:
        """EMA-update the selected training-only privileged projection."""
        if self.anatomical_auxiliary_pool is None:
            return
        update_teacher = getattr(
            self.anatomical_auxiliary_pool,
            "update_teacher",
            None,
        )
        if callable(update_teacher):
            update_teacher(momentum)

    def set_anatomical_attention_gate_active(
        self,
        active: bool,
    ) -> None:
        """Toggle retrieval gating while keeping RGB attention trainable."""
        for adapter in (
            self.anatomical_attention_adapter,
            self.anatomical_fine_attention_adapter,
        ):
            if adapter is not None:
                adapter.set_gate_active(active)

    def _anatomical_deployment_outputs(
        self,
        local_tokens: torch.Tensor,
        fine_tokens: torch.Tensor,
        local_visibility_logits: torch.Tensor,
        fine_visibility_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ]:
        """Fuse six RGB anatomical tokens into a visibility-gated descriptor."""
        expected_shape = (
            local_tokens.shape[0],
            6,
            self.anatomical_token_dim,
        )
        if local_tokens.shape != expected_shape or fine_tokens.shape != expected_shape:
            raise RuntimeError(
                "anatomical deployment expects local/fine tokens with shape "
                f"{expected_shape}, got {tuple(local_tokens.shape)} and "
                f"{tuple(fine_tokens.shape)}"
            )
        if local_visibility_logits.shape != expected_shape[:2] or fine_visibility_logits.shape != expected_shape[:2]:
            raise RuntimeError(f"anatomical deployment visibility must have shape {expected_shape[:2]}")
        fused = self.anatomical_deployment_projection(
            self.anatomical_deployment_norm(torch.cat((local_tokens, fine_tokens), dim=-1))
        )
        outputs = tuple(neck(fused[:, part_index]) for part_index, neck in enumerate(self.anatomical_deployment_necks))
        bn_parts = torch.stack(
            [output[0] for output in outputs],
            dim=1,
        )
        raw_parts = torch.stack(
            [output[2] for output in outputs],
            dim=1,
        )
        logits = tuple(output[1] for output in outputs)
        visibility = torch.sigmoid(0.5 * (local_visibility_logits + fine_visibility_logits))
        descriptor_visibility = visibility.detach() if self.training else visibility
        weighted_parts = F.normalize(bn_parts, p=2, dim=-1) * descriptor_visibility.clamp_min(0).sqrt()[..., None]
        descriptor = F.normalize(
            weighted_parts.flatten(1),
            p=2,
            dim=1,
        )
        return descriptor, raw_parts, logits, visibility

    @staticmethod
    def _normalize_head_parts(head_parts) -> tuple[int, ...]:
        if isinstance(head_parts, str):
            values = [part for part in head_parts.replace(";", ",").split(",") if part.strip()]
        elif isinstance(head_parts, int):
            values = [head_parts]
        else:
            values = list(head_parts or (1, 2))
        normalized = tuple(dict.fromkeys(int(part) for part in values))
        if not normalized:
            raise ValueError("CSL-TinyViT head_parts must not be empty")
        if any(part < 1 for part in normalized):
            raise ValueError(f"CSL-TinyViT head_parts must be positive, got {normalized}")
        if 1 not in normalized:
            raise ValueError(f"CSL-TinyViT head_parts must include 1 for the global branch, got {normalized}")
        return normalized

    @staticmethod
    def _build_branch_specs(head_parts: tuple[int, ...]) -> list[tuple[str, int, int]]:
        specs = [("global", 1, 0)]
        part_index = 0
        for granularity in head_parts:
            if granularity == 1:
                continue
            for stripe_index in range(granularity):
                specs.append((f"part{part_index}", granularity, stripe_index))
                part_index += 1
        return specs

    @staticmethod
    def _semantic_part_count(head_parts: tuple[int, ...]) -> int:
        count = sum(part for part in head_parts if part > 1)
        if count < 1:
            raise ValueError("semantic_parts pooling requires at least one local part in head_parts")
        return count

    @staticmethod
    def _bn_attr(key: str) -> str:
        return "bn_global" if key == "global" else f"bn_{key}"

    @staticmethod
    def _pool_attr(granularity: int) -> str:
        if granularity == 1:
            return "global_pool"
        if granularity == 2:
            return "partial_pool"
        return f"part_pool_{granularity}"

    def _descriptor_scale(self, granularity: int) -> float:
        """Give every enabled spatial scale equal total descriptor energy."""
        scale = 1.0 / math.sqrt(granularity) if self.scale_balanced_branches else 1.0
        if self.has_multiscale_channels:
            scale *= math.sqrt(1.0 - self.multiscale_channel_alpha**2)
        return scale

    def _declared_feature_dim(self, feature_name: str) -> int | None:
        """Return a statically known training descriptor width.

        The standard head builds every retrieval descriptor from registered
        neck widths, so training setup never needs a random forward merely to
        discover a shape. Specialized subclasses with custom forwards fall
        back to the trainer's state-preserving probe until they declare their
        own dimensions.
        """
        if self.__class__.forward is not MultiBranchHead.forward:
            return None
        branch_dims = {key: int(getattr(self, self._bn_attr(key)).bn.num_features) for key, _, _ in self.branch_specs}
        global_dim = branch_dims["global"]
        base_concat_dim = sum(branch_dims.values())
        specialist_count = (
            int(self.has_stage2_pg)
            + 2 * int(self.has_stage2_channels)
            + 6 * int(self.has_multiscale_channels)
            + int(self.has_suppressed_global)
        )
        feature_dims = {
            "global": global_dim,
            "raw_mean": global_dim,
            "raw_concat": base_concat_dim + 6 * self.SPECIALIST_DIM * int(self.has_multiscale_channels),
            "concat_bn": base_concat_dim,
            "norm_concat_bn": base_concat_dim
            + specialist_count * self.SPECIALIST_DIM
            + 6 * self.anatomical_deployment_dim * int(self.anatomical_deployment_enabled),
            "coarse_concat": sum(
                branch_dims[key] for key, granularity, _ in self.branch_specs if granularity in {1, 2}
            ),
        }
        return feature_dims.get(str(feature_name))

    @property
    def metric_dim(self) -> int | None:
        """Width of the configured metric descriptor, when statically known."""
        return self._declared_feature_dim(self.metric_feature)

    @property
    def classifier_dim(self) -> int | None:
        """Width consumed by an optional margin-based classifier."""
        return self.metric_dim

    @property
    def center_dim(self) -> int | None:
        """Width consumed by center loss under the active branch policy."""
        if self.scale_balanced_branches:
            return self.metric_dim
        return self._declared_feature_dim("global")

    def _fast_norm_concat_descriptor(
        self,
        pooled_by_granularity: dict[int, torch.Tensor],
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the common retrieval descriptor through one stable path."""
        normalized_branches = []
        for key, granularity, stripe_index in self.branch_specs:
            pooled = pooled_by_granularity[granularity]
            if granularity > 1:
                pooled = pooled[:, :, stripe_index : stripe_index + 1, :]
            bn_feature = getattr(
                self,
                self._bn_attr(key),
            ).forward_inference(pooled)
            normalized_branches.append(F.normalize(bn_feature, p=2, dim=1) * self._descriptor_scale(granularity))
        for _, branch_output, weight in self._specialist_outputs(
            global_feature,
            local_feature,
            fine_feature,
        ):
            normalized_branches.append(F.normalize(branch_output[0], p=2, dim=1) * weight)
        return F.normalize(
            torch.cat(normalized_branches, dim=1),
            p=2,
            dim=1,
        )

    @staticmethod
    def _make_pool(head_pool: str, output_size: tuple[int, int]) -> nn.Module:
        if head_pool == "avg":
            return nn.AdaptiveAvgPool2d(output_size)
        if head_pool == "gem":
            return GeM(output_size)
        if head_pool == "dse":
            return DSELitePool(output_size)
        if head_pool == "gelu_gem":
            return ActivatedGeM(nn.GELU(), output_size)
        if head_pool == "relu_gem":
            return ActivatedGeM(nn.ReLU(inplace=False), output_size)
        if head_pool == "softplus_gem":
            return ActivatedGeM(nn.Softplus(), output_size)
        raise ValueError(f"Unsupported CSL-TinyViT head_pool: {head_pool}")

    def set_pooling(self, head_pool: str) -> None:
        head_pool = str(head_pool).lower()
        granularities = (1,) if self.part_pooling in {"tokens", "semantic_parts"} else self.head_parts
        for granularity in granularities:
            output_size = (1, 1) if self.part_pooling == "overlap_stripes" and granularity > 1 else (granularity, 1)
            setattr(
                self,
                self._pool_attr(granularity),
                self._make_pool(head_pool, output_size),
            )
        self.head_pool = head_pool

    @staticmethod
    def _overlap_window_bounds(height: int, granularity: int) -> list[tuple[int, int]]:
        if granularity <= 1:
            return [(0, height)]
        stride = height / granularity
        window = min(height, max(1, int(math.ceil(stride * 1.5))))
        bounds = []
        for index in range(granularity):
            center = (index + 0.5) * stride
            start = int(round(center - window / 2))
            start = max(0, min(start, height - window))
            end = min(height, start + window)
            bounds.append((start, end))
        return bounds

    def _pool_overlap_stripes(
        self,
        feature: torch.Tensor,
        granularity: int,
        pool: nn.Module,
    ) -> torch.Tensor:
        stripes = [
            pool(feature[:, :, start:end, :])
            for start, end in self._overlap_window_bounds(feature.shape[-2], granularity)
        ]
        return torch.cat(stripes, dim=2)

    def set_branch_metric(self, branch_metric: bool) -> None:
        self.branch_metric = bool(branch_metric)

    def _needs_dse_descriptor(self) -> bool:
        return self.metric_feature in {"dse_weighted", "dse_mix"} or self.inference_feature in {
            "dse_weighted",
            "dse_mix",
        }

    def _add_dse_descriptors(self, raw_features: dict[str, torch.Tensor], source: torch.Tensor) -> None:
        if not self._needs_dse_descriptor():
            return
        dse_weighted = self.dse_descriptor_pool(source).flatten(1)
        raw_features["dse_weighted"] = dse_weighted
        raw_features["dse_mix"] = torch.cat(
            (
                F.normalize(raw_features["raw_mean"], p=2, dim=1),
                F.normalize(dse_weighted, p=2, dim=1),
                F.normalize(raw_features["raw_concat"], p=2, dim=1),
            ),
            dim=1,
        )

    def _compact_descriptor(
        self,
        pooled_by_granularity: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build one deployment vector from global, coarse, and fine scales."""
        if not self.compact_deployment_head:
            raise RuntimeError("Compact descriptor requested while compact_deployment_head is disabled")
        scale_vectors = [
            pooled_by_granularity[1].flatten(1),
            pooled_by_granularity[2].mean(dim=(2, 3)),
            pooled_by_granularity[4].mean(dim=(2, 3)),
        ]
        raw = self.compact_reduction(torch.cat(scale_vectors, dim=1))
        bn = self.compact_bn(raw)
        logits = self.compact_classifier(bn) if self.training else None
        return raw, bn, logits

    @staticmethod
    def _gated_branch_output(
        module: BNNeck | BNNeck3,
        pooled: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        bn_feature, logits, raw_feature = module(pooled)
        if gate is None:
            return bn_feature, logits, raw_feature
        gated_feature = bn_feature * gate
        gated_logits = module.classifier(gated_feature) if module.training else None
        return gated_feature, gated_logits, raw_feature

    def _specialist_outputs(
        self,
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
        fine_feature: torch.Tensor | None = None,
    ) -> list[tuple[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor | float]]:
        """Return the configured channel/global specialist branches."""
        if not self.has_specialists:
            return []
        gate = None
        if self.specialist_gate is not None:
            gate_input = torch.cat(
                (global_feature.mean(dim=(2, 3)), local_feature.mean(dim=(2, 3))),
                dim=1,
            )
            gate = self.specialist_gate(gate_input).sigmoid()

        outputs = []
        if self.has_stage2_pg:
            activated = self.stage2_pg_activation(local_feature)
            pooled = 0.5 * (self.stage2_pg_gem(activated) + self.stage2_pg_max(activated))
            output = self._gated_branch_output(self.bn_stage2_pg, pooled, gate)
            weight = gate if gate is not None else self.STAGE2_GLOBAL_WEIGHT
            outputs.append(("stage2_pg", output, weight))
        if self.has_stage2_channels:
            pooled = self.stage2_channel_pool(local_feature)
            for index, channel_half in enumerate(pooled.chunk(2, dim=1)):
                projected = self.stage2_channel_shared(channel_half)
                output = self._gated_branch_output(self.bn_stage2_channel_shared, projected, gate)
                weight = gate if gate is not None else self.STAGE2_CHANNEL_WEIGHT
                outputs.append((f"stage2_c{index + 1}", output, weight))
        if self.has_multiscale_channels:
            if fine_feature is None:
                raise RuntimeError("Multi-scale channel specialists require the fine map")
            channel_weight = self.multiscale_channel_alpha / math.sqrt(2.0)
            scale_features = {
                "global": global_feature,
                "coarse": local_feature,
                "fine": fine_feature,
            }
            for scale, feature in scale_features.items():
                pooled = self.multiscale_channel_pool(feature)
                for index, channel_half in enumerate(pooled.chunk(2, dim=1)):
                    projected = self.multiscale_channel_projections[scale](channel_half)
                    output = self.multiscale_channel_necks[scale](projected)
                    outputs.append(
                        (
                            f"{scale}_c{index + 1}",
                            output,
                            channel_weight,
                        )
                    )
        if self.has_suppressed_global:
            pooled = self.suppressed_global_pool(self.suppressed_global(global_feature))
            output = self.bn_suppressed_global(pooled)
            outputs.append(("suppressed_global", output, self.SUPPRESSED_GLOBAL_WEIGHT))
        return outputs

    def forward(
        self,
        x,
        *,
        pids: torch.Tensor | None = None,
        anatomical_pose: torch.Tensor | None = None,
        anatomical_query_masks: torch.Tensor | None = None,
    ):
        # x: (B, C, H, W) or (global_map, local_map) for split-map ablations.
        fine_source = None
        if isinstance(x, tuple) and len(x) == 3:
            global_source, local_source, fine_source = x
        elif isinstance(x, tuple):
            global_source, local_source = x
        else:
            global_source = local_source = x
        anatomical_source = local_source
        anatomical_fine_source = fine_source
        anatomical_teacher_source = fine_source if self.anatomical_pose_teacher_enabled else None
        global_feature = self.global_adapter(global_source)
        local_feature = self.local_adapter(local_source)
        if fine_source is not None:
            fine_source = self.fine_adapter(fine_source)
        mcpt_diagnostics = None
        if self.mcpt is not None:
            local_feature, fine_source, mcpt_diagnostics = self.mcpt(
                local_feature,
                fine_source,
            )
        anatomical_outputs = None
        anatomical_foreground_logits = None
        anatomical_fine_foreground_logits = None
        anatomical_gate_scale = None
        anatomical_fine_gate_scale = None
        if self.anatomical_auxiliary_runtime_active and self.anatomical_privileged_attention_enabled:
            if (
                self.anatomical_attention_adapter is None
                or self.anatomical_fine_attention_adapter is None
                or fine_source is None
            ):
                raise RuntimeError("Privileged mask-pose attention requires local and fine RGB feature maps")
            anatomical_attention_source = local_feature
            anatomical_fine_attention_source = fine_source
            (
                local_feature,
                anatomical_student_tokens,
                anatomical_attention,
                anatomical_visibility_logits,
                anatomical_foreground_logits,
                anatomical_gate_scale,
            ) = self.anatomical_attention_adapter(local_feature)
            (
                fine_source,
                anatomical_fine_student_tokens,
                anatomical_fine_attention,
                anatomical_fine_visibility_logits,
                anatomical_fine_foreground_logits,
                anatomical_fine_gate_scale,
            ) = self.anatomical_fine_attention_adapter(fine_source)
            if self.training:
                anatomical_outputs = (
                    anatomical_attention_source,
                    anatomical_student_tokens,
                    anatomical_attention,
                    anatomical_visibility_logits,
                    anatomical_fine_attention_source,
                    anatomical_fine_student_tokens,
                    anatomical_fine_attention,
                    anatomical_fine_visibility_logits,
                )
        pooled_by_granularity = {1: self.global_pool(global_feature)}
        token_parts = None
        semantic_parts = None
        semantic_visibility = None
        semantic_rarity = None
        semantic_role_logits = None
        semantic_nullness = None
        if self.part_pooling == "tokens":
            token_parts = self.part_token_pool(local_feature)
        elif self.part_pooling == "semantic_parts":
            (
                semantic_parts,
                semantic_visibility,
                semantic_rarity,
                semantic_role_logits,
                semantic_nullness,
            ) = self.semantic_part_pool(local_feature)
        else:
            pooled_by_granularity.update(
                {
                    granularity: (
                        self._pool_overlap_stripes(
                            fine_source if fine_source is not None and granularity >= 4 else local_feature,
                            granularity,
                            getattr(self, self._pool_attr(granularity)),
                        )
                        if self.part_pooling == "overlap_stripes"
                        else getattr(self, self._pool_attr(granularity))(
                            fine_source if fine_source is not None and granularity >= 4 else local_feature
                        )
                    )
                    for granularity in self.head_parts
                    if granularity > 1
                }
            )
        visibility_by_key = {"global": None}
        visibility_values = semantic_visibility
        if visibility_values is not None:
            visibility_by_key.update(
                {key: visibility_values[:, index : index + 1] for index, key in enumerate(self.part_keys)}
            )
        if self.visibility_gate is not None:
            visibility = self.visibility_gate(pooled_by_granularity[self.visibility_granularity])
            visibility_values = visibility
            visibility_by_key.update(
                {
                    key: visibility[:, index : index + 1]
                    for index, (key, _, _) in enumerate(spec for spec in self.branch_specs if spec[0] != "global")
                }
            )

        compact_raw = compact_bn = compact_logits = None
        if self.compact_deployment_head:
            compact_raw, compact_bn, compact_logits = self._compact_descriptor(pooled_by_granularity)
            if not self.training:
                # Export/tracking traverses only this compact path. Teacher
                # BNNecks, classifiers, and the distillation decoder are
                # therefore absent from the exported execution graph.
                return F.normalize(compact_bn, p=2, dim=1)

        # Retrieval-only fast path for the common fixed-stripe descriptor. It
        # deliberately avoids constructing raw/metric features, classifier
        # lists, branch dictionaries, and means that are only consumed during
        # training. Classifiers already remain inactive inside BNNeck3.eval().
        if (
            not self.training
            and self.inference_feature == "norm_concat_bn"
            and self.part_pooling in {"stripes", "overlap_stripes"}
            and self.branch_attention is None
            and self.branch_set_attention is None
            and self.multiscale_query_decoder is None
            and not self.anatomical_deployment_enabled
            and not self.emit_late_interaction_packet
            and all(value is None for value in visibility_by_key.values())
        ):
            return self._fast_norm_concat_descriptor(
                pooled_by_granularity,
                global_feature,
                local_feature,
                fine_source,
            )

        anatomical_student_only_outputs = None
        if self.training and self.anatomical_auxiliary_runtime_active and self.anatomical_auxiliary_pool is not None:
            if self.anatomical_pose_teacher_enabled:
                if anatomical_pose is not None and anatomical_teacher_source is None:
                    raise RuntimeError("Fine-map anatomical teacher requires a fine feature map")
                if anatomical_pose is not None:
                    anatomical_outputs = self.anatomical_auxiliary_pool(
                        anatomical_source,
                        teacher_x=anatomical_teacher_source,
                        pose_keypoints=anatomical_pose,
                    )
                    if self.anatomical_decoupled_query_teacher_enabled and anatomical_query_masks is None:
                        raise RuntimeError(
                            "Decoupled pose-parsing teacher requires training-only anatomical query masks"
                        )
            else:
                anatomical_outputs = self.anatomical_auxiliary_pool(
                    anatomical_source,
                    fine_x=anatomical_fine_source,
                )
        if self.anatomical_deployment_enabled and anatomical_outputs is None:
            if anatomical_teacher_source is None:
                raise RuntimeError("anatomical deployment requires a fine feature map")
            anatomical_student_only_outputs = self.anatomical_auxiliary_pool.student_forward(
                anatomical_source,
                fine_x=anatomical_teacher_source,
            )

        branch_outputs = {}
        bn_features_list = []
        raw_features_list = []
        normalized_bn_features_list = []
        normalized_raw_features_list = []
        coarse_normalized_raw_features_list = []
        cls_scores = []
        raw_features = {}
        if self.training and self.hpgrd_part_packet_runtime_active:
            # The trainer performs fixed mask-weighted pooling.  Returning the
            # already-computed deployed feature map lets privileged gradients
            # reach the backbone directly instead of being absorbed by a
            # disposable learned anatomy adapter.
            raw_features["_hpgrd_feature_map"] = local_feature
        if self.multilevel_suppression is not None and self.multilevel_suppression.active and pids is not None:
            if fine_source is None:
                raise RuntimeError("Multilevel suppression requires a fine spatial map")
            coarse_keys = tuple(key for key, granularity, _ in self.branch_specs if granularity == 2)
            fine_keys = tuple(key for key, granularity, _ in self.branch_specs if granularity == 4)
            # Run saliency before the clean branch BNNecks. The frozen
            # running-stat scorer therefore observes the state accumulated by
            # prior batches, never statistics from the current co-batch.
            suppression_output = self.multilevel_suppression(
                global_feature=global_feature,
                coarse_feature=local_feature,
                fine_feature=fine_source,
                pids=pids,
                global_pool=self.global_pool,
                coarse_pool=getattr(self, self._pool_attr(2)),
                fine_pool=getattr(self, self._pool_attr(4)),
                global_neck=getattr(self, self._bn_attr("global")),
                coarse_necks=tuple(getattr(self, self._bn_attr(key)) for key in coarse_keys),
                fine_necks=tuple(getattr(self, self._bn_attr(key)) for key in fine_keys),
            )
            raw_features["_multilevel_suppression_logits"] = {
                "coarse": suppression_output.coarse_logits,
                "fine": suppression_output.fine_logits,
            }
            raw_features["_multilevel_suppression_active"] = {
                "coarse": suppression_output.coarse_active,
                "fine": suppression_output.fine_active,
            }
            raw_features["_multilevel_suppression_diagnostics"] = suppression_output.diagnostics
        if self.training and self.jpm is not None:
            jpm_logits, jpm_features = self.jpm(
                local_feature,
                global_feature,
            )
            raw_features["_jpm_logits"] = jpm_logits
            raw_features["_jpm_features"] = jpm_features
        if mcpt_diagnostics is not None:
            raw_features.update(
                {
                    "_mcpt_smoothness": mcpt_diagnostics.smoothness,
                    "_mcpt_identity": mcpt_diagnostics.identity,
                    "_mcpt_mean_abs_displacement": (mcpt_diagnostics.mean_abs_displacement),
                    "_mcpt_boundary_mean": mcpt_diagnostics.boundary_mean,
                    "_mcpt_boundary_std": mcpt_diagnostics.boundary_std,
                    "_mcpt_cap_fraction": mcpt_diagnostics.cap_fraction,
                    "_mcpt_local_gate": mcpt_diagnostics.local_gate,
                    "_mcpt_fine_gate": mcpt_diagnostics.fine_gate,
                }
            )
        base_normalized_bn_features = {}
        pooled_features = {}
        for key, granularity, stripe_index in self.branch_specs:
            if key == "global":
                pooled = pooled_by_granularity[1]
            elif self.part_pooling == "tokens":
                pooled = token_parts[:, stripe_index]
            elif self.part_pooling == "semantic_parts":
                pooled = semantic_parts[:, stripe_index]
            else:
                pooled = pooled_by_granularity[granularity]
                pooled = pooled[:, :, stripe_index : stripe_index + 1, :]
            pooled_features[key] = pooled

        identity_communication = True
        if self.branch_set_attention is not None:
            branch_keys = tuple(key for key, _, _ in self.branch_specs)
            pooled_branches = torch.stack(
                tuple(pooled_features[key].flatten(1) for key in branch_keys),
                dim=1,
            )
            refined_branches = self.branch_set_attention(pooled_branches)
            identity_communication = not self.training and self.branch_set_attention.has_identity_output()
            if not identity_communication:
                pooled_features.update(
                    {
                        key: refined_branches[:, index, :].unsqueeze(-1).unsqueeze(-1)
                        for index, key in enumerate(branch_keys)
                    }
                )

        if self.multiscale_query_decoder is not None:
            if fine_source is None:
                raise RuntimeError("multi-scale query decoder requires global, coarse, and fine spatial maps")
            branch_keys = tuple(key for key, _, _ in self.branch_specs)
            pooled_branches = torch.stack(
                tuple(pooled_features[key].flatten(1) for key in branch_keys),
                dim=1,
            )
            refined_branches = self.multiscale_query_decoder(
                pooled_branches,
                (global_feature, local_feature, fine_source),
            )
            identity_communication = not self.training and self.multiscale_query_decoder.has_identity_output()
            if not identity_communication:
                pooled_features.update(
                    {
                        key: refined_branches[:, index, :].unsqueeze(-1).unsqueeze(-1)
                        for index, key in enumerate(branch_keys)
                    }
                )

        projected_features = {
            key: getattr(self, self._bn_attr(key)).project(pooled_features[key]) for key, _, _ in self.branch_specs
        }

        if self.branch_attention is not None:
            keys_by_granularity = {
                granularity: tuple(
                    key for key, branch_granularity, _ in self.branch_specs if branch_granularity == granularity
                )
                for granularity in (1, 2, 4)
            }
            refined_global, refined_coarse, refined_fine = self.branch_attention(
                projected_features[keys_by_granularity[1][0]],
                tuple(projected_features[key] for key in keys_by_granularity[2]),
                tuple(projected_features[key] for key in keys_by_granularity[4]),
            )
            identity_communication = not self.training and self.branch_attention.has_identity_output()
            if not identity_communication:
                projected_features[keys_by_granularity[1][0]] = refined_global
                projected_features.update(
                    zip(
                        keys_by_granularity[2],
                        refined_coarse,
                        strict=True,
                    )
                )
                projected_features.update(
                    zip(
                        keys_by_granularity[4],
                        refined_fine,
                        strict=True,
                    )
                )

        if (
            not self.training
            and identity_communication
            and (
                self.branch_attention is not None
                or self.branch_set_attention is not None
                or self.multiscale_query_decoder is not None
            )
            and self.inference_feature == "norm_concat_bn"
            and not self.anatomical_deployment_enabled
            and not self.emit_late_interaction_packet
            and all(value is None for value in visibility_by_key.values())
        ):
            return self._fast_norm_concat_descriptor(
                pooled_by_granularity,
                global_feature,
                local_feature,
                fine_source,
            )

        for key, granularity, _ in self.branch_specs:
            branch_output = getattr(self, self._bn_attr(key)).forward_projected(projected_features[key])
            branch_outputs[key] = branch_output
            confidence = visibility_by_key.get(key)
            base_bn_feature = branch_output[0]
            base_raw_feature = branch_output[2]
            bn_feature = base_bn_feature
            raw_feature = base_raw_feature
            normalized_bn_feature = F.normalize(base_bn_feature, p=2, dim=1)
            normalized_raw_feature = F.normalize(base_raw_feature, p=2, dim=1)
            base_normalized_bn_features[key] = normalized_bn_feature
            if confidence is not None:
                bn_feature = bn_feature * confidence
                raw_feature = raw_feature * confidence
                normalized_bn_feature = normalized_bn_feature * confidence
                normalized_raw_feature = normalized_raw_feature * confidence
            descriptor_scale = self._descriptor_scale(granularity)
            bn_features_list.append(bn_feature * descriptor_scale)
            normalized_bn_features_list.append(normalized_bn_feature * descriptor_scale)
            cls_scores.append(branch_output[1])
            raw_features_list.append(raw_feature)
            normalized_raw_features_list.append(normalized_raw_feature * descriptor_scale)
            if granularity in {1, 2}:
                coarse_normalized_raw_features_list.append(normalized_raw_feature * descriptor_scale)
            raw_features[key] = raw_feature
        specialist_normalized_bn_features = []
        specialist_normalized_raw_features = []
        for key, branch_output, weight in self._specialist_outputs(
            global_feature,
            local_feature,
            fine_source,
        ):
            raw_features[key] = branch_output[2]
            cls_scores.append(branch_output[1])
            specialist_normalized_bn_features.append(F.normalize(branch_output[0], p=2, dim=1) * weight)
            if self.has_multiscale_channels:
                specialist_normalized_raw_features.append(F.normalize(branch_output[2], p=2, dim=1) * weight)
        if visibility_values is not None:
            raw_features["_visibility"] = visibility_values
        if semantic_rarity is not None:
            raw_features["_rarity"] = semantic_rarity
        if semantic_role_logits is not None:
            raw_features["_role_logits"] = semantic_role_logits
        if semantic_nullness is not None:
            raw_features["_nullness"] = semantic_nullness
        anatomical_student_tokens = None
        anatomical_visibility_logits = None
        anatomical_fine_student_tokens = None
        anatomical_fine_visibility_logits = None
        if anatomical_outputs is not None:
            if self.anatomical_pose_teacher_enabled:
                (
                    anatomical_feature_map,
                    anatomical_teacher_feature_map,
                    anatomical_online_teacher_feature_map,
                    anatomical_student_tokens,
                    anatomical_attention,
                    anatomical_visibility_logits,
                    anatomical_fine_feature_map,
                    anatomical_fine_student_tokens,
                    anatomical_fine_attention,
                    anatomical_fine_visibility_logits,
                ) = anatomical_outputs
                raw_features["_anatomical_teacher_feature_map"] = anatomical_teacher_feature_map
                if anatomical_online_teacher_feature_map is not None:
                    raw_features["_anatomical_online_teacher_feature_map"] = anatomical_online_teacher_feature_map
                if self.anatomical_semantic_teacher_enabled:
                    (
                        semantic_foreground_logits,
                        semantic_part_logits,
                        semantic_fine_foreground_logits,
                        semantic_fine_part_logits,
                    ) = self.anatomical_auxiliary_pool.semantic_predictions(
                        anatomical_feature_map,
                        anatomical_fine_feature_map,
                    )
                    raw_features["_anatomical_semantic_foreground_logits"] = semantic_foreground_logits
                    raw_features["_anatomical_semantic_part_logits"] = semantic_part_logits
                    raw_features["_anatomical_semantic_fine_foreground_logits"] = semantic_fine_foreground_logits
                    raw_features["_anatomical_semantic_fine_part_logits"] = semantic_fine_part_logits
                if self.anatomical_decoupled_query_teacher_enabled:
                    (
                        query_student_tokens,
                        query_teacher_tokens,
                        query_teacher_valid,
                        query_visibility_logits,
                        query_foreground_logits,
                        query_part_logits,
                        query_fine_student_tokens,
                        query_fine_teacher_tokens,
                        query_fine_teacher_valid,
                        query_fine_visibility_logits,
                        query_fine_foreground_logits,
                        query_fine_part_logits,
                    ) = self.anatomical_auxiliary_pool.decoupled_query_outputs(
                        anatomical_feature_map,
                        anatomical_fine_feature_map,
                        anatomical_source,
                        anatomical_teacher_source,
                        anatomical_query_masks,
                    )
                    raw_features["_anatomical_query_student_tokens"] = query_student_tokens
                    raw_features["_anatomical_query_teacher_tokens"] = query_teacher_tokens
                    raw_features["_anatomical_query_teacher_valid"] = query_teacher_valid
                    raw_features["_anatomical_query_visibility_logits"] = query_visibility_logits
                    raw_features["_anatomical_query_foreground_logits"] = query_foreground_logits
                    raw_features["_anatomical_query_part_logits"] = query_part_logits
                    raw_features["_anatomical_query_fine_student_tokens"] = query_fine_student_tokens
                    raw_features["_anatomical_query_fine_teacher_tokens"] = query_fine_teacher_tokens
                    raw_features["_anatomical_query_fine_teacher_valid"] = query_fine_teacher_valid
                    raw_features["_anatomical_query_fine_visibility_logits"] = query_fine_visibility_logits
                    raw_features["_anatomical_query_fine_foreground_logits"] = query_fine_foreground_logits
                    raw_features["_anatomical_query_fine_part_logits"] = query_fine_part_logits
            else:
                (
                    anatomical_feature_map,
                    anatomical_student_tokens,
                    anatomical_attention,
                    anatomical_visibility_logits,
                    anatomical_fine_feature_map,
                    anatomical_fine_student_tokens,
                    anatomical_fine_attention,
                    anatomical_fine_visibility_logits,
                ) = anatomical_outputs
            raw_features["_anatomical_feature_map"] = anatomical_feature_map
            raw_features["_anatomical_student_tokens"] = anatomical_student_tokens
            raw_features["_anatomical_attention"] = anatomical_attention
            raw_features["_anatomical_visibility_logits"] = anatomical_visibility_logits
            if anatomical_foreground_logits is not None:
                raw_features["_anatomical_foreground_logits"] = anatomical_foreground_logits
                raw_features["_anatomical_gate_scale"] = anatomical_gate_scale
            if anatomical_fine_feature_map is not None:
                raw_features["_anatomical_fine_feature_map"] = anatomical_fine_feature_map
                raw_features["_anatomical_fine_student_tokens"] = anatomical_fine_student_tokens
                raw_features["_anatomical_fine_attention"] = anatomical_fine_attention
                raw_features["_anatomical_fine_visibility_logits"] = anatomical_fine_visibility_logits
                if anatomical_fine_foreground_logits is not None:
                    raw_features["_anatomical_fine_foreground_logits"] = anatomical_fine_foreground_logits
                    raw_features["_anatomical_fine_gate_scale"] = anatomical_fine_gate_scale
            if self.anatomical_branch_distill_enabled:
                features_by_granularity = {
                    granularity: tuple(
                        base_normalized_bn_features[key]
                        for key, branch_granularity, _ in self.branch_specs
                        if branch_granularity == granularity
                    )
                    for granularity in (1, 2, 4)
                }
                raw_features["_anatomical_branch_features"] = (
                    features_by_granularity[1][0],
                    features_by_granularity[2],
                    features_by_granularity[4],
                )
        elif anatomical_student_only_outputs is not None:
            (
                _,
                anatomical_student_tokens,
                _,
                anatomical_visibility_logits,
                _,
                anatomical_fine_student_tokens,
                _,
                anatomical_fine_visibility_logits,
            ) = anatomical_student_only_outputs

        bn_features = torch.cat(bn_features_list, dim=1)
        raw_features["raw_mean"] = (
            raw_features_list[0] if self.hierarchical_scales else torch.stack(raw_features_list, dim=0).mean(dim=0)
        )
        raw_features["raw_concat"] = torch.cat(
            normalized_raw_features_list + specialist_normalized_raw_features,
            dim=1,
        )
        if self.return_cross_scale_features:
            raw_features["_cross_scale_features"] = tuple(
                F.normalize(
                    torch.cat(
                        [
                            raw_features[key]
                            for key, branch_granularity, _ in self.branch_specs
                            if branch_granularity == granularity
                        ],
                        dim=1,
                    ),
                    p=2,
                    dim=1,
                )
                for granularity in (1, 2, 4)
            )
        if self.return_treeboost_features:
            features_by_granularity = {
                granularity: tuple(
                    base_normalized_bn_features[key]
                    for key, branch_granularity, _ in self.branch_specs
                    if branch_granularity == granularity
                )
                for granularity in (1, 2, 4)
            }
            raw_features["_treeboost_features"] = (
                features_by_granularity[1][0],
                features_by_granularity[2],
                features_by_granularity[4],
            )
        raw_features["coarse_concat"] = torch.cat(coarse_normalized_raw_features_list, dim=1)
        raw_features["concat_bn"] = bn_features
        base_descriptor = F.normalize(
            torch.cat(normalized_bn_features_list + specialist_normalized_bn_features, dim=1),
            p=2,
            dim=1,
        )
        raw_features["norm_concat_bn"] = base_descriptor
        if self.anatomical_deployment_enabled:
            if (
                anatomical_student_tokens is None
                or anatomical_visibility_logits is None
                or anatomical_fine_student_tokens is None
                or anatomical_fine_visibility_logits is None
            ):
                raise RuntimeError("anatomical deployment did not receive local/fine RGB tokens")
            (
                deployed_parts,
                deployed_raw_parts,
                deployed_logits,
                deployed_visibility,
            ) = self._anatomical_deployment_outputs(
                anatomical_student_tokens,
                anatomical_fine_student_tokens,
                anatomical_visibility_logits,
                anatomical_fine_visibility_logits,
            )
            raw_features["_anatomical_base_descriptor"] = base_descriptor
            raw_features["_anatomical_deployment_parts"] = deployed_raw_parts
            raw_features["_anatomical_deployment_logits"] = deployed_logits
            raw_features["_anatomical_deployment_visibility"] = deployed_visibility
            raw_features["_anatomical_deployment_descriptor"] = deployed_parts
            raw_features["norm_concat_bn"] = F.normalize(
                torch.cat(
                    (
                        base_descriptor,
                        self.anatomical_deployment_alpha * deployed_parts,
                    ),
                    dim=1,
                ),
                p=2,
                dim=1,
            )
        if anatomical_outputs is not None and self.anatomical_descriptor_distill_enabled:
            raw_features["_anatomical_final_student"] = self.anatomical_auxiliary_pool.project_descriptor(
                raw_features["norm_concat_bn"]
            )
        self._add_dse_descriptors(raw_features, local_feature)

        if self.late_interaction_matcher is not None:
            features_by_granularity = {
                granularity: tuple(
                    base_normalized_bn_features[key]
                    for key, branch_granularity, _ in self.branch_specs
                    if branch_granularity == granularity
                )
                for granularity in (1, 2, 4)
            }
            raw_features["_late_interaction_features"] = (
                features_by_granularity[1][0],
                features_by_granularity[2],
                features_by_granularity[4],
            )

        if self.compact_deployment_head:
            raw_features["_compact_logits"] = compact_logits
            raw_features["_compact_student"] = compact_raw
            raw_features["_compact_student_bn"] = compact_bn
            raw_features["_compact_teacher"] = raw_features["norm_concat_bn"].detach()
            raw_features["_compact_decoded"] = self.compact_distill_decoder(compact_bn)

        if not self.training:
            if self.emit_late_interaction_packet:
                hierarchy = raw_features.get("_late_interaction_features")
                if not isinstance(hierarchy, tuple):
                    raise RuntimeError("late-interaction evaluation packet requested without a matcher")
                global_branch, coarse_branches, fine_branches = hierarchy
                packet_descriptor = raw_features["norm_concat_bn"]
                if (
                    self.inference_feature == "norm_concat_bn"
                    and not self.anatomical_deployment_enabled
                    and all(value is None for value in visibility_by_key.values())
                ):
                    packet_descriptor = self._fast_norm_concat_descriptor(
                        pooled_by_granularity,
                        global_feature,
                        local_feature,
                        fine_source,
                    )
                return torch.cat(
                    (
                        packet_descriptor,
                        global_branch,
                        *coarse_branches,
                        *fine_branches,
                    ),
                    dim=1,
                )
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature == "visibility_weighted_parts":
                part_keys = self.part_keys
                if visibility_values is None:
                    visibility_values = torch.ones(
                        global_source.shape[0],
                        len(part_keys),
                        device=global_source.device,
                        dtype=base_normalized_bn_features["global"].dtype,
                    )
                return torch.cat(
                    [
                        base_normalized_bn_features["global"],
                        *[base_normalized_bn_features[key] for key in part_keys],
                        visibility_values.to(dtype=base_normalized_bn_features["global"].dtype),
                    ],
                    dim=1,
                )
            if self.inference_feature == "evidence_sinkhorn":
                part_keys = self.part_keys
                dtype = base_normalized_bn_features["global"].dtype
                device = global_source.device
                if visibility_values is None:
                    visibility_values = torch.ones(
                        global_source.shape[0],
                        len(part_keys),
                        device=device,
                        dtype=dtype,
                    )
                if semantic_rarity is None:
                    semantic_rarity = torch.ones(
                        global_source.shape[0],
                        len(part_keys),
                        device=device,
                        dtype=dtype,
                    )
                if semantic_role_logits is None:
                    role_probs = torch.full(
                        (
                            global_source.shape[0],
                            len(part_keys),
                            self.evidence_num_roles,
                        ),
                        1.0 / max(self.evidence_num_roles, 1),
                        device=device,
                        dtype=dtype,
                    )
                else:
                    role_probs = F.softmax(semantic_role_logits, dim=-1).to(dtype=dtype)
                if semantic_nullness is None:
                    semantic_nullness = torch.zeros(
                        global_source.shape[0],
                        len(part_keys),
                        device=device,
                        dtype=dtype,
                    )
                return torch.cat(
                    [
                        base_normalized_bn_features["global"],
                        *[base_normalized_bn_features[key] for key in part_keys],
                        visibility_values.to(dtype=dtype),
                        semantic_rarity.to(dtype=dtype),
                        role_probs.flatten(1),
                        semantic_nullness.to(dtype=dtype),
                    ],
                    dim=1,
                )
            if self.inference_feature in {"dse_weighted", "dse_mix"}:
                return raw_features[self.inference_feature]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.compact_deployment_head:
            feats = raw_features
        elif (
            self.branch_metric
            or self.return_cross_scale_features
            or self.return_treeboost_features
            or self.return_auxiliary_features
            or self.multilevel_suppression_enabled
            or self.late_interaction_matcher is not None
            or self.anatomical_auxiliary_enabled
            or self.hpgrd_part_packet_runtime_active
            or self.retrieval_packet_runtime_active
            or self.mcpt is not None
            or self.jpm is not None
        ):
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature in {"global", "coarse_concat", "dse_weighted", "dse_mix"}:
            feats = raw_features[self.metric_feature]
        else:
            feats = raw_features["raw_mean"]
        if self.drop_global_aux_enabled:
            dropped = self.drop_global_aux(global_source)
            aux_output = self.bn_drop_global_aux(self.global_pool(dropped))
            cls_scores.append(aux_output[1])
        return cls_scores, feats


class GPCLiteMultiBranchHead(MultiBranchHead):
    """Global/part/channel head with CE on every branch and global metric supervision."""

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "norm_concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        head_parts: tuple[int, ...] = (1, 3),
    ):
        super().__init__(
            in_ch=in_ch,
            feat_dim=feat_dim,
            num_classes=num_classes,
            metric_feature=metric_feature,
            inference_feature=inference_feature,
            head_pool=head_pool,
            branch_metric=branch_metric,
            head_parts=head_parts,
            part_pooling="stripes",
            decouple_patterns=False,
            stripe_visibility=False,
        )
        if in_ch % 2 != 0:
            raise ValueError(f"GPC-lite channel split requires even channels, got {in_ch}")
        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_shared = nn.Sequential(
            nn.Conv2d(in_ch // 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.bn_ch0 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)
        self.bn_ch1 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        pooled_by_granularity = {
            granularity: getattr(self, self._pool_attr(granularity))(x) for granularity in self.head_parts
        }
        branch_outputs = {"global": getattr(self, self._bn_attr("global"))(pooled_by_granularity[1])}
        for key, granularity, stripe_index in self.branch_specs:
            if key == "global":
                continue
            pooled = pooled_by_granularity[granularity][:, :, stripe_index : stripe_index + 1, :]
            branch_outputs[key] = getattr(self, self._bn_attr(key))(pooled)

        channel_0, channel_1 = torch.chunk(self.channel_pool(x), chunks=2, dim=1)
        branch_outputs["ch0"] = self.bn_ch0(self.channel_shared(channel_0))
        branch_outputs["ch1"] = self.bn_ch1(self.channel_shared(channel_1))

        ordered_keys = ["global", *self.part_keys, "ch0", "ch1"]
        bn_features_list = [branch_outputs[key][0] for key in ordered_keys]
        raw_features_list = [branch_outputs[key][2] for key in ordered_keys]
        cls_scores = [branch_outputs[key][1] for key in ordered_keys]
        bn_features = torch.cat(bn_features_list, dim=1)
        raw_features = {key: branch_outputs[key][2] for key in ordered_keys}
        # GPC-lite deliberately applies metric and center losses only to the
        # global raw descriptor while every branch retains CE supervision.
        raw_features["raw_mean"] = raw_features["global"]
        raw_features["raw_concat"] = torch.cat(
            [F.normalize(feature, p=2, dim=1) for feature in raw_features_list],
            dim=1,
        )
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(
                [F.normalize(feature, p=2, dim=1) for feature in bn_features_list],
                dim=1,
            ),
            p=2,
            dim=1,
        )

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature in raw_features:
                return raw_features[self.inference_feature]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature in raw_features:
            feats = raw_features[self.metric_feature]
        else:
            feats = raw_features["raw_mean"]
        return cls_scores, feats


class LMBNStyleMultiBranchHead(MultiBranchHead):
    """LMBN-style head with drop-global and channel split branches."""

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        head_parts: tuple[int, ...] = (1, 2),
        drop_h_ratio: float = 0.33,
    ):
        super().__init__(
            in_ch=in_ch,
            feat_dim=feat_dim,
            num_classes=num_classes,
            metric_feature=metric_feature,
            inference_feature=inference_feature,
            head_pool=head_pool,
            branch_metric=branch_metric,
            head_parts=head_parts,
        )
        if in_ch % 2 != 0:
            raise ValueError(f"LMBN-style channel split requires even channels, got {in_ch}")
        self.drop_global = SpatialTopDrop(h_ratio=drop_h_ratio)
        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_shared = nn.Sequential(
            nn.Conv2d(in_ch // 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.bn_drop_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_ch0 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)
        self.bn_ch1 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        pooled_by_granularity = {
            granularity: getattr(self, self._pool_attr(granularity))(x) for granularity in self.head_parts
        }
        branch_outputs = {"global": getattr(self, self._bn_attr("global"))(pooled_by_granularity[1])}
        dropped = self.drop_global(x)
        branch_outputs["drop_global"] = self.bn_drop_global(getattr(self, self._pool_attr(1))(dropped))
        branch_outputs["part_global"] = self.bn_part_global(pooled_by_granularity[1])

        for key, granularity, stripe_index in self.branch_specs:
            if key == "global" or granularity <= 1:
                continue
            pooled = pooled_by_granularity[granularity][:, :, stripe_index : stripe_index + 1, :]
            branch_outputs[key] = getattr(self, self._bn_attr(key))(pooled)

        pooled_channel = self.channel_pool(x)
        channel_0, channel_1 = torch.chunk(pooled_channel, chunks=2, dim=1)
        channel_0 = self.channel_shared(channel_0)
        channel_1 = self.channel_shared(channel_1)
        branch_outputs["ch0"] = self.bn_ch0(channel_0)
        branch_outputs["ch1"] = self.bn_ch1(channel_1)

        ordered_keys = ["global", "drop_global", "part_global", *self.part_keys, "ch0", "ch1"]
        bn_features_list = [branch_outputs[key][0] for key in ordered_keys]
        cls_scores = [branch_outputs[key][1] for key in ordered_keys]
        raw_features_list = [branch_outputs[key][2] for key in ordered_keys]

        bn_features = torch.stack(bn_features_list, dim=2).flatten(1, 2)
        raw_features = {key: branch_outputs[key][2] for key in ordered_keys}
        raw_features["raw_mean"] = torch.stack(raw_features_list, dim=0).mean(dim=0)
        raw_features["raw_concat"] = torch.cat(
            [F.normalize(feature, p=2, dim=1) for feature in raw_features_list],
            dim=1,
        )
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(
                [F.normalize(feature, p=2, dim=1) for feature in bn_features_list],
                dim=1,
            ),
            p=2,
            dim=1,
        )

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature in raw_features:
                return raw_features[self.inference_feature]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature != "raw_mean" and self.metric_feature in raw_features:
            feats = raw_features[self.metric_feature]
        else:
            feats = [
                raw_features["global"],
                raw_features["drop_global"],
                raw_features["part_global"],
            ]
        return cls_scores, feats
