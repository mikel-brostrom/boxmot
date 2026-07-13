# BoxMOT AGPL-3.0 license

from __future__ import annotations

import torch
from torch import nn

from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    BasicLayer,
    ConvLayer,
    LayerNorm2d,
    PatchEmbed,
    PatchMerging,
    _shift_for_window,
    _to_2tuple,
)
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
    PostFusionLocalMixer,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    GPCLiteMultiBranchHead,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
)

__all__ = ["CSLTinyViT"]


class CSLTinyViT(ReIDBackbone):
    """CSL-TinyViT: hybrid CNN-Transformer ReID backbone.

    Combines efficient MBConv early stages with windowed self-attention
    later stages, producing multi-granularity features via a multi-branch head.

    Input: 3×384×128 (H×W)
    Output:
      - Inference: num_branches × feat_dim feature vector
      - Training: (cls_scores_per_branch, features)
    """

    def __init__(
        self,
        num_classes: int,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        *,
        img_size: tuple[int, int] = (384, 128),
        in_chans: int = 3,
        embed_dims: list[int] = None,
        depths: list[int] = None,
        num_heads: list[int] = None,
        window_sizes: list[int | tuple[int, int]] = None,
        attention_window_layout: str = "legacy",
        attention_bias: str = "absolute",
        attention_mask: bool = False,
        attention_shift: bool = False,
        stage3_global: bool = False,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        feat_dim: int = 512,
        neck_dim: int = 512,
        inference_feature: str = "concat_bn",
        feature_fusion: str = "final",
        post_fusion_mixer: str = "none",
        post_fusion_mixer_reduction: int = 4,
        post_fusion_mixer_kernel: tuple[int, int] = (5, 3),
        post_fusion_mixer_gamma_init: float = 0.0,
        head_pool: str = "avg",
        head_parts: tuple[int, ...] = (1, 2),
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        head_type: str = "standard",
        stripe_visibility: bool = False,
        drop_global_aux: bool = False,
        drop_global_aux_ratio: float = 0.25,
        evidence_num_roles: int = 8,
        reid_adapter_stages: tuple[int, ...] = (),
        reid_adapter_reduction: int = 4,
        branch_metric: bool = False,
        lmbn_style_head: bool = False,
        drop_h_ratio: float = 0.33,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 160, 320]
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [2, 4, 5, 10]
        if window_sizes is None:
            attention_window_layout = str(attention_window_layout).lower()
            if attention_window_layout == "legacy":
                window_sizes = [7, 7, 14, 7]
            elif attention_window_layout == "rect":
                window_sizes = [7, (12, 4), (12, 8), (12, 8)]
            else:
                raise ValueError(f"Unsupported CSL-TinyViT attention_window_layout: {attention_window_layout}")
        else:
            attention_window_layout = str(attention_window_layout).lower()

        self.loss = loss
        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = float(drop_path_rate)
        self.window_sizes = tuple(_to_2tuple(size) for size in window_sizes)
        self.attention_window_layout = attention_window_layout
        self.attention_bias = str(attention_bias).lower()
        self.attention_mask = bool(attention_mask)
        self.attention_shift = bool(attention_shift)
        self.stage3_global = bool(stage3_global)
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = _to_2tuple(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.head_type = "lmbn" if lmbn_style_head else str(head_type).lower()
        if self.head_type not in {"standard", "gpc_lite", "lmbn"}:
            raise ValueError(f"Unsupported CSL-TinyViT head_type: {head_type}")
        if drop_global_aux and self.head_type != "standard":
            raise ValueError("drop_global_aux requires CSL-TinyViT head_type='standard'")
        self.reid_adapter_stages = self._normalize_adapter_stages(reid_adapter_stages)
        self.reid_adapter_reduction = int(reid_adapter_reduction)
        self.evidence_num_roles = int(evidence_num_roles)
        if self.reid_adapter_reduction < 1:
            raise ValueError("reid_adapter_reduction must be positive")
        if self.evidence_num_roles < 1:
            raise ValueError(f"evidence_num_roles must be positive, got {evidence_num_roles}")
        self.pretrained_match_count: int | None = None
        self.pretrained_total_count: int | None = None
        self.pretrained_url: str | None = None

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0], img_size=img_size, activation=activation
        )
        patches_resolution = self.patch_embed.patches_resolution

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            input_resolution = (
                patches_resolution[0] // (2 ** (i_layer if i_layer < 2 else 2)),
                patches_resolution[1] // (2 ** (i_layer if i_layer < 2 else 2)),
            )
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=input_resolution,
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer_window_size = window_sizes[i_layer]
                layer_shift_size = 0
                if self.attention_shift and i_layer in (1, 2):
                    shift_size = _shift_for_window(layer_window_size)
                    layer_shift_size = [
                        (0, 0) if block_index % 2 == 0 else shift_size for block_index in range(depths[i_layer])
                    ]
                if self.stage3_global and i_layer == self.num_layers - 1:
                    layer_window_size = [
                        layer_window_size if block_index < depths[i_layer] - 1 else input_resolution
                        for block_index in range(depths[i_layer])
                    ]
                    layer_shift_size = [(0, 0) for _ in range(depths[i_layer])]
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=layer_window_size,
                    shift_size=layer_shift_size,
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    attention_bias=self.attention_bias,
                    attention_mask=self.attention_mask,
                    adapter_reduction_ratio=(
                        self.reid_adapter_reduction if i_layer in self.reid_adapter_stages else None
                    ),
                    **kwargs,
                )
            self.layers.append(layer)

        # Feature neck: project to consistent dim
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], neck_dim, kernel_size=1, bias=False),
            LayerNorm2d(neck_dim),
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dim),
        )
        fusion_stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(self.feature_fusion)
        fusion_path_channels = {
            index: embed_dims[min(index + 1, len(embed_dims) - 1)] for index in fusion_stage_indices
        }
        self.feature_fusion_module = CSLTinyViTFeatureFusion.from_mode(
            mode=self.feature_fusion,
            path_channels=fusion_path_channels,
            out_channels=neck_dim,
        )
        self._fusion_stage_indices = self.feature_fusion_module.stage_indices
        if self.post_fusion_mixer == "dwconv":
            self.post_fusion_mixer_module = PostFusionLocalMixer(
                channels=neck_dim,
                reduction=self.post_fusion_mixer_reduction,
                kernel_size=self.post_fusion_mixer_kernel,
                gamma_init=self.post_fusion_mixer_gamma_init,
            )
        else:
            self.post_fusion_mixer_module = nn.Identity()

        # Multi-branch ReID head.
        # For standard CSL-TinyViT, MS loss trains on the same concatenated BN
        # embedding used at inference. For LMBN-style heads, keep LightMBN-like
        # metric supervision on the three raw branch features
        # (global/drop-global/part-global) regardless of loss type.
        metric_feature = "concat_bn" if loss == "ms" else "raw_mean"
        if self.head_type == "lmbn":
            metric_feature = "raw_mean"
        if self.head_type == "lmbn":
            self.head = LMBNStyleMultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
                drop_h_ratio=drop_h_ratio,
            )
        elif self.head_type == "gpc_lite":
            self.head = GPCLiteMultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
            )
        else:
            self.head = MultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                part_pooling=part_pooling,
                num_part_tokens=num_part_tokens,
                decouple_patterns=decouple_patterns,
                pattern_adapter_dim=pattern_adapter_dim,
                stripe_visibility=stripe_visibility,
                drop_global_aux=drop_global_aux,
                drop_global_aux_ratio=drop_global_aux_ratio,
                evidence_num_roles=self.evidence_num_roles,
                branch_metric=branch_metric,
                hierarchical_scales=self.feature_fusion == "global_final_parts_hierarchical_fpn",
            )

        # Initialize weights
        self.apply(self._init_weights)
        self.feature_fusion_module.initialize_dynamic_gate()
        self._reset_reid_specific_initialization()

    @staticmethod
    def _normalize_adapter_stages(stages) -> tuple[int, ...]:
        if stages is None:
            return ()
        if isinstance(stages, str):
            if stages.lower() in {"", "none", "off"}:
                return ()
            values = [part for part in stages.replace(";", ",").split(",") if part.strip()]
        elif isinstance(stages, int):
            values = [stages]
        else:
            values = list(stages)
        normalized = tuple(dict.fromkeys(int(stage) for stage in values))
        invalid = [stage for stage in normalized if stage not in {1, 2, 3}]
        if invalid:
            raise ValueError(f"CSL-TinyViT ReID adapters only support attention stages 1, 2, 3; got {invalid}")
        return normalized

    @staticmethod
    def _normalize_post_fusion_mixer(mixer: str) -> str:
        normalized = str(mixer).lower()
        if normalized in {"", "none", "off", "identity"}:
            return "none"
        if normalized in {"dwconv", "local", "dwconv5x3"}:
            return "dwconv"
        raise ValueError(f"Unsupported CSL-TinyViT post_fusion_mixer: {mixer}")

    @property
    def fusion_scales(self) -> nn.ParameterDict:
        return self.feature_fusion_module.residual_scales

    @property
    def fusion_weights(self) -> nn.Parameter | None:
        return self.feature_fusion_module.fusion_weights

    def _normalized_fusion_weights(self) -> torch.Tensor:
        return self.feature_fusion_module.normalized_weights()

    @property
    def blocks(self) -> nn.ModuleList:
        """Compatibility alias for generic ViT trainer logic without state duplication."""
        return self.layers

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
        old_to_new_prefixes = {
            "fusion_projections.": "feature_fusion_module.projections.",
            "fusion_scales.": "feature_fusion_module.residual_scales.",
        }
        for old_prefix, new_prefix in old_to_new_prefixes.items():
            old_full_prefix = f"{prefix}{old_prefix}"
            for key in list(state_dict.keys()):
                if key.startswith(old_full_prefix):
                    new_key = f"{prefix}{new_prefix}{key[len(old_full_prefix) :]}"
                    state_dict.setdefault(new_key, state_dict[key])
                    del state_dict[key]

        # Older CSL-TinyViT checkpoints registered self.blocks = self.layers,
        # which serialized duplicate top-level blocks.* keys. Keep loading those
        # checkpoints without retaining the duplicate alias in new state_dicts.
        old_blocks_prefix = f"{prefix}blocks."
        new_layers_prefix = f"{prefix}layers."
        for key in list(state_dict.keys()):
            if key.startswith(old_blocks_prefix):
                new_key = f"{new_layers_prefix}{key[len(old_blocks_prefix) :]}"
                state_dict.setdefault(new_key, state_dict[key])
                del state_dict[key]

        old_weight_key = f"{prefix}fusion_weights"
        new_weight_key = f"{prefix}feature_fusion_module.fusion_weights"
        if old_weight_key in state_dict:
            state_dict.setdefault(new_weight_key, state_dict[old_weight_key])
            del state_dict[old_weight_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _reset_reid_specific_initialization(self) -> None:
        reset = getattr(self.head, "reset_reid_initialization", None)
        if reset is not None:
            reset()

    def forward_features(self, x):
        """Extract spatial feature map from backbone."""
        x = self.patch_embed(x)
        out_size = (x.shape[2], x.shape[3])
        fusion_features: dict[int, tuple[torch.Tensor, tuple[int, int]]] = {}

        # Stage 0 (conv layer operates on 4D tensor)
        x, out_size = self.layers[0](x, out_size)
        if 0 in self._fusion_stage_indices:
            fusion_features[0] = (x, out_size)

        # Stages 1+ (attention layers operate on 3D tokens)
        for i in range(1, len(self.layers)):
            x, out_size = self.layers[i](x, out_size)
            if i in self._fusion_stage_indices:
                fusion_features[i] = (x, out_size)

        # Reshape back to spatial for neck
        B, _, C = x.size()
        x = x.view(B, out_size[0], out_size[1], C).permute(0, 3, 1, 2)
        x = self.neck(x)
        path_features: dict[int, torch.Tensor] = {}
        for index in self._fusion_stage_indices:
            stage_tokens, stage_size = fusion_features[index]
            stage = stage_tokens.view(B, stage_size[0], stage_size[1], -1)
            path_features[index] = stage.permute(0, 3, 1, 2)
        x = self.feature_fusion_module(x, path_features)
        if isinstance(x, tuple):
            return tuple(self.post_fusion_mixer_module(feature) for feature in x)
        return self.post_fusion_mixer_module(x)

    def forward_head(self, features):
        return self.head(features)
