# BoxMOT AGPL-3.0 license

from __future__ import annotations

import torch
from torch import nn

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    BasicLayer,
    BodySlotReadWrite,
    ConvLayer,
    IdentityRegisterCommunication,
    LayerNorm2d,
    NormPreservingWidthMerge,
    PatchEmbed,
    PatchMerging,
    _shift_for_window,
    _to_2tuple,
)
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
    PostFusionLocalMixer,
    make_spatial_conv,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    BodySlotFeatures,
    BodySlotHead,
    GPCLiteMultiBranchHead,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
)
from boxmot.reid.backbones.head_registry import (
    HeadImplementation,
    get_reid_head_spec,
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
        interpolate_pretrained_attention_bias: bool = False,
        attention_mask: bool = False,
        attention_shift: bool = False,
        stage3_global: bool = False,
        stage3_downsample: bool = False,
        stage2_width_merge_after: int = 0,
        stage2_mlp_ratio: float = 4.0,
        stage3_mlp_ratio: float = 4.0,
        stage2_depth: int = 6,
        stage3_depth: int = 2,
        width_first_hierarchy: bool = False,
        identity_registers: bool = False,
        identity_register_count: int = 4,
        identity_register_dim: int = 128,
        identity_register_num_heads: int = 4,
        identity_register_dropout: float = 0.10,
        identity_register_gate_init: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        feat_dim: int = 512,
        neck_dim: int = 512,
        inference_feature: str = "concat_bn",
        feature_fusion: str = "final",
        pyramid_resize_mode: str = "bilinear",
        spatial_conv_mode: str = "standard",
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
        multiscale_channel_alpha: float = 0.5,
        body_slot_mode: str = "recurrent_read",
        body_slot_alpha: float = 0.45,
        body_slot_visibility_floor: float = 0.05,
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
        reid_adapter_stages: tuple[int, ...] = (),
        reid_adapter_reduction: int = 4,
        reid_adapter_suppression_tau: float = 0.0,
        multilevel_suppression: bool = False,
        multilevel_suppression_ratio: float = 0.15,
        branch_metric: bool = False,
        scale_balanced_branches: bool = False,
        native_branch_widths: bool = False,
        fine_map_dim: int = 0,
        compact_deployment_head: bool = False,
        return_cross_scale_features: bool = False,
        return_treeboost_features: bool = False,
        return_auxiliary_features: bool = False,
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
        lmbn_style_head: bool = False,
        drop_h_ratio: float = 0.33,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 160, 320]
        if depths is None:
            depths = [2, 2, 6, 2]
        else:
            depths = list(depths)
        pretrained_total_depth = sum(depths)
        stage2_depth = int(stage2_depth)
        if stage2_depth < 1:
            raise ValueError(f"CSL-TinyViT stage2_depth must be positive, got {stage2_depth}")
        depths[-2] = stage2_depth
        stage3_depth = int(stage3_depth)
        if stage3_depth < 1:
            raise ValueError(f"CSL-TinyViT stage3_depth must be positive, got {stage3_depth}")
        depths[-1] = stage3_depth
        if num_heads is None:
            num_heads = [2, 4, 5, 10]
        custom_window_sizes = window_sizes is not None
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
        window_sizes = list(window_sizes)
        stage2_width_merge_after = int(stage2_width_merge_after)
        if stage2_width_merge_after < 0 or stage2_width_merge_after >= depths[-2]:
            if stage2_width_merge_after != 0:
                raise ValueError(
                    "stage2_width_merge_after must be zero (disabled) or fall before the final Stage-2 block; "
                    f"got {stage2_width_merge_after} for Stage-2 depth={depths[-2]}"
                )
        if stage2_width_merge_after and stage3_downsample:
            raise ValueError("stage2_width_merge_after and stage3_downsample are alternative token-reduction paths")
        use_reduced_stage3_window = (
            (stage3_downsample or stage2_width_merge_after)
            and attention_window_layout == "rect"
            and not custom_window_sizes
        )
        if use_reduced_stage3_window:
            window_sizes[-1] = (12, 4)

        self.loss = loss
        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = float(drop_path_rate)
        self.window_sizes = tuple(_to_2tuple(size) for size in window_sizes)
        self.attention_window_layout = attention_window_layout
        self.attention_bias = str(attention_bias).lower()
        self.interpolate_pretrained_attention_bias = bool(interpolate_pretrained_attention_bias)
        self.attention_mask = bool(attention_mask)
        self.attention_shift = bool(attention_shift)
        self.stage3_global = bool(stage3_global)
        self.stage3_downsample = bool(stage3_downsample)
        self.stage2_width_merge_after = stage2_width_merge_after
        self.stage2_mlp_ratio = float(stage2_mlp_ratio)
        self.stage3_mlp_ratio = float(stage3_mlp_ratio)
        self.stage2_depth = stage2_depth
        self.stage3_depth = stage3_depth
        self.width_first_hierarchy = bool(width_first_hierarchy)
        self.identity_registers_enabled = bool(identity_registers)
        self.identity_register_count = int(identity_register_count)
        self.identity_register_dim = int(identity_register_dim)
        self.identity_register_num_heads = int(
            identity_register_num_heads
        )
        self.identity_register_dropout = float(
            identity_register_dropout
        )
        self.identity_register_gate_init = float(
            identity_register_gate_init
        )
        if self.stage2_mlp_ratio <= 0:
            raise ValueError(f"CSL-TinyViT stage2_mlp_ratio must be positive, got {stage2_mlp_ratio}")
        if self.stage3_mlp_ratio <= 0:
            raise ValueError(f"CSL-TinyViT stage3_mlp_ratio must be positive, got {stage3_mlp_ratio}")
        if self.identity_register_count < 2:
            raise ValueError("identity_register_count must be at least two")
        if self.identity_register_dim < 1:
            raise ValueError("identity_register_dim must be positive")
        if self.identity_register_num_heads < 1:
            raise ValueError(
                "identity_register_num_heads must be positive"
            )
        if not 0 <= self.identity_register_dropout < 1:
            raise ValueError(
                "identity_register_dropout must be in [0, 1)"
            )
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)
        self.pyramid_resize_mode = CSLTinyViTFeatureFusion.normalize_resize_mode(pyramid_resize_mode)
        self.spatial_conv_mode = CSLTinyViTFeatureFusion.normalize_spatial_conv_mode(spatial_conv_mode)
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = _to_2tuple(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.head_type = "lmbn" if lmbn_style_head else str(head_type).lower()
        self.head_spec = get_reid_head_spec(
            self.head_type,
            family="csl_tinyvit",
        )
        self.multilevel_suppression_enabled = bool(
            multilevel_suppression
        )
        self.multilevel_suppression_ratio = float(
            multilevel_suppression_ratio
        )
        if (
            self.multilevel_suppression_enabled
            and self.head_spec.implementation
            != HeadImplementation.MULTI_BRANCH
        ):
            raise ValueError(
                "Multilevel suppression requires the standard CSL-TinyViT head"
            )
        if (
            self.multilevel_suppression_enabled
            and self.feature_fusion
            != "global_final_parts_stage0_semantic_fine"
        ):
            raise ValueError(
                "Multilevel suppression requires the Stage-0 semantic-fine "
                "feature map"
            )
        self.multiscale_channel_alpha = float(
            multiscale_channel_alpha
        )
        self.scale_balanced_branches = bool(scale_balanced_branches)
        self.body_slot_mode = str(body_slot_mode).lower()
        self.body_slot_alpha = float(body_slot_alpha)
        self.body_slot_visibility_floor = float(
            body_slot_visibility_floor
        )
        self.body_slots_enabled = (
            self.head_spec.implementation
            == HeadImplementation.BODY_SLOT
        )
        if self.body_slot_mode not in {"recurrent_read", "recurrent_read_write"}:
            raise ValueError(
                "body_slot_mode must be 'recurrent_read' or "
                "'recurrent_read_write'"
            )
        if not 0 < self.body_slot_alpha < 1:
            raise ValueError("body_slot_alpha must satisfy 0 < value < 1")
        if not 0 <= self.body_slot_visibility_floor < 1:
            raise ValueError(
                "body_slot_visibility_floor must satisfy 0 <= value < 1"
            )
        self.native_branch_widths = bool(native_branch_widths)
        self.fine_map_dim = int(fine_map_dim)
        self.compact_deployment_head = bool(compact_deployment_head)
        self.mcpt_mode = str(mcpt_mode).lower()
        self.mcpt_hidden_dim = int(mcpt_hidden_dim)
        self.mcpt_max_displacement = float(mcpt_max_displacement)
        self.mcpt_start_epoch = int(mcpt_start_epoch)
        self.mcpt_ramp_end_epoch = int(mcpt_ramp_end_epoch)
        if self.fine_map_dim < 0:
            raise ValueError(f"fine_map_dim must be non-negative, got {fine_map_dim}")
        if self.fine_map_dim:
            if self.feature_fusion not in {
                "global_final_parts_stage0_semantic_fine",
                "global_final_parts_stage0_fine_lite",
            }:
                raise ValueError("fine_map_dim requires optimized or lite Stage-0 fine fusion")
            if self.fine_map_dim > neck_dim:
                raise ValueError("fine_map_dim must not exceed neck_dim")
            if part_pooling != "stripes" or tuple(head_parts) != (1, 2, 4):
                raise ValueError("fine_map_dim requires stripe head_parts=(1, 2, 4)")
            if self.head_type != "standard":
                raise ValueError("fine_map_dim requires the standard multi-branch head")
            if self.post_fusion_mixer != "none":
                raise ValueError("fine_map_dim does not support a shared post-fusion mixer")
        if self.feature_fusion == "global_final_parts_stage0_pool_first":
            if part_pooling != "stripes" or tuple(head_parts) != (1, 2, 4):
                raise ValueError("pool-first fusion requires stripe head_parts=(1, 2, 4)")
            if self.head_type != "standard":
                raise ValueError("pool-first fusion requires the standard multi-branch head")
            if self.post_fusion_mixer != "none":
                raise ValueError("pool-first fusion does not support a shared post-fusion mixer")
        if self.stage2_width_merge_after:
            if self.attention_window_layout != "rect":
                raise ValueError("stage2_width_merge_after requires rectangular attention windows")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("stage2_width_merge_after requires optimized Stage-0 semantic-fine fusion")
        if self.width_first_hierarchy:
            if self.attention_window_layout != "rect":
                raise ValueError(
                    "width_first_hierarchy requires rectangular windows"
                )
            if self.feature_fusion != (
                "global_final_parts_stage0_semantic_fine"
            ):
                raise ValueError(
                    "width_first_hierarchy requires optimized Stage-0 "
                    "semantic-fine fusion"
                )
            if self.stage2_width_merge_after or self.stage3_downsample:
                raise ValueError(
                    "width_first_hierarchy cannot be combined with later "
                    "spatial reduction paths"
                )
        if self.identity_registers_enabled and (
            self.stage2_width_merge_after or self.stage3_downsample
        ):
            raise ValueError(
                "identity registers require unreduced 24x8 Stage-2/3 maps"
            )
        if self.identity_registers_enabled and (
            self.head_type != "standard"
            or tuple(head_parts) != (1, 2, 4)
            or part_pooling != "stripes"
        ):
            raise ValueError(
                "identity registers require the unchanged standard "
                "global/2-stripe/4-stripe head"
            )
        if self.body_slots_enabled:
            if tuple(embed_dims) != (64, 128, 256, 448):
                raise ValueError(
                    "body_slot head is implemented for the 11M "
                    "(64,128,256,448) CSL-TinyViT only"
                )
            if neck_dim != BodySlotHead.GLOBAL_DIM:
                raise ValueError("body_slot head requires neck_dim=512")
            if inference_feature != "norm_concat_bn":
                raise ValueError(
                    "body_slot head requires inference_feature='norm_concat_bn'"
                )
            if self.identity_registers_enabled:
                raise ValueError(
                    "body slots and identity registers are independent "
                    "recurrent-memory treatments"
                )
            if self.stage2_width_merge_after or self.stage3_downsample:
                raise ValueError(
                    "body slots require unreduced Stage-2/3 spatial memories"
                )
            if any(
                (
                    hierarchical_branch_attention,
                    branch_set_attention,
                    multiscale_query_decoder,
                    hierarchical_late_interaction,
                    compact_deployment_head,
                )
            ):
                raise ValueError(
                    "body_slot head replaces stripe communication and compact "
                    "deployment treatments"
                )
            if anatomical_target_type != "body_slot_privileged_ema":
                raise ValueError(
                    "body_slot head requires "
                    "anatomical_target_type='body_slot_privileged_ema'"
                )
        if self.compact_deployment_head:
            if self.head_type != "standard":
                raise ValueError("compact_deployment_head requires the standard multi-branch head")
            if self.feature_fusion not in {
                "global_final_parts_stage0_semantic_fine_reference",
                "global_final_parts_stage0_semantic_fine",
            }:
                raise ValueError("compact_deployment_head requires Stage-0 semantic-fine fusion")
            if part_pooling != "stripes" or tuple(head_parts) != (1, 2, 4):
                raise ValueError("compact_deployment_head requires stripe head_parts=(1, 2, 4)")
            if not self.scale_balanced_branches:
                raise ValueError("compact_deployment_head requires scale-balanced teacher branches")
        if self.native_branch_widths:
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("native_branch_widths requires optimized Stage-0 semantic-fine fusion")
            if not self.stage3_downsample:
                raise ValueError("native_branch_widths requires stage3_downsample")
            if part_pooling != "stripes" or tuple(head_parts) != (1, 2, 4):
                raise ValueError("native_branch_widths requires stripe head_parts=(1, 2, 4)")
            if self.head_type != "standard":
                raise ValueError("native_branch_widths requires the standard multi-branch head")
            if self.post_fusion_mixer != "none":
                raise ValueError("native_branch_widths does not support a shared post-fusion mixer")
        if (
            self.scale_balanced_branches
            and not self.head_spec.supports_scale_balance
        ):
            raise ValueError("scale-balanced branches require a standard CSL-TinyViT multi-scale head")
        if self.scale_balanced_branches and branch_metric:
            raise ValueError("scale-balanced branches use one selected metric descriptor, not branch metric losses")
        if drop_global_aux and self.head_type != "standard":
            raise ValueError("drop_global_aux requires CSL-TinyViT head_type='standard'")
        self.reid_adapter_stages = self._normalize_adapter_stages(reid_adapter_stages)
        self.reid_adapter_reduction = int(reid_adapter_reduction)
        self.reid_adapter_suppression_tau = float(reid_adapter_suppression_tau)
        self.evidence_num_roles = int(evidence_num_roles)
        if self.reid_adapter_reduction < 1:
            raise ValueError("reid_adapter_reduction must be positive")
        if not 0.0 <= self.reid_adapter_suppression_tau <= 1.0:
            raise ValueError(
                "reid_adapter_suppression_tau must be in [0, 1], got "
                f"{reid_adapter_suppression_tau}"
            )
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

        # Preserve the pretrained two-block Stage-3 drop-path positions when a
        # speed tier removes its second block; otherwise all earlier stages
        # would receive subtly different regularization in the depth ablation.
        dpr_total_depth = max(sum(depths), pretrained_total_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, dpr_total_depth)]

        # Build stages. The optional final downsample is applied only after
        # retaining Stage-2's pre-merge 24x8 tokens for the local branch.
        self.layers = nn.ModuleList()
        self.stage1_width_merge = (
            NormPreservingWidthMerge()
            if self.width_first_hierarchy
            else None
        )
        input_resolution = patches_resolution
        for i_layer in range(self.num_layers):
            has_downsample = i_layer < self.num_layers - 1
            out_dim = embed_dims[min(i_layer + 1, len(embed_dims) - 1)]
            downsample_stride = None
            if has_downsample:
                downsample_stride = 1 if out_dim in (320, 448, 576) else 2
                if self.stage3_downsample and i_layer == self.num_layers - 2:
                    downsample_stride = 2
                if self.width_first_hierarchy and i_layer == 1:
                    downsample_stride = (2, 1)
                if (
                    self.width_first_hierarchy
                    and i_layer == self.num_layers - 2
                ):
                    downsample_stride = 1
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=input_resolution,
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[:i_layer]) + depths[i_layer]],
                downsample=PatchMerging if has_downsample else None,
                downsample_stride=downsample_stride,
                use_checkpoint=False,
                out_dim=out_dim,
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer_window_size = window_sizes[i_layer]
                if (
                    self.width_first_hierarchy
                    and i_layer == 1
                    and self.attention_window_layout == "rect"
                    and not custom_window_sizes
                ):
                    stage1_windows = ((12, 4), (16, 4))
                    layer_window_size = [
                        stage1_windows[
                            block_index % len(stage1_windows)
                        ]
                        for block_index in range(depths[i_layer])
                    ]
                if (
                    self.stage2_width_merge_after
                    and i_layer == self.num_layers - 2
                    and self.attention_window_layout == "rect"
                    and not custom_window_sizes
                ):
                    layer_window_size = [
                        (12, 8) if block_index < self.stage2_width_merge_after else (12, 4)
                        for block_index in range(depths[i_layer])
                    ]
                layer_shift_size = 0
                if self.attention_shift and i_layer in (1, 2):
                    if isinstance(layer_window_size, list):
                        layer_shift_size = [
                            (0, 0) if block_index % 2 == 0 else _shift_for_window(block_window)
                            for block_index, block_window in enumerate(layer_window_size)
                        ]
                    else:
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
                    mlp_ratio=(
                        self.stage3_mlp_ratio
                        if i_layer == self.num_layers - 1
                        else self.stage2_mlp_ratio
                        if i_layer == self.num_layers - 2
                        else self.mlp_ratio
                    ),
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    attention_bias=self.attention_bias,
                    attention_mask=self.attention_mask,
                    adapter_reduction_ratio=(
                        self.reid_adapter_reduction if i_layer in self.reid_adapter_stages else None
                    ),
                    adapter_suppression_tau=self.reid_adapter_suppression_tau,
                    width_merge_after_blocks=(self.stage2_width_merge_after if i_layer == self.num_layers - 2 else 0),
                    **kwargs,
                )
            self.layers.append(layer)
            if has_downsample:
                stride_pair = (
                    (downsample_stride, downsample_stride)
                    if isinstance(downsample_stride, int)
                    else downsample_stride
                )
                input_resolution = tuple(
                    (size + stride - 1) // stride
                    for size, stride in zip(
                        input_resolution,
                        stride_pair,
                        strict=True,
                    )
                )
            if self.width_first_hierarchy and i_layer == 0:
                if input_resolution[1] % 2:
                    raise ValueError(
                        "width_first_hierarchy requires an even Stage-0 width"
                    )
                input_resolution = (
                    input_resolution[0],
                    input_resolution[1] // 2,
                )
            if self.stage2_width_merge_after and i_layer == self.num_layers - 2:
                input_resolution = (input_resolution[0], input_resolution[1] // 2)

        # Feature neck: project to consistent dim
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], neck_dim, kernel_size=1, bias=False),
            LayerNorm2d(neck_dim),
            make_spatial_conv(neck_dim, mode=self.spatial_conv_mode),
            LayerNorm2d(neck_dim),
        )
        fusion_stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(self.feature_fusion)
        fusion_path_channels = {
            index: (
                embed_dims[index]
                if (self.stage3_downsample or self.stage2_width_merge_after) and index == self.num_layers - 2
                else embed_dims[min(index + 1, len(embed_dims) - 1)]
            )
            for index in fusion_stage_indices
        }
        # Fusion arms contain different numbers of randomly initialized layers.
        # Keep their construction from shifting the RNG state seen by the
        # shared post-fusion mixer and ReID head so same-seed ablations retain
        # identical initialization outside the treatment module.
        with torch.random.fork_rng(devices=[]):
            self.feature_fusion_module = CSLTinyViTFeatureFusion.from_mode(
                mode=self.feature_fusion,
                path_channels=fusion_path_channels,
                out_channels=neck_dim,
                resize_mode=self.pyramid_resize_mode,
                spatial_conv_mode=self.spatial_conv_mode,
                native_branch_widths=self.native_branch_widths,
                fine_map_dim=self.fine_map_dim,
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
        if self.head_spec.implementation == HeadImplementation.LMBN:
            metric_feature = "raw_mean"
        if self.head_spec.implementation == HeadImplementation.LMBN:
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
        elif self.head_spec.implementation == HeadImplementation.BODY_SLOT:
            self.head = BodySlotHead(
                neck_dim,
                num_classes=num_classes,
                head_pool=head_pool,
                alpha=self.body_slot_alpha,
                visibility_floor=self.body_slot_visibility_floor,
            )
        elif self.head_spec.implementation == HeadImplementation.GPC_LITE:
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
            head_in_channels = (
                (
                    neck_dim,
                    self.feature_fusion_module.local_channels,
                    self.feature_fusion_module.fine_output_channels,
                )
                if self.native_branch_widths
                or self.fine_map_dim
                or self.feature_fusion == "global_final_parts_stage0_pool_first"
                else neck_dim
            )
            self.head = MultiBranchHead(
                head_in_channels,
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
                anatomical_auxiliary=anatomical_auxiliary,
                anatomical_token_dim=anatomical_token_dim,
                anatomical_descriptor_distill=(anatomical_descriptor_distill),
                anatomical_branch_distill=anatomical_branch_distill,
                anatomical_multiscale=anatomical_multiscale,
                anatomical_target_type=anatomical_target_type,
                anatomical_accessory_query=anatomical_accessory_query,
                anatomical_deployment=anatomical_deployment,
                anatomical_deployment_dim=anatomical_deployment_dim,
                anatomical_deployment_alpha=anatomical_deployment_alpha,
                branch_metric=branch_metric,
                scale_balanced_branches=self.scale_balanced_branches,
                compact_deployment_head=self.compact_deployment_head,
                specialist_mode=self.head_type,
                multiscale_channel_alpha=(
                    self.multiscale_channel_alpha
                ),
                hierarchical_scales=CSLTinyViTFeatureFusion.uses_hierarchical_scales(self.feature_fusion),
                return_cross_scale_features=return_cross_scale_features,
                return_treeboost_features=return_treeboost_features,
                return_auxiliary_features=(
                    self.identity_registers_enabled
                    or return_auxiliary_features
                ),
                multilevel_suppression=(
                    self.multilevel_suppression_enabled
                ),
                multilevel_suppression_ratio=(
                    self.multilevel_suppression_ratio
                ),
                hierarchical_branch_attention=hierarchical_branch_attention,
                branch_attention_token_dim=branch_attention_token_dim,
                branch_attention_num_heads=branch_attention_num_heads,
                branch_attention_num_layers=branch_attention_num_layers,
                branch_attention_mlp_ratio=branch_attention_mlp_ratio,
                branch_attention_dropout=branch_attention_dropout,
                branch_set_attention=branch_set_attention,
                branch_set_attention_token_dim=branch_set_attention_token_dim,
                branch_set_attention_num_heads=branch_set_attention_num_heads,
                branch_set_attention_num_layers=branch_set_attention_num_layers,
                branch_set_attention_mlp_ratio=branch_set_attention_mlp_ratio,
                branch_set_attention_dropout=branch_set_attention_dropout,
                multiscale_query_decoder=multiscale_query_decoder,
                query_decoder_dim=query_decoder_dim,
                query_decoder_num_heads=query_decoder_num_heads,
                query_decoder_num_layers=query_decoder_num_layers,
                query_decoder_mlp_ratio=query_decoder_mlp_ratio,
                query_decoder_dropout=query_decoder_dropout,
                hierarchical_late_interaction=hierarchical_late_interaction,
                late_interaction_dim=late_interaction_dim,
                late_interaction_num_heads=late_interaction_num_heads,
                late_interaction_num_layers=late_interaction_num_layers,
                late_interaction_sinkhorn_iters=late_interaction_sinkhorn_iters,
                late_interaction_null_tokens=late_interaction_null_tokens,
                late_interaction_base_score_init=late_interaction_base_score_init,
                mcpt_mode=self.mcpt_mode,
                mcpt_hidden_dim=self.mcpt_hidden_dim,
                mcpt_max_displacement=self.mcpt_max_displacement,
                mcpt_start_epoch=self.mcpt_start_epoch,
                mcpt_ramp_end_epoch=self.mcpt_ramp_end_epoch,
                jpm=jpm,
                jpm_num_groups=jpm_num_groups,
                jpm_shift=jpm_shift,
                jpm_token_dim=jpm_token_dim,
                jpm_num_heads=jpm_num_heads,
                jpm_mlp_ratio=jpm_mlp_ratio,
                jpm_dropout=jpm_dropout,
            )

        self.body_slot_modules = nn.ModuleList()
        if self.body_slots_enabled:
            self.body_slot_seed = nn.Parameter(
                torch.empty(
                    1,
                    BodySlotHead.NUM_SLOTS,
                    BodySlotHead.SLOT_DIM,
                )
            )
            self.body_slot_roles = nn.Parameter(
                torch.empty(
                    1,
                    BodySlotHead.NUM_SLOTS,
                    BodySlotHead.SLOT_DIM,
                )
            )
            writeback = self.body_slot_mode == "recurrent_read_write"
            for spatial_dim in (
                embed_dims[1],
                embed_dims[-1],
                embed_dims[-1],
            ):
                self.body_slot_modules.append(
                    BodySlotReadWrite(
                        spatial_dim,
                        slot_dim=BodySlotHead.SLOT_DIM,
                        num_slots=BodySlotHead.NUM_SLOTS,
                        num_heads=4,
                        mlp_ratio=2.0,
                        dropout=0.0,
                        writeback=writeback,
                        gate_init=0.0,
                    )
                )
        else:
            self.register_parameter("body_slot_seed", None)
            self.register_parameter("body_slot_roles", None)

        self.identity_register_modules = nn.ModuleList()
        if self.identity_registers_enabled:
            register_dim = self.identity_register_dim
            if register_dim % self.identity_register_num_heads:
                raise ValueError(
                    f"Register width {register_dim} must divide "
                    "identity_register_num_heads="
                    f"{self.identity_register_num_heads}"
                )
            rng_state = torch.random.get_rng_state()
            try:
                self.identity_register_seed = nn.Parameter(
                    torch.empty(
                        1,
                        self.identity_register_count,
                        register_dim,
                    )
                )
                for stage_index in (2, 3):
                    self.identity_register_modules.append(
                        IdentityRegisterCommunication(
                            embed_dims[-1],
                            register_dim=register_dim,
                            num_registers=self.identity_register_count,
                            num_heads=self.identity_register_num_heads,
                            window_size=_to_2tuple(
                                window_sizes[stage_index]
                            ),
                            dropout=self.identity_register_dropout,
                            gate_init=self.identity_register_gate_init,
                        )
                    )
            finally:
                torch.random.set_rng_state(rng_state)
        else:
            self.register_parameter("identity_register_seed", None)
        self._last_identity_register_tokens: tuple[
            torch.Tensor,
            ...,
        ] = ()

        # Initialize weights
        self.apply(self._init_weights)
        self.feature_fusion_module.initialize_dynamic_gate()
        self._reset_reid_specific_initialization()
        if self.body_slot_seed is not None:
            nn.init.trunc_normal_(self.body_slot_seed, std=0.02)
            nn.init.trunc_normal_(self.body_slot_roles, std=0.02)
            for module in self.body_slot_modules:
                module.reset_teacher()
        if self.identity_register_seed is not None:
            nn.init.trunc_normal_(
                self.identity_register_seed,
                std=0.02,
            )

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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _reset_reid_specific_initialization(self) -> None:
        reset = getattr(self.head, "reset_reid_initialization", None)
        if reset is not None:
            reset()

    def forward_features(
        self,
        x,
        *,
        body_slot_masks: torch.Tensor | None = None,
    ):
        """Extract spatial feature map from backbone."""
        x = self.patch_embed(x)
        out_size = (x.shape[2], x.shape[3])
        fusion_features: dict[int, tuple[torch.Tensor, tuple[int, int]]] = {}
        body_slot_state = None
        body_slot_states = []
        body_slot_attentions = []
        body_slot_visibility = []
        body_slot_teacher_slots = []
        body_slot_teacher_valid = []
        body_slot_teacher_attentions = []

        # Stage 0 (conv layer operates on 4D tensor)
        x, out_size = self.layers[0](x, out_size)
        if self.body_slots_enabled:
            body_slot_state = self.body_slot_seed.expand(x.shape[0], -1, -1)
            (
                x,
                body_slot_state,
                slot_visibility,
                slot_attention,
                teacher_slots,
                teacher_valid,
                teacher_attention,
            ) = self.body_slot_modules[0](
                x,
                out_size,
                body_slot_state,
                self.body_slot_roles,
                teacher_masks=body_slot_masks,
            )
            body_slot_states.append(body_slot_state)
            body_slot_attentions.append(slot_attention)
            body_slot_visibility.append(slot_visibility)
            if teacher_slots is not None:
                body_slot_teacher_slots.append(teacher_slots)
                body_slot_teacher_valid.append(teacher_valid)
                body_slot_teacher_attentions.append(teacher_attention)
        if 0 in self._fusion_stage_indices:
            fusion_features[0] = (x, out_size)
        if self.stage1_width_merge is not None:
            x, out_size = self.stage1_width_merge(x, out_size)

        # Stages 1+ (attention layers operate on 3D tokens)
        register_state = None
        register_outputs = []
        for i in range(1, len(self.layers)):
            retain_pre_width_merge = (
                self.stage2_width_merge_after > 0 and i == len(self.layers) - 2 and i in self._fusion_stage_indices
            )
            retain_pre_merge = self.stage3_downsample and i == len(self.layers) - 2 and i in self._fusion_stage_indices
            if retain_pre_width_merge:
                x, out_size, stage_tokens, stage_size = self.layers[i](
                    x,
                    out_size,
                    return_pre_width_merge=True,
                )
            elif retain_pre_merge:
                x, out_size, stage_tokens, stage_size = self.layers[i](
                    x,
                    out_size,
                    return_pre_downsample=True,
                )
            else:
                x, out_size = self.layers[i](x, out_size)
                stage_tokens = x
                stage_size = out_size
            if self.identity_registers_enabled and i in (2, 3):
                if register_state is None:
                    register_state = self.identity_register_seed.expand(
                        x.shape[0],
                        -1,
                        -1,
                    )
                register_module = self.identity_register_modules[i - 2]
                x, register_state = register_module(
                    x,
                    out_size,
                    register_state,
                )
                register_outputs.append(register_state)
                if not retain_pre_width_merge and not retain_pre_merge:
                    stage_tokens = x
                    stage_size = out_size
            if self.body_slots_enabled and i in (2, 3):
                if body_slot_state is None:
                    raise RuntimeError(
                        "Body-slot state was not initialized at Stage 0"
                    )
                (
                    x,
                    body_slot_state,
                    slot_visibility,
                    slot_attention,
                    teacher_slots,
                    teacher_valid,
                    teacher_attention,
                ) = self.body_slot_modules[i - 1](
                    x,
                    out_size,
                    body_slot_state,
                    self.body_slot_roles,
                    teacher_masks=body_slot_masks,
                )
                body_slot_states.append(body_slot_state)
                body_slot_attentions.append(slot_attention)
                body_slot_visibility.append(slot_visibility)
                if teacher_slots is not None:
                    body_slot_teacher_slots.append(teacher_slots)
                    body_slot_teacher_valid.append(teacher_valid)
                    body_slot_teacher_attentions.append(teacher_attention)
                stage_tokens = x
                stage_size = out_size
            if i in self._fusion_stage_indices:
                fusion_features[i] = (stage_tokens, stage_size)
        self._last_identity_register_tokens = (
            tuple(register_outputs) if self.training else ()
        )

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
            x = tuple(
                self.post_fusion_mixer_module(feature)
                for feature in x
            )
        else:
            x = self.post_fusion_mixer_module(x)
        if self.body_slots_enabled:
            if body_slot_state is None or len(body_slot_states) != 3:
                raise RuntimeError(
                    "Body-slot backbone did not produce Stage-0/2/3 states"
                )
            global_map = x[0] if isinstance(x, tuple) else x
            has_teacher = len(body_slot_teacher_slots) == 3
            return BodySlotFeatures(
                global_map=global_map,
                slots=body_slot_state,
                visibility_logits=body_slot_visibility[-1],
                stage_slots=tuple(body_slot_states),
                stage_attentions=tuple(body_slot_attentions),
                stage_visibility_logits=tuple(body_slot_visibility),
                teacher_slots=(
                    tuple(body_slot_teacher_slots)
                    if has_teacher
                    else None
                ),
                teacher_valid=(
                    tuple(body_slot_teacher_valid)
                    if has_teacher
                    else None
                ),
                teacher_attentions=(
                    tuple(body_slot_teacher_attentions)
                    if has_teacher
                    else None
                ),
            )
        return x

    def forward_head(
        self,
        features,
        *,
        pids: torch.Tensor | None = None,
        anatomical_pose: torch.Tensor | None = None,
        anatomical_query_masks: torch.Tensor | None = None,
    ):
        if isinstance(features, BodySlotFeatures):
            return self.head(features)
        if (
            pids is None
            and anatomical_pose is None
            and anatomical_query_masks is None
        ):
            return self.head(features)
        return self.head(
            features,
            pids=pids,
            anatomical_pose=anatomical_pose,
            anatomical_query_masks=anatomical_query_masks,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        pids: torch.Tensor | None = None,
        anatomical_pose: torch.Tensor | None = None,
        anatomical_query_masks: torch.Tensor | None = None,
    ):
        """Run the RGB model."""
        output = self.forward_head(
            self.forward_features(
                x,
                body_slot_masks=(
                    anatomical_query_masks
                    if self.body_slots_enabled
                    else None
                ),
            ),
            pids=pids,
            anatomical_pose=anatomical_pose,
            anatomical_query_masks=anatomical_query_masks,
        )
        if (
            self.training
            and self._last_identity_register_tokens
            and isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[1], dict)
        ):
            output[1]["_identity_register_tokens"] = (
                self._last_identity_register_tokens
            )
        self._last_identity_register_tokens = ()
        return output

    def set_multilevel_suppression_progress(
        self,
        progress: float,
    ) -> None:
        """Set the training-only classifier suppression schedule."""
        setter = getattr(
            self.head,
            "set_multilevel_suppression_progress",
            None,
        )
        if setter is not None:
            setter(progress)

    def set_anatomical_auxiliary_active(self, active: bool) -> None:
        """Forward the runtime-only anatomy schedule to the ReID head."""
        setter = getattr(
            self.head,
            "set_anatomical_auxiliary_active",
            None,
        )
        if setter is not None:
            setter(active)

    @torch.no_grad()
    def update_body_slot_teacher(self, momentum: float) -> None:
        """EMA-update the training-only masked RGB slot projections."""
        if not self.body_slots_enabled:
            raise RuntimeError("Body-slot teacher update requested while disabled")
        for module in self.body_slot_modules:
            module.update_teacher(momentum)
