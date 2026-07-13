# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.blocks import LayerNorm2d, _to_2tuple

__all__ = ["CSLTinyViTFeatureFusion", "PostFusionLocalMixer"]


class CSLTinyViTFeatureFusion(nn.Module):
    """Swappable spatial feature fusion module for CSL-TinyViT stage outputs."""

    _VALID_MODES = {
        "final",
        "last2",
        "last3",
        "last4_layer0_target",
        "last3_stage2_target",
        "last3_stage1_concat",
        "global_final_parts_stage1_concat",
        "global_final_parts_fpn_layer0",
        "last3_fpn_stage1_add",
        "last3_fpn_stage1_split",
        "last3_panet_stage1_split",
        "last3_panet_stage1_shared",
        "last3_panet_stage1_scale_aware",
        "last3_bifpn_stage1_split",
        "last3_bifpn_stage1_branch_aware",
        "global_final_parts_hierarchical_fpn",
        "last3_fpn_stage2",
        "last3_pafpn_stage2",
        "last4_fpn_layer0_target",
        "global_final_parts_stage2",
        "late_concat_stage2",
        "weighted_last2",
        "weighted_last3",
        "normpres_last2",
        "normpres_last3",
        "dynamic_last3",
        "dynamic_last3_scale_token",
    }
    _VALID_FUSION_TYPES = {
        "final",
        "residual",
        "weighted",
        "norm_preserved",
        "dynamic",
        "dynamic_scale_token",
        "fpn",
        "pafpn",
        "split_global_local",
        "late_concat",
        "concat_compress",
        "split_stage1_concat",
        "split_fpn_layer0",
        "fpn_topdown",
        "panet",
        "bifpn",
        "hierarchical_fpn",
    }

    def __init__(
        self,
        fusion_type: str,
        stage_indices: tuple[int, ...],
        path_channels: dict[int, int],
        out_channels: int,
        target_stage_index: int | None = None,
        mode: str | None = None,
    ):
        super().__init__()
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in self._VALID_FUSION_TYPES:
            raise ValueError(f"Unsupported CSL-TinyViT feature fusion type: {fusion_type}")
        self.mode = str(mode or self.fusion_type).lower()
        self.stage_indices = tuple(stage_indices)
        self.target_stage_index = target_stage_index
        if self.fusion_type == "final" and self.stage_indices:
            raise ValueError("CSL-TinyViT final feature fusion must not define path stages")
        if self.target_stage_index is not None and self.target_stage_index not in self.stage_indices:
            raise ValueError(
                "CSL-TinyViT feature fusion target stage must be one of the fused path stages, "
                f"got target_stage_index={self.target_stage_index}, stage_indices={self.stage_indices}"
            )
        self.weighted = self.fusion_type == "weighted"
        self.norm_preserved = self.fusion_type == "norm_preserved"
        self.dynamic = self.fusion_type in {"dynamic", "dynamic_scale_token"}
        self.fpn = self.fusion_type == "fpn"
        self.pafpn = self.fusion_type == "pafpn"
        self.split_global_local = self.fusion_type == "split_global_local"
        self.late_concat = self.fusion_type == "late_concat"
        self.concat_compress = self.fusion_type == "concat_compress"
        self.split_stage1_concat = self.fusion_type == "split_stage1_concat"
        self.split_fpn_layer0 = self.fusion_type == "split_fpn_layer0"
        self.fpn_topdown = self.fusion_type == "fpn_topdown"
        self.panet = self.fusion_type == "panet"
        self.bifpn = self.fusion_type == "bifpn"
        self.hierarchical_fpn = self.fusion_type == "hierarchical_fpn"
        self.use_scale_token = self.fusion_type == "dynamic_scale_token"

        missing = [index for index in self.stage_indices if index not in path_channels]
        if missing:
            raise ValueError(f"Missing CSL-TinyViT fusion path channels for stages: {missing}")

        self.projections = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.Conv2d(path_channels[index], out_channels, kernel_size=1, bias=False),
                    LayerNorm2d(out_channels),
                )
                for index in self.stage_indices
            }
        )
        self.residual_scales = nn.ParameterDict(
            {
                str(index): nn.Parameter(torch.zeros(()))
                for index in (
                    self.stage_indices
                    if self.fusion_type in {"residual", "split_global_local", "split_stage1_concat", "split_fpn_layer0", "hierarchical_fpn"}
                    else ()
                )
            }
        )
        if self.weighted:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0, *([1e-3] * len(self.stage_indices))]))
        else:
            self.register_parameter("fusion_weights", None)

        if self.dynamic:
            num_paths = 1 + len(self.stage_indices)
            gate_hidden_dim = max(out_channels // 4, 64)
            scale_token_dim = max(min(out_channels // 16, 64), 16)
            if self.use_scale_token:
                self.scale_token_projection = nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.Linear(out_channels, scale_token_dim),
                    nn.GELU(),
                )
                self.scale_tokens = nn.Parameter(torch.empty(num_paths, scale_token_dim))
                nn.init.trunc_normal_(self.scale_tokens, std=0.02)
                self.scale_token_norm = nn.LayerNorm(scale_token_dim)
                gate_input_dim = out_channels + num_paths * scale_token_dim
            else:
                self.scale_token_projection = None
                self.register_parameter("scale_tokens", None)
                self.scale_token_norm = None
                gate_input_dim = out_channels
            self.dynamic_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, num_paths),
            )
            self.initialize_dynamic_gate()
        else:
            self.scale_token_projection = None
            self.register_parameter("scale_tokens", None)
            self.scale_token_norm = None
            self.dynamic_gate = None

        if self.pafpn:
            if self.stage_indices != (2, 1) or self.target_stage_index != 2:
                raise ValueError("CSL-TinyViT PAFPN fusion currently supports last3_pafpn_stage2 only")
            self.pafpn_top_down = nn.ModuleDict(
                {
                    "2": self._make_pafpn_block(out_channels),
                    "1": self._make_pafpn_block(out_channels),
                }
            )
            self.pafpn_bottom_up = nn.ModuleDict({"2": self._make_pafpn_block(out_channels)})
        else:
            self.pafpn_top_down = nn.ModuleDict()
            self.pafpn_bottom_up = nn.ModuleDict()

        if self.concat_compress or self.split_stage1_concat:
            self.concat_projection = nn.Sequential(
                nn.Conv2d(out_channels * (1 + len(self.stage_indices)), out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            self.concat_projection = nn.Identity()

        if self.fpn_topdown:
            if self.stage_indices != (2, 1) or self.target_stage_index != 1:
                raise ValueError("Top-down FPN currently supports Stage-1 additive modes only")
            self.fpn_global_output = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels), nn.GELU(),
            )
            self.fpn_output = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            self.fpn_global_output = nn.Identity()
            self.fpn_output = nn.Identity()

        if self.split_fpn_layer0:
            if self.stage_indices != (2, 1, 0) or self.target_stage_index != 0:
                raise ValueError("Layer-0 split FPN requires final -> Stage 2 -> Stage 1 -> Stage 0")
            self.layer0_fpn_outputs = nn.ModuleDict({
                str(index): nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(out_channels), nn.GELU(),
                ) for index in (2, 1, 0)
            })
        else:
            self.layer0_fpn_outputs = nn.ModuleDict()

        if self.panet:
            if self.stage_indices != (2, 1) or self.target_stage_index != 1:
                raise ValueError("PANet currently supports the Stage-1 split/shared modes only")
            self.panet_downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                LayerNorm2d(out_channels), nn.GELU(),
            )
            self.panet_output = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels), nn.GELU(),
            )
            self.panet_scale_gate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
            nn.init.zeros_(self.panet_scale_gate.weight)
            nn.init.constant_(self.panet_scale_gate.bias, math.log(0.7 / 0.3))
        else:
            self.panet_downsample = nn.Identity()
            self.panet_output = nn.Identity()
            self.panet_scale_gate = nn.Identity()

        if self.bifpn:
            if self.stage_indices != (2, 1) or self.target_stage_index != 1:
                raise ValueError("BiFPN currently supports last3_bifpn_stage1_split only")
            self.bifpn_weights = nn.ParameterDict({
                "top_low": nn.Parameter(torch.ones(2)), "top_high": nn.Parameter(torch.ones(2)),
                "bottom_low": nn.Parameter(torch.ones(3)),
            })
            self.bifpn_blocks = nn.ModuleDict({key: self._make_bifpn_block(out_channels) for key in self.bifpn_weights})
            if self.mode == "last3_bifpn_stage1_branch_aware":
                self.bifpn_branch_weights = nn.ParameterDict({
                    "global": nn.Parameter(torch.ones(3)),
                    "local": nn.Parameter(torch.ones(3)),
                })
                self.bifpn_branch_blocks = nn.ModuleDict({
                    "global": self._make_bifpn_block(out_channels),
                    "local": self._make_bifpn_block(out_channels),
                })
            else:
                self.bifpn_branch_weights = nn.ParameterDict()
                self.bifpn_branch_blocks = nn.ModuleDict()
        else:
            self.bifpn_weights = nn.ParameterDict()
            self.bifpn_blocks = nn.ModuleDict()
            self.bifpn_branch_weights = nn.ParameterDict()
            self.bifpn_branch_blocks = nn.ModuleDict()

        if self.hierarchical_fpn:
            self.layer0_fpn_outputs = nn.ModuleDict({
                str(index): nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), LayerNorm2d(out_channels), nn.GELU()
                ) for index in (2, 1, 0)
            })

    @classmethod
    def from_mode(
        cls,
        mode: str,
        path_channels: dict[int, int],
        out_channels: int,
    ) -> CSLTinyViTFeatureFusion:
        normalized_mode = cls.normalize_mode(mode)
        module = cls(
            fusion_type=cls.fusion_type_for_mode(normalized_mode),
            stage_indices=cls.stage_indices_for_mode(normalized_mode),
            path_channels=path_channels,
            out_channels=out_channels,
            target_stage_index=cls.target_stage_for_mode(normalized_mode),
            mode=normalized_mode,
        )
        module.mode = normalized_mode
        return module

    @classmethod
    def normalize_mode(cls, mode: str) -> str:
        mode = str(mode).lower()
        if mode not in cls._VALID_MODES:
            raise ValueError(f"Unsupported CSL-TinyViT feature_fusion: {mode}")
        return mode

    @staticmethod
    def fusion_type_for_mode(mode: str) -> str:
        if mode == "final":
            return "final"
        if mode.startswith("weighted_"):
            return "weighted"
        if mode.startswith("normpres_"):
            return "norm_preserved"
        if mode == "dynamic_last3":
            return "dynamic"
        if mode == "dynamic_last3_scale_token":
            return "dynamic_scale_token"
        if mode in {"last3_fpn_stage2", "last4_fpn_layer0_target"}:
            return "fpn"
        if mode == "last3_pafpn_stage2":
            return "pafpn"
        if mode == "global_final_parts_stage2":
            return "split_global_local"
        if mode == "late_concat_stage2":
            return "late_concat"
        if mode == "last3_stage1_concat":
            return "concat_compress"
        if mode == "global_final_parts_stage1_concat":
            return "split_stage1_concat"
        if mode == "global_final_parts_fpn_layer0":
            return "split_fpn_layer0"
        if mode in {"last3_fpn_stage1_add", "last3_fpn_stage1_split"}:
            return "fpn_topdown"
        if mode in {"last3_panet_stage1_split", "last3_panet_stage1_shared", "last3_panet_stage1_scale_aware"}:
            return "panet"
        if mode in {"last3_bifpn_stage1_split", "last3_bifpn_stage1_branch_aware"}:
            return "bifpn"
        if mode == "global_final_parts_hierarchical_fpn":
            return "hierarchical_fpn"
        return "residual"

    @staticmethod
    def stage_indices_for_mode(mode: str) -> tuple[int, ...]:
        if mode in {"last2", "weighted_last2"}:
            return (2,)
        if mode in {"last3", "weighted_last3"}:
            return (1, 2)
        if mode == "last4_layer0_target":
            return (0, 1, 2)
        if mode in {"last3_stage2_target", "last3_stage1_concat", "global_final_parts_stage1_concat"}:
            return (1, 2)
        if mode == "last3_fpn_stage2":
            # FPN-style semantic order: final, stage 2, then stage 1 resized to stage 2.
            return (2, 1)
        if mode in {"last3_fpn_stage1_add", "last3_fpn_stage1_split", "last3_panet_stage1_split", "last3_panet_stage1_shared", "last3_panet_stage1_scale_aware", "last3_bifpn_stage1_split", "last3_bifpn_stage1_branch_aware"}:
            return (2, 1)
        if mode == "last3_pafpn_stage2":
            # PAFPN-style semantic order: final -> stage 2 -> stage 1, then bottom-up to stage 2.
            return (2, 1)
        if mode == "last4_fpn_layer0_target":
            # FPN-style semantic order: final, stage 2, stage 1, then stage 0 at layer0 resolution.
            return (2, 1, 0)
        if mode in {"global_final_parts_fpn_layer0", "global_final_parts_hierarchical_fpn"}:
            return (2, 1, 0)
        if mode == "global_final_parts_stage2":
            return (1, 2)
        if mode == "late_concat_stage2":
            return (2,)
        if mode == "normpres_last2":
            return (2,)
        if mode == "normpres_last3":
            return (1, 2)
        if mode in {"dynamic_last3", "dynamic_last3_scale_token"}:
            # Dynamic fusion follows the semantic order final, stage 2, stage 1.
            return (2, 1)
        return ()

    @staticmethod
    def target_stage_for_mode(mode: str) -> int | None:
        if mode in {
            "last4_layer0_target",
            "last3_stage2_target",
            "last3_stage1_concat",
            "global_final_parts_stage1_concat",
            "last3_fpn_stage1_add",
            "last3_fpn_stage1_split",
            "last3_panet_stage1_split",
            "last3_panet_stage1_shared",
            "last3_panet_stage1_scale_aware",
            "last3_bifpn_stage1_split",
            "last3_bifpn_stage1_branch_aware",
            "last3_fpn_stage2",
            "last3_pafpn_stage2",
            "last4_fpn_layer0_target",
            "global_final_parts_stage2",
            "global_final_parts_fpn_layer0",
            "global_final_parts_hierarchical_fpn",
            "late_concat_stage2",
        }:
            if mode in {"last4_layer0_target", "last4_fpn_layer0_target", "global_final_parts_fpn_layer0", "global_final_parts_hierarchical_fpn"}:
                return 0
            if mode in {
                "last3_stage1_concat", "global_final_parts_stage1_concat", "last3_fpn_stage1_add",
                "last3_fpn_stage1_split",
                "last3_panet_stage1_split", "last3_panet_stage1_shared", "last3_bifpn_stage1_split",
                "last3_panet_stage1_scale_aware", "last3_bifpn_stage1_branch_aware",
            }:
                return 1
            return 2
        return None

    def _residual_fuse_to_size(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        fused = self._resize_feature(final_feature, output_size)
        for stage_index in self.stage_indices:
            path_feature = self._project_path(stage_index, path_features[stage_index], output_size)
            fused = fused + self.residual_scales[str(stage_index)] * path_feature
        return fused

    def initialize_dynamic_gate(self) -> None:
        """Initialize dynamic fusion with a stable 80/10/10 path mixture."""
        if self.dynamic_gate is None:
            return
        output = self.dynamic_gate[-1]
        nn.init.trunc_normal_(output.weight, std=1e-3)
        with torch.no_grad():
            initial_weights = output.bias.new_tensor([0.8, *([0.1] * len(self.stage_indices))])
            output.bias.copy_(initial_weights.log())

    def normalized_weights(self) -> torch.Tensor:
        if self.fusion_weights is None:
            raise RuntimeError("Normalized fusion weights are only available for weighted feature fusion")
        weights = F.relu(self.fusion_weights)
        return weights / (weights.sum() + 1e-4)

    def _project_path(self, stage_index: int, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        feature = self.projections[str(stage_index)](feature)
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)
        return feature

    @staticmethod
    def _make_pafpn_block(out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _make_bifpn_block(out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            LayerNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels), nn.GELU(),
        )

    @staticmethod
    def _fast_normalized_fusion(features: list[torch.Tensor], weights: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        positive_weights = F.relu(weights)
        normalized_weights = positive_weights / (positive_weights.sum() + epsilon)
        return sum(weight * feature for weight, feature in zip(normalized_weights, features, strict=True))

    @staticmethod
    def _resize_feature(feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == output_size:
            return feature
        return F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)

    def _output_size(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> tuple[int, int]:
        if self.target_stage_index is None:
            return final_feature.shape[-2:]
        return path_features[self.target_stage_index].shape[-2:]

    @staticmethod
    def _half_size(size: tuple[int, int]) -> tuple[int, int]:
        """Return the next coarser pyramid level for ReID feature maps."""
        return max(size[0] // 2, 1), max(size[1] // 2, 1)

    def _stage2_pyramid_inputs(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Build the semantic 12x4 level below the native Stage-1 24x8 map.

        CSL-TinyViT preserves spatial resolution across its last stages. The
        detector necks adapted here require an actual coarser semantic level,
        so both final and Stage-2 maps are resized to one half of Stage 1.
        """
        low_size = self._half_size(path_features[1].shape[-2:])
        final_low = self._resize_feature(final_feature, low_size)
        stage2_low = self._project_path(2, path_features[2], low_size)
        return final_low, stage2_low, low_size

    @staticmethod
    def _pooled_descriptor(feature: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)

    def _ordered_features(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> list[torch.Tensor]:
        output_size = self._output_size(final_feature, path_features)
        return [
            self._resize_feature(final_feature, output_size),
            *[
                self._project_path(stage_index, path_features[stage_index], output_size)
                for stage_index in self.stage_indices
            ],
        ]

    def dynamic_weights(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Return per-image softmax weights in fusion path order."""
        if self.dynamic_gate is None:
            raise RuntimeError("Dynamic weights are only available for dynamic feature fusion")
        features = self._ordered_features(final_feature, path_features)
        return self._dynamic_weights_from_features(features)

    def _dynamic_weights_from_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Compute dynamic weights from already projected, ordered features."""
        descriptors = [self._pooled_descriptor(feature) for feature in features]
        final_descriptor = descriptors[0]
        gate_inputs = [final_descriptor]
        if self.scale_token_projection is not None:
            scale_descriptors = torch.stack(
                [self.scale_token_projection(descriptor) for descriptor in descriptors],
                dim=1,
            )
            scale_queries = self.scale_tokens.unsqueeze(0).expand(scale_descriptors.shape[0], -1, -1)
            scale_key_values = scale_descriptors + scale_queries
            attention = (scale_queries @ scale_key_values.transpose(1, 2) / math.sqrt(scale_queries.shape[-1])).softmax(
                dim=-1
            )
            scale_context = self.scale_token_norm(scale_queries + attention @ scale_key_values)
            gate_inputs.append(scale_context.flatten(1))
        return self.dynamic_gate(torch.cat(gate_inputs, dim=1)).softmax(dim=1)

    def forward(self, final_feature: torch.Tensor, path_features: dict[int, torch.Tensor]) -> torch.Tensor:
        if not self.stage_indices:
            return final_feature

        if self.split_global_local:
            global_feature = self._residual_fuse_to_size(
                final_feature,
                path_features,
                final_feature.shape[-2:],
            )
            local_feature = self._project_path(
                self.target_stage_index,
                path_features[self.target_stage_index],
                path_features[self.target_stage_index].shape[-2:],
            )
            return global_feature, local_feature

        if self.split_stage1_concat:
            global_size = self._half_size(path_features[1].shape[-2:])
            global_feature = self._residual_fuse_to_size(final_feature, path_features, global_size)
            output_size = path_features[self.target_stage_index].shape[-2:]
            local_features = [self._resize_feature(final_feature, output_size), *[
                self._project_path(stage_index, path_features[stage_index], output_size)
                for stage_index in self.stage_indices
            ]]
            return global_feature, self.concat_projection(torch.cat(local_features, dim=1))

        if self.split_fpn_layer0:
            global_size = self._half_size(path_features[1].shape[-2:])
            global_feature = self._resize_feature(final_feature, global_size)
            for stage_index in (1, 2):
                projected = self._project_path(stage_index, path_features[stage_index], global_size)
                global_feature = global_feature + self.residual_scales[str(stage_index)] * projected
            final_low, stage2_low, _ = self._stage2_pyramid_inputs(final_feature, path_features)
            pyramid = self.layer0_fpn_outputs["2"](final_low + stage2_low)
            for stage_index in (1, 0):
                stage = self._project_path(stage_index, path_features[stage_index], path_features[stage_index].shape[-2:])
                pyramid = self.layer0_fpn_outputs[str(stage_index)](
                    stage + self._resize_feature(pyramid, stage.shape[-2:])
                )
            return global_feature, pyramid

        if self.late_concat:
            local_feature = self._project_path(
                self.target_stage_index,
                path_features[self.target_stage_index],
                path_features[self.target_stage_index].shape[-2:],
            )
            return final_feature, local_feature

        if self.dynamic:
            features = self._ordered_features(final_feature, path_features)
            weights = self._dynamic_weights_from_features(features)
            return sum(
                weight[:, None, None, None] * feature
                for weight, feature in zip(weights.unbind(dim=1), features, strict=True)
            )

        if self.norm_preserved:
            features = self._ordered_features(final_feature, path_features)
            mean_feature = torch.stack(features, dim=0).mean(dim=0)
            max_norm = (
                torch.stack(
                    [feature.norm(p=2, dim=1, keepdim=True) for feature in features],
                    dim=0,
                )
                .max(dim=0)
                .values
            )
            return F.normalize(mean_feature, p=2, dim=1) * max_norm

        if self.fpn:
            features = self._ordered_features(final_feature, path_features)
            return torch.stack(features, dim=0).mean(dim=0)

        if self.pafpn:
            stage2 = self._project_path(
                2,
                path_features[2],
                path_features[2].shape[-2:],
            )
            stage1 = self._project_path(
                1,
                path_features[1],
                path_features[1].shape[-2:],
            )
            top_down_stage2 = self.pafpn_top_down["2"](
                torch.cat([stage2, self._resize_feature(final_feature, stage2.shape[-2:])], dim=1)
            )
            top_down_stage1 = self.pafpn_top_down["1"](
                torch.cat([stage1, self._resize_feature(top_down_stage2, stage1.shape[-2:])], dim=1)
            )
            return self.pafpn_bottom_up["2"](
                torch.cat([top_down_stage2, self._resize_feature(top_down_stage1, top_down_stage2.shape[-2:])], dim=1)
            )

        if self.concat_compress:
            return self.concat_projection(torch.cat(self._ordered_features(final_feature, path_features), dim=1))

        if self.fpn_topdown:
            final_low, stage2, _ = self._stage2_pyramid_inputs(final_feature, path_features)
            pyramid2 = stage2 + final_low
            stage1 = self._project_path(1, path_features[1], path_features[1].shape[-2:])
            pyramid1 = stage1 + self._resize_feature(pyramid2, stage1.shape[-2:])
            if self.mode == "last3_fpn_stage1_split":
                return self.fpn_global_output(pyramid2), self.fpn_output(pyramid1)
            return self.fpn_output(pyramid1)

        if self.panet:
            final_low, stage2, _ = self._stage2_pyramid_inputs(final_feature, path_features)
            pyramid2 = stage2 + final_low
            stage1 = self._project_path(1, path_features[1], path_features[1].shape[-2:])
            pyramid1 = stage1 + self._resize_feature(pyramid2, stage1.shape[-2:])
            bottom_up = self.panet_downsample(pyramid1) if pyramid1.shape[-2:] != pyramid2.shape[-2:] else pyramid1
            bottom_up = self._resize_feature(bottom_up, pyramid2.shape[-2:])
            semantic_global = self.panet_output(pyramid2 + bottom_up)
            if self.mode == "last3_panet_stage1_shared":
                return semantic_global
            if self.mode == "last3_panet_stage1_scale_aware":
                semantic_local = self._resize_feature(semantic_global, pyramid1.shape[-2:])
                gate = torch.sigmoid(self.panet_scale_gate(torch.cat([pyramid1, semantic_local], dim=1)))
                return semantic_global, gate * pyramid1 + (1.0 - gate) * semantic_local
            return semantic_global, pyramid1

        if self.bifpn:
            final_low, stage2, _ = self._stage2_pyramid_inputs(final_feature, path_features)
            stage1 = self._project_path(1, path_features[1], path_features[1].shape[-2:])
            if self.mode == "last3_bifpn_stage1_branch_aware":
                stage1_low = self._resize_feature(stage1, stage2.shape[-2:])
                global_feature = self.bifpn_branch_blocks["global"](self._fast_normalized_fusion(
                    [final_low, stage2, stage1_low], self.bifpn_branch_weights["global"]
                ))
                local_feature = self.bifpn_branch_blocks["local"](self._fast_normalized_fusion(
                    [
                        self._resize_feature(final_low, stage1.shape[-2:]),
                        self._resize_feature(stage2, stage1.shape[-2:]),
                        stage1,
                    ],
                    self.bifpn_branch_weights["local"],
                ))
                return global_feature, local_feature
            top_low = self.bifpn_blocks["top_low"](self._fast_normalized_fusion(
                [stage2, final_low], self.bifpn_weights["top_low"]
            ))
            top_high = self.bifpn_blocks["top_high"](self._fast_normalized_fusion(
                [stage1, self._resize_feature(top_low, stage1.shape[-2:])], self.bifpn_weights["top_high"]
            ))
            bottom_up = (
                F.max_pool2d(top_high, kernel_size=3, stride=2, padding=1)
                if top_high.shape[-2:] != stage2.shape[-2:]
                else top_high
            )
            bottom_up = self._resize_feature(bottom_up, stage2.shape[-2:])
            bottom_low = self.bifpn_blocks["bottom_low"](self._fast_normalized_fusion(
                [stage2, top_low, bottom_up],
                self.bifpn_weights["bottom_low"],
            ))
            return bottom_low, top_high

        if self.hierarchical_fpn:
            final_low, stage2_low, _ = self._stage2_pyramid_inputs(final_feature, path_features)
            coarse = self.layer0_fpn_outputs["2"](final_low + stage2_low)
            stage1 = self._project_path(1, path_features[1], path_features[1].shape[-2:])
            coarse = self.layer0_fpn_outputs["1"](stage1 + self._resize_feature(coarse, stage1.shape[-2:]))
            stage0 = self._project_path(0, path_features[0], path_features[0].shape[-2:])
            fine = self.layer0_fpn_outputs["0"](stage0 + self._resize_feature(coarse, stage0.shape[-2:]))
            return final_low + self.residual_scales["1"] * self._project_path(1, path_features[1], final_low.shape[-2:]) + self.residual_scales["2"] * stage2_low, coarse, fine

        output_size = self._output_size(final_feature, path_features)
        fused = self._resize_feature(final_feature, output_size)
        weighted_features = [fused]
        for stage_index in self.stage_indices:
            path_feature = self._project_path(stage_index, path_features[stage_index], output_size)
            if self.weighted:
                weighted_features.append(path_feature)
            else:
                fused = fused + self.residual_scales[str(stage_index)] * path_feature

        if not self.weighted:
            return fused

        fusion_weights = self.normalized_weights()
        fused = fusion_weights[0] * weighted_features[0]
        for weight, feature in zip(fusion_weights[1:], weighted_features[1:], strict=True):
            fused = fused + weight * feature
        return fused


class PostFusionLocalMixer(nn.Module):
    """Zero-gated local spatial mixer after CSL-TinyViT feature fusion."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: tuple[int, int] = (5, 3),
        gamma_init: float = 0.0,
    ):
        super().__init__()
        reduction = int(reduction)
        if reduction < 1:
            raise ValueError(f"post_fusion_mixer_reduction must be positive, got {reduction}")
        kernel_h, kernel_w = _to_2tuple(kernel_size)
        if kernel_h <= 0 or kernel_w <= 0 or kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError(f"post_fusion_mixer_kernel must contain positive odd values, got {(kernel_h, kernel_w)}")
        hidden_channels = max(int(channels) // reduction, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(kernel_h, kernel_w),
            padding=(kernel_h // 2, kernel_w // 2),
            groups=hidden_channels,
            bias=False,
        )
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.expand(self.dwconv(self.act(self.reduce(x))))
        return x + self.gamma * mixed
