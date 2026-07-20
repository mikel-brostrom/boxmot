# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.pooling import (
    ActivatedGeM,
    DSELitePool,
    GeM,
    LearnedPartTokenPool,
    PatternAdapter,
    SemanticVisibilityPartPool,
    SpatialTopDrop,
    SpatialTopSuppression,
    StripeVisibilityGate,
)
from boxmot.reid.backbones.heads.bnneck import BNNeck, BNNeck3

__all__ = ["GPCLiteMultiBranchHead", "LMBNStyleMultiBranchHead", "MultiBranchHead"]


class MultiBranchHead(nn.Module):
    """Multi-granularity head with fixed stripes or learned part tokens.

    Produces:
      - Training: (cls_scores_list, features_tensor)
      - Inference: (B, feat_dim × num_branches) concatenated features
    """

    SPECIALIST_MODES = {
        "standard",
        "stage2_channel2",
        "stage2_pg",
        "stage2_gpc_lite",
        "stage2_gpc_lite_gate",
        "stage2_pg_gate",
        "suppressed_global",
    }
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
        hierarchical_scales: bool = False,
        compact_deployment_head: bool = False,
        specialist_mode: str = "standard",
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
        self.hierarchical_scales = bool(hierarchical_scales)
        self.compact_deployment_head = bool(compact_deployment_head)
        self.head_parts = self._normalize_head_parts(head_parts)
        self.specialist_mode = str(specialist_mode).lower()
        if self.specialist_mode not in self.SPECIALIST_MODES:
            raise ValueError(f"Unsupported CSL-TinyViT specialist head: {specialist_mode}")
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
        self.has_specialist_gate = self.specialist_mode in {
            "stage2_pg_gate",
            "stage2_gpc_lite_gate",
        }
        self.has_suppressed_global = self.specialist_mode == "suppressed_global"
        self.has_specialists = self.specialist_mode != "standard"
        if self.has_specialists:
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("G/P/C specialists require hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("G/P/C specialists require fixed stripe pooling")
        if self.evidence_num_roles < 1:
            raise ValueError(f"evidence_num_roles must be positive, got {evidence_num_roles}")
        if self.scale_balanced_branches and self.part_pooling not in {"stripes", "overlap_stripes"}:
            raise ValueError("scale-balanced branches require fixed or overlapping stripe pooling")
        if self.hierarchical_scales and self.head_parts != (1, 2, 4):
            raise ValueError("hierarchical FPN requires head_parts=(1, 2, 4)")
        if self.compact_deployment_head:
            if not self.hierarchical_scales or self.head_parts != (1, 2, 4):
                raise ValueError("compact deployment head requires hierarchical head_parts=(1, 2, 4)")
            if self.part_pooling != "stripes":
                raise ValueError("compact deployment head requires fixed stripe pooling")
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
        if self.specialist_gate is not None:
            nn.init.zeros_(self.specialist_gate.weight)
            initial_logit = math.log(self.SPECIALIST_GATE_INIT / (1.0 - self.SPECIALIST_GATE_INIT))
            nn.init.constant_(self.specialist_gate.bias, initial_logit)

    def reset_reid_initialization(self) -> None:
        """Restore ReID-specific head initialization after global model init."""
        for module in self.modules():
            if isinstance(module, (BNNeck, BNNeck3)):
                module.reset_reid_initialization()
        if self.semantic_part_pool is not None:
            self.semantic_part_pool.reset_metadata_initialization()
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
        if not self.scale_balanced_branches:
            return 1.0
        return 1.0 / math.sqrt(granularity)

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
    ) -> list[tuple[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor | float]]:
        """Return auxiliary CE/retrieval branches without entering the metric descriptor."""
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
        if self.has_suppressed_global:
            pooled = self.suppressed_global_pool(self.suppressed_global(global_feature))
            output = self.bn_suppressed_global(pooled)
            outputs.append(("suppressed_global", output, self.SUPPRESSED_GLOBAL_WEIGHT))
        return outputs

    def forward(self, x):
        # x: (B, C, H, W) or (global_map, local_map) for split-map ablations.
        fine_source = None
        if isinstance(x, tuple) and len(x) == 3:
            global_source, local_source, fine_source = x
        elif isinstance(x, tuple):
            global_source, local_source = x
        else:
            global_source = local_source = x
        global_feature = self.global_adapter(global_source)
        local_feature = self.local_adapter(local_source)
        if fine_source is not None:
            fine_source = self.fine_adapter(fine_source)
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
            and all(value is None for value in visibility_by_key.values())
        ):
            normalized_branches = []
            for key, granularity, stripe_index in self.branch_specs:
                pooled = pooled_by_granularity[granularity]
                if granularity > 1:
                    pooled = pooled[:, :, stripe_index : stripe_index + 1, :]
                bn_feature = getattr(self, self._bn_attr(key))(pooled)[0]
                normalized_branches.append(
                    F.normalize(bn_feature, p=2, dim=1) * self._descriptor_scale(granularity)
                )
            for _, branch_output, weight in self._specialist_outputs(global_feature, local_feature):
                normalized_branches.append(F.normalize(branch_output[0], p=2, dim=1) * weight)
            return F.normalize(torch.cat(normalized_branches, dim=1), p=2, dim=1)

        branch_outputs = {}
        bn_features_list = []
        raw_features_list = []
        normalized_bn_features_list = []
        normalized_raw_features_list = []
        coarse_normalized_raw_features_list = []
        cls_scores = []
        raw_features = {}
        base_normalized_bn_features = {}
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
            branch_output = getattr(self, self._bn_attr(key))(pooled)
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
        for key, branch_output, weight in self._specialist_outputs(global_feature, local_feature):
            raw_features[key] = branch_output[2]
            cls_scores.append(branch_output[1])
            specialist_normalized_bn_features.append(
                F.normalize(branch_output[0], p=2, dim=1) * weight
            )
        if visibility_values is not None:
            raw_features["_visibility"] = visibility_values
        if semantic_rarity is not None:
            raw_features["_rarity"] = semantic_rarity
        if semantic_role_logits is not None:
            raw_features["_role_logits"] = semantic_role_logits
        if semantic_nullness is not None:
            raw_features["_nullness"] = semantic_nullness

        bn_features = torch.cat(bn_features_list, dim=1)
        raw_features["raw_mean"] = (
            raw_features_list[0]
            if self.hierarchical_scales
            else torch.stack(raw_features_list, dim=0).mean(dim=0)
        )
        raw_features["raw_concat"] = torch.cat(normalized_raw_features_list, dim=1)
        raw_features["coarse_concat"] = torch.cat(coarse_normalized_raw_features_list, dim=1)
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(normalized_bn_features_list + specialist_normalized_bn_features, dim=1),
            p=2,
            dim=1,
        )
        self._add_dse_descriptors(raw_features, local_feature)

        if self.compact_deployment_head:
            raw_features["_compact_logits"] = compact_logits
            raw_features["_compact_student"] = compact_raw
            raw_features["_compact_student_bn"] = compact_bn
            raw_features["_compact_teacher"] = raw_features["norm_concat_bn"].detach()
            raw_features["_compact_decoded"] = self.compact_distill_decoder(compact_bn)

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
        elif self.branch_metric:
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
