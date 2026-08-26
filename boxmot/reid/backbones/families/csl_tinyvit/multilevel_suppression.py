# BoxMOT AGPL-3.0 license

"""Training-only classifier-guided suppression for hierarchical ReID maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from boxmot.reid.backbones.heads.bnneck import BNNeck3

__all__ = [
    "MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION",
    "MultilevelClassifierSuppression",
    "MultilevelSuppressionOutput",
    "stripe_top_quantile_mask",
]

MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION = 2


class MultilevelSuppressionOutput(NamedTuple):
    """Auxiliary logits and detached monitoring values."""

    coarse_logits: tuple[torch.Tensor, ...]
    fine_logits: tuple[torch.Tensor, ...]
    coarse_active: torch.Tensor
    fine_active: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def stripe_top_quantile_mask(
    saliency: torch.Tensor,
    *,
    num_stripes: int,
    ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a hard keep-mask and per-sample receiving-stripe activity.

    A spatially constant or non-finite saliency stripe carries no ranking
    information, so it is deliberately left intact instead of erasing
    arbitrary tied positions. The requested ratio is rounded up to a whole
    location, and diagnostics report the resulting discrete erase fraction.
    """
    if saliency.ndim != 4 or saliency.shape[1] != 1:
        raise ValueError(
            "Classifier saliency must have shape (B, 1, H, W), got "
            f"{tuple(saliency.shape)}"
        )
    if num_stripes < 1 or saliency.shape[2] < num_stripes:
        raise ValueError(
            f"Cannot split height {saliency.shape[2]} into {num_stripes} stripes"
        )
    if not 0 <= ratio < 1:
        raise ValueError(f"Suppression ratio must be in [0, 1), got {ratio}")

    batch_size, _, height, _ = saliency.shape
    mask = torch.ones_like(saliency, memory_format=torch.contiguous_format)
    active_by_stripe = torch.zeros(
        batch_size,
        num_stripes,
        device=saliency.device,
        dtype=torch.bool,
    )
    if ratio == 0:
        return mask.detach(), active_by_stripe.detach()

    for stripe_index in range(num_stripes):
        start = stripe_index * height // num_stripes
        end = (stripe_index + 1) * height // num_stripes
        stripe = saliency[:, :, start:end, :]
        flat = stripe.reshape(batch_size, -1)
        finite = torch.isfinite(flat).all(dim=1, keepdim=True)
        dynamic = (flat.amax(dim=1, keepdim=True) - flat.amin(dim=1, keepdim=True)) > eps
        active = finite & dynamic
        active_by_stripe[:, stripe_index] = active.squeeze(1)

        erase_count = min(
            flat.shape[1],
            max(1, math.ceil(float(ratio) * flat.shape[1])),
        )
        indices = flat.topk(erase_count, dim=1, largest=True).indices
        erased = torch.zeros_like(flat, dtype=torch.bool)
        erased.scatter_(
            1,
            indices,
            active.expand(-1, erase_count),
        )
        mask[:, :, start:end, :] = (~erased).reshape_as(stripe).to(mask.dtype)

    return mask.detach(), active_by_stripe.detach()


class MultilevelClassifierSuppression(nn.Module):
    """Teach finer maps from target evidence already used by coarser heads.

    The target-class Grad-CAM masks and all clean-head parameters are detached.
    Consequently, the auxiliary losses update only the receiving coarse/fine
    feature maps. Frozen running-stat batch normalization keeps both saliency
    and auxiliary logits sample-independent without mutating the clean BN state
    or adding training-only parameters to deployment checkpoints.
    """

    def __init__(self, ratio: float = 0.15) -> None:
        super().__init__()
        if not 0 < ratio < 1:
            raise ValueError(
                "multilevel_suppression_ratio must be in (0, 1), got "
                f"{ratio}"
            )
        self.ratio = float(ratio)
        self._progress = 0.0

    @property
    def progress(self) -> float:
        """Return the current schedule multiplier."""
        return self._progress

    @property
    def effective_ratio(self) -> float:
        """Return the scheduled spatial erasure ratio."""
        return self.ratio * self._progress

    @property
    def active(self) -> bool:
        """Return whether the training-only auxiliary path should run."""
        return self.training and self.effective_ratio > 0

    def set_progress(self, progress: float) -> None:
        """Set a validated schedule multiplier in ``[0, 1]``."""
        progress = float(progress)
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError(
                "Multilevel suppression progress must be finite and in [0, 1], "
                f"got {progress}"
            )
        self._progress = progress

    @staticmethod
    def _target_gradcam(
        source: torch.Tensor,
        logits: Sequence[torch.Tensor],
        pids: torch.Tensor,
        *,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """Compute detached per-sample Grad-CAM with one summed VJP.

        The supplied logits must be sample-independent. The controller builds
        them with frozen running-stat batch normalization below, so summing the
        selected scores preserves each sample's diagonal gradient while
        avoiding one backward traversal per image.
        """
        if not source.requires_grad:
            raise RuntimeError(
                "Classifier-guided suppression requires differentiable feature "
                "maps; schedule it after backbone freezing has ended"
            )
        if not logits:
            raise ValueError("At least one clean classifier logit is required")
        targets = pids.to(device=source.device, dtype=torch.long).reshape(-1)
        if targets.shape[0] != source.shape[0]:
            raise ValueError(
                "PID batch does not match feature batch: "
                f"{targets.shape[0]} != {source.shape[0]}"
            )
        num_classes = logits[0].shape[1]
        if bool(((targets < 0) | (targets >= num_classes)).any()):
            raise ValueError(
                f"PIDs must lie in [0, {num_classes - 1}] for saliency"
            )
        target_score = source.new_zeros(())
        for branch_logits in logits:
            if branch_logits.shape != (source.shape[0], num_classes):
                raise ValueError(
                    "All saliency logits must have shape "
                    f"({source.shape[0]}, {num_classes}), got "
                    f"{tuple(branch_logits.shape)}"
                )
            target_score = target_score + branch_logits.gather(
                1,
                targets[:, None],
            ).sum()
        gradient = torch.autograd.grad(
            target_score,
            source,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )[0]
        channel_weight = gradient.detach().float().mean(
            dim=(2, 3),
            keepdim=True,
        )
        saliency = (
            channel_weight * source.detach().float()
        ).sum(dim=1, keepdim=True)
        return saliency.relu_()

    @staticmethod
    def _frozen_pool(pool: nn.Module, feature: torch.Tensor) -> torch.Tensor:
        """Run a shared pool while detaching any learnable pool parameters."""
        state = {
            name: parameter.detach()
            for name, parameter in pool.named_parameters()
        }
        state.update(
            {
                name: buffer.detach().clone()
                for name, buffer in pool.named_buffers()
            }
        )
        if not state:
            return pool(feature)
        return functional_call(pool, state, (feature,), strict=True)

    @staticmethod
    def _frozen_classify(
        pooled: torch.Tensor,
        neck: BNNeck3,
    ) -> torch.Tensor:
        """Classify a stripe without updating the clean head.

        Suppression scoring uses detached running statistics so every sample's
        score depends only on its own feature map and masked-out samples cannot
        influence active auxiliary logits through batch statistics.
        """
        reduction = neck.reduction
        projected = F.conv2d(
            pooled,
            reduction.weight.detach(),
            reduction.bias.detach() if reduction.bias is not None else None,
            stride=reduction.stride,
            padding=reduction.padding,
            dilation=reduction.dilation,
            groups=reduction.groups,
        ).flatten(1)
        normalized = F.batch_norm(
            projected,
            neck.bn.running_mean.detach().clone(),
            neck.bn.running_var.detach().clone(),
            neck.bn.weight.detach() if neck.bn.weight is not None else None,
            neck.bn.bias.detach() if neck.bn.bias is not None else None,
            training=False,
            momentum=0.0,
            eps=neck.bn.eps,
        )
        return F.linear(
            normalized,
            neck.classifier.weight.detach(),
            neck.classifier.bias.detach()
            if neck.classifier.bias is not None
            else None,
        )

    def _saliency_logits(
        self,
        feature: torch.Tensor,
        *,
        pool: nn.Module,
        necks: Sequence[BNNeck3],
    ) -> tuple[torch.Tensor, ...]:
        """Return sample-independent target scorers for a clean feature map."""
        pooled = self._frozen_pool(pool, feature)
        if pooled.shape[2] != len(necks):
            raise RuntimeError(
                "Saliency stripe pool produced "
                f"{pooled.shape[2]} stripes for {len(necks)} classifiers"
            )
        return tuple(
            self._frozen_classify(
                pooled[:, :, index : index + 1, :],
                neck,
            )
            for index, neck in enumerate(necks)
        )

    @classmethod
    def _stitched_target_gradcam(
        cls,
        source: torch.Tensor,
        logits: Sequence[torch.Tensor],
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Stitch independent classifier CAMs into their source stripes."""
        if len(logits) < 1:
            raise ValueError("At least one stripe classifier logit is required")
        height = source.shape[2]
        stitched = torch.zeros(
            source.shape[0],
            1,
            height,
            source.shape[3],
            device=source.device,
            dtype=torch.float32,
        )
        for stripe_index, branch_logits in enumerate(logits):
            branch_saliency = cls._target_gradcam(
                source,
                (branch_logits,),
                pids,
                retain_graph=stripe_index + 1 < len(logits),
            )
            start = stripe_index * height // len(logits)
            end = (stripe_index + 1) * height // len(logits)
            stitched[:, :, start:end, :] = branch_saliency[
                :, :, start:end, :
            ]
        return stitched

    @staticmethod
    def _resize_striped_saliency(
        saliency: torch.Tensor,
        *,
        output_size: tuple[int, int],
        num_stripes: int,
    ) -> torch.Tensor:
        """Resize parent stripes independently without crossing boundaries."""
        if saliency.ndim != 4 or saliency.shape[1] != 1:
            raise ValueError(
                "Classifier saliency must have shape (B, 1, H, W), got "
                f"{tuple(saliency.shape)}"
            )
        output_height, output_width = output_size
        if (
            num_stripes < 1
            or saliency.shape[2] < num_stripes
            or output_height < num_stripes
            or output_width < 1
        ):
            raise ValueError(
                "Saliency resize requires positive source/output stripe "
                "extents, got "
                f"source_height={saliency.shape[2]}, "
                f"output_size={output_size}, num_stripes={num_stripes}"
            )

        source_height = saliency.shape[2]
        resized = []
        for stripe_index in range(num_stripes):
            source_start = stripe_index * source_height // num_stripes
            source_end = (stripe_index + 1) * source_height // num_stripes
            output_start = stripe_index * output_height // num_stripes
            output_end = (stripe_index + 1) * output_height // num_stripes
            resized.append(
                F.interpolate(
                    saliency[:, :, source_start:source_end, :],
                    size=(output_end - output_start, output_width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(resized, dim=2)

    def _auxiliary_logits(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        *,
        pool: nn.Module,
        necks: Sequence[BNNeck3],
    ) -> tuple[torch.Tensor, ...]:
        pooled = self._frozen_pool(pool, feature * mask.to(feature.dtype))
        if pooled.shape[2] != len(necks):
            raise RuntimeError(
                "Suppressed stripe pool produced "
                f"{pooled.shape[2]} stripes for {len(necks)} classifiers"
            )
        return tuple(
            self._frozen_classify(
                pooled[:, :, index : index + 1, :],
                neck,
            )
            for index, neck in enumerate(necks)
        )

    def forward(
        self,
        *,
        global_feature: torch.Tensor,
        coarse_feature: torch.Tensor,
        fine_feature: torch.Tensor,
        pids: torch.Tensor,
        global_pool: nn.Module,
        coarse_pool: nn.Module,
        fine_pool: nn.Module,
        global_neck: BNNeck3,
        coarse_necks: Sequence[BNNeck3],
        fine_necks: Sequence[BNNeck3],
    ) -> MultilevelSuppressionOutput:
        """Build private suppressed copies and reclassify their stripes."""
        if not self.active:
            raise RuntimeError(
                "Inactive multilevel suppression must be bypassed by its caller"
            )
        if len(coarse_necks) != 2:
            raise ValueError("Multilevel suppression requires two coarse branches")
        if len(fine_necks) != 4:
            raise ValueError("Multilevel suppression requires four fine branches")

        global_logits = self._saliency_logits(
            global_feature,
            pool=global_pool,
            necks=(global_neck,),
        )[0]
        global_saliency = self._target_gradcam(
            global_feature,
            (global_logits,),
            pids,
        )
        global_saliency = F.interpolate(
            global_saliency,
            size=coarse_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_mask, coarse_active = stripe_top_quantile_mask(
            global_saliency,
            num_stripes=2,
            ratio=self.effective_ratio,
        )
        suppressed_coarse_logits = self._auxiliary_logits(
            coarse_feature,
            coarse_mask,
            pool=coarse_pool,
            necks=coarse_necks,
        )

        coarse_logits = self._saliency_logits(
            coarse_feature,
            pool=coarse_pool,
            necks=coarse_necks,
        )
        coarse_saliency = self._stitched_target_gradcam(
            coarse_feature,
            coarse_logits,
            pids,
        )
        coarse_saliency = self._resize_striped_saliency(
            coarse_saliency,
            output_size=fine_feature.shape[-2:],
            num_stripes=2,
        )
        fine_mask, fine_active = stripe_top_quantile_mask(
            coarse_saliency,
            num_stripes=4,
            ratio=self.effective_ratio,
        )
        suppressed_fine_logits = self._auxiliary_logits(
            fine_feature,
            fine_mask,
            pool=fine_pool,
            necks=fine_necks,
        )

        diagnostics = {
            "effective_ratio": torch.tensor(
                self.effective_ratio,
                device=fine_feature.device,
                dtype=torch.float32,
            ).detach(),
            "coarse_erased_fraction": (1.0 - coarse_mask.float())
            .mean()
            .detach(),
            "fine_erased_fraction": (1.0 - fine_mask.float())
            .mean()
            .detach(),
            "global_cam_active_fraction": coarse_active.float().mean().detach(),
            "coarse_cam_active_fraction": fine_active.float().mean().detach(),
        }
        return MultilevelSuppressionOutput(
            suppressed_coarse_logits,
            suppressed_fine_logits,
            coarse_active,
            fine_active,
            diagnostics,
        )
