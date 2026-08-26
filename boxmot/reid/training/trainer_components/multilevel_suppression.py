"""Training-only multilevel classifier-guided feature suppression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.training.losses import CrossEntropyLabelSmooth


class _MultilevelSuppressionMixin:
    """Schedule and score the CSL-TinyViT suppression-only ID branches."""

    _MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS = (
        "effective_ratio",
        "coarse_erased_fraction",
        "fine_erased_fraction",
        "global_cam_active_fraction",
        "coarse_cam_active_fraction",
    )

    def _multilevel_suppression_progress(self, epoch: int) -> float:
        """Return the suppression/loss strength for a one-indexed epoch."""
        if not self.multilevel_suppression:
            return 0.0
        if epoch <= self.multilevel_suppression_start_epoch:
            return 0.0
        if epoch < self.multilevel_suppression_ramp_end_epoch:
            return (epoch - self.multilevel_suppression_start_epoch) / (
                self.multilevel_suppression_ramp_end_epoch - self.multilevel_suppression_start_epoch
            )
        if epoch <= self.multilevel_suppression_decay_start_epoch:
            return 1.0
        if epoch < self.multilevel_suppression_decay_end_epoch:
            return 1.0 - (epoch - self.multilevel_suppression_decay_start_epoch) / (
                self.multilevel_suppression_decay_end_epoch - self.multilevel_suppression_decay_start_epoch
            )
        return 0.0

    def _set_multilevel_suppression_progress(self, model, epoch: int) -> float:
        """Push the current scheduled strength into the enabled model."""
        progress = self._multilevel_suppression_progress(epoch)
        self._current_multilevel_suppression_progress = progress
        if not self.multilevel_suppression:
            return progress

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        setter = getattr(unwrapped, "set_multilevel_suppression_progress", None)
        if not callable(setter):
            raise RuntimeError("Enabled multilevel suppression model is missing its progress hook")
        setter(progress)
        return progress

    def _effective_multilevel_suppression_loss_weight(self) -> float:
        """Return the scheduled auxiliary CE coefficient for this epoch."""
        progress = getattr(self, "_current_multilevel_suppression_progress", 0.0)
        return self.multilevel_suppression_loss_weight * float(progress)

    def _multilevel_suppression_loss(
        self,
        criterion_id,
        features,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 0.5 * (mean coarse CE + mean fine CE), and nothing else."""
        zero = pids.new_zeros((), dtype=torch.float32)
        if self._effective_multilevel_suppression_loss_weight() <= 0:
            return zero
        if not isinstance(features, Mapping):
            raise RuntimeError("Enabled multilevel suppression requires a feature mapping")
        logits_by_scale = features.get("_multilevel_suppression_logits")
        if not isinstance(logits_by_scale, Mapping):
            raise RuntimeError("Enabled multilevel suppression model did not return auxiliary logits")
        active_by_scale = features.get("_multilevel_suppression_active")
        if not isinstance(active_by_scale, Mapping):
            raise RuntimeError("Enabled multilevel suppression model did not return auxiliary activity masks")

        coarse = logits_by_scale.get("coarse")
        fine = logits_by_scale.get("fine")
        self._validate_multilevel_suppression_logits(coarse, "coarse", 2)
        self._validate_multilevel_suppression_logits(fine, "fine", 4)
        coarse_active = active_by_scale.get("coarse")
        fine_active = active_by_scale.get("fine")
        self._validate_multilevel_suppression_active(
            coarse_active,
            "coarse",
            batch_size=pids.shape[0],
            expected=2,
        )
        self._validate_multilevel_suppression_active(
            fine_active,
            "fine",
            batch_size=pids.shape[0],
            expected=4,
        )
        coarse_loss = self._masked_multilevel_suppression_scale_loss(
            criterion_id,
            coarse,
            coarse_active,
            pids,
        )
        fine_loss = self._masked_multilevel_suppression_scale_loss(
            criterion_id,
            fine,
            fine_active,
            pids,
        )
        return 0.5 * (coarse_loss + fine_loss)

    def _masked_multilevel_suppression_scale_loss(
        self,
        criterion_id: nn.Module,
        logits: Sequence[torch.Tensor],
        active: torch.Tensor,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Average branch CE after zeroing samples whose CAM erased nothing."""
        branch_losses = tuple(
            self._masked_multilevel_suppression_ce(
                criterion_id,
                branch_logits,
                pids,
                active[:, branch_index],
            )
            for branch_index, branch_logits in enumerate(logits)
        )
        return torch.stack(branch_losses).mean()

    @staticmethod
    def _masked_multilevel_suppression_ce(
        criterion_id: nn.Module,
        logits: torch.Tensor,
        pids: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact CE over active samples and differentiable zero if none."""
        if isinstance(criterion_id, CrossEntropyLabelSmooth):
            log_probs = F.log_softmax(logits, dim=1)
            negative_log_likelihood = -log_probs.gather(
                1,
                pids[:, None],
            ).squeeze(1)
            uniform_smoothing = -log_probs.mean(dim=1)
            per_sample = (
                (1.0 - criterion_id.epsilon) * negative_log_likelihood
                + criterion_id.epsilon * uniform_smoothing
            )
            normalizer = active.to(dtype=per_sample.dtype)
        elif isinstance(criterion_id, nn.CrossEntropyLoss):
            valid = active & pids.ne(criterion_id.ignore_index)
            per_sample = F.cross_entropy(
                logits,
                pids,
                weight=criterion_id.weight,
                ignore_index=criterion_id.ignore_index,
                reduction="none",
                label_smoothing=criterion_id.label_smoothing,
            )
            normalizer = valid.to(dtype=per_sample.dtype)
            if criterion_id.weight is not None:
                normalizer = normalizer * criterion_id.weight[pids.clamp_min(0)]
            active = valid
        else:
            raise TypeError(
                "Multilevel suppression requires CrossEntropyLabelSmooth or "
                "torch.nn.CrossEntropyLoss"
            )

        active_weight = active.to(dtype=per_sample.dtype)
        return (per_sample * active_weight).sum() / normalizer.sum().clamp_min(1.0)

    def _multilevel_suppression_diagnostics(self, features) -> torch.Tensor:
        """Return validated scalar diagnostics in a stable metric order."""
        values = torch.zeros(
            len(self._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS),
            device=self.device,
            dtype=torch.float32,
        )
        if self._effective_multilevel_suppression_loss_weight() <= 0:
            return values
        if not isinstance(features, Mapping):
            raise RuntimeError("Enabled multilevel suppression requires a feature mapping")
        diagnostics = features.get("_multilevel_suppression_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("Enabled multilevel suppression model did not return diagnostics")

        scalars = []
        effective_ratio_atol = 1e-6
        for key in self._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS:
            value = diagnostics.get(key)
            if not torch.is_tensor(value) or value.numel() != 1:
                raise RuntimeError(f"Multilevel suppression diagnostic {key!r} must be a scalar tensor")
            if key == "effective_ratio" and value.is_floating_point():
                # Accommodate diagnostics produced inside an fp16/bf16 autocast
                # region. The canonical model emits fp32, but keeping the trainer
                # tolerant avoids rejecting correctly rounded third-party heads.
                effective_ratio_atol = max(
                    effective_ratio_atol,
                    float(torch.finfo(value.dtype).eps) / 2,
                )
            scalar = (
                value.detach()
                .to(
                    device=self.device,
                    dtype=torch.float32,
                )
                .reshape(())
            )
            scalars.append(scalar)
        values = torch.stack(scalars)
        expected_ratio = self.multilevel_suppression_ratio * self._current_multilevel_suppression_progress
        expected_ratio_matches = torch.isclose(
            values[0],
            values.new_tensor(expected_ratio),
            rtol=0.0,
            atol=effective_ratio_atol,
        )
        finite = torch.isfinite(values)
        in_range = (values >= 0) & (values <= 1)
        # Collapse all device-side checks before crossing the device boundary.
        # This is one synchronization per active batch instead of one per scalar.
        diagnostics_valid = finite.all() & in_range.all() & expected_ratio_matches
        if not bool(diagnostics_valid):
            cpu_values = values.cpu()
            cpu_finite = torch.isfinite(cpu_values)
            if not bool(cpu_finite.all()):
                invalid_index = int((~cpu_finite).nonzero()[0])
                invalid_key = self._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS[invalid_index]
                raise RuntimeError(f"Multilevel suppression diagnostic {invalid_key!r} must be finite")
            cpu_in_range = (cpu_values >= 0) & (cpu_values <= 1)
            if not bool(cpu_in_range.all()):
                invalid_index = int((~cpu_in_range).nonzero()[0])
                invalid_key = self._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS[invalid_index]
                raise RuntimeError(f"Multilevel suppression diagnostic {invalid_key!r} must lie in [0, 1]")
            raise RuntimeError("Multilevel suppression effective-ratio diagnostic does not match the trainer schedule")
        return values

    @staticmethod
    def _validate_multilevel_suppression_logits(
        logits: object,
        scale: str,
        expected: int,
    ) -> None:
        if (
            not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes))
            or len(logits) != expected
            or not all(torch.is_tensor(value) for value in logits)
        ):
            raise RuntimeError(f"Multilevel suppression {scale} logits must contain exactly {expected} tensors")

    @staticmethod
    def _validate_multilevel_suppression_active(
        active: object,
        scale: str,
        *,
        batch_size: int,
        expected: int,
    ) -> None:
        if (
            not torch.is_tensor(active)
            or active.dtype is not torch.bool
            or active.shape != (batch_size, expected)
        ):
            raise RuntimeError(
                f"Multilevel suppression {scale} activity must be a bool tensor "
                f"with shape ({batch_size}, {expected})"
            )
