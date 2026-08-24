"""Identity-only GlobalAP loss backed by an instance-indexed memory bank.

The component is deliberately independent of the trainer and model.  The
caller passes the descriptor that is actually deployed at retrieval time
(currently ``features["norm_concat_bn"]`` for CSL-TinyViT), so gradients train
the representation used by evaluation rather than a training-only proxy.

Typical training integration::

    global_ap = IdentityGlobalAP(
        memory_size=len(train_dataset),
        feature_dim=deployment_dim,
    ).to(device)

    global_ap_loss = global_ap(features["norm_concat_bn"], sample_indices, pids)
    loss = loss + global_ap_weight * global_ap_loss
    # After the loss has been formed (and normally after optimizer.step()):
    global_ap.update(features["norm_concat_bn"], sample_indices, pids)

All memory tensors and the age clock are registered buffers.  Consequently,
``state_dict()`` contains the complete resume state; a resumable training
checkpoint only needs to save and restore that state dictionary.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("IdentityGlobalAP",)


class IdentityGlobalAP(nn.Module):
    """Smooth-AP over a hard neighborhood from an instance memory bank.

    Each memory row is owned by one stable dataset sample index.  Every non-self
    sample of the same identity is a positive and every non-self sample of a
    different identity is a negative.  Camera metadata is deliberately absent:
    the deployed tracker embedding is a single-stream identity descriptor.  The
    differentiable ranking objective is the Smooth-AP relaxation evaluated on
    every eligible positive and the ``top_k`` most similar eligible negatives.
    Retaining positives outside the hard-negative cutoff is essential: otherwise
    the worst-ranked queries would receive no corrective gradient.

    Args:
        memory_size: Number of stable training-sample indices.
        feature_dim: Width of the deployed retrieval descriptor.
        top_k: Maximum number of hard negatives per query.  ``None`` uses every
            eligible negative.  Every eligible same-identity positive is always
            retained independently of this limit.
        temperature: Logistic temperature for the pairwise rank relaxation.
        max_age: Optional hard time-to-live in memory-update steps.  Entries
            older than this value do not participate in mining.
        staleness_tau: Optional exponential age-decay constant.  It downweights
            stale entries in the relaxed ranks while ``max_age`` remains the
            hard exclusion boundary.
        memory_momentum: Momentum used when a sample index is seen again.
        strict_metadata: Reject a PID change for an occupied sample index.  This
            catches unstable dataset indexing across resume.
        eps: Epsilon used by descriptor normalization.
    """

    _EMPTY_ID = -1
    _EMPTY_STEP = -1

    def __init__(
        self,
        memory_size: int,
        feature_dim: int,
        *,
        top_k: int | None = 512,
        temperature: float = 0.05,
        max_age: int | None = 2_000,
        staleness_tau: float | None = None,
        memory_momentum: float = 0.0,
        strict_metadata: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if max_age is not None and max_age < 0:
            raise ValueError("max_age must be non-negative or None")
        if staleness_tau is not None and (not math.isfinite(staleness_tau) or staleness_tau <= 0):
            raise ValueError("staleness_tau must be finite and positive or None")
        if not math.isfinite(memory_momentum) or not 0 <= memory_momentum < 1:
            raise ValueError("memory_momentum must lie in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.memory_size = int(memory_size)
        self.feature_dim = int(feature_dim)
        self.top_k = top_k
        self.temperature = float(temperature)
        self.max_age = max_age
        self.staleness_tau = staleness_tau
        self.memory_momentum = float(memory_momentum)
        self.strict_metadata = bool(strict_metadata)
        self.eps = float(eps)

        self.register_buffer(
            "memory_features",
            torch.zeros(memory_size, feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "memory_pids",
            torch.full((memory_size,), self._EMPTY_ID, dtype=torch.long),
        )
        self.register_buffer(
            "memory_last_update",
            torch.full((memory_size,), self._EMPTY_STEP, dtype=torch.long),
        )
        self.register_buffer(
            "memory_valid",
            torch.zeros(memory_size, dtype=torch.bool),
        )
        self.register_buffer("memory_step", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        deployed_descriptors: torch.Tensor,
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
    ) -> torch.Tensor:
        """Return a mean differentiable AP loss for queries with valid positives.

        Memory entries are stop-gradient by construction.  A query with no
        eligible same-identity non-self positive in fresh memory is skipped.  If
        the whole batch has no such query, the result is a differentiable FP32
        zero connected to ``deployed_descriptors``.
        """
        self._validate_batch(
            deployed_descriptors,
            sample_indices,
            pids,
        )
        if deployed_descriptors.device != self.memory_features.device:
            raise ValueError(
                "deployed_descriptors and GlobalAP memory must share a device; "
                "move the component with .to(descriptors.device)"
            )

        # Disable an enclosing autocast region explicitly: pairwise ranks can
        # saturate prematurely in fp16, especially at a small temperature.
        with torch.autocast(
            device_type=deployed_descriptors.device.type,
            enabled=False,
        ):
            queries = F.normalize(
                deployed_descriptors.float(),
                p=2,
                dim=1,
                eps=self.eps,
            )
            zero = queries.sum() * 0.0
            if queries.shape[0] == 0:
                return zero

            fresh = self.memory_valid
            ages = self.memory_step - self.memory_last_update
            if self.max_age is not None:
                fresh = fresh & ages.le(self.max_age)
            candidate_slots = torch.nonzero(fresh, as_tuple=False).flatten()
            if candidate_slots.numel() == 0:
                return zero

            candidate_features = self.memory_features.index_select(
                0,
                candidate_slots,
            )
            candidate_pids = self.memory_pids.index_select(0, candidate_slots)
            candidate_ages = ages.index_select(0, candidate_slots).clamp_min(0)
            similarities = queries @ candidate_features.transpose(0, 1)

            indices = sample_indices.to(
                device=candidate_slots.device,
                dtype=torch.long,
            )
            query_pids = pids.to(device=candidate_slots.device, dtype=torch.long)
            freshness_weights = self._freshness_weights(candidate_ages)
            query_losses: list[torch.Tensor] = []
            for query_index in range(queries.shape[0]):
                same_identity = candidate_pids.eq(query_pids[query_index])
                # Stable-index self-exclusion is independent of identity
                # metadata and protects against a stale copy of the query row.
                non_self = candidate_slots.ne(indices[query_index])
                positive = same_identity & non_self
                positive_slots = torch.nonzero(
                    positive,
                    as_tuple=False,
                ).flatten()
                if positive_slots.numel() == 0:
                    continue
                negative_slots = torch.nonzero(
                    ~same_identity & non_self,
                    as_tuple=False,
                ).flatten()
                selected_negatives = self._hard_negative_neighborhood(
                    similarities[query_index],
                    negative_slots,
                )
                selected = torch.cat(
                    (positive_slots, selected_negatives),
                    dim=0,
                )
                selected_positive = positive.index_select(0, selected)
                query_losses.append(
                    self._smooth_ap_loss(
                        similarities[query_index].index_select(0, selected),
                        selected_positive,
                        freshness_weights.index_select(0, selected),
                    )
                )

            if not query_losses:
                return zero
            return torch.stack(query_losses).mean()

    @torch.no_grad()
    def update(
        self,
        deployed_descriptors: torch.Tensor,
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
        *,
        step: int | None = None,
    ) -> None:
        """Store detached normalized descriptors at their stable sample rows.

        Repeated indices in a batch are averaged and normalized once.  Without
        an explicit ``step``, each call advances the persistent age clock by
        one.  Supplying a nondecreasing external optimizer step is useful when
        several updates belong to one gradient-accumulation step.
        """
        self._validate_batch(
            deployed_descriptors,
            sample_indices,
            pids,
        )
        if deployed_descriptors.device != self.memory_features.device:
            raise ValueError(
                "deployed_descriptors and GlobalAP memory must share a device; "
                "move the component with .to(descriptors.device)"
            )
        if deployed_descriptors.shape[0] == 0:
            return

        descriptors = F.normalize(
            deployed_descriptors.detach().float(),
            p=2,
            dim=1,
            eps=self.eps,
        )
        if not bool(torch.isfinite(descriptors).all()):
            raise ValueError("deployed_descriptors must be finite")
        if not bool(descriptors.norm(p=2, dim=1).gt(self.eps).all()):
            raise ValueError("deployed_descriptors must have non-zero norm")

        indices = sample_indices.to(
            device=self.memory_features.device,
            dtype=torch.long,
        )
        batch_pids = pids.to(device=self.memory_features.device, dtype=torch.long)
        unique_indices, inverse = torch.unique(
            indices,
            sorted=True,
            return_inverse=True,
        )
        unique_features = descriptors.new_zeros(
            unique_indices.shape[0],
            self.feature_dim,
        )
        unique_features.index_add_(0, inverse, descriptors)
        counts = torch.bincount(
            inverse,
            minlength=unique_indices.shape[0],
        ).to(dtype=unique_features.dtype)
        unique_features = F.normalize(
            unique_features / counts[:, None],
            p=2,
            dim=1,
            eps=self.eps,
        )

        unique_pids = torch.empty_like(unique_indices)
        # Duplicate stable indices are uncommon, and this small loop makes a
        # metadata conflict explicit instead of silently choosing one label.
        for offset in range(unique_indices.numel()):
            members = inverse.eq(offset)
            group_pids = torch.unique(batch_pids[members])
            if group_pids.numel() != 1:
                stable_index = int(unique_indices[offset])
                raise ValueError(f"Conflicting PID metadata for stable sample index {stable_index}")
            unique_pids[offset] = group_pids[0]

        occupied = self.memory_valid.index_select(0, unique_indices)
        if self.strict_metadata and bool(occupied.any()):
            stored_pids = self.memory_pids.index_select(0, unique_indices)
            mismatch = occupied & stored_pids.ne(unique_pids)
            if bool(mismatch.any()):
                offset = int(torch.nonzero(mismatch, as_tuple=False)[0])
                stable_index = int(unique_indices[offset])
                raise ValueError(f"PID metadata changed for occupied stable sample index {stable_index}")

        update_step = self._resolve_update_step(step)
        if self.memory_momentum > 0 and bool(occupied.any()):
            old_features = self.memory_features.index_select(0, unique_indices)
            blended = self.memory_momentum * old_features + (1.0 - self.memory_momentum) * unique_features
            degenerate = blended.norm(p=2, dim=1).le(self.eps)
            blended = F.normalize(
                blended,
                p=2,
                dim=1,
                eps=self.eps,
            )
            blended = torch.where(
                degenerate[:, None],
                unique_features,
                blended,
            )
            unique_features = torch.where(
                occupied[:, None],
                blended,
                unique_features,
            )

        self.memory_features.index_copy_(0, unique_indices, unique_features)
        self.memory_pids.index_copy_(0, unique_indices, unique_pids)
        self.memory_last_update.index_fill_(0, unique_indices, update_step)
        self.memory_valid.index_fill_(0, unique_indices, True)

    @torch.no_grad()
    def reset(self) -> None:
        """Clear every memory row and reset the persistent age clock."""
        self.memory_features.zero_()
        self.memory_pids.fill_(self._EMPTY_ID)
        self.memory_last_update.fill_(self._EMPTY_STEP)
        self.memory_valid.zero_()
        self.memory_step.zero_()

    @property
    def num_valid(self) -> int:
        """Return the number of occupied instance-memory rows."""
        return int(self.memory_valid.sum())

    def _hard_negative_neighborhood(
        self,
        query_similarities: torch.Tensor,
        negative_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Return at most ``top_k`` highest-similarity negative slots."""
        if self.top_k is None or negative_slots.numel() <= self.top_k:
            return negative_slots
        negative_similarities = query_similarities.index_select(
            0,
            negative_slots,
        )
        offsets = torch.topk(
            negative_similarities,
            k=self.top_k,
            largest=True,
            sorted=False,
        ).indices
        return negative_slots.index_select(0, offsets)

    def _smooth_ap_loss(
        self,
        scores: torch.Tensor,
        positive: torch.Tensor,
        freshness: torch.Tensor,
    ) -> torch.Tensor:
        positive_positions = torch.nonzero(
            positive,
            as_tuple=False,
        ).flatten()
        positive_scores = scores.index_select(0, positive_positions)
        # [candidate, positive]: probability that the candidate ranks above
        # the positive under the logistic Smooth-AP relaxation.
        pairwise_before = torch.sigmoid((scores[:, None] - positive_scores[None, :]) / self.temperature)
        comparison_mask = torch.ones_like(pairwise_before)
        comparison_mask[
            positive_positions,
            torch.arange(
                positive_positions.numel(),
                device=positive_positions.device,
            ),
        ] = 0.0
        weighted_before = pairwise_before * comparison_mask * freshness[:, None]
        rank_all = 1.0 + weighted_before.sum(dim=0)
        rank_positive = 1.0 + (weighted_before * positive[:, None]).sum(dim=0)
        precision = rank_positive / rank_all.clamp_min(self.eps)
        positive_freshness = freshness.index_select(0, positive_positions)
        average_precision = (precision * positive_freshness).sum() / positive_freshness.sum().clamp_min(self.eps)
        return 1.0 - average_precision

    def _freshness_weights(self, ages: torch.Tensor) -> torch.Tensor:
        if self.staleness_tau is None:
            return torch.ones_like(ages, dtype=torch.float32)
        return torch.exp(-ages.float() / self.staleness_tau)

    def _resolve_update_step(self, step: int | None) -> int:
        current = int(self.memory_step)
        if step is None:
            resolved = current + 1
        else:
            resolved = int(step)
            if resolved < current:
                raise ValueError(f"update step must be nondecreasing: current={current}, received={resolved}")
        self.memory_step.fill_(resolved)
        return resolved

    def _validate_batch(
        self,
        deployed_descriptors: torch.Tensor,
        sample_indices: torch.Tensor,
        pids: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(deployed_descriptors):
            raise TypeError("deployed_descriptors must be a tensor")
        if deployed_descriptors.ndim != 2:
            raise ValueError("deployed_descriptors must have shape [batch, feature_dim]")
        if deployed_descriptors.shape[1] != self.feature_dim:
            raise ValueError(
                "deployed descriptor width does not match feature_dim: "
                f"{deployed_descriptors.shape[1]} != {self.feature_dim}"
            )
        if not deployed_descriptors.is_floating_point():
            raise TypeError("deployed_descriptors must be floating point")
        batch_size = deployed_descriptors.shape[0]
        for name, values in (
            ("sample_indices", sample_indices),
            ("pids", pids),
        ):
            if not torch.is_tensor(values):
                raise TypeError(f"{name} must be a tensor")
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [{batch_size}]")
            if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
                raise TypeError(f"{name} must use an integer dtype")
        if sample_indices.numel() > 0:
            minimum = int(sample_indices.min())
            maximum = int(sample_indices.max())
            if minimum < 0 or maximum >= self.memory_size:
                raise IndexError(
                    f"stable sample indices must lie in [0, {self.memory_size}): min={minimum}, max={maximum}"
                )
