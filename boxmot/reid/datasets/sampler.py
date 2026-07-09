"""PK sampler for ReID training.

Samples P identities and K instances per identity in each mini-batch,
which is the standard setup for triplet-based ReID training.
"""

from __future__ import annotations

import copy
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List

from torch.utils.data import Sampler

from boxmot.reid.datasets.base import ReIDSample


@dataclass(frozen=True)
class SourceBalanceGroup:
    """One source-balanced PK sub-batch definition."""

    sources: tuple[str, ...]
    p: int
    k: int

    @property
    def batch_size(self) -> int:
        return self.p * self.k


def normalize_source_name(source: str) -> str:
    """Normalize source labels for matching sampler specs to dataset names."""
    return str(source).lower().replace("-", "").replace("_", "").replace(" ", "")


def parse_source_balance(source_balance: str | None) -> tuple[SourceBalanceGroup, ...]:
    """Parse a source-balanced PK sampler specification.

    Format:
        ``source_a+source_b:P,K;source_c:P,K``

    Example:
        ``market1501+dukemtmcreid:8,4;mot17_1501:8,4``
    """
    spec = str(source_balance or "").strip()
    if not spec:
        return ()

    groups: list[SourceBalanceGroup] = []
    for raw_group in spec.split(";"):
        group = raw_group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(
                "source_balance groups must use 'sources:p,k', "
                f"got {raw_group!r}"
            )
        raw_sources, raw_pk = group.split(":", 1)
        sources = tuple(
            normalize_source_name(part)
            for part in re.split(r"[+,]", raw_sources)
            if part.strip()
        )
        pk_tokens = [part for part in re.split(r"[xX,* ]+", raw_pk.strip()) if part]
        if len(pk_tokens) != 2:
            raise ValueError(
                "source_balance group size must be 'p,k' or 'p x k', "
                f"got {raw_pk!r}"
            )
        p, k = (int(pk_tokens[0]), int(pk_tokens[1]))
        if not sources:
            raise ValueError(f"source_balance group has no sources: {raw_group!r}")
        if p <= 0 or k <= 0:
            raise ValueError("source_balance p and k values must be positive")
        groups.append(SourceBalanceGroup(sources=sources, p=p, k=k))

    if not groups:
        raise ValueError("source_balance did not contain any valid groups")
    return tuple(groups)


class PKSampler(Sampler[int]):
    """Randomly samples P identities, then K images per identity.

    Args:
        samples: List of ``ReIDSample`` from the training set.
        p: Number of identities per batch.
        k: Number of instances per identity.
    """

    def __init__(self, samples: List[ReIDSample], p: int = 16, k: int = 4, seed: int = 0):
        self.samples = samples
        self.p = p
        self.k = k
        self.seed = int(seed)
        self.epoch = 0

        self._pid_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, s in enumerate(samples):
            self._pid_to_indices[s.pid].append(idx)
        self._pids = list(self._pid_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic sampling stream for one training epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pids = copy.deepcopy(self._pids)
        rng.shuffle(pids)

        batch_indices: List[int] = []
        for pid in pids:
            idxs = copy.deepcopy(self._pid_to_indices[pid])
            if len(idxs) < self.k:
                idxs = idxs * (self.k // len(idxs) + 1)
            rng.shuffle(idxs)
            batch_indices.extend(idxs[: self.k])

            if len(batch_indices) >= self.p * self.k:
                yield from batch_indices[: self.p * self.k]
                batch_indices = batch_indices[self.p * self.k :]

        # Yield remaining complete batches
        bs = self.p * self.k
        while len(batch_indices) >= bs:
            yield from batch_indices[:bs]
            batch_indices = batch_indices[bs:]

    def __len__(self) -> int:
        return (len(self._pids) // self.p) * self.p * self.k


class SourceBalancedPKSampler(Sampler[int]):
    """PK sampler that composes each batch from fixed source groups.

    Each group independently samples P identities and K images per identity.
    The yielded order keeps the group sub-batches contiguous inside each
    training batch, while the source and identity order remains deterministic
    per seed and epoch.
    """

    def __init__(
        self,
        samples: List[ReIDSample],
        groups: tuple[SourceBalanceGroup, ...] | str,
        seed: int = 0,
    ):
        self.samples = samples
        self.groups = parse_source_balance(groups) if isinstance(groups, str) else tuple(groups)
        if not self.groups:
            raise ValueError("SourceBalancedPKSampler requires at least one source group")
        self.seed = int(seed)
        self.epoch = 0

        self._pid_to_indices: dict[int, list[int]] = defaultdict(list)
        self._pid_to_source: dict[int, str] = {}
        self._sources_seen: set[str] = set()
        for idx, sample in enumerate(samples):
            source = normalize_source_name(sample.source)
            self._sources_seen.add(source)
            self._pid_to_indices[sample.pid].append(idx)
            existing = self._pid_to_source.setdefault(sample.pid, source)
            if existing != source:
                raise ValueError(
                    f"PID {sample.pid} appears in multiple sources: "
                    f"{existing!r} and {source!r}"
                )

        self._group_pids: list[list[int]] = []
        for group in self.groups:
            source_set = set(group.sources)
            pids = [
                pid
                for pid, source in self._pid_to_source.items()
                if source in source_set
            ]
            if not pids:
                available = ", ".join(sorted(self._sources_seen)) or "(none)"
                requested = "+".join(group.sources)
                raise ValueError(
                    f"source_balance group '{requested}' matched no training samples. "
                    f"Available sources: {available}"
                )
            if len(pids) < group.p:
                requested = "+".join(group.sources)
                raise ValueError(
                    f"source_balance group '{requested}' has only {len(pids)} IDs, "
                    f"but p={group.p} was requested"
                )
            self._group_pids.append(pids)

    @property
    def batch_size(self) -> int:
        return sum(group.batch_size for group in self.groups)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic sampling stream for one training epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        group_batches = [
            self._build_group_batches(rng, pids, group)
            for pids, group in zip(self._group_pids, self.groups)
        ]
        n_batches = min(len(batches) for batches in group_batches)
        for batch_index in range(n_batches):
            for batches in group_batches:
                yield from batches[batch_index]

    def __len__(self) -> int:
        n_batches = min(
            len(pids) // group.p
            for pids, group in zip(self._group_pids, self.groups)
        )
        return n_batches * self.batch_size

    def _build_group_batches(
        self,
        rng: random.Random,
        pids: list[int],
        group: SourceBalanceGroup,
    ) -> list[list[int]]:
        shuffled_pids = list(pids)
        rng.shuffle(shuffled_pids)

        batches: list[list[int]] = []
        batch_indices: list[int] = []
        target_size = group.batch_size
        for pid in shuffled_pids:
            idxs = copy.deepcopy(self._pid_to_indices[pid])
            if len(idxs) < group.k:
                idxs = idxs * (group.k // len(idxs) + 1)
            rng.shuffle(idxs)
            batch_indices.extend(idxs[: group.k])

            if len(batch_indices) >= target_size:
                batches.append(batch_indices[:target_size])
                batch_indices = batch_indices[target_size:]
        return batches
