"""Collation helpers that preserve variable-size canonical samples."""

from __future__ import annotations

from collections.abc import Sequence

from .cached import DatasetSample


def collate_samples(samples: Sequence[DatasetSample]) -> list[DatasetSample]:
    """Return samples as a list; detection and image sizes are intentionally ragged."""

    return list(samples)


__all__ = ("collate_samples",)
