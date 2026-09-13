"""Shared keyword contracts for concrete tracker constructors.

Keep algorithm-specific parameters on their constructors. These optional
keyword groups describe only settings forwarded to the shared tracker bases.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from typing_extensions import TypedDict


class TrackerMetadataOptions(TypedDict, total=False):
    """Observation history and detector class metadata."""

    max_obs: int
    class_ids: Iterable[int] | None
    class_names: Mapping[int, str] | None


class AssociationTrackerOptions(TrackerMetadataOptions, total=False):
    """Metadata plus configurable geometric association."""

    asso_func: str


class BoxTrackerOptions(AssociationTrackerOptions, total=False):
    """Shared box geometry, lifecycle, and display settings."""

    max_age: int
    min_hits: int
    iou_threshold: float
    per_class: bool
    is_obb: bool


class KalmanTrackerOptions(BoxTrackerOptions, total=False):
    """Box settings with supported capture timing."""

    variable_dt: bool


class CommonTrackerOptions(KalmanTrackerOptions, total=False):
    """Shared settings for trackers accepting the base detection threshold."""

    det_thresh: float


class OccluBoostOptions(CommonTrackerOptions, total=False):
    """Additional BoostTrack parameters forwarded by OccluBoost."""

    use_cmc: bool
    min_box_area: int
    aspect_ratio_thresh: float
    cmc_method: str
    lambda_iou: float
    lambda_mhd: float
    lambda_shape: float
    use_dlo_boost: bool
    use_duo_boost: bool
    dlo_boost_coef: float
    s_sim_corr: bool
    use_rich_s: bool
    use_sb: bool
    use_vt: bool


__all__ = (
    "AssociationTrackerOptions",
    "BoxTrackerOptions",
    "CommonTrackerOptions",
    "KalmanTrackerOptions",
    "OccluBoostOptions",
    "TrackerMetadataOptions",
)
