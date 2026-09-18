"""Canonical algorithm settings for BotSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class BotSortConfig(TrackerConfig):
    """Configure confidence stages, lost-track retention, and appearance matching.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        track_high_thresh: Detection confidence threshold for first-pass matching.
        track_low_thresh: Lower confidence bound for second-pass candidates.
        new_track_thresh: Minimum confidence required to initialize a new track.
        track_buffer: Lost-track lifetime in frames at 30 FPS, scaled by frame_rate.
        match_thresh: Maximum assignment cost for first-pass matching.
        proximity_thresh: Maximum geometry distance that permits appearance matching.
        appearance_thresh: Maximum embedding distance accepted for appearance matching.
        use_cmc: Enable camera-motion compensation; requires image frames.
        cmc_method: Camera-motion compensation method used when CMC is enabled.
        frame_rate: Frame rate used to scale track_buffer relative to 30 FPS.
        fuse_first_associate: Fuse detection confidence into the first-pass geometry cost.
        use_embeddings: Use supplied appearance embeddings, generating missing
            embeddings from image frames with the configured ReID backend.
        second_match_thresh: Maximum assignment cost for low-confidence detections.
        unconfirmed_match_thresh: Maximum assignment cost for tentative tracks.
        unconfirmed_emb_scale: Divisor applied to tentative-track embedding distances.
        removed_stracks_buffer: Maximum number of removed tracks retained in history.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    track_high_thresh: float = field(default=0.6296854875023994, metadata={"ge": 0.0, "le": 1.0})
    track_low_thresh: float = field(default=0.1014392537025336, metadata={"ge": 0.0, "le": 1.0})
    new_track_thresh: float = field(default=0.6246494191492591, metadata={"ge": 0.0, "le": 1.0})
    track_buffer: int = field(default=40, metadata={"ge": 0})
    match_thresh: float = field(default=0.7722224024589055, metadata={"ge": 0})
    proximity_thresh: float = field(default=0.6084297894561342, metadata={"ge": 0})
    appearance_thresh: float = field(default=0.6188818853936099, metadata={"ge": 0})
    use_cmc: bool = True
    cmc_method: str = field(default="sof", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    frame_rate: int = field(default=30, metadata={"gt": 0})
    fuse_first_associate: bool = True
    use_embeddings: bool = True
    second_match_thresh: float = field(default=0.28795081514328974, metadata={"ge": 0})
    unconfirmed_match_thresh: float = field(default=0.41148010638233784, metadata={"ge": 0})
    unconfirmed_emb_scale: float = field(default=2.5445206391993294, metadata={"gt": 0})
    removed_stracks_buffer: int = field(default=329, metadata={"ge": 0})
    det_thresh: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("BotSortConfig",)
