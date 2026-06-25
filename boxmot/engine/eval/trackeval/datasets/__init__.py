"""BoxMOT TrackEval dataset adapters."""

from .base import CustomMotChallengeBase
from .mot_challenge_2d_box import CustomMotChallenge2DBox
from .mot_challenge_obb import CustomMotChallengeOBB

__all__ = [
    "CustomMotChallenge2DBox",
    "CustomMotChallengeBase",
    "CustomMotChallengeOBB",
]
