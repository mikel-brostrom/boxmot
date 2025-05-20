import pytest
from pathlib import Path
from boxmot.utils import WEIGHTS

from boxmot import (
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack,
)

from tests.test_config import (
    MOTION_ONLY_TRACKING_METHODS,
    MOTION_N_APPEARANCE_TRACKING_METHODS,
)

@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    Tracker(
        reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        device="cpu",
        half=True,
    )

@pytest.mark.parametrize("Tracker", MOTION_ONLY_TRACKING_METHODS)
def test_motion_only_trackers_instantiation(Tracker):
    Tracker()