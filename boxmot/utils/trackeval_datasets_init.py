from .kitti_2d_box import Kitti2DBox
from .kitti_mots import KittiMOTS
from .mot_challenge_2d_box import MotChallenge2DBox
from .mots_challenge import MOTSChallenge
from .bdd100k import BDD100K
from .davis import DAVIS
from .tao import TAO
from .tao_ow import TAO_OW
try:
    from .burst import BURST
    from .burst_ow import BURST_OW
except ImportError as err:
    print(f"Error importing BURST due to missing underlying dependency: {err}")
from .youtube_vis import YouTubeVIS
from .head_tracking_challenge import HeadTrackingChallenge
from .rob_mots import RobMOTS
from .person_path_22 import PersonPath22

try:
    from .mmot_rgb import mmot_RGB
    from .mmot_8ch import mmot_8ch
except ImportError as err:
    print(f"Skipping MMOT dataset adapters due to missing optional dependency: {err}")

    class _MissingMMOTDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MMOT TrackEval adapters require optional dependencies that are not installed."
            ) from err

    mmot_RGB = _MissingMMOTDependency
    mmot_8ch = _MissingMMOTDependency
