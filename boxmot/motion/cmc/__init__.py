# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from boxmot.motion.cmc.ecc import ECC
from boxmot.motion.cmc.orb import ORB
from boxmot.motion.cmc.sift import SIFT
from boxmot.motion.cmc.sof import SOF


def get_cmc_method(cmc_method):
    if cmc_method == "ecc":
        return ECC
    elif cmc_method == "orb":
        return ORB
    elif cmc_method == "sof":
        return SOF
    elif cmc_method == "sift":
        return SIFT
    else:
        return None
