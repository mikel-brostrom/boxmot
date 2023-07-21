from boxmot.motion.cmc.ecc import ECCStrategy
from boxmot.motion.cmc.orb import ORBStrategy
from boxmot.motion.cmc.sof import SparseOptFlowStrategy


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECCStrategy
    elif cmc_method == 'orb':
        return ORBStrategy
    elif cmc_method == 'sof':
        return SparseOptFlowStrategy
    else:
        return None
