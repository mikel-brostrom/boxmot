from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
#from .gaussian_mine_sum_triplet_loss import TripletLoss
#from .soft_margin_triplet_loss import TripletLoss
#from .gaussian_mine_triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ranked_loss import RankedLoss



def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
