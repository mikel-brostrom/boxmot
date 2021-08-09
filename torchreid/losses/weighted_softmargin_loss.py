from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class WeightedSoftMarginLoss(nn.Module):
   
    def __init__(self, const=0.3):
        super(WeightedSoftMarginLoss, self).__init__()
        self.const = const

    def forward(self, dist, y): 
        loss = torch.log(1 + torch.exp(-(self.const + dist) * y)).mean()
        
        return loss 


