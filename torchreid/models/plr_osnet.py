"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['plr_osnet']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .osnet import *
import copy
import random
import math
from .attention_module import Attention_Module


class PLR_OSNet(nn.Module):

    def __init__(self, num_classes, fc_dims=None, loss=None,  pretrained=True, **kwargs):
        super(PLR_OSNet, self).__init__()
        
        osnet = osnet_x1_0(pretrained=pretrained)
        
        self.loss = loss
        
        self.layer0 = nn.Sequential(
            osnet.conv1,
            osnet.maxpool
            )
        self.layer1 = osnet.conv2
        self.attention_module1 = Attention_Module(256)
        self.layer2 = osnet.conv3
        self.attention_module2 = Attention_Module(384)
        self.layer30 = osnet.conv4
        self.layer31 = nn.Sequential(copy.deepcopy(self.layer30))
      
        self.layer40 = osnet.conv5 
        self.layer41 = nn.Sequential(copy.deepcopy(self.layer40))        


        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc2 = nn.Linear(fc_dims, 512)
 
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)

        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
              
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)

        nn.init.normal_(self.fc2.weight, 0, 0.01)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)


    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.attention_module1(x)
        x = self.layer2(x)
        x = self.attention_module2(x)
        x1 = self.layer30(x)
        x2 = self.layer31(x)
        x1 = self.layer40(x1)
        x2 = self.layer41(x2)
        return x1, x2

    def forward(self, x):
        f1, f2 = self.featuremaps(x)
        B, C, H, W = f1.size()
        f11 = f1[:, :, :H//4, :]
        f12 = f1[:, :, H//4:H//2, :]
        f13 = f1[:, :, H//2:(3*H//4), :]
        f14 = f1[:, :, (3*H//4):, :]
        
        v11 = self.global_avgpool(f11)
        v12 = self.global_avgpool(f12)
        v13 = self.global_avgpool(f13)
        v14 = self.global_avgpool(f14)
        v2 = self.global_maxpool(f2)
       
        v11 = v11.view(v11.size(0), -1)
        v12 = v12.view(v12.size(0), -1)
        v13 = v13.view(v13.size(0), -1)
        v14 = v14.view(v14.size(0), -1)
        v1 = torch.cat([v11, v12, v13, v14], 1)
        v2 = v2.view(v2.size(0), -1)

        v2 = self.fc2(v2)

        fea = [v1, v2]
       
        v1 = self.bn1(v1)
        v2 = self.bn2(v2)
        
        if not self.training:
           v1 = F.normalize(v1, p=2, dim=1)
           v2 = F.normalize(v2, p=2, dim=1)
           return torch.cat([v1, v2], 1)
   
        y1 = self.classifier1(v1)
        y2 = self.classifier2(v2)
 
        if self.loss == 'softmax':
            return y1, y2
        elif self.loss == 'triplet':
            return y1, y2, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def plr_osnet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PLR_OSNet(
        num_classes=num_classes,
        fc_dims=512,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
    return model
