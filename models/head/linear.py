import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from models.registry import CLASSIFIER
from models.head.base import BaseClassifier

@CLASSIFIER.register("linear")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr),
            nn.BatchNorm1d(nattr) if bn else nn.Identity()
        )


    def forward(self, feature, label=None):

        if len(feature.shape) == 3:  # for vit (bt, nattr, c)

            bt, hw, c = feature.shape
            # NOTE ONLY USED FOR INPUT SIZE (256, 192)
            h = 14
            # w = 12
            w = 14
            feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        feat = self.pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)

        return x, feature
    
    
@CLASSIFIER.register("linear_vtb")
class VTBClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()
        self.nattr = nattr
        self.weight_layer = nn.ModuleList([nn.Linear(c_in, 1) for i in range(self.nattr)])
        self.bn = nn.BatchNorm1d(nattr)

    def forward(self, feature, label=None):
        logits = torch.cat([self.weight_layer[i](feature[:, i, :]) for i in range(self.nattr)], dim=1)
        logits = self.bn(logits)
        return [logits], feature
    

@CLASSIFIER.register("linear_internimage")
class ConvHead(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()
        self.nattr = nattr
        self.c_in = c_in
        self.cls_scale = 1.5
        self.head = nn.Linear(int(self.c_in * self.cls_scale), nattr) 

    def forward(self, feature, label=None):
        logits = self.head(feature)
        return [logits], feature
