import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from models.registry import CLASSIFIER
from models.head.base import BaseClassifier

@CLASSIFIER.register("cosine")
class NormClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()

        self.logits = nn.Parameter(torch.FloatTensor(nattr, c_in))

        stdv = 1. / math.sqrt(self.logits.data.size(1))
        self.logits.data.uniform_(-stdv, stdv)

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, feature, label=None):
        feat = self.pool(feature).view(feature.size(0), -1)
        feat_n = F.normalize(feat, dim=1)
        weight_n = F.normalize(self.logits, dim=1)
        x = torch.matmul(feat_n, weight_n.t())
        return x, feat_n

