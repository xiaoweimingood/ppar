# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
from models.registry import CLASSIFIER
from models.head.transformer import build_transformer
from models.head.position_encoding import build_position_encoding
from models.head.base import BaseClassifier

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

@CLASSIFIER.register("q2l")
class Qeruy2Label(BaseClassifier):
    def __init__(self, 
                 nattr,
                 c_in,
                 **kargs):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.transformer = build_transformer()
        self.position_embedding = build_position_encoding()
        self.num_class = nattr
        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(c_in, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(self.num_class, hidden_dim)
        self.fc = GroupWiseLinear(self.num_class, hidden_dim, bias=True)
        
    def forward(self, x, label=None):
        # print('q2l', x.size())
        pos = self.position_embedding(x).to(x.dtype)
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(x), query_input, pos)[0] # B,K,d
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        return out, pos

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

