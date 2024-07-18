import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self,
                 sample_weight=None,
                 size_sum=True,
                 scale=None,
                 tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):

        # logits = logits[0]
        # bs * label num
        # print(logits.shape, targets.shape)
        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (
                1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits,
                                                    targets,
                                                    reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1),
                                   torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()
        # print(loss.shape, loss_m.shape)
        return [loss], [loss_m]


@LOSSES.register("bceloss_vtb")
class BCELoss_VTB(nn.Module):

    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(BCELoss_VTB, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        loss_m = F.binary_cross_entropy_with_logits(logits,
                                                    targets,
                                                    reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1),
                                   torch.zeros(1))
        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[
                    1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx],
                                      self.sample_weight)
                loss_m = loss_m[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            # import pdb;pdb.set_trace()
            loss_m = (loss_m * weight.cuda())

        loss = loss_m.sum() / batch_size if self.size_average else loss_m.sum()

        return [loss], [loss_m]


@LOSSES.register("bceloss_l2l")
class BCELoss_L2L(nn.Module):

    def __init__(self,
                 sample_weight=None,
                 size_sum=True,
                 scale=None,
                 tb_writer=None,
                 ratio=0.5,
                 pos_weight=1,
                 size_average=True):
    
        super(BCELoss_L2L, self).__init__()
        self.ratio = ratio
        self.sample_weight = sample_weight
        self.size_average = size_average
        self.pos_weight = pos_weight
        # logger.warning('loss ratio for mask is 0')
        
    def forward(self, logits, targets, mask=None):
        batch_size = logits.shape[0]
        mask = torch.ones_like(logits) if mask == None else torch.where(
            mask == 0, torch.ones_like(mask), self.ratio *
            torch.ones_like(mask))
        loss_m= F.binary_cross_entropy_with_logits(logits,
                                                  targets,
                                                  reduction='none',
                                                  pos_weight=self.pos_weight *
                                                  torch.ones_like(logits))
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1),
                                   torch.zeros(1))
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            loss_m = (loss_m * weight.cuda() * mask.cuda())
            # loss_m = (loss_m * weight.cuda())
        loss = loss_m.sum() / batch_size if self.size_average else loss_m.sum()

        return [loss], [loss_m]
