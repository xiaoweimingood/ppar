from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES
import timm
import math
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def build_backbone(key, multi_scale=False):

    model_dict = {
        'resnet34': 512,
        'resnet18': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'tresnet': 2432,
        'swin_s': 768,
        'swin_b': 1024,
        'swin_b_solider': 1024,
        'swin_t_solider': 768,
        'vit_s': 768,
        'vit_b': 768,
        'vit_b_path': 768,
        'vit_b_dinov1': 768,
        'vit_b_dinov2': 768,
        'vit_l': 1024,
        'bninception': 1024,
        'tresnetM': 2048,
        'tresnetL': 2048,
        'convnext_b': 1024,
        'intern_image_b': 1344,
        'intern_image_t': 768

    }
    if key in model_dict:
        model = BACKBONE[key]()
        output_d = model_dict[key]
    else:
        model = timm.create_model(key, pretrained=True)
        model.forward = model.forward_features
        output_d = model.num_features

    return model, output_d

def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True, dim=768,  attr_num=88):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd


        self.attr_num = attr_num

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, x, word_vec=None, label=None):
        features = self.backbone(x)
        cls_res = self.classifier(features, label)
        return cls_res

