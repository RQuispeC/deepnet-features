from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['ResNet50']

class ResNet50(nn.Module):
    """
    Code imported from https://github.com/KaiyangZhou/deep-person-reid
    """
    def __init__(self,  **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        features = x.view(x.size(0), -1)

        return features