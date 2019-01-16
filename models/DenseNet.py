from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['DenseNet121']

class DenseNet121(nn.Module):
    """
    Code imported from https://github.com/KaiyangZhou/deep-person-reid
    """
    def __init__(self, **kwargs):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        features = x.view(x.size(0), -1)

        return features
						