from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .Xception import *
from .Inception import *
from .ResNeXt import *

__factory = {
    'resnet50': ResNet50,
    'densenet121': DenseNet121,
    'xception': Xception,
    'inceptionv4': InceptionV4ReID,
    'resnext101': ResNeXt101_32x4d,
}


def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)