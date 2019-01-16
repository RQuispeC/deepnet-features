from __future__ import print_function, absolute_import

import argparse
import models

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())

args = parser.parse_args()

if __name__ == '__main__':
    model = models.init_model(name=args.arch)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))