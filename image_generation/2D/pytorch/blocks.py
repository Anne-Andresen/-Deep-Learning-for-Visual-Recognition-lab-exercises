import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# two ways of writing blocks
def block(in_channels, out_channels, normalize=True):
    layers = [nn.Linear(in_channels, out_channels)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_channels, 0.2))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    return layers


def discriminator_block(in_channels, hidden=64): # in_channels = image shape

    layers = [nn.Linear(in_channels, hidden*8),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(hidden*8, hidden*4),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(hidden*4, hidden*2),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(hidden*2, 1),
              nn.Sigmoid()]
    return layers