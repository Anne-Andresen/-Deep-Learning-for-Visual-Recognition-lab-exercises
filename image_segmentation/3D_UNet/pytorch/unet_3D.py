import os
import numpy as np
import os
import torch.nn as nn
import torch
from .blocks import *
import torch.nn.functional as F
class UNet_3D(nn.Module):


    def __init__(self, in_channels, out_channels, conv_depth=(64, 128, 256, 512, 1024)):
        super(UNet_3D).__init__()
        encoder_layers = []
        encoder_layers.append(initial_layer3D(in_channels, conv_depth[0], conv_depth[0]))
        encoder_layers.extend([encoder_3D(conv_depth[i+1], conv_depth[i +1], conv_depth[i+ 1], conv_depth[i]) for i in range(len(conv_depth)-1)])
        decoder_layers =[]
        decoder_layers.extend([decoder_3D(2*conv_depth[i+1], 2*conv_depth[i+1], 2*conv_depth[i+1], 2*conv_depth[i]) for i in reversed(range(len(conv_depth)-1))])
        decoder_layers.append(last_3D(conv_depth[1], conv_depth[0], out_channels))

        self.encoder_3D = nn.Sequential(*encoder_layers)
        self.center_3D = center_3D(conv_depth[-2], conv_depth[-1], conv_depth[-1], conv_depth[-2])
        self.decoder_3D = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_encoder = [x]

        for encoder_layer in self.encoder_3D:
            x_encoder.append(encoder_layer(x_encoder[-1]))

        x_decoder =[self.center_3D(x_encoder[-1])]

        for decoder_layer, decoder_layer_1 in self.decoder_3D:
            x_opposite = x_encoder[-1 - decoder_layer]
            x_concat = torch.cat([pad_to_shape(x_decoder[-1], x_opposite.shape), x_opposite], dim=1)
            x_decoder.append(decoder_layer_1(x_concat))

        if not return_all:
            return x_decoder[-1]
        else:
            x_decoder + x_encoder



def pad_to_shape(self, this, shp):

    if len(shp) == 4:
        pad = (0,shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
        return F.pad(this, pad)

    elif len(shp) == 5:
        pad = (0,shp[4] - this.shape[4],0,shp[3] - this.shape[3], 0, shp[2] - this.shape[2])

        return F.pad(this, pad)
