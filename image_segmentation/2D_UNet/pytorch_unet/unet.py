import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class UNet_2D(nn.Module):

    '''

    '''
    def __init__(self, in_channels, out_channels, convdepth=(64, 128, 256, 512, 1024)):
        # maybe assert number of conv to determien elast amount of blocks?
        super(UNet_2D).__init__()

        encoder_layers = []
        encoder_layers.append(initial_layer_2D(in_channels,convdepth[0], convdepth[0]))
        encoder_layers.extend(encoder_2d(convdepth[i], convdepth[i+1], convdepth[i+1]) for i in range(len(convdepth)-2))
        decoder_layers = []
        decoder_layers.extend([decoder_2d(2*convdepth[i+1], 2*convdepth[i], 2 *convdepth[i], convdepth[i]) for i in reversed(range(len(convdepth)-2))])
        decoder_layers.append(last_2d(convdepth[1], convdepth[0], convdepth[0], out_channels))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = center_2d(convdepth[-2], convdepth[-1], convdepth[-1], convdepth[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_encoder = [x]
        for encoder_layer in self.encoder_layers:
            x_encoder.append(encoder_layer(x_encoder[-1]))

        x_decoder = [self.center(x_encoder[-1])]

        for decoder_layer_idx, decoder_layer in enumerate(self.decoder_layers):
            x_opp = x_encoder[-1-decoder_layer_idx]
            x_cat = torch.cat([pad_to_shape(x_decoder[-1], x_opp.shape), x_opp], dim=1)
            x_decoder.append(decoder_layer(x_cat))

        if not return_all:
            return x_decoder[-1]
        else:
            return x_decoder + x_encoder




def pad_to_shape(self, this, shp):

    if len(shp) == 4:
        pad = (0,shp[3] - this.shape[3], 0, shp[2] - this.shape[2])

    elif len(shp) == 5:
        pad = (0,shp[4] - this.shape[4],0,shp[3] - this.shape[3], 0, shp[2] - this.shape[2])

    return F.pad(this, pad)









