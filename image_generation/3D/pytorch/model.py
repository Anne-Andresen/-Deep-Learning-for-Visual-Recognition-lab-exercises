import torch.nn as nn
import torch
import numpy as np
import parameter_file as pf
import blocks


def conv_layer(in_channels, out_channels):
    conv_layers = [
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ELU(inplace=True)]
    return nn.Sequential(*conv_layers)


def out_layer(in_channels):
    out_layer = [nn.Conv3d(in_channels, 1, kernel_size=3, padding=(1, 1, 1), bias=False),
                 nn.Softmax()]
    return nn.Sequential(*out_layer)


class Generator(nn.Module):

    def __init__(self, args, in_channels, hidden, out_channels):
        super(Generator, self).__init__()
        self.args = args
        self.bias = pf.bias
        self.cube_len = pf.cube_len
        self.z_dim = pf.z_dim
        self.f_dim = pf.f_dim

        self.hidden = hidden
        # One variation of the

        self.conv_layer = conv_layer
        self.out_layer = out_layer

        # block can be written i various of ways we can also just write it straight to the self command
        #self.layer_out = #nn.Sequential(
            #nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False),
            #nn.Softmax())
        #output_layer = nn.Sequential(
         #   nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False),
         #   nn.Softmax())

        if self.f_dim == 32:
            pad = (1, 1, 1)
        else:
            pad = (0, 0, 0)
        self.layer1 = self.conv_layer(self.z_dim,self.z_dim*32) # should reverse reduce number of filters
        self.layer2 = self.conv_layer(self.z_dim*32, self.z_dim * 16)
        self.layer3 = self.conv_layer(self.z_dim*16, self.z_dim * 8)
        self.layer4 = self.conv_layer(self.z_dim*8, self.z_dim*4)
        self.layer5 = self.conv_layer(self.z_dim*4, self.z_dim*2)
        self.layer_out = self.out_layer(self.z_dim*2)

    def forward(self,x):
        x = x.view(-1, self.z_dim, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_out(x)
        x = np.squeeze(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.cube_len = pf.cube_len
        self.lv = pf.leak_value
        self.bias = pf.bias
        #m new con layer for discirminator no trnapose only cnv, bn and LR
        if self.cube_len ==32:
            pad = (1, 1, 1)
        else:
            pad = (0, 0, 0)

        self.f_dim = 32
        self.conv_layer = conv_layer

        self.layer1_d = self.conv_layer(self.f_dim, self.f_dim*32)
        self.layer2_d = self.conv_layer(self.f_dim *32, self.f_dim *16)
        self.layer3_d = self.conv_layer(self.f_dim*16, self.f_dim*8)
        self.layer4_d = self.conv_layer(self.f_dim*8, self.f_dim*4)
        self.layer5_d = self.conv_layer(self.f_dim*4, self.f_dim*2)
        self.layer_out_d = self.out_layer(self.f_dim*2)

    def forward(self, x):
        x = x.view(-1, self.f_dim, 1, 1, 1)
        x = self.layer1_d(x)
        x = self.layer2_d(x)
        x = self.layer3_d(x)
        x = self.layer4_d(x)
        x = self.layer5_d(x)
        x = self.layer_out_d(x)

        return x








