import torch.nn as nn


class block(nn.Module):

    def __init__(self, in_channels, hidden, out_channels):
        self.hidden = hidden
        # One variation of the
        conv_layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True)]
        self.conv_layer = nn.Sequential(*conv_layers)


        # block can be written i various of ways we can also just write it straight to the self command
        self.layer_out = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False), nn.Softmax())
        output_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=2, padding=(1, 1, 1), bias=False), nn.Softmax())

        return conv_layers, output_layer



