import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss


class soft_dice_loss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(soft_dice_loss).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_label):
        numerator = torch.sum(y_pred*y_label)
        denominator = torch.sum(y_pred*y_pred + y_label*y_label)
        softDiceLoss = numerator/denominator

        return softDiceLoss

class initial_layer3D(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channels, dropout=False):
        super(initial_layer3D, self).__init__()

        layers = [nn.Conv3d(in_channels, hidden_lay, kernel_size=3, padding=1), # unet architecture suggest two conv and then pooling
                  nn.BatchNorm3d(hidden_lay),
                  nn.ELU(inplace=True),
                  nn.Conv3d(hidden_lay, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm3d(out_channels),
                  nn.ELU(inplace=True)]#replacing relu with lELU TO METIGATE PROBLEMS WITH VANISHING GRADIENT AND RELU ZERO


        if dropout:
            layers.append(nn.Dropout3d(p=0.2))

        self.first_block = nn.Sequential(*layers)

    def forward(self,x):
        return self.first_block(x)


class encoder_3D(nn.Module):

    def __init__(self, in_channels, hidden_lays, out_channel, dropout=False, down_sample_kernel=2):
        super(encoder_3D, self).__init__()


        layers = [nn.MaxPool3d(kernel_size=down_sample_kernel),
                  nn.Conv3d(in_channels, hidden_lays, kernel_size=3, padding=1),
                  nn.BatchNorm3d(hidden_lays),
                  nn.ELU(inplace=True),#replacing relu with ELU TO METIGATE PROBLEMS WITH VANISHING GRADIENT AND RELU ZERO
                  nn.Conv3d(hidden_lays, out_channel, kernel_size=3, padding=1), # unet architecture suggest two conv and then pooling
                  nn.BatchNorm3d(hidden_lays),
                  nn.ELU(inplace=True)]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout has to be  between 0 and 1'
            layers.append(nn.Dropout3d(p=0.2))
        self.encoder_block_3D = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder_block_3D(x)


class center_3D(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channels, deconv_channel, dropout=False, down_sample_kernel=2, deconv_kernel=2):
        super(center_3D).__init__()

        layers = [nn.MaxPool3d(kernel_size=down_sample_kernel),
                  nn.Conv3d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm3d(hidden_lay),
                  nn.ELU(inplace=True),#replacing relu with ELU TO METIGATE PROBLEMS WITH VANISHING GRADIENT AND RELU ZERO
                  nn.Conv3d(hidden_lay, out_channels, kernel_size=3, padding=1), # unet architeture suggest two conv and then pooling
                  nn.BatchNorm3d(out_channels),
                  nn.ELU(inplace=True),
                  nn.ConvTranspose3d(out_channels, deconv_channel, kernel_size=deconv_kernel, stride=2)]

        if dropout:
            assert  0 <= dropout <= 1, 'dropout has to be between 0 and 1'
            layers.append(nn.Dropout3d(p=0.2))

        self.center_block_3D = nn.Sequential(*layers)

    def forward(self, x):
        return self.center_block_3D(x)

class decoder_3D(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channel, deconv_channel, dropout=False, deconv_kernel=2): # pooled using kernel=2  therefore we now deconv using same kernel size to obtain the original resolution
        super(decoder_3D).__init__()

        layers = [nn.Conv3d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm3d(hidden_lay),
                  nn.ELU(inplace=True),
                  nn.Conv3d(hidden_lay, out_channel, kernel_size=3, padding=1),
                  nn.BatchNorm3d(out_channel),
                  nn.ELU(inplace=True),
                  nn.ConvTranspose3d(out_channel, deconv_channel, kernel_size=deconv_kernel, stride=2)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout has to be between 0 and 1'
            layers.append(nn.Dropout3d(p=0.2))
        self.decoder_block_3D = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder_block_3D(x)

class last_3D(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channel):
        super(last_3D).__init__()

        layers = [nn.Conv3d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm3d(hidden_lay),
                  nn.ELU(inplace=True),
                  nn.Conv3d(hidden_lay, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm3d(hidden_lay),
                  nn.ELU(inplace=True),
                  nn.Conv3d(hidden_lay, out_channel, kernel_size=1),
                  nn.Softmax(dim=1)]

        self.last_block_3D = nn.Sequential(*layers)

    def forward(self,x):
        return self.last_block_3D(x)





