import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch
class SoftDiceLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        numerator = torch.sum(y_pred *y_gt)
        denominator = torch.sum(y_pred*y_pred + y_gt*y_gt)
        soft_dice_loss = numerator/denominator
        return soft_dice_loss

class initial_layer_2D(nn.Module):
    def __init__(self, in_channels, hidden_lay, out_channels, dropout=False):
        super(initial_layer_2D, self).__init__()

        layers = [nn.Conv2d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_lay),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_lay, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if dropout:
            assert 0 <= dropout <= 1,'Dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)

class encoder_2d(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channels, dropout=False, downsample_kernel=2):
        super(encoder_2d).__init__()

        layers = [nn.MaxPool2d(kernel_size=downsample_kernel),
                  nn.Conv2d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_lay),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_lay, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]

        if dropout:
            assert 0<= dropout <= 1, 'Dropout has to be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)
    def forward(self,x):
        return self.encoder(x)

class center_2d(nn.Module):
    def __init__(self, in_channels, hidden_lay, out_channels, deconv_channels, dropout=False):
        super(center_2d).__init__()

        layers = [nn.MaxPool2d(kernel_size=2),
                  nn.Conv2d(in_channels,hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_lay),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_lay, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)]

        if dropout:
            assert 0 <= dropout <=1, 'dropout has to be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center_2d = nn.Sequential(*layers)
    def forward(self, x):
        return self.center_2d(x)

class decoder_2d(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channels, deconv_channels, dropout=False):
        super(decoder_2d).__init__()
        layers = [nn.Conv2d(in_channels, hidden_lay, kernel_size=3, padding=1),
                   nn.BatchNorm2d(hidden_lay),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(hidden_lay, out_channels, kernel_size=3, padding=1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)]

        if dropout:
            assert 0 <= dropout <= 1, 'Drop between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder_2d = nn.Sequential(*layers)
    def forward(self, x):
        return self.decoder_2d(x)

class last_2d(nn.Module):
    def __init__(self, in_channels, hidden_lay, out_channels, deconv_channels, dropout=False):
        super(last_2d).__init__()

        layers = [nn.Conv2d(in_channels, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_lay),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_lay, hidden_lay, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_lay),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_lay, out_channels, kernel_size=1),
                  nn.Softmax(dim=1)]

        self.last = nn.Sequential(*layers)

    def forward(self, x):
        return self.last(x)

class first_3d(nn.Module):

    def __init__(self, in_channels, hidden_lay, out_channels, dropout=False):
        super(first_3d).__init__()
