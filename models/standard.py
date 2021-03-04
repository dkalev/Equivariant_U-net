from collections import OrderedDict
import torch.nn as nn
from .base import BaseUNet

class UNet(BaseUNet):

    def get_head(self, out_channels):
        return nn.Sequential(OrderedDict({
            f'head-conv': nn.Conv2d(self.features['decoder4'][1], out_channels, kernel_size=3, padding=1, bias=False),
            f'final-act': nn.Sigmoid()
        }))

    def get_encoder(self, name):
        in_channels, out_channels = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(in_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def get_decoder(self, name):
        in_channels, out_channels = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv2': nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def get_bottleneck(self, name='bottleneck'):
        channels, _  = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))
