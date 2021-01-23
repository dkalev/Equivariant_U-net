import torch
import torch.nn as nn
from collections import OrderedDict


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, features=64):
        super().__init__()

        self.features = [ features * i for i in [1,2,4,8] ]

        self.encoder1 = self.block(in_channels, self.features[0], 'encoder1')
        self.encoder2 = self.block(self.features[0], self.features[1], 'encoder2')
        self.encoder3 = self.block(self.features[1], self.features[2], 'encoder3')
        self.encoder4 = self.block(self.features[2], self.features[3], 'encoder4')

        self.bottleneck = self.block(self.features[3], self.features[3], 'bottleneck')

        # in_channels is multiplied by 2 because encoder outputs are concatenated as well (see forward method)
        self.decoder1 = self.block(self.features[3]*2, self.features[2], 'decoder1')
        self.decoder2 = self.block(self.features[2]*2, self.features[1], 'decoder2')
        self.decoder3 = self.block(self.features[1]*2, self.features[0], 'decoder3')
        self.decoder4 = self.block(self.features[0]*2, self.features[0], 'decoder4')

        self.head = nn.Sequential(OrderedDict({
            f'head-conv': nn.Conv2d(self.features[0], out_channels, kernel_size=3, padding=1, bias=False),
            f'final-act': nn.Sigmoid()
        }))

    def forward(self, x):
        print('input', x.shape)
        enc1 = self.encoder1(x)
        print('enc1', enc1.shape)
        enc2 = self.encoder2(enc1)
        print('enc2', enc2.shape)
        enc3 = self.encoder3(enc2)
        print('enc3', enc3.shape)
        enc4 = self.encoder4(enc3)
        print('enc4', enc4.shape)

        x = self.bottleneck(enc4)
        print('botleneck', x.shape)

        x = self.decoder1(torch.cat([x, enc4], dim=1))
        print('dec1', x.shape)
        x = self.decoder2(torch.cat([x, enc3], dim=1))
        print('dec2', x.shape)
        x = self.decoder3(torch.cat([x, enc2], dim=1))
        print('dec3', x.shape)
        x = self.decoder4(torch.cat([x, enc1], dim=1))
        print('dec4', x.shape)

        x = self.head(x)
        print('head', x.shape)
        return x

    def block(self, in_channels, out_channels, name):
        if in_channels < out_channels:
            return self._down_block(in_channels, out_channels, name)
        elif in_channels > out_channels:
            return self._up_block(in_channels, out_channels, name)
        else:
            return self._bottleneck(in_channels, name)

    @staticmethod
    def _down_block(in_channels, out_channels, name='down-block'):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    @staticmethod
    def _up_block(in_channels, out_channels, name='up-block'):
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv2': nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    @staticmethod
    def _bottleneck(channels, name='bottleneck'):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm2d(channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm2d(channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))



if __name__ == '__main__':
    import torch
    from torchvision.transforms import ToTensor
    from PIL import Image

    device = torch.device('cuda')
    trfm = ToTensor()

    model = UNet(3,1).to(device)

    x = Image.open('training/images/21_training.tif').resize((512,512)).convert('RGB')
    batch = trfm(x).unsqueeze(0).to(device)

    pred = model(batch)
    print(pred.shape)
    print(pred)

