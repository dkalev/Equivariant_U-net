import torch
import torch.nn as nn
from collections import OrderedDict
from e2cnn.nn import R2Conv, R2ConvTransposed, R2Upsampling, GroupPooling
from e2cnn.nn import ReLU, FieldType, GeometricTensor, PointwiseMaxPool
from e2cnn.nn import InnerBatchNorm
from e2cnn import gspaces


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

    def forward(self, x, verbose=False):
        if verbose: print('input', x.shape)
        enc1 = self.encoder1(x)
        if verbose: print('enc1', enc1.shape)
        enc2 = self.encoder2(enc1)
        if verbose: print('enc2', enc2.shape)
        enc3 = self.encoder3(enc2)
        if verbose: print('enc3', enc3.shape)
        enc4 = self.encoder4(enc3)
        if verbose: print('enc4', enc4.shape)

        x = self.bottleneck(enc4)
        if verbose: print('botleneck', x.shape)

        x = self.decoder1(torch.cat([x, enc4], dim=1))
        if verbose: print('dec1', x.shape)
        x = self.decoder2(torch.cat([x, enc3], dim=1))
        if verbose: print('dec2', x.shape)
        x = self.decoder3(torch.cat([x, enc2], dim=1))
        if verbose: print('dec3', x.shape)
        x = self.decoder4(torch.cat([x, enc1], dim=1))
        if verbose: print('dec4', x.shape)

        x = self.head(x)
        if verbose: print('head', x.shape)
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


class C4UNet(nn.Module):

    def __init__(self, gspace, in_channels, out_channels, features=64):
        super().__init__()
        self.gspace = gspace
        self.feat_types = self.get_feature_types(in_channels, features)

        self.encoder1 = self.get_encoder('encoder1')
        self.encoder2 = self.get_encoder('encoder2')
        self.encoder3 = self.get_encoder('encoder3')
        self.encoder4 = self.get_encoder('encoder4')

        self.bottleneck = self.get_bottleneck()

        self.decoder1 = self.get_decoder('decoder1')
        self.decoder2 = self.get_decoder('decoder2')
        self.decoder3 = self.get_decoder('decoder3')
        self.decoder4 = self.get_decoder('decoder4')

        self.gpool = GroupPooling(self.feat_types['decoder4'][1])

        out = self.gpool.out_type.size

        self.head = nn.Sequential(OrderedDict({
            f'head-conv': nn.Conv2d(out, out_channels, kernel_size=3, padding=1, bias=False),
            f'final-act': nn.Sigmoid()
        }))

    def get_feature_types(self, in_channels, n_features):
        n_features_down = [n_features * 2**i for i in range(0,4)]
        # in_channels is multiplied by 2 because encoder outputs are concatenated as well (see forward method)
        n_features_up = [(2*n, max(1,n//2)) for n in reversed(n_features_down)]

        features_down = [( 'encoder1',
                          (
                            FieldType(self.gspace, in_channels*[self.gspace.trivial_repr]),
                            FieldType(self.gspace, n_features*[self.gspace.regular_repr])
                          )
        )] + [( f'encoder{i}',
               (
                 FieldType(self.gspace, n_in*[self.gspace.regular_repr]),
                 FieldType(self.gspace, n_out*[self.gspace.regular_repr])
               )
        ) for i, (n_in, n_out) in enumerate(zip(n_features_down, n_features_down[1:]), start=2)]

        feats_bottleneck = [( 'bottleneck',
               (
                 FieldType(self.gspace, n_features_down[-1]*[self.gspace.regular_repr]),
                 FieldType(self.gspace, n_features_down[-1]*[self.gspace.regular_repr])
               )
        )]

        features_up = [( f'decoder{i}',
                        (
                         FieldType(self.gspace, n_in*[self.gspace.regular_repr]),
                         FieldType(self.gspace, n_out*[self.gspace.regular_repr])
                        )
        ) for i, (n_in, n_out) in enumerate(n_features_up, start=1)]

        return OrderedDict(features_down + feats_bottleneck + features_up)

    def get_encoder(self, name):
        feat_type_in, feat_type_out = self.feat_types[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': R2Conv(feat_type_in, feat_type_out, kernel_size=3, stride=2, padding=1, bias=False),
            f'{name}-bn1': InnerBatchNorm(feat_type_out),
            f'{name}-relu1': ReLU(feat_type_out, inplace=True),
#             f'{name}-maxpool': PointwiseMaxPool(feat_type_out, kernel_size=3, stride=2, padding=1),
            f'{name}-conv2': R2Conv(feat_type_out, feat_type_out, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': InnerBatchNorm(feat_type_out),
            f'{name}-relu2': ReLU(feat_type_out, inplace=True),
        }))

    def get_decoder(self, name):
        feat_type_in, feat_type_out = self.feat_types[name]
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': R2ConvTransposed(feat_type_in, feat_type_out, kernel_size=3, stride=2, output_padding=1, bias=False),
#             f'{name}-upsample': R2Upsampling(feat_type_in, scale_factor=2),
#             f'{name}-conv1': R2Conv(feat_type_in, feat_type_out, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': InnerBatchNorm(feat_type_out),
            f'{name}-relu1': ReLU(feat_type_out, inplace=True),
            f'{name}-conv2': R2Conv(feat_type_out, feat_type_out, kernel_size=3, bias=False),
#             f'{name}-conv2': R2Conv(feat_type_out, feat_type_out, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': InnerBatchNorm(feat_type_out),
            f'{name}-relu2': ReLU(feat_type_out, inplace=True),
        }))

    def get_bottleneck(self, name='bottleneck'):
        feat_type, _ = self.feat_types[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': InnerBatchNorm(feat_type),
            f'{name}-relu1': ReLU(feat_type, inplace=True),
            f'{name}-conv2': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': InnerBatchNorm(feat_type),
            f'{name}-relu2': ReLU(feat_type, inplace=True),
        }))

    def forward(self, x, verbose=False):
        x = GeometricTensor(x, self.feat_types['encoder1'][0])
        if verbose: print('input', x.shape)
        enc1 = self.encoder1(x)
        if verbose: print('enc1', enc1.shape)
        enc2 = self.encoder2(enc1)
        if verbose: print('enc2', enc2.shape)
        enc3 = self.encoder3(enc2)
        if verbose: print('enc3', enc3.shape)
        enc4 = self.encoder4(enc3)
        if verbose: print('enc4', enc4.shape)

        x = self.bottleneck(enc4)
        if verbose: print('botleneck', x.shape)

        x = GeometricTensor(torch.cat([x.tensor, enc4.tensor], dim=1), self.feat_types['decoder1'][0])
        x = self.decoder1(x)
        if verbose: print('dec1', x.shape)
        x = GeometricTensor(torch.cat([x.tensor, enc3.tensor], dim=1), self.feat_types['decoder2'][0])
        x = self.decoder2(x)
        if verbose: print('dec2', x.shape)
        x = GeometricTensor(torch.cat([x.tensor, enc2.tensor], dim=1), self.feat_types['decoder3'][0])
        x = self.decoder3(x)
        if verbose: print('dec3', x.shape)
        x = GeometricTensor(torch.cat([x.tensor, enc1.tensor], dim=1), self.feat_types['decoder4'][0])
        x = self.decoder4(x)
        if verbose: print('dec4', x.shape)

        x = self.gpool(x)
        x = x.tensor
        if verbose: print('gpool', x.shape)

        x = self.head(x)
        if verbose: print('head', x.shape)
        return x



if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    from PIL import Image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trfm = ToTensor()

    r2_act = gspaces.Rot2dOnR2(4)
    model = C4UNet(r2_act, 3,1, features=2).to(device)

    x = Image.open('training/images/21_training.tif').resize((512,512)).convert('RGB')
    batch = trfm(x).unsqueeze(0).to(device)

    pred = model(batch)
    print(pred.shape)
    print(pred)

