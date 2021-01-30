import torch
import torch.nn as nn
from collections import OrderedDict
from e2cnn.nn import R2Conv, R2ConvTransposed, R2Upsampling, GroupPooling
from e2cnn.nn import ReLU, FieldType, GeometricTensor, PointwiseMaxPool
from e2cnn.nn import InnerBatchNorm
from e2cnn import gspaces
from .base import BaseEquivUNet


class C4UNet(BaseEquivUNet):

    def get_features(self, in_channels, n_features):
        features = super().get_features(in_channels, n_features)

        for module, (n_in, n_out) in features.items():
            if module == 'encoder1':
                features[module] = (
                    FieldType(self.gspace, n_in*[self.gspace.trivial_repr]),
                    FieldType(self.gspace, n_out*[self.gspace.regular_repr])
                )
            else:
                features[module] = (
                    FieldType(self.gspace, n_in*[self.gspace.regular_repr]),
                    FieldType(self.gspace, n_out*[self.gspace.regular_repr])
                )
        return features

    def get_encoder(self, name):
        feat_type_in, feat_type_out = self.features[name]
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
        feat_type_in, feat_type_out = self.features[name]
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
        feat_type, _ = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': InnerBatchNorm(feat_type),
            f'{name}-relu1': ReLU(feat_type, inplace=True),
            f'{name}-conv2': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': InnerBatchNorm(feat_type),
            f'{name}-relu2': ReLU(feat_type, inplace=True),
        }))
