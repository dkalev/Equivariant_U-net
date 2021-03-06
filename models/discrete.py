import torch.nn as nn
from collections import OrderedDict
from e2cnn.nn import R2Conv, R2ConvTransposed
from e2cnn.nn import ReLU, FieldType
from e2cnn.nn import InnerBatchNorm
from .base import BaseEquivUNet
from argparse import ArgumentParser
from e2cnn import gspaces


class C4UNet(BaseEquivUNet):

    @staticmethod
    def get_gspace(kwargs):
        group_type, N = kwargs['group_type'], kwargs['N']
        del kwargs['group_type']
        del kwargs['N']
        if group_type == 'circle':
            return gspaces.Rot2dOnR2(N)
        elif group_type == 'dihedral':
            return gspaces.FlipRot2dOnR2(N)
        else:
            raise ValueError('Discrete group argument "group" must be one of ["circle", "dihedral"]')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--N', default=4, type=int, help='N for circle and dihedral groups')
        parser.add_argument('--group', type=str, default='circle', help='Type of discrete group. One of [circle, dihedral]')
        return parser

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
