from collections import OrderedDict
import torch.nn as nn
from e2cnn.nn import R2Conv, R2ConvTransposed, GeometricTensor, GNormBatchNorm, NormNonLinearity, FieldType
from .base import BaseEquivUNet, InvariantHead


class HarmonicUNet(BaseEquivUNet):
        
    def get_features(self, in_channels, n_features):
        features = super().get_features(in_channels, n_features)

        for module, (n_in, n_out) in features.items():
            if module == 'encoder1':
                features[module] = (
                    FieldType(self.gspace, n_in*[self.gspace.trivial_repr]),
                    FieldType(self.gspace, n_out*list(self.gspace.irreps.values()))
                )
            else:
                features[module] = (
                    FieldType(self.gspace, n_in*list(self.gspace.irreps.values())),
                    FieldType(self.gspace, n_out*list(self.gspace.irreps.values()))
                )
        return features

    def get_encoder(self, name):
        feat_type_in, feat_type_out = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': R2Conv(feat_type_in, feat_type_out, kernel_size=3, stride=2, padding=1, bias=False),
            f'{name}-bn1': GNormBatchNorm(feat_type_out),
            f'{name}-relu1': NormNonLinearity(feat_type_out, bias=False),
            f'{name}-conv2': R2Conv(feat_type_out, feat_type_out, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': GNormBatchNorm(feat_type_out),
            f'{name}-relu2': NormNonLinearity(feat_type_out, bias=False)
        }))
    
    def get_decoder(self, name):
        feat_type_in, feat_type_out = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': R2ConvTransposed(feat_type_in, feat_type_out, kernel_size=3, stride=2, output_padding=1, bias=False),
            f'{name}-bn1': GNormBatchNorm(feat_type_out),
            f'{name}-relu1': NormNonLinearity(feat_type_out, bias=False),
            f'{name}-conv2': R2Conv(feat_type_out, feat_type_out, kernel_size=3, bias=False),
            f'{name}-bn2': GNormBatchNorm(feat_type_out),
            f'{name}-relu2': NormNonLinearity(feat_type_out, bias=False),
        }))
    
    def get_bottleneck(self, name='bottleneck'):
        feat_type, _ = self.features[name]
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': GNormBatchNorm(feat_type),
            f'{name}-relu1': NormNonLinearity(feat_type, bias=False),
            f'{name}-conv2': R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': GNormBatchNorm(feat_type),
            f'{name}-relu2': NormNonLinearity(feat_type, bias=False),
        }))
    
    def get_head(self, out_channels):
        return InvariantHead(self.features['decoder4'][1], out_channels, pool_type='norm')