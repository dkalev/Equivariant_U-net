from collections import OrderedDict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import DiceLoss

from e2cnn.group import directsum
from e2cnn.nn import GeometricTensor, FieldType, R2Conv, GNormBatchNorm, NormNonLinearity, NormPool

class SteerableCNN(pl.LightningModule):
    def __init__(self, gspace, in_channels, out_channels, n_blocks=4, n_features=20, irrep_type='all', lr=1e-3):
        super().__init__()
        self.gspace = gspace
        self.crit = DiceLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1()
        self.lr = lr
        
        self.in_type = FieldType(gspace, in_channels*[gspace.trivial_repr])
        out_type = FieldType(gspace, n_features*self.get_irreps(irrep_type))
        
        norm_pool = NormPool(out_type)
        
        self.model = nn.Sequential()
        self.model.add_module('input-block', nn.Sequential(OrderedDict({
            'conv1': R2Conv(self.in_type, out_type, kernel_size=3, padding=1),
            'bn1': GNormBatchNorm(out_type),
            'relu1': NormNonLinearity(out_type),
        })))
        for i in range(n_blocks):
            self.model.add_module(f'block{i+1}', self.get_conv_block(out_type, i+1))
        self.model.add_module('norm-pool', norm_pool)
        
        self.head = nn.Conv2d(norm_pool.out_type.size, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = GeometricTensor(x, self.in_type)
        x = self.model(x)
        x = x.tensor
        x = self.head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss, split='valid')
    
    def log_metrics(self, preds, targs, loss, split='train'):
        self.log(f'{split}_loss', loss)
        self.log(f'{split}_acc', self.accuracy(preds, targs))
        self.log(f'{split}_f1', self.f1(preds, targs))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    @staticmethod
    def get_conv_block(feat_type, idx):
        return nn.Sequential(OrderedDict({
            f'conv{idx}': R2Conv(feat_type, feat_type, kernel_size=3, padding=1),
            f'bn{idx}': GNormBatchNorm(feat_type),
            f'relu{idx}': NormNonLinearity(feat_type),
        }))
    
    def get_irreps(self, irrep_type='all'):
        if irrep_type == 'all':
            irreps = self.gspace.irreps
        elif irrep_type == 'even':
            even_keys = list(self.gspace.irreps.keys())[::2]
            irreps = { k:v for k,v in self.gspace.irreps.items() if k in even_keys }
        elif irrep_type == 'odd':
            odd_keys = list(self.gspace.irreps.keys())[1::2]
            irreps = { k:v for k,v in self.gspace.irreps.items() if k in odd_keys }
        else:
            raise ValueError(f'irrep_type must be one of ["all", "even", "odd"], {irrep_type} given')
        irreps = list(irreps.values())
        return [directsum(irreps)]
