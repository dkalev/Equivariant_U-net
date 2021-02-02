import torch
import torch.nn as nn
from collections import OrderedDict
from e2cnn.nn import GeometricTensor, GroupPooling, NormPool
import e2cnn
import pytorch_lightning as pl
from loss import DiceLoss
from typing import Iterable, Union

class BaseUNet(pl.LightningModule):

    def __init__(self, in_channels:int, out_channels:int, *args, n_features:int=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = DiceLoss()
        self.lr = 1e-3

        self.features = self.get_features(in_channels, n_features)

        self.encoder1 = self.get_encoder('encoder1')
        self.encoder2 = self.get_encoder('encoder2')
        self.encoder3 = self.get_encoder('encoder3')
        self.encoder4 = self.get_encoder('encoder4')

        self.bottleneck = self.get_bottleneck()

        self.decoder1 = self.get_decoder('decoder1')
        self.decoder2 = self.get_decoder('decoder2')
        self.decoder3 = self.get_decoder('decoder3')
        self.decoder4 = self.get_decoder('decoder4')

        self.head = self.get_head(out_channels)

    @staticmethod 
    def cat(tensors:Iterable[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        return torch.cat(tensors, *args, **kwargs)

    def get_head(self, out_channels:str):
        raise NotImplementedError("Implement this method")

    def get_encoder(self, name:str):
        raise NotImplementedError("Implement this method")

    def get_bottleneck(self, name:str='bottleneck'):
        raise NotImplementedError("Implement this method")

    def get_decoder(self, name:str):
        raise NotImplementedError("Implement this method")

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

        x = self.decoder1(self.cat([x, enc4], dim=1))
        if verbose: print('dec1', x.shape)
        x = self.decoder2(self.cat([x, enc3], dim=1))
        if verbose: print('dec2', x.shape)
        x = self.decoder3(self.cat([x, enc2], dim=1))
        if verbose: print('dec3', x.shape)
        x = self.decoder4(self.cat([x, enc1], dim=1))
        if verbose: print('dec4', x.shape)

        x = self.head(x)
        if verbose: print('head', x.shape)
        return x

    def get_features(self, in_channels:int, n_features:int) -> OrderedDict:
        n_features_down = [n_features * 2**i for i in range(0,4)]
        # in_channels is multiplied by 2 because encoder outputs are concatenated as well (see forward method)
        n_features_up = [(2*n, max(1,n//2)) for n in reversed(n_features_down)]

        features_down = [( 'encoder1', ( in_channels, n_features) )] + [
            ( f'encoder{i}', ( n_in, n_out) ) 
            for i, (n_in, n_out) in enumerate(zip(n_features_down, n_features_down[1:]), start=2)]

        feats_bottleneck = [( 'bottleneck', ( n_features_down[-1], n_features_down[-1]))]

        features_up = [( f'decoder{i}', ( n_in, n_out)) for i, (n_in, n_out) in enumerate(n_features_up, start=1)]

        return OrderedDict(features_down + feats_bottleneck + features_up)
        
    def training_step(self, batch:Union[torch.Tensor, Iterable[torch.Tensor]], batch_idx:int) -> torch.Tensor:
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch:Union[torch.Tensor, Iterable[torch.Tensor]], batch_idx:int):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log('valid_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class InvariantHead(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, *args, pool_type:str='group', **kwargs):
        super().__init__(*args, **kwargs)

        if pool_type == 'group':
            self.pool = GroupPooling(in_channels)
        elif pool_type == 'norm':
            self.pool = NormPool(in_channels)
        else:
            raise ValueError(f'Unsupported pooling: pool_type={pool_type}')

        out = self.pool.out_type.size
        self.final = nn.Sequential(OrderedDict({
            f'head-conv': nn.Conv2d(out, out_channels, kernel_size=3, padding=1, bias=False),
            f'final-act': nn.Sigmoid()
        }))
    
    def forward(self, x:GeometricTensor, verbose:bool=False) -> torch.Tensor:
        x = self.pool(x)
        x = x.tensor
        if verbose: print('pool', x.shape)
        x = self.final(x)
        if verbose: print('head', x.shape)
        return x

class BaseEquivUNet(BaseUNet):

    def __init__(self, gspace:e2cnn.gspaces, in_channels:int, out_channels:int, *args, **kwargs):
        self.gspace = gspace
        super().__init__(in_channels, out_channels, *args, **kwargs)

    def get_head(self, out_channels:int) -> InvariantHead:
        return InvariantHead(self.features['decoder4'][1], out_channels)

    @staticmethod
    def cat(gtensors:Iterable[GeometricTensor], *args, **kwargs) -> GeometricTensor:
        tensors = [ t.tensor for t in gtensors]
        tensors_cat = torch.cat(tensors, *args, **kwargs)
        feature_type = sum([t.type for t in gtensors[1:]], start=gtensors[0].type)
        return GeometricTensor(tensors_cat, feature_type)

    def forward(self, x:torch.Tensor, verbose:bool=False) -> torch.Tensor:
        x = GeometricTensor(x, self.features['encoder1'][0])
        return super().forward(x)