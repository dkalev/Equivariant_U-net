import torch
from collections import OrderedDict
import pytorch_lightning as pl
from loss import DiceLoss

class BaseUNet(pl.LightningModule):

    def __init__(self, in_channels, out_channels, n_features=64):
        super().__init__()
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
    def cat(tensors, *args, **kwargs):
        return torch.cat(tensors, *args, **kwargs)

    def get_head(self, out_channels):
        raise NotImplementedError("Implement this method")

    def get_encoder(self, name):
        raise NotImplementedError("Implement this method")

    def get_bottleneck(self, name='bottleneck'):
        raise NotImplementedError("Implement this method")

    def get_decoder(self, name):
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

    def get_features(self, in_channels, n_features):
        n_features_down = [n_features * 2**i for i in range(0,4)]
        # in_channels is multiplied by 2 because encoder outputs are concatenated as well (see forward method)
        n_features_up = [(2*n, max(1,n//2)) for n in reversed(n_features_down)]

        features_down = [( 'encoder1', ( in_channels, n_features) )] + [
            ( f'encoder{i}', ( n_in, n_out) ) 
            for i, (n_in, n_out) in enumerate(zip(n_features_down, n_features_down[1:]), start=2)]

        feats_bottleneck = [( 'bottleneck', ( n_features_down[-1], n_features_down[-1]))]

        features_up = [( f'decoder{i}', ( n_in, n_out)) for i, (n_in, n_out) in enumerate(n_features_up, start=1)]

        return OrderedDict(features_down + feats_bottleneck + features_up)
        
    def training_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
