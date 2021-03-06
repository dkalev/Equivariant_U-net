import argparse
from models import C4UNet, UNet, HarmonicUNet, SteerableCNN
from e2cnn import gspaces
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import RetinalDataModule
from parse_hyperparams import parse_hparams

def get_model(hparams):
    if hparams['model'] == 'discrete':
        return C4UNet(hparams['in_channels'], hparams['out_channels'],
                        group_type=hparams['group'],
                        N=hparams['N'],
                        n_features=hparams['n_features'],
                        loss_type=hparams['loss_func'],
                        lr=hparams['lr'])
    elif hparams['model'] == 'standard':
        return UNet(hparams['in_channels'], hparams['out_channels'], n_features=hparams['n_features'], loss_type=hparams['loss_func'], lr=hparams['lr'])
    elif hparams['model'] == 'harmonic':
        return HarmonicUNet(hparams['in_channels'],
                            hparams['out_channels'],
                            n_features=hparams['n_features'],
                            group_type=hparams['group'],
                            max_freq=hparams['max_freq'],
                            loss_type=hparams['loss_func'],
                            lr=hparams['lr'])
    elif hparams['model'] == 'steerable':
        gspace = gspaces.Rot2dOnR2(-1, maximum_frequency=hparams['max_freq'])
        return SteerableCNN(gspace,
                        hparams['in_channels'],
                        hparams['out_channels'],
                        n_blocks=hparams['n_blocks'],
                        n_features=hparams['n_features'],
                        irrep_type=hparams['irrep_type'],
                        loss_type=hparams['loss_func'],
                        lr=hparams['lr'])
    else:
        raise ValueError(f'Unsupported model type: {hparams["model"]}')

def get_exp_name(hparams):
    return ('_').join([f'{key}={val}' for key, val in hparams.items()])

def train(hparams):
    rdm = RetinalDataModule()

    model = get_model(hparams)
    logger = TensorBoardLogger('logs', name=get_exp_name(hparams), default_hp_metric=False)
    # log hparams to tensorboard
    logger.log_hyperparams(hparams, {
        'train_acc': 0,
        'train_f1': 0,
        'train_loss': 0,
        'valid_acc': 0,
        'valid_f1': 0,
        'valid_loss': 0,
        })
    trainer = pl.Trainer(gpus=1,
                        min_epochs=50,
                        max_epochs=hparams['n_epochs'],
                        logger=logger,
                        callbacks=[
                            EarlyStopping(monitor='valid_loss', patience=10, mode='min'),
                            ModelCheckpoint(monitor='valid_loss')
                        ])
    trainer.fit(model, rdm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model")
    hparams = parse_hparams(parser)
    train(hparams)


