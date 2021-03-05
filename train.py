import argparse
from models import C4UNet, UNet, HarmonicUNet, SteerableCNN
from e2cnn import gspaces
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import RetinalDataModule

def get_model(config):
    if config['model'] == 'discrete':
        if config['group'] == 'circle':
            gspace = gspaces.Rot2dOnR2(config['N'])
        elif config['group'] == 'dihedral':
            gspace = gspaces.FlipRot2dOnR2(config['N'])
        else:
            raise ValueError('Discrete group argument "group" must be one of ["circle", "dihedral"]')
        return C4UNet(gspace, config['in_channels'], config['out_channels'], n_features=config['n_features'], lr=config['lr'])
    elif config['model'] == 'standard':
        return UNet(config['in_channels'], config['out_channels'], n_features=config['n_features'], lr=config['lr'])
    elif config['model'] == 'harmonic':
        gspace = gspaces.Rot2dOnR2(-1, maximum_frequency=config['max_freq'])
        return HarmonicUNet(gspace, config['in_channels'], config['out_channels'], n_features=config['n_features'], lr=config['lr'])
    elif config['model'] == 'steerable':
        gspace = gspaces.Rot2dOnR2(-1, maximum_frequency=config['max_freq'])
        return SteerableCNN(gspace,
                        config['in_channels'],
                        config['out_channels'],
                        n_blocks=config['n_blocks'],
                        n_features=config['n_features'],
                        irrep_type=config['irrep_type'],
                        lr=config['lr'])
    else:
        raise ValueError(f'Unsupported model type: {config["model"]}')

def get_exp_name(config):
    return f"{config['model']}_{config['n_features']}_{config['lr']}_{config['batch_size']}" 

def train(config):
    rdm = RetinalDataModule()

    model = get_model(config)
    logger = TensorBoardLogger('logs', name=get_exp_name(config), default_hp_metric=False)
    logger.log_hyperparams(config, {
        'train_acc': 0,
        'train_f1': 0,
        'train_loss': 0,
        'valid_acc': 0,
        'valid_f1': 0,
        'valid_loss': 0,
        })
    trainer = pl.Trainer(gpus=1, max_epochs=config['n_epochs'], logger=logger, callbacks=[
        EarlyStopping(monitor='valid_loss', patience=10, mode='min'),
        ModelCheckpoint(monitor='valid_loss')])
    trainer.fit(model, rdm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model")
    parser.add_argument('--model', type=str, default='discrete', help='Model type. One of [discrete, standard, harmonic, steerable]')
    parser.add_argument('--irrep_type', type=str, default='all', help='Steerable model irrep type. One of [all, even, odd]')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for the Adam optimizer')
    parser.add_argument('--N', default=4, type=int, help='N for circle and dihedral groups')
    parser.add_argument('--group', type=str, default='circle', help='Type of discrete group. One of [circle, dihedral]')
    parser.add_argument('--n_features', type=int, default=32, help='Number of features per block')
    parser.add_argument('--n_blocks', type=int, default=10, help='Number of conv blocks for steerable model')
    parser.add_argument('--max_freq',  type=int, default=2,help='Max frequency for steerable model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--image_dir_train', type=str, default='data/training/images', help='Path to training dataset')
    parser.add_argument('--label_dir_train', type=str, default='data/training/1st_manual', help='Path to training dataset manually segmented masks')
    parser.add_argument('--image_dir_valid', type=str, default='data/valid/images', help='Path to valid dataset')
    parser.add_argument('--label_dir_valid', type=str, default='data/valid/1st_manual', help='Path to valid dataset manually segmented masks')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    config = parser.parse_args()
    config = vars(config)

    train(config)


