from models import C4UNet, UNet, HarmonicUNet
from e2cnn import gspaces
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import Experiment, HyperOptArgumentParser
from dataset import RetinalDataModule
from torch.nn import BCEWithLogitsLoss

def get_model(config):
    if config.model == 'discrete':
        r2_act = gspaces.Rot2dOnR2(8)
        return C4UNet(r2_act, config.in_channels, config.out_channels, n_features=config.n_features, lr=config.lr)
    elif config.model == 'standard':
        return UNet(config.in_channels, config.out_channels, n_features=config.n_features, lr=config.lr)
    elif config.model == 'harmonic':
        r2_act = gspaces.Rot2dOnR2(-1, maximum_frequency=1)
        return HarmonicUNet(r2_act, config.in_channels, config.out_channels, n_features=config.n_features, lr=config.lr)
    else:
        raise ValueError(f'Unsupported model type: {config.model}')

def get_exp_name(config):
    return f'{config.model}_{config.n_features}_{config.lr}_{config.batch_size}' 

def train(config):
    rdm = RetinalDataModule()

    model = get_model(config)
    model.crit = BCEWithLogitsLoss()
    checkpoint_cb = ModelCheckpoint(monitor='valid_loss')
    logger = TensorBoardLogger('logs', name=get_exp_name(config))
    trainer = pl.Trainer(gpus=1, max_epochs=config.n_epochs, logger=logger, callbacks=[checkpoint_cb])
    trainer.fit(model, rdm)

def grid_search(config):
    rdm = RetinalDataModule()
    rdm.prepare_data()
    rdm.setup()

    exp = Experiment(name=get_exp_name(config), debug=False, save_dir='test_tube')
    for hparam in config.trials(10):
        model = get_model(hparam)
        train_loader = rdm.train_dataloader(batch_size=config.batch_size)
        valid_loader = rdm.val_dataloader(batch_size=config.batch_size)
        trainer = pl.Trainer(gpus=1, max_epochs=10)
        trainer.fit(model, train_loader, valid_loader)
        exp.log(trainer.logged_metrics)

    exp.save()

if __name__ == '__main__':
    parser = HyperOptArgumentParser(description="Train a U-Net model", strategy='grid_search')
    parser.add_argument('--grid_search', default=False, help='Grid search')
    parser.add_argument('--model', type=str, default='discrete', help='Model type. One of [discrete, standard, harmonic]')
    parser.opt_range('--lr', default=1e-3, type=float, low=1e-6, high=1e-1, nb_samples=10, tunable=True)
    parser.opt_list('--n_features', default=32, type=int, tunable=True, options=[8, 16, 32, 64])
    parser.opt_list('--batch_size', type=int, default=4, options=[4, 8, 16], help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--image_dir_train', type=str, default='data/training/images', help='Path to training dataset')
    parser.add_argument('--label_dir_train', type=str, default='data/training/1st_manual', help='Path to training dataset manually segmented masks')
    parser.add_argument('--image_dir_valid', type=str, default='data/valid/images', help='Path to valid dataset')
    parser.add_argument('--label_dir_valid', type=str, default='data/valid/1st_manual', help='Path to valid dataset manually segmented masks')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    config = parser.parse_args()

    if config.grid_search:
        grid_search(config)
    else:
        train(config)


