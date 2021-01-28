from dataset import RetinalDataset
from models import C4UNet, UNet
from torch.utils.data import DataLoader
from e2cnn import gspaces
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment, HyperOptArgumentParser

def get_model(config):
    if config.model == 'discrete':
        r2_act = gspaces.Rot2dOnR2(8)
        model = C4UNet(r2_act, config.in_channels, config.out_channels, n_features=config.n_features)
        model.lr = config.learning_rate
        return model
    elif config.model == 'standard':
        model = UNet(config.in_channels, config.out_channels, n_features=config.n_features)
        model.lr = config.learning_rate
        return model
    else:
        raise ValueError(f'Unsupported model type: {config.model}')

def train(config):
    ds = RetinalDataset(config.image_dir, config.label_dir)
    train_loader = DataLoader(ds, batch_size=config.bs, shuffle=True, num_workers=12, pin_memory=True)

    model = get_model(config)
    checkpoint_cb = ModelCheckpoint(monitor='train_loss')
    trainer = pl.Trainer(gpus=1, max_epochs=config.n_epochs, callbacks=[checkpoint_cb])
    # trainer = pl.Trainer(auto_lr_find=True, gpus=1, callbacks=[checkpoint_cb])
    # trainer.tune(model, train_loader)
    trainer.fit(model, train_loader)

def grid_search(config):
    ds = RetinalDataset(config.image_dir, config.label_dir)

    exp = Experiment(name=f'{config.model}-{config.n_features}-{config.bs}-{config.learning_rate}',
                    debug=False, save_dir='test_tube')
    for hparam in config.trials(10):
        model = get_model(hparam)
        train_loader = DataLoader(ds, batch_size=hparam.bs, shuffle=True, num_workers=12, pin_memory=True)
        trainer = pl.Trainer(gpus=1, max_epochs=10)
        trainer.fit(model, train_loader)
        exp.log(trainer.logged_metrics)

    exp.save()

if __name__ == '__main__':
    parser = HyperOptArgumentParser(description="Train a U-Net model", strategy='grid_search')
    parser.add_argument('--grid_search', default=False, help='Grid search')
    parser.add_argument('--model', type=str, default='discrete', help='Model type. One of [discrete, standard]')
    parser.opt_range('--learning_rate', default=1e-3, type=float, low=1e-6, high=1e-1, nb_samples=10, tunable=True)
    parser.opt_list('--n_features', default=32, type=int, tunable=True, options=[2, 8, 16])
    parser.opt_list('--bs', type=int, default=4, options=[4, 8, 16, 32], help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--image_dir', type=str, default='training/images', help='Path to training dataset')
    parser.add_argument('--label_dir', type=str, default='training/1st_manual', help='Path to training dataset manually segmented masks')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    config = parser.parse_args()

    if config.grid_search:
        grid_search(config)
    else:
        train(config)


